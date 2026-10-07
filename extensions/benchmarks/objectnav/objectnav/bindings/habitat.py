"""ObjectNav task binding for the Habitat simulator plugins.

Runs inside the habitat conda environment; imports habitat lazily so that
`describe`-level tooling elsewhere in this repo never needs ML dependencies.

Contract mapping:
- habitat discrete task actions stop/move_forward/turn_left/turn_right map to
  the platform codecs stop/primitive(forward 0.25 m, protocol-selected turn);
  other increments are rejected as method policy violations, not truncated.
- the goal is an object category handed to the method through EpisodeContext
  (Goal kind `object_category`); the objectgoal index sensor stays private to
  the environment worker.
- rgb/depth observations travel as lossless base64 tensor envelopes
  (nav_eval.tensorcode); depth is unnormalised metres.
- `sim_time_s` is null; `control_tick` counts actions for this discrete task.
- evaluator evidence (habitat-objectnav-evidence/1) records per-step distance
  to goal as produced by habitat's official DistanceToGoal measure
  (VIEW_POINTS geodesic distance, the same quantity upstream success/SPL use),
  the GT trajectory, and habitat's own reference measurements so the offline
  evaluator can be cross-checked (benchmark parity layer).
- `oracle_follower_action` exposes a privileged greedy geodesic follower that
  targets the closest goal view point, for pipeline validation runs only; the
  rollout manifest flags every such run.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from nav_eval.contracts import (
    SCHEMA_VERSION,
    ContractError,
    EpisodeContext,
    Goal,
    PolicyViolation,
    SensorSpec,
    validate_action,
)
from extensions.benchmarks.shared.dispatch import SimulatorDispatch
from nav_eval.tensorcode import encode_array

EVIDENCE_SCHEMA = "habitat-objectnav-evidence/1"
FORWARD_STEP_M = 0.25
TURN_ANGLE_RAD = math.radians(30.0)
MAX_EPISODE_STEPS = 500

# benchmark_id -> (dataset family, base task yaml, dataset version dir, split)
# Only families with an official upstream episode release are integrated:
# mp3d v1 and hm3d v2 (habitat-challenge edition). Gibson ObjectNav episodes
# have no official download and are therefore not shipped; see docs/benchmarks.md.
BENCHMARK_SPLITS = {
    "objectnav_mp3d_val": ("mp3d", "objectnav_mp3d.yaml", "v1", "val"),
    "objectnav_hm3d_val": ("hm3d", "objectnav_hm3d.yaml", "v2", "val"),
}
SCENE_PREFIX = "data/scene_datasets/"


class HabitatObjectNavBenchmarkService:
    def __init__(self, data_root, benchmark_id="objectnav_mp3d_val", episode_limit=None,
                 gpu_device_id=0, sensors=("rgb", "depth"), success_distance_m=0.1,
                 width=640, height=480, hfov=79, max_depth_m=10.0, enable_oracle=True,
                 simulator_version=None, backend=None, episode_ids=None,
                 turn_angle_rad=TURN_ANGLE_RAD):
        if benchmark_id not in BENCHMARK_SPLITS:
            raise ValueError(f"unknown benchmark; choose from {sorted(BENCHMARK_SPLITS)}")
        unknown = set(sensors) - {"rgb", "depth"}
        if unknown:
            raise ValueError(f"unsupported sensors: {sorted(unknown)}")
        if not sensors:
            raise ValueError("observation policy requires at least one sensor")
        self.benchmark_id = benchmark_id
        self.family, self._task_yaml, self._dataset_version, self.split = BENCHMARK_SPLITS[benchmark_id]
        self.data_root = Path(data_root).resolve()
        self.episode_limit = episode_limit
        self.gpu_device_id = gpu_device_id
        self.sensors = tuple(sensors)
        self.success_distance_m = success_distance_m
        if turn_angle_rad not in (math.radians(15), math.radians(30)):
            raise ValueError("Habitat ObjectNav supports 15 or 30 degree turns")
        self.turn_angle_rad = turn_angle_rad
        self.width, self.height, self.hfov = int(width), int(height), int(hfov)
        self.max_depth_m = max_depth_m
        self.enable_oracle = enable_oracle
        # Declared by the launcher (which knows which conda env it used); verified
        # against habitat.__version__ at env init so a mismatch cannot pass silently.
        self.declared_version = simulator_version
        self.actual_version = None
        from extensions.simulators.habitat.backend import HabitatBackend
        self.backend = backend or HabitatBackend({"version": simulator_version or "0.3.0"})
        self.requested_ids = episode_ids
        self.env = None
        self.follower = None
        self.session = None  # single active session; the dev transport is per-attempt
        # habitat-sim's GL context must be created and used on one thread; the
        # HTTP transport serves requests on arbitrary threads, so every op is
        # dispatched onto this dedicated simulation thread.
        self._dispatcher = SimulatorDispatch(self._handle)

    def call(self, operation, payload):
        return self._dispatcher.call(operation, payload)

    # -- dataset -----------------------------------------------------------
    @property
    def _dataset_path(self):
        return (self.data_root / "datasets" / "objectnav" / self.family
                / self._dataset_version / self.split / f"{self.split}.json.gz")

    def _episode_manifest(self):
        from extensions.benchmarks.objectnav.objectnav.dataset import select_episodes
        if not hasattr(self, "_selected_episodes"):
            self._selected_episodes = select_episodes(self._dataset_path, self.data_root,
                self.episode_limit, self.requested_ids)
        return list(self._selected_episodes)

    # -- habitat bootstrap --------------------------------------------------
    def _ensure_env(self):
        if self.env is not None:
            return
        import habitat  # heavy import stays inside the worker process

        actual = str(habitat.__version__)
        if self.declared_version and actual != self.declared_version:
            raise ContractError(
                f"launcher declared habitat {self.declared_version} but worker runs {actual}")
        self.actual_version = actual

        self._episodes = self._episode_manifest()
        if not self._episodes:
            raise ContractError("no episodes with locally available scenes; copy scene dirs first")
        self._episode_order = {str(e["episode_id"]): index for index, e in enumerate(self._episodes)}

        overrides = [
            f"habitat.dataset.data_path={self._dataset_path}",
            f"habitat.dataset.scenes_dir={self.data_root / 'scene_datasets'}",
            f"habitat.dataset.split={self.split}",
            "habitat.dataset.type=ObjectNav-v1",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={self.gpu_device_id}",
            "habitat.simulator.forward_step_size=0.25",
            f"habitat.simulator.turn_angle={round(math.degrees(self.turn_angle_rad))}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={self.width}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={self.height}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov={self.hfov}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width={self.width}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height={self.height}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov={self.hfov}",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth={self.max_depth_m}",
            "habitat.environment.max_episode_steps=500",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.max_scene_repeat_steps=50000",
            f"habitat.task.measurements.success.success_distance={self.success_distance_m}",
        ]
        config = habitat.get_config(f"benchmark/nav/objectnav/{self._task_yaml}", overrides)
        self.env = self.backend.initialize(config)

    def _agent_position(self):
        return self.backend.position()

    def _distance_to_goal(self):
        """Habitat's official DistanceToGoal measure (VIEW_POINTS geodesic)."""
        value = self.env.get_metrics()["distance_to_goal"]
        try:
            import numpy as np
            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {"distance_m": None, "fallback": "unavailable_measure"}
        return {"distance_m": float(value), "fallback": None}

    def _goal_category(self):
        episode = self.env.current_episode
        return episode.goals[0].object_category if episode.goals else episode.object_category

    def _goal_position(self):
        goals = self.env.current_episode.goals
        if not goals:
            raise ContractError("episode has no goal")
        return [float(v) for v in goals[0].position]

    # -- observation policy -------------------------------------------------
    def _observation(self, obs, sequence):
        payloads, specs = {}, {}
        if "rgb" in self.sensors:
            payloads["rgb"] = encode_array(obs["rgb"])
            specs["rgb"] = asdict(SensorSpec("rgb", "uint8", (self.height, self.width, 3), "srgb",
                                             "camera_optical", "rendered",
                                             {"hfov_deg": self.hfov, "width": self.width,
                                              "height": self.height}))
        if "depth" in self.sensors:
            payloads["depth"] = encode_array(obs["depth"])
            specs["depth"] = asdict(SensorSpec("depth", "float32", (self.height, self.width, 1), "m",
                                               "camera_optical", "rendered",
                                               {"hfov_deg": self.hfov, "width": self.width,
                                                "height": self.height, "min_depth_m": 0.0,
                                                "max_depth_m": self.max_depth_m,
                                                "convention": "z_depth_along_optical_axis"}))
        return {"episode_id": self.session["episode_id"], "sequence": sequence,
                "sim_time_s": None, "control_tick": self.session["steps"], "sensors": payloads,
                "sensor_specs": specs}

    # -- action mapping -----------------------------------------------------
    def _to_habitat_action(self, action):
        if action.kind == "stop":
            return "stop"
        name, amount = action.values["name"], action.values["amount"]
        if name == "forward" and abs(amount - FORWARD_STEP_M) > 1e-3:
            raise PolicyViolation(f"forward amount must be {FORWARD_STEP_M} m on this benchmark")
        if name in {"left", "right"} and abs(amount - self.turn_angle_rad) > 1e-3:
            raise PolicyViolation(f"turn amount must be {math.degrees(self.turn_angle_rad):g} degrees "
                                  f"({self.turn_angle_rad:.6f} rad) on this benchmark")
        return {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}[name]

    # -- service surface ----------------------------------------------------
    def _handle(self, operation, payload):
        if operation == "prepare":
            ids = [str(e["episode_id"]) for e in self._episode_manifest()]
            if not ids:
                raise ContractError("empty episode selection")
            self._handle("reset", {"episode_id": ids[0], "session_id": "preflight", "seed": 0})
            self._handle("close_episode", {"session_id": "preflight"})
            return {}
        if operation == "shutdown":
            self.backend.close()
            return {}
        if operation == "describe":
            version = getattr(self, "actual_version", None) or self.declared_version or "0.3.0"
            return {
                "schema_version": SCHEMA_VERSION, "role": "benchmark",
                "id": f"habitat_{self.benchmark_id}", "benchmark_kind": "object_goal_navigation",
                "benchmark_id": self.benchmark_id,
                "dataset_family": f"objectnav/{self.family}", "split": self.split,
                "simulator": f"habitat-sim/habitat-lab {version}",
                "simulator_version": version, "real": True,
                "accepts_actions": ["primitive", "stop"],
                "offers_sensors": list(self.sensors),
                "sensor_geometry": {"width": self.width, "height": self.height, "hfov_deg": self.hfov},
                "coordinate_frame": "habitat_world_y_up",
                "clock": "discrete_control_ticks",
                "action_semantics": {"forward_m": FORWARD_STEP_M, "turn_rad": self.turn_angle_rad},
                "capture_schema": EVIDENCE_SCHEMA,
                "success_distance_m": self.success_distance_m,
                "goal": {"kind": "object_category"},
                "oracle_follower_available": bool(self.enable_oracle),
                "note": "Discrete task: sim_time_s is null; control_tick counts actions. "
                        "Goal distances use habitat's DistanceToGoal (view-point geodesic).",
            }
        if operation == "episodes":
            # Group by scene (alphabetical) to minimise simulator reconfigures;
            # the selection set itself is the first N episodes in habitat order.
            manifest = self._episode_manifest()
            manifest.sort(key=lambda e: (e["scene_id"], str(e["episode_id"])))
            return [str(e["episode_id"]) for e in manifest]

        if operation == "oracle_follower_action":
            if not self.enable_oracle:
                raise ContractError("oracle follower disabled")
            state = self._active_session(payload)
            follower = self._ensure_follower()
            action = follower.next_action_along(self._follower_target())
            if action is None:
                raise ContractError("oracle follower could not find a path")
            mapping = {
                "stop": {"kind": "stop", "values": {}},
                "move_forward": {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
                "turn_left": {"kind": "primitive", "values": {"name": "left", "amount": self.turn_angle_rad}},
                "turn_right": {"kind": "primitive", "values": {"name": "right", "amount": self.turn_angle_rad}},
            }
            if action not in mapping:
                raise ContractError(f"oracle follower returned unknown action: {action}")
            return {"action": mapping[action]}

        session_id = payload.get("session_id")
        if operation == "reset":
            self._ensure_env()
            if self.session is not None:
                raise ContractError("active session already exists")
            episode_id = str(payload["episode_id"])
            if episode_id not in self._episode_order:
                raise ContractError(f"episode {episode_id} not in the frozen local selection")
            target = self._select_episode(episode_id)
            self.follower = None
            obs = self.backend.reset(target, int(payload.get("seed", 0)))
            try:
                category = self._goal_category()
                goal = self._goal_position()
                start = self._agent_position()
                first_distance = self._distance_to_goal()
            except Exception:
                self.session = None
                raise
            self.session = {
                "session_id": session_id, "episode_id": episode_id,
                "steps": 0, "stopped": False, "truncated": False,
                "positions": [start], "goal_distances": [first_distance],
                "goal": goal, "start": start, "category": category,
            }
            context = EpisodeContext(
                episode_id, Goal("object_category", category), "habitat_objectnav_v1",
                int(payload.get("seed", 0)),
            )
            return {"context": asdict(context), "observation": self._observation(obs, 0)}

        state = self._active_session(payload)
        if operation == "step":
            if state["stopped"] or state["truncated"]:
                raise ContractError("episode already ended")
            if payload["expected_sequence"] != state["steps"]:
                raise ContractError("stale or duplicate step: episode must be restarted")
            action = validate_action(payload["action"])
            if action.kind not in {"primitive", "stop"}:
                raise PolicyViolation(f"{action.kind} unsupported by this benchmark")
            habitat_action = self._to_habitat_action(action)
            obs = self.backend.step(habitat_action)
            state["steps"] += 1
            position = self._agent_position()
            state["positions"].append(position)
            state["goal_distances"].append(self._distance_to_goal())
            if action.kind == "stop":
                # Execute STOP through habitat so the upstream success/SPL
                # reference measures (end_on_success) reflect the stop claim.
                state["stopped"] = True
            state["truncated"] = (self.env.episode_over or state["steps"] >= MAX_EPISODE_STEPS) and not state["stopped"]
            return {"observation": self._observation(obs, state["steps"]),
                    "terminated": state["stopped"], "truncated": state["truncated"]}

        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            positions = state["positions"]
            path_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
            goal = state["goal"]
            final = math.dist(positions[-1], goal)
            distances = [d["distance_m"] for d in state["goal_distances"]]
            min_distance = min((d for d in distances if d is not None), default=None)
            record = {
                "episode_id": state["episode_id"], "status": "completed",
                "termination": "stop" if state["stopped"] else "budget",
                "steps": state["steps"], "control_tick": state["steps"], "sim_time_s": None,
                "sim_clock": "discrete_control_ticks",
            }
            evidence = {
                "schema": EVIDENCE_SCHEMA,
                "reference": {
                    "scene_id": self.env.current_episode.scene_id,
                    "goal_object_category": state["category"],
                    "goal_xyz_m": goal, "start_xyz_m": state["start"],
                    "geodesic_start_to_goal_m": distances[0],
                    "success_radius_m": self.success_distance_m,
                    "distance_definition": "habitat_distance_to_goal_view_points",
                },
                "trajectory": {
                    "positions_xyz_m": positions, "control_ticks": list(range(len(positions))),
                    "goal_distances_m": distances,
                    "goal_distance_fallbacks": [d["fallback"] for d in state["goal_distances"]],
                    "path_length_m": path_length,
                },
                "final_goal_distance_m": final,
                "min_goal_distance_m": min_distance,
                "benchmark_reference_metrics": self._reference_metrics(),
            }
            return {"record": record, "evidence": evidence}

        if operation == "close_episode":
            self.session = None
            self.follower = None
            return {}
        raise ContractError(f"unsupported benchmark operation: {operation}")

    def _active_session(self, payload):
        if self.session is None or self.session["session_id"] != payload.get("session_id"):
            raise ContractError("reset session first")
        return self.session

    def _select_episode(self, episode_id):
        """Pin the exact episode via habitat's public currentepisode setter.

        The episode iterator walks scene groups in alphabetical order and cannot
        go back, so iterating resets is both slow and unable to reach arbitrary
        ids; the setter disables iterator advancement for the next reset.
        """
        for episode in self.env.episodes:
            if str(episode.episode_id) == episode_id:
                return episode
        raise ContractError(f"episode {episode_id} not in the loaded dataset")

    def _follower_target(self):
        """Nearest goal view point by geodesic distance from the agent."""
        goal = self.env.current_episode.goals[0]
        agent = self._agent_position()
        best, best_distance = None, None
        for view in goal.view_points:
            candidate = [float(v) for v in view.agent_state.position]
            distance = self.backend.geodesic(agent, candidate)
            if distance is None:
                continue
            if best is None or distance < best_distance:
                best, best_distance = candidate, distance
        if best is None:
            raise ContractError("no reachable goal view point")
        return best

    def _ensure_follower(self):
        if self.session is None:
            raise ContractError("reset session first")
        if self.follower is None:
            from habitat_sim.nav import GreedyGeodesicFollower

            self.follower = GreedyGeodesicFollower(
                self.env.sim.pathfinder, self.env.sim.get_agent(0),
                goal_radius=0.5,
                stop_key="stop", forward_key="move_forward",
                left_key="turn_left", right_key="turn_right",
            )
        return self.follower

    def _reference_metrics(self):
        metrics = self.env.get_metrics()

        _DROP = object()

        def clean(value):
            # numpy scalars (np.float64/np.bool_) carry .shape == () and must be
            # unwrapped; real arrays/tensors are dropped, not silently coerced.
            try:
                import numpy as np

                if isinstance(value, np.generic):
                    return value.item()
            except ImportError:
                pass
            if isinstance(value, (int, float, bool)):
                return value
            if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
                return value.item()
            return _DROP

        keys = ("success", "spl", "soft_spl", "distance_to_goal")
        cleaned = {k: clean(v) for k, v in metrics.items() if k in keys}
        return {k: v for k, v in cleaned.items() if v is not _DROP}

    def prepare(self):
        self.call("prepare", {})

    def asset_files(self):
        files = {self._dataset_path}
        content_dir = self._dataset_path.parent / "content"
        if content_dir.is_dir():
            files.update(content_dir.glob("*.json.gz"))
        for episode in self._episode_manifest():
            scene = self.data_root / "scene_datasets" / episode["scene_id"].removeprefix(SCENE_PREFIX)
            files.add(scene)
            files.update(scene.parent.glob("*.navmesh"))
        return {str(path.relative_to(self.data_root)): path for path in sorted(files)}

    def runtime_identity(self):
        import habitat_sim
        return {"habitat_lab": self.backend.version, "habitat_sim": habitat_sim.__version__}

    def close(self):
        self._dispatcher.close()


def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]
    backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"])
    settings.update(config["observation"])
    settings["turn_angle_rad"] = config["capabilities"]["action_semantics"]["turn_rad"]
    return HabitatObjectNavBenchmarkService(backend=backend, **settings)
