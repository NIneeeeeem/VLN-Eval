"""R2R/RxR task binding for the Habitat simulator plugins.

Runs inside the habitat conda environment; imports habitat lazily so that
`describe`-level tooling elsewhere in this repo never needs ML dependencies.

Contract mapping:
- habitat discrete task actions stop/move_forward/turn_left/turn_right map to
  the platform codecs stop/primitive(forward 0.25 m, protocol-selected turn);
  other increments are rejected as method policy violations, not truncated.
- rgb/depth observations travel as lossless base64 tensor envelopes
  (nav_eval.tensorcode); depth is unnormalised metres.
- `sim_time_s` is null; `control_tick` counts actions for this discrete task.
- evaluator evidence (habitat-r2r-evidence/1) records the GT trajectory,
  per-step geodesic distance to goal and habitat's own reference measurements
  so the offline evaluator can be cross-checked (benchmark parity layer).
- `oracle_follower_action` exposes a privileged greedy geodesic follower for
  pipeline validation runs only; the rollout manifest flags every such run.
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

EVIDENCE_SCHEMA = "habitat-r2r-evidence/1"
FORWARD_STEP_M = 0.25
TURN_ANGLE_RAD = math.radians(15.0)
DEFAULT_MAX_NAVIGATION_STEPS = 500

# benchmark_id -> (dataset family, split, episode filename)
BENCHMARK_SPLITS = {
    "r2r_val_unseen": ("r2r", "val_unseen", "{split}.json.gz"),
    "r2r_val_seen": ("r2r", "val_seen", "{split}.json.gz"),
    "rxr_val_unseen": ("rxr", "val_unseen", "{split}_guide.json.gz"),
}
DATASET_TYPE = {"r2r": "R2RVLN-v1", "rxr": "RxR-VLN-CE-v1"}


class HabitatR2RBenchmarkService:
    def __init__(self, data_root, benchmark_id="r2r_val_unseen", episode_limit=None,
                 gpu_device_id=0, sensors=("rgb", "depth"), success_distance_m=3.0,
                 width=640, height=480, hfov=79, max_depth_m=10.0, enable_oracle=True,
                 languages=(), simulator_version=None, backend=None, episode_ids=None,
                 allow_tilt=False, turn_angle_rad=TURN_ANGLE_RAD,
                 max_navigation_steps=DEFAULT_MAX_NAVIGATION_STEPS, max_control_steps=None):
        if benchmark_id not in BENCHMARK_SPLITS:
            raise ValueError(f"unknown benchmark; choose from {sorted(BENCHMARK_SPLITS)}")
        unknown = set(sensors) - {"rgb", "depth", "pose"}
        if unknown:
            raise ValueError(f"unsupported sensors: {sorted(unknown)}")
        if not sensors:
            raise ValueError("observation policy requires at least one sensor")
        self.benchmark_id = benchmark_id
        self.dataset_family, self.split, self._episode_filename = BENCHMARK_SPLITS[benchmark_id]
        self.data_root = Path(data_root).resolve()
        self.episode_limit = episode_limit
        self.languages = frozenset(languages)
        self.gpu_device_id = gpu_device_id
        self.sensors = tuple(sensors)
        self.success_distance_m = success_distance_m
        if turn_angle_rad not in (math.radians(15), math.radians(30)):
            raise ValueError("Habitat VLN supports 15 or 30 degree turns")
        self.turn_angle_rad = turn_angle_rad
        # Camera tilt probes (look_up/look_down) are opt-in: methods that need a
        # pixel goal under the horizon (e.g. dual-system VLA) request them via
        # benchmark_settings.allow_tilt; the tilt granularity is 15 degrees and
        # every probe consumes a control tick from the episode budget.
        self.allow_tilt = bool(allow_tilt)
        self.tilt_angle_rad = math.radians(15.0)
        if type(max_navigation_steps) is not int or not 1 <= max_navigation_steps <= 100000:
            raise ValueError("max_navigation_steps must be an integer in [1, 100000]")
        if max_control_steps is not None and (type(max_control_steps) is not int or
                                              not 1 <= max_control_steps <= 100000):
            raise ValueError("max_control_steps must be an integer in [1, 100000] or null")
        self.max_navigation_steps = max_navigation_steps
        self.max_control_steps = (max_control_steps if max_control_steps is not None
                                  else (5000 if self.allow_tilt else max_navigation_steps))
        self.width, self.height, self.hfov = int(width), int(height), int(hfov)
        self.max_depth_m = max_depth_m
        self.enable_oracle = enable_oracle
        # Declared by the launcher (which knows which conda env it used); verified
        # against habitat.__version__ at env init so a mismatch cannot pass silently.
        self.declared_version = simulator_version
        self.actual_version = None
        from extensions.simulators.habitat.backend import HabitatBackend
        self.backend = backend or HabitatBackend({"version": simulator_version or "0.2.4"})
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
        name = self._episode_filename.format(split=self.split)
        return self.data_root / "datasets" / self.dataset_family / self.split / name

    def _episode_manifest(self):
        from extensions.benchmarks.vln.r2r_ce.dataset import select_episodes
        if not hasattr(self, "_selected_episodes"):
            self._selected_episodes = select_episodes(self._dataset_path, self.data_root,
                self.languages, self.episode_limit, self.requested_ids)
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

        data_path = str(self._dataset_path)
        overrides = [
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.scenes_dir={self.data_root / 'scene_datasets'}",
            f"habitat.dataset.split={self.split}",
            f"habitat.dataset.type={DATASET_TYPE[self.dataset_family]}",
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
            f"habitat.environment.max_episode_steps={self.max_control_steps}",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.max_scene_repeat_steps=50000",
            f"habitat.task.measurements.success.success_distance={self.success_distance_m}",
        ]
        if self.allow_tilt:
            from extensions.simulators.habitat.backend import tilt_settings
            overrides += [
                "+habitat/task/actions@habitat.task.actions.look_up=look_up",
                "+habitat/task/actions@habitat.task.actions.look_down=look_down",
            ]
            overrides += [f"{key}={value}" for key, value in tilt_settings(actual, 15).items()]
        config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides)
        if self.dataset_family == "rxr":
            import extensions.benchmarks.vln.rxr_ce.dataset  # noqa: F401  registers RxR-VLN-CE-v1
        self.env = self.backend.initialize(config)

    def _agent_position(self):
        return self.backend.position()

    def _goal_position(self):
        goals = self.env.current_episode.goals
        if not goals:
            raise ContractError("episode has no goal")
        return [float(v) for v in goals[0].position]

    def _geodesic_to_goal(self, position):
        distance = self.backend.geodesic(position, self._goal_position())
        return {"geodesic_m": distance, "fallback": "unreachable" if distance is None else None}

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
        if "pose" in self.sensors:
            import numpy as np

            payloads["pose"] = encode_array(np.asarray(self.backend.state(), dtype=np.float32))
            specs["pose"] = asdict(SensorSpec("state", "float32", (7,), "m+rad",
                                              "habitat_world_y_up", "measured", {}))
        return {"episode_id": self.session["episode_id"], "sequence": sequence,
                "sim_time_s": None, "control_tick": self.session["steps"], "sensors": payloads,
                "sensor_specs": specs}

    # -- action mapping -----------------------------------------------------
    def _to_habitat_action(self, action):
        if action.kind == "stop":
            return "stop"
        if action.kind == "camera_tilt":
            if not self.allow_tilt:
                raise PolicyViolation("camera tilt is disabled; set benchmark_settings.allow_tilt")
            delta = action.values["delta_rad"]
            if abs(abs(delta) - self.tilt_angle_rad) > 1e-3:
                raise PolicyViolation(f"camera tilt must be {math.degrees(self.tilt_angle_rad):g} degrees")
            return "look_up" if delta > 0 else "look_down"
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
            version = getattr(self, "actual_version", None) or self.declared_version or "0.2.4"
            return {
                "schema_version": SCHEMA_VERSION, "role": "benchmark",
                "id": f"habitat_{self.benchmark_id}", "benchmark_kind": "vln_continuous",
                "benchmark_id": self.benchmark_id,
                "dataset_family": self.dataset_family, "split": self.split,
                "language_filter": sorted(self.languages) if self.languages else None,
                "simulator": f"habitat-sim/habitat-lab {version}",
                "simulator_version": version, "real": True,
                "accepts_actions": ["primitive", "stop"]
                                 + (["camera_tilt"] if self.allow_tilt else []),
                "offers_sensors": list(self.sensors),
                "sensor_geometry": {"width": self.width, "height": self.height, "hfov_deg": self.hfov},
                "coordinate_frame": "habitat_world_y_up",
                "clock": "discrete_control_ticks",
                "action_semantics": {"forward_m": FORWARD_STEP_M, "turn_rad": self.turn_angle_rad},
                "max_navigation_steps": self.max_navigation_steps,
                "max_control_steps": self.max_control_steps,
                "decision_limit": self.max_control_steps,
                "capture_schema": EVIDENCE_SCHEMA,
                "success_distance_m": self.success_distance_m,
                "oracle_follower_available": bool(self.enable_oracle),
                "note": "Discrete task: sim_time_s is null; control_tick counts actions.",
            }
        if operation == "episodes":
            # Group by scene (alphabetical) to minimise simulator reconfigures;
            # the selection set itself is the first N episodes by numeric id.
            manifest = self._episode_manifest()
            manifest.sort(key=lambda e: (e["scene_id"], str(e["episode_id"])))
            return [str(e["episode_id"]) for e in manifest]

        if operation == "oracle_follower_action":
            if not self.enable_oracle:
                raise ContractError("oracle follower disabled")
            state = self._active_session(payload)
            follower = self._ensure_follower()
            action = follower.next_action_along(state["goal"])
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
                goal = self._goal_position()
                start = self._agent_position()
                first_distance = self._geodesic_to_goal(start)
            except Exception:
                self.session = None
                raise
            self.session = {
                "session_id": session_id, "episode_id": episode_id,
                "steps": 0, "navigation_steps": 0, "control_steps": 0,
                "stopped": False, "truncated": False, "environment_end_reason": None,
                "positions": [start], "goal_distances": [first_distance],
                "goal": goal, "start": start,
            }
            context = EpisodeContext(
                episode_id, Goal("language", self.env.current_episode.instruction.instruction_text),
                "habitat_rp2c", int(payload.get("seed", 0)),
            )
            return {"context": asdict(context), "observation": self._observation(obs, 0),
                    "decision_limit": self.max_control_steps}

        state = self._active_session(payload)
        if operation == "step":
            if state["stopped"] or state["truncated"]:
                raise ContractError("episode already ended")
            if payload["expected_sequence"] != state["steps"]:
                raise ContractError("stale or duplicate step: episode must be restarted")
            action = validate_action(payload["action"])
            if action.kind not in {"primitive", "stop"} and not (
                    action.kind == "camera_tilt" and self.allow_tilt):
                raise PolicyViolation(f"{action.kind} unsupported by this benchmark")
            habitat_action = self._to_habitat_action(action)
            obs = self.backend.step(habitat_action)
            state["steps"] += 1
            state["control_steps"] += 1
            if action.kind in {"primitive", "stop"}:
                state["navigation_steps"] += 1
            position = self._agent_position()
            state["positions"].append(position)
            state["goal_distances"].append(self._geodesic_to_goal(position))
            if action.kind == "stop":
                state["stopped"] = True
                state["environment_end_reason"] = "stop"
            elif self.env.episode_over:
                state["truncated"] = True
                state["environment_end_reason"] = "environment_episode_over"
            elif state["navigation_steps"] >= self.max_navigation_steps:
                state["truncated"] = True
                state["environment_end_reason"] = "navigation_budget"
            elif state["control_steps"] >= self.max_control_steps:
                state["truncated"] = True
                state["environment_end_reason"] = "control_budget"
            return {"observation": self._observation(obs, state["steps"]),
                    "terminated": state["stopped"], "truncated": state["truncated"]}

        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            positions = state["positions"]
            path_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
            goal = state["goal"]
            final = math.dist(positions[-1], goal)
            distances = [d["geodesic_m"] for d in state["goal_distances"]]
            min_distance = min((d for d in distances if d is not None), default=None)
            record = {
                "episode_id": state["episode_id"], "status": "completed",
                "termination": "stop" if state["stopped"] else "budget",
                "steps": state["steps"], "control_tick": state["steps"], "sim_time_s": None,
                "navigation_steps": state["navigation_steps"], "control_steps": state["control_steps"],
                "environment_end_reason": state["environment_end_reason"],
                "sim_clock": "discrete_control_ticks",
            }
            evidence = {
                "schema": EVIDENCE_SCHEMA,
                "reference": {
                    "scene_id": self.env.current_episode.scene_id,
                    "goal_xyz_m": goal, "start_xyz_m": state["start"],
                    "geodesic_start_to_goal_m": distances[0],
                    "success_radius_m": self.success_distance_m,
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
                "budget": {
                    "max_navigation_steps": self.max_navigation_steps,
                    "max_control_steps": self.max_control_steps,
                    "navigation_steps": state["navigation_steps"],
                    "control_steps": state["control_steps"],
                    "environment_end_reason": state["environment_end_reason"],
                },
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
        """Pin the exact episode via habitat's public current_episode setter.

        The episode iterator walks scene groups in alphabetical order and cannot
        go back, so iterating resets is both slow and unable to reach arbitrary
        ids; the setter disables iterator advancement for the next reset.
        """
        for episode in self.env.episodes:
            if str(episode.episode_id) == episode_id:
                return episode
        raise ContractError(f"episode {episode_id} not in the loaded dataset")

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

        keys = ("success", "spl", "distance_to_goal", "oracle_success")
        cleaned = {k: clean(v) for k, v in metrics.items() if k in keys}
        return {k: v for k, v in cleaned.items() if v is not _DROP}

    def prepare(self):
        self.call("prepare", {})

    def asset_files(self):
        files = {self._dataset_path}
        for episode in self._episode_manifest():
            scene = self.data_root / "scene_datasets" / episode["scene_id"]
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
    return HabitatR2RBenchmarkService(backend=backend, **settings)
