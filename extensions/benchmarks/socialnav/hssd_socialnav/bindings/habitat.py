"""Habitat 3.0 social navigation binding (HSSD, Spot robot + simulated human).

Runs inside the habitat conda environment; imports habitat lazily so that
`describe`-level tooling elsewhere in this repo never needs ML dependencies.

This binding follows the upstream ``benchmark/multi_agent/hssd_spot_human_social_nav.yaml``
task (RearrangePddlSocialNavTask-v0): a Spot robot (agent_0) must navigate to a
simulated human (agent_1) that wanders through an HSSD scene via habitat's
scripted OracleNavRandCoordAction.

Contract mapping:
- platform primitive(forward 0.25 m / turn) actions are converted by this
  binding into one ``agent_0_base_velocity`` control step with the exact
  physical displacement of the primitive (velocity = displacement *
  ``ctrl_freq``, normalised and clipped exactly like habitat's BaseVelAction);
  ``stop`` ends the episode without a habitat action.
- the human is stepped internally every control step by the upstream scripted
  oracle wander policy; the method never sees or controls the human.
- rgb/depth observations are the robot's head sensors, transported as lossless
  base64 tensor envelopes (nav_eval.tensorcode); depth is unnormalised metres.
- `sim_time_s` is null; `control_tick` counts control steps.
- evaluator evidence (habitat-socialnav-evidence/1) records per-step robot and
  human positions and habitat's official DistToGoal (robot-to-human) measure,
  plus the upstream seek-success measure and social-nav statistics in the
  parity layer. Requires the habitat 3.0 assets bundle (robots, humanoids,
  YCB/AB/Google object configs) and HSSD scenes; see docs/benchmarks.md.

Validation level is `unverified` until a real episode has been run against the
downloaded upstream assets.
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

EVIDENCE_SCHEMA = "habitat-socialnav-evidence/1"
FORWARD_STEP_M = 0.25
TURN_ANGLE_RAD = math.radians(15.0)
TASK_YAML = "benchmark/multi_agent/hssd_spot_human_social_nav.yaml"

ADDITIONAL_OBJECT_PATHS = (
    "objects/ycb/configs/",
    "objects/amazon_berkeley/configs/",
    "objects/google_object_dataset/configs/",
)
ROBOT_URDF = "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
HUMAN_URDF = "humanoids/humanoid_data/female_2/female_2.urdf"


class HabitatSocialNavBenchmarkService:
    def __init__(self, data_root, benchmark_id="socialnav_hssd_val", episode_limit=None,
                 gpu_device_id=0, sensors=("rgb", "depth"), success_distance_m=1.5,
                 max_task_steps=750, task_instruction=None, width=640, height=480,
                 hfov=79, max_depth_m=10.0, enable_oracle=True, simulator_version=None,
                 backend=None, episode_ids=None, turn_angle_rad=TURN_ANGLE_RAD):
        if benchmark_id != "socialnav_hssd_val":
            raise ValueError("unknown benchmark; choose from ['socialnav_hssd_val']")
        unknown = set(sensors) - {"rgb", "depth"}
        if unknown:
            raise ValueError(f"unsupported sensors: {sorted(unknown)}")
        if not sensors:
            raise ValueError("observation policy requires at least one sensor")
        self.benchmark_id = benchmark_id
        self.data_root = Path(data_root).resolve()
        self.episode_limit = episode_limit
        self.gpu_device_id = gpu_device_id
        self.sensors = tuple(sensors)
        self.success_distance_m = success_distance_m
        self.max_task_steps = int(max_task_steps)
        self.task_instruction = task_instruction or (
            "Find the person in this home and navigate close to them, then stop.")
        if turn_angle_rad not in (math.radians(15), math.radians(30)):
            raise ValueError("Habitat social nav supports 15 or 30 degree turns")
        self.turn_angle_rad = turn_angle_rad
        self.width, self.height, self.hfov = int(width), int(height), int(hfov)
        self.max_depth_m = max_depth_m
        self.enable_oracle = enable_oracle
        self.declared_version = simulator_version
        self.actual_version = None
        from extensions.simulators.habitat.backend import HabitatBackend
        self.backend = backend or HabitatBackend({"version": simulator_version or "0.3.0"})
        self.requested_ids = episode_ids
        self.env = None
        self.session = None
        self._obs_keys = {}
        self._dispatcher = SimulatorDispatch(self._handle)

    def call(self, operation, payload):
        return self._dispatcher.call(operation, payload)

    # -- dataset -----------------------------------------------------------
    @property
    def _dataset_path(self):
        return self.data_root / "datasets" / "hssd" / "rearrange" / "val" / "social_rearrange.json.gz"

    def _episode_manifest(self):
        from extensions.benchmarks.socialnav.hssd_socialnav.dataset import (
            select_episodes,
        )
        if not hasattr(self, "_selected_episodes"):
            self._selected_episodes = select_episodes(self._dataset_path, self.data_root,
                self.episode_limit, self.requested_ids)
        return list(self._selected_episodes)

    # -- habitat bootstrap --------------------------------------------------
    def _ensure_env(self):
        if self.env is not None:
            return
        import habitat

        actual = str(habitat.__version__)
        if self.declared_version and actual != self.declared_version:
            raise ContractError(
                f"launcher declared habitat {self.declared_version} but worker runs {actual}")
        self.actual_version = actual

        self._episodes = self._episode_manifest()
        if not self._episodes:
            raise ContractError("no episodes with locally available scenes; copy scene dirs first")
        self._episode_order = {str(e["episode_id"]): i for i, e in enumerate(self._episodes)}

        object_paths = ",".join(f"'{self.data_root / rel}'"
                                for rel in ADDITIONAL_OBJECT_PATHS)
        overrides = [
            f"habitat.dataset.data_path={self._dataset_path}",
            f"habitat.dataset.scenes_dir={self.data_root / 'scene_datasets'}",
            "habitat.dataset.split=val",
            "habitat.dataset.type=RearrangeDataset-v0",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={self.gpu_device_id}",
            f"habitat.simulator.additional_object_paths=[{object_paths}]",
            f"habitat.simulator.agents.agent_0.articulated_agent_urdf={self.data_root / ROBOT_URDF}",
            f"habitat.simulator.agents.agent_1.articulated_agent_urdf={self.data_root / HUMAN_URDF}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.width={self.width}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.height={self.height}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.hfov={self.hfov}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.width={self.width}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.height={self.height}",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.hfov={self.hfov}",
            "habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.normalize_depth=False",
            f"habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.max_depth={self.max_depth_m}",
            f"habitat.environment.max_episode_steps={self.max_task_steps}",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.max_scene_repeat_steps=100000",
            "habitat.task.measurements.nav_to_pos_succ.success_distance="
            f"{self.success_distance_m}",
        ]
        config = habitat.get_config(TASK_YAML, overrides)
        self.env = self.backend.initialize(config)

    # -- state helpers ------------------------------------------------------
    def _agent_base(self, index):
        return [float(v) for v in
                self.env.sim.get_agent_data(index).articulated_agent.base_pos]

    def _distance_to_goal(self):
        value = self.env.get_metrics()["dist_to_goal"]
        try:
            import numpy as np
            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {"distance_m": None, "fallback": "unavailable_measure"}
        return {"distance_m": float(value), "fallback": None}

    def _velocity_command(self, name, amount):
        """Normalised base_vel reproducing the primitive's physical displacement.

        habitat's BaseVelAction clips the command to [-1, 1] and multiplies by
        the configured lin/ang speed; one control step integrates velocity for
        1/ctrl_freq seconds. Displacement d therefore needs command
        d * ctrl_freq / speed, clipped exactly like the upstream action.
        """
        ctrl_freq = float(self.env.sim.ctrl_freq)
        if name == "forward":
            raw = amount * ctrl_freq / self._robot_lin_speed
            return min(1.0, raw)  # allow_back is False upstream; forward only
        raw = amount * ctrl_freq / self._robot_ang_speed
        return math.copysign(min(1.0, abs(raw)), 1.0 if name == "left" else -1.0)

    # -- observation policy -------------------------------------------------
    def _robot_sensor_key(self, kind):
        preferred = f"agent_0_head_{kind}"
        if preferred in self._obs_keys:
            return preferred
        candidates = [key for key in self._obs_keys
                      if key.startswith("agent_0_") and kind in key
                      and "arm" not in key and "panoptic" not in key]
        if len(candidates) != 1:
            raise ContractError(
                f"cannot identify robot {kind} sensor; available keys: {sorted(self._obs_keys)}")
        return candidates[0]

    def _observation(self, obs, sequence):
        payloads, specs = {}, {}
        if "rgb" in self.sensors:
            payloads["rgb"] = encode_array(obs[self._robot_sensor_key("rgb")])
            specs["rgb"] = asdict(SensorSpec("rgb", "uint8", (self.height, self.width, 3), "srgb",
                                             "camera_optical", "rendered",
                                             {"hfov_deg": self.hfov, "width": self.width,
                                              "height": self.height, "mount": "spot_head"}))
        if "depth" in self.sensors:
            payloads["depth"] = encode_array(obs[self._robot_sensor_key("depth")])
            specs["depth"] = asdict(SensorSpec("depth", "float32", (self.height, self.width, 1), "m",
                                               "camera_optical", "rendered",
                                               {"hfov_deg": self.hfov, "width": self.width,
                                                "height": self.height, "min_depth_m": 0.0,
                                                "max_depth_m": self.max_depth_m,
                                                "convention": "z_depth_along_optical_axis",
                                                "mount": "spot_head"}))
        return {"episode_id": self.session["episode_id"], "sequence": sequence,
                "sim_time_s": None, "control_tick": self.session["steps"], "sensors": payloads,
                "sensor_specs": specs}

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
                "id": f"habitat_{self.benchmark_id}", "benchmark_kind": "social_navigation",
                "benchmark_id": self.benchmark_id,
                "dataset_family": "hssd/rearrange_social_nav", "split": "val",
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
                "goal": {"kind": "language", "value": "person"},
                "human_policy": "oracle_nav_randcoord (upstream scripted wander)",
                "oracle_follower_available": False,
                "note": "Discrete control ticks; goal distance is habitat's DistToGoal "
                        "(robot to human). Requires habitat 3.0 assets bundle.",
            }
        if operation == "episodes":
            manifest = self._episode_manifest()
            manifest.sort(key=lambda e: (e["scene_id"], str(e["episode_id"])))
            return [str(e["episode_id"]) for e in manifest]

        session_id = payload.get("session_id")
        if operation == "reset":
            self._ensure_env()
            if self.session is not None:
                raise ContractError("active session already exists")
            episode_id = str(payload["episode_id"])
            if episode_id not in self._episode_order:
                raise ContractError(f"episode {episode_id} not in the frozen local selection")
            target = self._select_episode(episode_id)
            obs = self.backend.reset(target, int(payload.get("seed", 0)))
            self._obs_keys = dict(obs)
            # The upstream action config fixes these speeds; read them back so the
            # primitive->velocity conversion stays faithful to the composed config.
            self._robot_lin_speed = float(self.env.task.actions["agent_0_base_velocity"]
                                          ._lin_speed)
            self._robot_ang_speed = float(self.env.task.actions["agent_0_base_velocity"]
                                          ._ang_speed)
            try:
                start = self._agent_base(0)
                human_start = self._agent_base(1)
                first_distance = self._distance_to_goal()
            except Exception:
                self.session = None
                raise
            self.session = {
                "session_id": session_id, "episode_id": episode_id,
                "steps": 0, "stopped": False, "truncated": False,
                "positions": [start], "human_positions": [human_start],
                "goal_distances": [first_distance], "start": start,
            }
            context = EpisodeContext(
                episode_id, Goal("language", self.task_instruction), "habitat_spot_h3",
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
            if action.kind == "stop":
                # The stop decision consumes a control tick; the observation is
                # the unchanged last frame but the sequence must still advance.
                state["stopped"] = True
                state["steps"] += 1
                return {"observation": self._observation(dict(self._obs_keys), state["steps"]),
                        "terminated": True, "truncated": False}
            name, amount = action.values["name"], action.values["amount"]
            if name == "forward" and abs(amount - FORWARD_STEP_M) > 1e-3:
                raise PolicyViolation(f"forward amount must be {FORWARD_STEP_M} m on this benchmark")
            if name in {"left", "right"} and abs(amount - self.turn_angle_rad) > 1e-3:
                raise PolicyViolation(f"turn amount must be {math.degrees(self.turn_angle_rad):g} degrees "
                                      f"({self.turn_angle_rad:.6f} rad) on this benchmark")
            import numpy as np
            command = self._velocity_command(name, amount)
            base_vel = np.array([command, 0.0] if name == "forward"
                                 else [0.0, command], dtype=np.float32)
            habitat_action = {
                "action": ("agent_0_base_velocity", "agent_1_oracle_nav_randcoord_action"),
                "action_args": {
                    "agent_0_base_vel": base_vel,
                    "agent_1_oracle_nav_randcoord_action": np.array([0.0], dtype=np.float32),
                },
            }
            obs = self.backend.step(habitat_action)
            self._obs_keys = dict(obs)
            state["steps"] += 1
            state["positions"].append(self._agent_base(0))
            state["human_positions"].append(self._agent_base(1))
            state["goal_distances"].append(self._distance_to_goal())
            state["truncated"] = (self.env.episode_over
                                  or state["steps"] >= self.max_task_steps) and not state["stopped"]
            return {"observation": self._observation(obs, state["steps"]),
                    "terminated": state["stopped"], "truncated": state["truncated"]}

        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            positions = state["positions"]
            path_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
            distances = [d["distance_m"] for d in state["goal_distances"]]
            min_distance = min((d for d in distances if d is not None), default=None)
            final = distances[-1] if distances and distances[-1] is not None else None
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
                    "goal_kind": "simulated_human",
                    "human_start_xyz_m": state["human_positions"][0],
                    "start_xyz_m": state["start"],
                    "geodesic_start_to_goal_m": distances[0],
                    "success_radius_m": self.success_distance_m,
                    "distance_definition": "habitat_dist_to_goal_robot_to_human",
                },
                "trajectory": {
                    "positions_xyz_m": positions, "control_ticks": list(range(len(positions))),
                    "human_positions_xyz_m": state["human_positions"],
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
            return {}
        raise ContractError(f"unsupported benchmark operation: {operation}")

    def _active_session(self, payload):
        if self.session is None or self.session["session_id"] != payload.get("session_id"):
            raise ContractError("reset session first")
        return self.session

    def _select_episode(self, episode_id):
        for episode in self.env.episodes:
            if str(episode.episode_id) == episode_id:
                return episode
        raise ContractError(f"episode {episode_id} not in the loaded dataset")

    def _reference_metrics(self):
        metrics = self.env.get_metrics()

        _DROP = object()

        def clean(value):
            try:
                import numpy as np
                if isinstance(value, np.generic):
                    return value.item()
            except ImportError:
                pass
            if isinstance(value, (int, float, bool)):
                return value
            if isinstance(value, dict):
                cleaned = {k: clean(v) for k, v in value.items()}
                return {k: v for k, v in cleaned.items() if v is not _DROP} or _DROP
            if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
                return value.item()
            return _DROP

        keys = ("nav_seek_success", "nav_to_pos_success", "rot_dist_to_goal",
                "did_agents_collide", "num_agents_collide", "social_nav_stats")
        cleaned = {k: clean(v) for k, v in metrics.items() if k in keys}
        return {k: v for k, v in cleaned.items() if v is not _DROP}

    def prepare(self):
        self.call("prepare", {})

    def asset_files(self):
        files = {self._dataset_path, self.data_root / ROBOT_URDF, self.data_root / HUMAN_URDF}
        for episode in self._episode_manifest():
            scene = self.data_root / "scene_datasets" / episode["scene_id"].removeprefix(
                "data/scene_datasets/")
            files.add(scene)
            files.update(scene.parent.glob("*.navmesh"))
        return {str(path.relative_to(self.data_root)): path for path in sorted(files)}

    def runtime_identity(self):
        import habitat_sim
        return {"habitat_lab": self.backend.version, "habitat_sim": habitat_sim.__version__,
                "task": "RearrangePddlSocialNavTask-v0 (habitat 3.0)"}

    def close(self):
        self._dispatcher.close()


def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]
    backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"])
    settings.update(config["observation"])
    settings["turn_angle_rad"] = config["capabilities"]["action_semantics"]["turn_rad"]
    return HabitatSocialNavBenchmarkService(backend=backend, **settings)
