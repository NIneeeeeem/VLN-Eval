"""Habitat-Lab 0.3 InstanceImageNav binding.

The only policy-visible task goal is the rendered ``instance_imagegoal``.  The
dataset's goal camera pose, object category/id, and all world positions remain
private; view-point geodesic distances are evaluator evidence only.
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

EVIDENCE_SCHEMA = "habitat-instance-imagenav-evidence/1"
TASK_YAML = "benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml"
FORWARD_STEP_M = 0.25
TURN_ANGLE_RAD = math.radians(30.0)
TILT_ANGLE_RAD = math.radians(30.0)
MAX_EPISODE_STEPS = 1000
SENSOR = "instance_imagegoal"

BENCHMARK_SPLITS = {
    "instance_imagenav_hm3d_val": "val",
    "instance_imagenav_hm3d_val_mini": "val_mini",
}


class HabitatInstanceImageNavBenchmarkService:
    def __init__(self, data_root, benchmark_id="instance_imagenav_hm3d_val", episode_limit=None,
                 gpu_device_id=0, sensors=("rgb", "depth", SENSOR), success_distance_m=0.1,
                 width=640, height=480, hfov=90, max_depth_m=10.0,
                 simulator_version=None, backend=None, episode_ids=None,
                 turn_angle_rad=TURN_ANGLE_RAD, tilt_angle_rad=TILT_ANGLE_RAD,
                 content_scenes_path=None):
        if benchmark_id not in BENCHMARK_SPLITS:
            raise ValueError(f"unknown benchmark; choose from {sorted(BENCHMARK_SPLITS)}")
        if content_scenes_path is not None:
            raise ValueError("nondefault content_scenes_path is unsupported by Habitat 0.3 loader")
        unknown = set(sensors) - {"rgb", "depth", SENSOR}
        if unknown or SENSOR not in sensors:
            raise ValueError(f"InstanceImageNav requires {SENSOR}; unsupported sensors: {sorted(unknown)}")
        if turn_angle_rad != TURN_ANGLE_RAD or tilt_angle_rad != TILT_ANGLE_RAD:
            raise ValueError("InstanceImageNav requires 30 degree turns and camera tilts")
        for name, value, lower, upper in (("success_distance_m", success_distance_m, 0.0, math.inf),
                                          ("hfov", hfov, 0.0, 360.0),
                                          ("max_depth_m", max_depth_m, 0.0, math.inf)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"{name} must be finite and in ({lower}, {upper}]")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (width, height)):
            raise ValueError("observation width and height must be positive integers")
        if hfov >= 180:
            raise ValueError("pinhole camera hfov must be less than 180 degrees")
        self.benchmark_id, self.split = benchmark_id, BENCHMARK_SPLITS[benchmark_id]
        self.data_root, self.episode_limit = Path(data_root).resolve(), episode_limit
        self.gpu_device_id, self.sensors = gpu_device_id, tuple(sensors)
        self.success_distance_m = float(success_distance_m)
        self.width, self.height, self.hfov, self.max_depth_m = width, height, float(hfov), max_depth_m
        self.declared_version, self.actual_version = simulator_version, None
        self.requested_ids = episode_ids
        from extensions.simulators.habitat.backend import HabitatBackend
        self.backend = backend or HabitatBackend({"version": simulator_version or "0.3.0"})
        self.env = self.session = None
        self._dispatcher = SimulatorDispatch(self._handle)

    def call(self, operation, payload):
        return self._dispatcher.call(operation, payload)

    @property
    def _dataset_path(self):
        return (self.data_root / "datasets" / "instance_imagenav" / "hm3d" / "v2"
                / self.split / f"{self.split}.json.gz")

    def _episode_manifest(self):
        from extensions.benchmarks.imagenav.instance_imagenav.dataset import (
            select_episodes,
        )
        if not hasattr(self, "_selected_episodes"):
            self._selected_episodes = select_episodes(
                self._dataset_path, self.data_root, self.episode_limit, self.requested_ids)
        return list(self._selected_episodes)

    def _ensure_env(self):
        if self.env is not None:
            return
        import habitat

        actual = str(habitat.__version__)
        if self.declared_version and actual != self.declared_version:
            raise ContractError(f"launcher declared habitat {self.declared_version} but worker runs {actual}")
        self.actual_version = actual
        self._episodes = self._episode_manifest()
        if not self._episodes:
            raise ContractError("no episodes with locally available scenes; copy scene dirs first")
        self._episode_order = {episode["_nav_eval_public_id"]: episode for episode in self._episodes}
        overrides = [
            f"habitat.dataset.data_path={self._dataset_path}",
            f"habitat.dataset.scenes_dir={self.data_root / 'scene_datasets'}",
            f"habitat.dataset.split={self.split}", "habitat.dataset.type=InstanceImageNav-v1",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={self.gpu_device_id}",
            "habitat.simulator.forward_step_size=0.25", "habitat.simulator.turn_angle=30",
            "habitat.environment.max_episode_steps=1000",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.max_scene_repeat_steps=100000",
            f"habitat.task.measurements.success.success_distance={self.success_distance_m}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width={self.width}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height={self.height}",
            f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov={self.hfov:g}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width={self.width}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height={self.height}",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov={self.hfov:g}",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False",
            f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth={self.max_depth_m}",
        ]
        from extensions.simulators.habitat.backend import tilt_settings
        overrides += [f"{key}={value}" for key, value in tilt_settings(actual, 30).items()]
        self.env = self.backend.initialize(habitat.get_config(TASK_YAML, overrides))

    def _distance_to_goal(self):
        value = self.env.get_metrics().get("distance_to_goal")
        try:
            import numpy as np
            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass
        return ({"distance_m": float(value), "fallback": None}
                if not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0
                else {"distance_m": None, "fallback": "unavailable_measure"})

    def _goal_hfov(self):
        value = self.session["obs"].get("instance_imagegoal_hfov")
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 360:
            raise ContractError("InstanceImageNav goal image HFOV is unavailable")
        return float(value)

    def _observation(self, obs, sequence):
        payloads, specs = {}, {}
        if "rgb" in self.sensors:
            payloads["rgb"] = encode_array(obs["rgb"])
            specs["rgb"] = asdict(SensorSpec("rgb", "uint8", (self.height, self.width, 3), "srgb", "camera_optical", "rendered", {"hfov_deg": self.hfov, "width": self.width, "height": self.height}))
        if "depth" in self.sensors:
            payloads["depth"] = encode_array(obs["depth"])
            specs["depth"] = asdict(SensorSpec("depth", "float32", (self.height, self.width, 1), "m", "camera_optical", "rendered", {"hfov_deg": self.hfov, "width": self.width, "height": self.height, "min_depth_m": 0.0, "max_depth_m": self.max_depth_m, "convention": "z_depth_along_optical_axis"}))
        image = obs.get(SENSOR)
        if image is None:
            raise ContractError("InstanceImageNav observation missing instance_imagegoal")
        shape = tuple(int(x) for x in image.shape)
        if len(shape) != 3 or shape[-1] != 3:
            raise ContractError("instance_imagegoal must be native HxWx3")
        payloads[SENSOR] = encode_array(image)
        specs[SENSOR] = asdict(SensorSpec("image_goal", "uint8", shape, "srgb", "camera_optical", "rendered", {"hfov_deg": self._goal_hfov(), "width": shape[1], "height": shape[0]}))
        return {"episode_id": self.session["episode_id"], "sequence": sequence, "sim_time_s": None,
                "control_tick": self.session["steps"], "sensors": payloads, "sensor_specs": specs}

    def _to_habitat_action(self, action):
        if action.kind == "stop":
            return "stop"
        if action.kind == "camera_tilt":
            amount = action.values["delta_rad"]
            if abs(abs(amount) - TILT_ANGLE_RAD) > 1e-3:
                raise PolicyViolation("camera tilt must be 30 degrees on this benchmark")
            return "look_up" if amount > 0 else "look_down"
        name, amount = action.values["name"], action.values["amount"]
        if name == "forward" and abs(amount - FORWARD_STEP_M) > 1e-3:
            raise PolicyViolation("forward amount must be 0.25 m on this benchmark")
        if name in {"left", "right"} and abs(amount - TURN_ANGLE_RAD) > 1e-3:
            raise PolicyViolation("turn amount must be 30 degrees on this benchmark")
        return {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}[name]

    def _handle(self, operation, payload):
        if operation == "shutdown":
            self.backend.close()
            return {}
        if operation == "describe":
            version = self.actual_version or self.declared_version or "0.3.0"
            return {"schema_version": SCHEMA_VERSION, "role": "benchmark", "id": f"habitat_{self.benchmark_id}",
                    "benchmark_kind": "instance_image_navigation", "benchmark_id": self.benchmark_id,
                    "dataset_family": "instance_imagenav/hm3d/v2", "split": self.split,
                    "simulator": f"habitat-sim/habitat-lab {version}", "simulator_version": version,
                    "real": True, "accepts_actions": ["primitive", "camera_tilt", "stop"],
                    "offers_sensors": list(self.sensors), "clock": "discrete_control_ticks",
                    "sensor_geometry": {"width": self.width, "height": self.height, "hfov_deg": self.hfov},
                    "action_semantics": {"forward_m": FORWARD_STEP_M, "turn_rad": TURN_ANGLE_RAD, "camera_tilt_rad": TILT_ANGLE_RAD},
                    "capture_schema": EVIDENCE_SCHEMA, "success_distance_m": self.success_distance_m,
                    "goal": {"kind": "image_reference", "value": SENSOR},
                    "note": "Goal image is rendered natively; goal camera pose and object metadata remain private."}
        if operation == "episodes":
            return [episode["_nav_eval_public_id"] for episode in self._episode_manifest()]
        if operation == "prepare":
            ids = self._handle("episodes", {})
            if not ids:
                raise ContractError("empty episode selection")
            self._handle("reset", {"episode_id": ids[0], "session_id": "preflight", "seed": 0})
            self._handle("close_episode", {"session_id": "preflight"})
            return {}
        if operation == "reset":
            self._ensure_env()
            if self.session is not None:
                raise ContractError("active session already exists")
            public_id = str(payload["episode_id"])
            if public_id not in self._episode_order:
                raise ContractError(f"episode {public_id} not in the frozen local selection")
            target = self._select_episode(self._episode_order[public_id])
            obs = self.backend.reset(target, int(payload.get("seed", 0)))
            self.session = {"session_id": payload.get("session_id"), "episode_id": public_id, "steps": 0,
                            "stopped": False, "truncated": False, "obs": obs,
                            "positions": [self.backend.position()], "goal_distances": [self._distance_to_goal()]}
            observation = self._observation(obs, 0)
            return {"context": asdict(EpisodeContext(public_id, Goal("image_reference", SENSOR), "habitat_instance_imagenav_v1", int(payload.get("seed", 0)))), "observation": observation}
        state = self._active_session(payload)
        if operation == "step":
            if state["stopped"] or state["truncated"]:
                raise ContractError("episode already ended")
            if payload["expected_sequence"] != state["steps"]:
                raise ContractError("stale or duplicate step: episode must be restarted")
            action = validate_action(payload["action"])
            if action.kind not in {"primitive", "camera_tilt", "stop"}:
                raise PolicyViolation(f"{action.kind} unsupported by this benchmark")
            obs = self.backend.step(self._to_habitat_action(action))
            state["obs"], state["steps"] = obs, state["steps"] + 1
            state["positions"].append(self.backend.position())
            state["goal_distances"].append(self._distance_to_goal())
            state["stopped"] = action.kind == "stop"
            state["truncated"] = (self.env.episode_over or state["steps"] >= MAX_EPISODE_STEPS) and not state["stopped"]
            return {"observation": self._observation(obs, state["steps"]), "terminated": state["stopped"], "truncated": state["truncated"]}
        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            distances = [entry["distance_m"] for entry in state["goal_distances"]]
            positions = state["positions"]
            return {"record": {"episode_id": state["episode_id"], "status": "completed", "termination": "stop" if state["stopped"] else "budget", "steps": state["steps"], "control_tick": state["steps"], "sim_time_s": None, "sim_clock": "discrete_control_ticks"},
                    "evidence": {"schema": EVIDENCE_SCHEMA, "reference": {"scene_id": self.env.current_episode.scene_id, "geodesic_start_to_goal_m": distances[0], "success_radius_m": self.success_distance_m, "distance_definition": "habitat_distance_to_goal_view_points"}, "trajectory": {"positions_xyz_m": positions, "control_ticks": list(range(len(positions))), "goal_distances_m": distances, "goal_distance_fallbacks": [entry["fallback"] for entry in state["goal_distances"]], "path_length_m": sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))}, "benchmark_reference_metrics": self._reference_metrics()}}
        if operation == "close_episode":
            self.session = None
            return {}
        raise ContractError(f"unsupported benchmark operation: {operation}")

    def _active_session(self, payload):
        if self.session is None or self.session["session_id"] != payload.get("session_id"):
            raise ContractError("reset session first")
        return self.session

    def _select_episode(self, manifest_episode):
        from extensions.benchmarks.imagenav.instance_imagenav.dataset import (
            scene_relative,
        )
        original_id = str(manifest_episode["episode_id"])
        root = self.data_root / "scene_datasets"
        scene = (root / scene_relative(manifest_episode["scene_id"])).resolve()
        def canonical(value):
            path = Path(value)
            return path.resolve() if path.is_absolute() else (root / scene_relative(value)).resolve()
        candidates = [episode for episode in self.env.episodes
                      if str(episode.episode_id) == original_id and canonical(episode.scene_id) == scene]
        if len(candidates) != 1:
            raise ContractError(f"could not uniquely select upstream episode {manifest_episode['_nav_eval_public_id']}")
        return candidates[0]

    def _reference_metrics(self):
        metrics = self.env.get_metrics()
        result = {}
        for key in ("success", "spl", "soft_spl", "distance_to_goal"):
            value = metrics.get(key)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float, bool)) and math.isfinite(value):
                result[key] = value
        return result

    def prepare(self):
        self.call("prepare", {})

    def asset_files(self):
        from extensions.benchmarks.imagenav.instance_imagenav.dataset import (
            scene_relative,
        )
        files = {self._dataset_path}
        content_dir = self._dataset_path.parent / "content"
        if content_dir.is_dir():
            files.update(content_dir.glob("*.json.gz"))
        for episode in self._episode_manifest():
            scene = self.data_root / "scene_datasets" / scene_relative(episode["scene_id"])
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
    return HabitatInstanceImageNavBenchmarkService(backend=backend, **settings)
