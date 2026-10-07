"""Conservative adapter for maintained Habitat benchmark repositories.

This module deliberately consumes the upstream ``Env`` rather than recreating
its dataset/task logic.  A benchmark binding supplies the import registrations
and public-goal projection; all poses, object ids, view points, and evaluator
state remain inside the worker.
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path, PurePosixPath

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

FORWARD_M = 0.25
TURN_RAD = math.radians(30)


def _scalar(value):
    return value.item() if hasattr(value, "item") and getattr(value, "ndim", 0) == 0 else value


def _utf8_tensor(value):
    import numpy as np
    return encode_array(np.frombuffer(value.encode("utf-8"), dtype=np.uint8))


class UpstreamHabitatService:
    """A one-session, one-thread adapter over an explicitly configured Env."""
    def __init__(self, *, benchmark_id, repo_root, task_config, dataset_path,
                 upstream_modules, goal_projector, sequence=False,
                 sensors=("rgb", "depth", "task_goal"), episode_limit=None,
                 episode_ids=None, gpu_device_id=0, width=640, height=480,
                 hfov=90, max_episode_steps=500, simulator_version="0.2.4",
                 backend=None, task_goal_sensor="task_goal",
                 subtask_stop_action="subtask_stop", forward_m=FORWARD_M, turn_rad=TURN_RAD,
                 camera_tilt=True, asset_paths=(), data_root=None):
        self.benchmark_id = benchmark_id
        self.repo_root = Path(repo_root).resolve()
        self._logical_asset_names = {}
        self.task_config = self._relative_file(task_config, "task_config")
        self.dataset_path = self._relative_file(dataset_path, "dataset_path")
        self.upstream_modules = tuple(upstream_modules)
        self.goal_projector, self.sequence = goal_projector, sequence
        self.sensors = tuple(sensors)
        allowed = {"rgb", "depth", "task_goal", "goal_embedding"}
        if not set(self.sensors) <= allowed or "task_goal" not in self.sensors:
            raise ValueError("sensors must be a declared subset including task_goal")
        self.episode_limit, self.requested_ids = episode_limit, episode_ids
        self.gpu_device_id, self.width, self.height, self.hfov = gpu_device_id, int(width), int(height), float(hfov)
        self.max_episode_steps = int(max_episode_steps)
        if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= 100000:
            raise ValueError("max_episode_steps must be an integer in [1, 100000]")
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError("width and height must be positive integers")
        if not math.isfinite(self.hfov) or not 0 < self.hfov < 180:
            raise ValueError("hfov must be finite and between 0 and 180")
        if episode_limit is not None and (type(episode_limit) is not int or episode_limit <= 0):
            raise ValueError("episode_limit must be a positive integer")
        if episode_ids is not None and (not isinstance(episode_ids, list) or not episode_ids
                or any(not isinstance(v, str) or not v for v in episode_ids)):
            raise ValueError("episode_ids must be a nonempty list of string IDs")
        self.extra_assets = []
        if not isinstance(asset_paths, (list, tuple)) or any(not isinstance(v, str) or not v for v in asset_paths):
            raise ValueError("asset_paths must be a list of nonempty relative paths")
        for value in asset_paths:
            path = self._relative_path(value, "asset_paths")
            if not path.exists():
                raise ValueError("asset_paths must name existing paths below repo_root")
            self.extra_assets.append(path)
        self.declared_version, self.actual_version = simulator_version, None
        if backend is None:
            from extensions.simulators.habitat.backend import HabitatBackend
            backend = HabitatBackend({"version": simulator_version})
        self.backend, self.task_goal_sensor = backend, task_goal_sensor
        self.subtask_stop_action = subtask_stop_action
        self.forward_m, self.turn_rad = float(forward_m), float(turn_rad)
        self.camera_tilt = bool(camera_tilt)
        self.env = self.session = None
        self._dispatcher = SimulatorDispatch(self._handle, name=f"{benchmark_id}-habitat")

    def _relative_file(self, value, name):
        path = self._relative_path(value, name)
        if not path.is_file():
            raise ValueError(f"{name} must name an existing file below repo_root")
        return path

    def _relative_path(self, value, name):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty path relative to repo_root")
        normalized = value.replace("\\", "/")
        logical = PurePosixPath(normalized)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError(f"{name} must be a safe relative path below repo_root")
        # Check containment before resolving: a repository-owned symlink may
        # legitimately point at a shared, licensed dataset volume.
        lexical = self.repo_root.joinpath(*logical.parts)
        path = lexical.resolve()
        self._logical_asset_names[path] = str(logical)
        return path

    def _asset_name(self, path):
        path = Path(path).resolve()
        if path in self._logical_asset_names:
            return self._logical_asset_names[path]
        for physical, logical in self._logical_asset_names.items():
            base = physical if physical.is_dir() else physical.parent
            name = PurePosixPath(logical) if physical.is_dir() else PurePosixPath(logical).parent
            try:
                suffix = path.relative_to(base)
            except ValueError:
                continue
            return str(name / suffix)
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError as error:
            raise ContractError("captured asset is outside configured logical repo paths") from error

    def call(self, operation, payload):
        return self._dispatcher.call(operation, payload)

    def prepare(self):
        self.call("prepare", {})

    def _load_config(self):
        import habitat
        return habitat.get_config(str(self.task_config))

    def _ensure_env(self):
        if self.env is not None:
            return
        for module in self.upstream_modules:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as error:
                raise ContractError(f"required upstream module {module!r} is not installed") from error
        import habitat
        actual = str(habitat.__version__)
        if actual != self.declared_version:
            raise ContractError(f"expected Habitat-Lab {self.declared_version}, found {actual}")
        self.actual_version = actual
        config = self._load_config()
        self._configure_live_task(config)
        self._live_config = config
        self._resolved_config = str(config)
        self.env = self.backend.initialize(config)
        indexed = {}
        for episode in self.env.episodes:
            scene = str(getattr(episode, "scene_id", ""))
            # Public identifiers are deterministic even when upstream IDs are
            # scene-local. Never mutate the native episode's id.
            if scene:
                scene_path = Path(scene)
                if scene_path.is_absolute():
                    try:
                        scene = scene_path.relative_to(self.repo_root).as_posix()
                    except ValueError as error:
                        raise ContractError("scene must reside below repo_root") from error
            key = f"{scene}::{episode.episode_id}" if scene else str(episode.episode_id)
            if key in indexed:
                raise ContractError("upstream exposes duplicate scene-qualified episode ids")
            indexed[key] = episode
        keys = sorted(indexed)
        if self.requested_ids is not None:
            wanted = set(self.requested_ids)
            missing = wanted - set(keys)
            if missing:
                raise ContractError(f"requested episodes absent upstream: {sorted(missing)}")
            keys = [key for key in keys if key in wanted]
        if self.episode_limit is not None:
            keys = keys[:self.episode_limit]
        if not keys:
            raise ContractError("upstream environment exposes no selected episodes")
        self._episodes = {key: indexed[key] for key in keys}

    def _configure_live_task(self, config):
        """Apply only physical execution settings, preserving upstream task logic."""
        from habitat.config import read_write
        from extensions.simulators.habitat.backend import tilt_settings
        values = {
            "habitat.dataset.data_path": str(self.dataset_path),
            "habitat.simulator.habitat_sim_v0.gpu_device_id": self.gpu_device_id,
            "habitat.simulator.forward_step_size": self.forward_m,
            "habitat.simulator.turn_angle": math.degrees(self.turn_rad),
            "habitat.environment.max_episode_steps": self.max_episode_steps,
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width": self.width,
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height": self.height,
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov": self.hfov,
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width": self.width,
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height": self.height,
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov": self.hfov,
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth": False,
        }
        if self.camera_tilt:
            values.update(tilt_settings(self.declared_version, math.degrees(self.turn_rad)))
        with read_write(config):
            for path, value in values.items():
                if ("rgb_sensor" in path and "rgb" not in self.sensors
                        or "depth_sensor" in path and "depth" not in self.sensors):
                    continue
                current = config
                parts = path.split(".")
                try:
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    previous = getattr(current, parts[-1])
                    if type(previous) is int and isinstance(value, float) and math.isclose(value, round(value), abs_tol=1e-9):
                        value = round(value)
                    setattr(current, parts[-1], value)
                except (AttributeError, KeyError, TypeError) as error:
                    raise ContractError(f"resolved task_config lacks writable {path}") from error

    def _active(self, payload):
        if self.session is None or payload.get("session_id") != self.session["session_id"]:
            raise ContractError("reset session first")
        return self.session

    def _native_action(self, action, subtask=False):
        if action.kind == "stop":
            return self.subtask_stop_action if subtask else "stop"
        if action.kind == "camera_tilt":
            if abs(abs(action.values["delta_rad"]) - self.turn_rad) > 1e-3:
                raise PolicyViolation("camera tilt has the wrong fixed increment")
            return "look_up" if action.values["delta_rad"] > 0 else "look_down"
        name, amount = action.values["name"], action.values["amount"]
        expected = self.forward_m if name == "forward" else self.turn_rad
        if abs(amount - expected) > 1e-3:
            raise PolicyViolation(f"{name} action has the wrong fixed increment")
        return {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}[name]

    def _projected_goals(self, episode):
        raw = self.goal_projector(episode)
        if not isinstance(raw, list) or not raw or any(not isinstance(x, dict) for x in raw):
            raise ContractError("upstream episode has no public goal projection")
        for goal in raw:
            if set(goal) != {"kind", "value"} or goal["kind"] not in {"language", "object_category", "image_reference", "embedding_reference"}:
                raise ContractError("goal projection must contain only public kind/value")
            if not isinstance(goal["value"], str) or not goal["value"]:
                raise ContractError("public goal value must be a nonempty string")
        return raw

    def _task_goal(self, state):
        active = state["goals"][state["active_goal"]]
        return json.dumps({"schema": "nav-eval-task-goal/1", "active_index": state["active_goal"],
                           "count": len(state["goals"]), **active}, ensure_ascii=True, separators=(",", ":"))

    def _upstream_active_goal(self):
        task = getattr(self.env, "task", None) or getattr(self.env, "_task", None)
        value = getattr(task, "active_subtask_idx", None)
        if type(value) is not int:
            raise ContractError("sequential benchmark task does not expose authoritative active_subtask_idx")
        return value

    def _find_obs(self, obs, candidates):
        for name in candidates:
            if name in obs:
                return obs[name]
        return None

    def _observation(self, obs, sequence):
        import numpy as np
        state = self.session
        payloads, specs = {}, {}
        for name, modality, unit in (("rgb", "rgb", "srgb"), ("depth", "depth", "m")):
            if name not in self.sensors:
                continue
            value = obs.get(name)
            if not isinstance(value, np.ndarray):
                raise ContractError(f"upstream observation missing {name}")
            if value.shape[:2] != (self.height, self.width):
                raise ContractError(f"upstream {name} geometry differs from declared width/height")
            if name == "rgb" and (value.ndim != 3 or value.shape[-1] != 3):
                raise ContractError("rgb must be HxWx3")
            if name == "depth" and value.ndim == 2:
                value = value[..., None]
            payloads[name] = encode_array(value)
            specs[name] = asdict(SensorSpec(modality, value.dtype.name, tuple(value.shape), unit, "camera_optical", "rendered",
                                             {"width": self.width, "height": self.height, "hfov_deg": self.hfov}))
        task_goal = self._task_goal(state)
        payloads["task_goal"] = _utf8_tensor(task_goal)
        specs["task_goal"] = asdict(SensorSpec("task_goal", "uint8", (len(task_goal.encode("utf-8")),), "utf8_json", "none", "measured", {"schema": "nav-eval-task-goal/1"}))
        return {"episode_id": state["episode_id"], "sequence": sequence, "sim_time_s": None,
                "control_tick": state["steps"], "sensors": payloads, "sensor_specs": specs}

    def _metrics(self):
        values = self.env.get_metrics()
        def public(value):
            value = _scalar(value)
            if isinstance(value, bool) or isinstance(value, int):
                return value
            if isinstance(value, float) and math.isfinite(value):
                return value
            if isinstance(value, dict):
                result = {str(key): public(item) for key, item in value.items()}
                return {key: item for key, item in result.items() if item is not None}
            if isinstance(value, (list, tuple)):
                return [public(item) for item in value]
            return None
        captured = {}
        for key, value in values.items():
            value = public(value)
            if value is not None:
                captured[str(key)] = value
        return captured

    def _handle(self, operation, payload):
        if operation == "shutdown":
            self.backend.close()
            return {}
        if operation == "describe":
            return {"schema_version": SCHEMA_VERSION, "role": "benchmark", "id": f"upstream_{self.benchmark_id}",
                    "benchmark_id": self.benchmark_id, "benchmark_kind": "upstream_habitat_navigation", "real": True,
                    "simulator": "Habitat-Lab", "simulator_version": self.actual_version or self.declared_version,
                    "accepts_actions": ["primitive", *( ["camera_tilt"] if self.camera_tilt else []), "stop"], "offers_sensors": list(self.sensors),
                    "sensor_geometry": {"width": self.width, "height": self.height, "hfov_deg": self.hfov},
                    "clock": "discrete_control_ticks", "action_semantics": {"forward_m": self.forward_m, "turn_rad": self.turn_rad, "camera_tilt_rad": self.turn_rad},
                    "capture_schema": "upstream-habitat-captured-metrics/1",
                    "goal": {"dynamic_sensor": "task_goal"},
                    "note": "Upstream metrics are captured verbatim for diagnostic provenance; Nav-Eval does not claim metric parity."}
        if operation == "episodes":
            self._ensure_env()
            return list(self._episodes)
        if operation == "prepare":
            self._ensure_env()
            return {}
        if operation == "reset":
            self._ensure_env()
            if self.session is not None:
                raise ContractError("active session already exists")
            episode_id = str(payload["episode_id"])
            if episode_id not in self._episodes:
                raise ContractError(f"episode {episode_id} not in frozen selection")
            obs = self.backend.reset(self._episodes[episode_id], int(payload.get("seed", 0)))
            goals = self._projected_goals(self.env.current_episode)
            active_goal = self._upstream_active_goal() if self.sequence else 0
            if active_goal < 0 or active_goal >= len(goals):
                raise ContractError("upstream active subtask index is outside public sequence")
            public_goal = json.dumps({"schema": "nav-eval-goal-sequence/1", "count": len(goals), "sensor": "task_goal"}, separators=(",", ":")) if self.sequence else goals[0]["value"]
            kind = "goal_sequence" if self.sequence else goals[0]["kind"]
            self.session = {"session_id": payload.get("session_id"), "episode_id": episode_id, "steps": 0,
                            "goals": goals, "active_goal": active_goal, "stopped": False, "truncated": False,
                            "upstream_metrics": []}
            self.session["upstream_metrics"].append(self._metrics())
            context = EpisodeContext(episode_id, Goal(kind, public_goal), f"habitat_upstream_{self.declared_version}", int(payload.get("seed", 0)))
            return {"context": asdict(context), "observation": self._observation(obs, 0),
                    "decision_limit": self.max_episode_steps}
        state = self._active(payload)
        if operation == "step":
            if state["stopped"] or state["truncated"]:
                raise ContractError("episode already ended")
            if payload.get("expected_sequence") != state["steps"]:
                raise ContractError("stale or duplicate step")
            action = validate_action(payload["action"])
            if action.kind not in ({"primitive", "camera_tilt", "stop"} if self.camera_tilt else {"primitive", "stop"}):
                raise PolicyViolation(f"{action.kind} is unsupported")
            subtask_stop = self.sequence and action.kind == "stop"
            obs = self.backend.step(self._native_action(action, subtask_stop))
            state["steps"] += 1
            if self.sequence:
                active_goal = self._upstream_active_goal()
                if active_goal < 0 or active_goal > len(state["goals"]):
                    raise ContractError("upstream active subtask index is outside public sequence")
                state["active_goal"] = min(active_goal, len(state["goals"]) - 1)
                state["stopped"] = active_goal >= len(state["goals"])
            elif action.kind == "stop":
                state["stopped"] = True
            state["truncated"] = not state["stopped"] and state["steps"] >= self.max_episode_steps
            # A failed FOUND may end MultiON before reaching its last goal.
            if getattr(self.env, "episode_over", False) and not state["truncated"]:
                state["stopped"] = True
            state["upstream_metrics"].append(self._metrics())
            return {"observation": self._observation(obs, state["steps"]), "terminated": state["stopped"], "truncated": state["truncated"]}
        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            return {"record": {"episode_id": state["episode_id"], "status": "completed", "steps": state["steps"],
                                "termination": "stop" if state["stopped"] else "budget", "control_tick": state["steps"],
                                "sim_time_s": None, "sim_clock": "discrete_control_ticks"},
                    "evidence": {"schema": "upstream-habitat-captured-metrics/1", "benchmark_id": self.benchmark_id, "provenance": {"kind": "upstream_captured", "simulator_version": self.declared_version, "task_config": self._asset_name(self.task_config), "dataset": self._asset_name(self.dataset_path)}, "upstream_metrics_by_tick": state["upstream_metrics"]}}
        if operation == "close_episode":
            self.session = None
            return {}
        raise ContractError(f"unsupported benchmark operation: {operation}")

    def asset_files(self):
        files = {self.task_config, self.dataset_path}
        files.update(self.dataset_path.parent.glob("content/*.json.gz"))
        for episode in getattr(self, "_episodes", {}).values():
            scene = getattr(episode, "scene_id", None)
            if scene:
                path = (self.repo_root / scene).resolve()
                if path.is_file():
                    files.add(path)
                    files.update(path.parent.glob("*.navmesh"))
        for path in self.extra_assets:
            if path.is_dir():
                files.update(p for p in path.rglob("*") if p.is_file())
            else:
                files.add(path)
        return {self._asset_name(path): path for path in sorted(files)}

    def runtime_identity(self):
        return {**getattr(self.backend, "runtime_identity", {"habitat_lab": self.actual_version or self.declared_version}),
                "resolved_task_config": getattr(self, "_resolved_config", None)}

    def close(self):
        self._dispatcher.close()
