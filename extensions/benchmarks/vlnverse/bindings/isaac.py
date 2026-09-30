"""VLNVerse task binding for InternUtopia + Isaac Sim 5.0.0.

Runs through the interpreter/resource mapping for the fixed Isaac 5.0 runtime.
The protocol binding is not a certificate of task or model parity.

The service mirrors the habitat benchmark surface so the same generic rollout
runner, evidence capture and offline evaluator apply:

- discrete actions map 1:1 onto the challenge's `move_by_discrete` controller
  (forward 0.25 m / turn 15 deg) and `stop`; each contract step loops the
  InternUtopia env until the controller reports the action finished, exactly
  like the upstream evaluator's env_step loop;
- per-episode warm-up (`stand_still` until finish_action) matches upstream;
- observations expose rgb (+camera pose) from the task's get_rgb_depth;
- evidence (vlnverse-evidence/2) records the trajectory (world pose minus
  env_offset), the full reference_path and the official metric inputs, so the
  offline metric set reproduces the challenge's VLNPEMetric formulas
  (NE/SR/OSR/SPL/nDTW over xy coordinates, geodesic = info.geodesic_distance
  or the reference path length).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import queue
from pathlib import Path

from nav_eval.contracts import (
    ContractError, EpisodeContext, Goal, PolicyViolation, SCHEMA_VERSION,
    SensorSpec, validate_action,
)
from nav_eval.tensorcode import encode_array
from dataclasses import asdict

EVIDENCE_SCHEMA = "vlnverse-evidence/2"
VLNVERSE_BENCHMARKS = {
    "vlnverse_fine_val_unseen": "fine_val_unseen.json.gz",
    "vlnverse_coarse_val_unseen": "coarse_val_unseen.json.gz",
}
ROBOT_OFFSET = [0.0, 0.0, 1.05]
WARM_UP_STEP = 30


def make_simulation_config(settings, episodes):
    from internutopia.core.config import Config, SimConfig

    # Pydantic keeps subclass instances but truncates subclass-only fields when
    # dictionaries are revalidated as TaskCfg/MetricCfg/RobotCfg base models.
    return Config(simulator=SimConfig(**settings), env_num=1, env_offset_size=100,
                  task_configs=episodes)


class VLNVerseBenchmarkService:
    def __init__(self, challenge_repo, data_root="data", benchmark_id="vlnverse_fine_val_unseen",
                 episode_limit=None, gpu_device_id=0, sensors=("rgb",), success_distance_m=3.0,
                 camera_resolution=(640, 480), backend=None, episode_ids=None,
                 max_control_steps=500, max_task_steps=25000):
        if benchmark_id not in VLNVERSE_BENCHMARKS:
            raise ValueError(f"unknown vlnverse benchmark; choose from {sorted(VLNVERSE_BENCHMARKS)}")
        if set(sensors) - {"rgb"}:
            raise ValueError("this adapter exposes rgb only (depth is normalised upstream and "
                             "not part of the challenge observation contract used here)")
        self.challenge_repo = Path(challenge_repo).resolve()
        self.data_root = Path(data_root).resolve()
        self.benchmark_id = benchmark_id
        self.episode_limit = episode_limit
        self.gpu_device_id = gpu_device_id
        self.sensors = tuple(sensors)
        self.success_distance_m = success_distance_m
        self.width, self.height = int(camera_resolution[0]), int(camera_resolution[1])
        if any(type(value) is not int or value <= 0 for value in (max_control_steps, max_task_steps)):
            raise ValueError("control and upstream task budgets must be positive integers")
        self.max_control_steps, self.max_task_steps = max_control_steps, max_task_steps
        self.env = None
        self.session = None
        self._mailbox = queue.Queue()
        # Isaac must be driven from the thread that created it, and the kit
        # bootstrap installs signal handlers that require the MAIN thread.
        # run_main() is therefore executed by the serve entrypoint on the main
        # thread; the HTTP server runs in a background thread instead.
        self._sim_thread = None
        from extensions.simulators.isaacsim500.backend import IsaacBackend
        self.backend = backend or IsaacBackend({"version": "5.0.0"})
        self.requested_ids = episode_ids

    def run_main(self):
        """Serve operations on the calling (main) thread; blocks forever."""
        while True:
            operation, payload, result = self._mailbox.get()
            try:
                result.put(("ok", self._handle(operation, payload)))
            except BaseException as error:
                result.put(("error", error))

    def call(self, operation, payload):
        result = queue.Queue()
        self._mailbox.put((operation, payload, result))
        status, value = result.get()
        if status == "error":
            raise value
        return value

    # -- episodes ------------------------------------------------------------
    @property
    def _dataset_path(self):
        return self.data_root / "datasets" / "vlnverse" / VLNVERSE_BENCHMARKS[self.benchmark_id]

    def _episode_manifest(self):
        with gzip.open(self._dataset_path, "rt", encoding="utf-8") as handle:
            episodes = json.load(handle)["episodes"]

        def sort_key(episode):
            # vlnverse ids are strings like "kujiale_0272_26_1"
            try:
                return (0, int(episode["episode_id"]), "")
            except (TypeError, ValueError):
                return (1, 0, str(episode["episode_id"]))

        episodes.sort(key=sort_key)
        if self.requested_ids is not None:
            episodes = [e for e in episodes if str(e["episode_id"]) in self.requested_ids]
            if set(self.requested_ids) - {str(e["episode_id"]) for e in episodes}:
                raise ContractError("requested VLNVerse episodes absent from split")
        return episodes[: self.episode_limit] if self.episode_limit is not None else episodes

    # -- internutopia env ------------------------------------------------------
    def _episode_data(self, raw):
        data = dict(raw)
        data["start_position"] = [v + o for v, o in zip(raw["start_position"], ROBOT_OFFSET)]
        if "reference_path" in raw and raw["reference_path"] is not None:
            data["reference_path"] = [
                [v + o for v, o in zip(point, ROBOT_OFFSET)] for point in raw["reference_path"]
            ]
        data["path_key"] = f"{raw['trajectory_id']}_{raw['episode_id']}"
        data["name"] = "nav_eval"
        return data

    def _ensure_env(self):
        if self.env is not None:
            return
        previous_cwd = os.getcwd()
        os.chdir(self.challenge_repo)
        import sys

        if str(self.challenge_repo) not in sys.path:
            sys.path.insert(0, str(self.challenge_repo))
        try:
            from vlnverse.configs.agent import AgentCfg
            from vlnverse.configs.evaluator import (
                EnvCfg, EvalCfg, EvalDatasetCfg, MetricCfg, SceneCfg, TaskCfg,
            )
            from vlnverse.configs.evaluator.default_config import get_config
            from vlnverse.evaluator.utils.common import load_kujiale_scene_usd
            from vlnverse.projects.internutopia_vln_extension import import_extensions
            from vlnverse.projects.internutopia_vln_extension.configs.metrics.vln_pe_metrics import (
                VLNPEMetricCfg,
            )
            from vlnverse.projects.internutopia_vln_extension.configs.tasks.vln_eval_task import (
                VLNEvalTaskCfg,
            )

            scene_data_dir = str(self.challenge_repo / "data" / "scene_data" / "vlnverse")
            eval_cfg = EvalCfg(
                agent=AgentCfg(server_port=0, model_name="strive", ckpt_path="",
                               model_settings={"env_num": 1, "proc_num": 1,
                                               "device": f"cuda:{self.gpu_device_id}"}),
                env=EnvCfg(env_type="vln_pe",
                           env_settings={"use_fabric": False, "headless": True}),
                task=TaskCfg(
                    task_name="nav_eval",
                    task_settings={"env_num": 1, "use_distributed": False, "proc_num": 1,
                                   "max_step": self.max_task_steps, "warm_up_step": WARM_UP_STEP,
                                   "check_fall_and_stuck": True, "robot_ankle_height": 0.0758,
                                   "fall_height_threshold": 0.5, "offset_size": 100},
                    scene=SceneCfg(scene_type="kujiale", scene_asset_path="",
                                   scene_data_dir=scene_data_dir),
                    robot_name="h1",
                    robot_flash=False,
                    robot_usd_path=str(self.challenge_repo / "data" / "Embodiments" / "vln-pe"
                                       / "h1" / "h1_internvla.usd"),
                    camera_resolution=[self.width, self.height],
                    camera_prim_path="torso_link/h1_1_25_down_30",
                    one_step_stand_still=True,
                    metric=MetricCfg(save_dir="",
                                     metric_setting={"type": "VLNPEMetric", "name": "VLNPEMetric",
                                                     "metric_config": VLNPEMetricCfg(
                                                         success_distance=self.success_distance_m,
                                                         shortest_to_goal_distance=999).model_dump()}),
                ),
                dataset=EvalDatasetCfg(dataset_type="kujiale",
                                       dataset_settings={"base_data_dir": "data/vlnverse",
                                                         "split_data_types": ["challenge"],
                                                         "filter_stairs": False}),
                eval_settings={"save_to_json": False, "vis_output": False},
            )
            cfg = get_config(eval_cfg)  # mutates eval_cfg: fills task.robot and task_settings
            metric_config = dict(eval_cfg.task.metric.metric_setting["metric_config"])
            if metric_config.get("name") is None:
                metric_config["name"] = "nav_eval_metric"  # upstream patches the same None name
            # rebuild as H1RobotCfg (the only variant with .update()); its
            # controllers take internutopia's FLAT controller dicts, which is
            # exactly what RobotCfg.controller_settings holds.
            from internutopia_extension.configs.robots.h1 import H1RobotCfg
            from internutopia_extension.configs.sensors import RepCameraCfg

            robot = H1RobotCfg(
                **eval_cfg.task.robot.robot_settings,
                controllers=[dict(c.controller_settings)
                             for c in eval_cfg.task.robot.controllers],
                sensors=[RepCameraCfg(**s.sensor_settings)
                         for s in eval_cfg.task.robot.sensors],
            )
            # Keep one Kit application for the worker's lifetime. InternUtopia
            # reset replaces the scene for the selected task; Env.close shuts
            # down Kit itself and cannot be used as a scene-switch operation.
            episodes = self._episode_manifest()
            task_episodes = []
            for raw in episodes:
                data = self._episode_data(raw)
                task_episodes.append(VLNEvalTaskCfg(
                    **eval_cfg.task.task_settings,
                    robot_flash=False, one_step_stand_still=True,
                    metrics=[VLNPEMetricCfg(**metric_config)],
                    scene_asset_path=load_kujiale_scene_usd(scene_data_dir, raw["scan"]),
                    scene_scale=(1, 1, 1),
                    robots=[robot.update(position=tuple(data["start_position"]),
                                         orientation=tuple(data["start_rotation"]))],
                    data=data, scene_type="kujiale",
                ))
            cfg.task.task_settings["episodes"] = task_episodes
            import_extensions()

            config = make_simulation_config(cfg.env.env_settings, task_episodes)
            self.env = self.backend.initialize(config)
            self._episode_indices = {str(raw["episode_id"]): index
                                     for index, raw in enumerate(episodes)}
        finally:
            os.chdir(previous_cwd)

    # -- helpers ---------------------------------------------------------------
    def _step_until_action_complete(self, action):
        return self.backend.step(action)

    def _warm_up(self):
        action = {self.backend.robot_name: {"stand_still": []}}
        for _ in range(WARM_UP_STEP * 4):
            robot_obs, ended = self._step_until_action_complete(action)
            if robot_obs.get("finish_action"):
                return robot_obs
        raise ContractError("vlnverse warm-up did not converge")

    def _observation(self, robot_obs, sequence):
        payloads, specs = {}, {}
        if "rgb" in self.sensors and "rgb" in robot_obs:
            payloads["rgb"] = encode_array(robot_obs["rgb"])
            specs["rgb"] = asdict(SensorSpec(
                "rgb", "uint8", (self.height, self.width, 3), "srgb", "camera_optical", "rendered",
                {"hfov_deg": 90, "width": self.width, "height": self.height,
                 "note": "challenge H1 head camera; intrinsics in obs camera_params"}))
        return {"episode_id": self.session["episode_id"], "sequence": sequence,
                "sim_time_s": None, "control_tick": self.session["steps"], "sensors": payloads,
                "sensor_specs": specs}

    # -- service surface ---------------------------------------------------------
    def _handle(self, operation, payload):
        if operation == "prepare":
            episodes = self._episode_manifest()
            if not episodes:
                raise ContractError("empty VLNVerse selection")
            self._handle("reset", {"episode_id": str(episodes[0]["episode_id"]), "session_id": "preflight", "seed": 0})
            self._handle("close_episode", {"session_id": "preflight"})
            return {}
        if operation == "shutdown":
            self.backend.close()
            return {}
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "benchmark",
                "id": self.benchmark_id, "benchmark_kind": "vln_continuous",
                "benchmark_id": self.benchmark_id,
                "simulator": "isaac-sim 5.0.0 (standalone) + InternUtopia",
                "simulator_version": "isaacsim-5.0.0",
                "real": True,
                "accepts_actions": ["primitive", "stop"],
                "offers_sensors": list(self.sensors),
                "sensor_geometry": {"width": self.width, "height": self.height, "hfov_deg": 90},
                "coordinate_frame": "isaac_world_minus_env_offset",
                "clock": "discrete_control_ticks",
                "max_control_steps": self.max_control_steps,
                "max_task_steps": self.max_task_steps,
                "capture_schema": EVIDENCE_SCHEMA,
                "success_distance_m": self.success_distance_m,
                "oracle_follower_available": False,
                "note": "H1 embodiment and discrete controller; fixed Isaac 5.0 binding, parity unverified.",
            }
        if operation == "episodes":
            manifest = self._episode_manifest()
            manifest.sort(key=lambda e: (e["scan"], str(e["episode_id"])))
            return [str(e["episode_id"]) for e in manifest]

        session_id = payload.get("session_id")
        if operation == "reset":
            if self.session is not None:
                raise ContractError("active session already exists")
            episode_id = str(payload["episode_id"])
            raw = next((e for e in self._episode_manifest()
                        if str(e["episode_id"]) == episode_id), None)
            if raw is None:
                raise ContractError(f"episode {episode_id} not in the frozen local selection")
            self._ensure_env()
            index = self._episode_indices[episode_id]
            self.backend.reset(index, int(payload.get("seed", 0)))
            robot_obs = self._warm_up()
            position = [float(v) for v in robot_obs["globalgps"]]
            self.session = {
                "session_id": session_id, "episode_id": episode_id,
                "raw": raw, "data": self._episode_data(raw),
                "steps": 0, "stopped": False, "truncated": False,
                "positions": [position], "last_obs": robot_obs,
            }
            context = EpisodeContext(
                episode_id, Goal("language", raw["instruction"]["instruction_text"]),
                "vlnverse_h1", int(payload.get("seed", 0)),
            )
            return {"context": asdict(context),
                    "observation": self._observation(robot_obs, 0)}

        state = self._active_session(payload)
        if operation == "step":
            if state["stopped"] or state["truncated"]:
                raise ContractError("episode already ended")
            if payload["expected_sequence"] != state["steps"]:
                raise ContractError("stale or duplicate step: episode must be restarted")
            action = validate_action(payload["action"])
            if action.kind not in {"primitive", "stop"}:
                raise PolicyViolation(f"{action.kind} unsupported by this benchmark")
            if action.kind == "primitive":
                name, amount = action.values["name"], action.values["amount"]
                if name == "forward" and abs(amount - 0.25) > 1e-3:
                    raise PolicyViolation("forward amount must be 0.25 m on this benchmark")
                if name in {"left", "right"} and abs(amount - 0.2617993877991494) > 1e-3:
                    raise PolicyViolation("turn amount must be 15 degrees on this benchmark")
                code = {"forward": 1, "left": 2, "right": 3}[name]
                command = {self.backend.robot_name: {"move_by_discrete": [code]}}
            else:
                command = {self.backend.robot_name: {"stop": []}}
            robot_obs, ended = self._step_until_action_complete(command)
            state["steps"] += 1
            state["positions"].append([float(v) for v in robot_obs["globalgps"]])
            state["last_obs"] = robot_obs
            if action.kind == "stop":
                state["stopped"] = True
            state["truncated"] = (ended or state["steps"] >= self.max_control_steps) and not state["stopped"]
            return {"observation": self._observation(robot_obs, state["steps"]),
                    "terminated": state["stopped"], "truncated": state["truncated"]}

        if operation == "finish":
            if not (state["stopped"] or state["truncated"]):
                raise ContractError("cannot finish an active episode")
            raw, data = state["raw"], state["data"]
            positions = state["positions"]
            reference = data.get("reference_path") or []
            goal = (reference[-1] if reference else raw["goals"]["position"])
            geodesic = raw.get("info", {}).get("geodesic_distance", -1)
            if geodesic is None or geodesic <= 0:
                geodesic = sum(math.dist(a, b) for a, b in zip(reference, reference[1:])) if reference else -1
            metrics = (state["last_obs"].get("metrics") or {})
            if not metrics:
                # Capture private measurement state if our control budget ends
                # before the upstream task. These data never enter policy RPCs.
                tasks = list(self.env.runner.current_tasks.values())
                if len(tasks) != 1:
                    raise ContractError("expected one active VLNVerse task at capture")
                metrics = tasks[0].calculate_metrics()
            reference_metrics = {}
            metric_positions, path_length = None, None
            for value in (metrics.values() if isinstance(metrics, dict) else []):
                entries = value if isinstance(value, list) else [value]
                for entry in entries:
                    if isinstance(entry, dict):
                        if "pred_path" in entry and "TL" in entry:
                            metric_positions = [[float(v) for v in p] for p in entry["pred_path"]]
                            path_length = float(entry["TL"])
                        reference_metrics = {k: v for k, v in entry.items()
                                             if isinstance(v, (int, float))}
            record = {
                "episode_id": state["episode_id"], "status": "completed",
                "termination": "stop" if state["stopped"] else "budget",
                "steps": state["steps"], "control_tick": state["steps"], "sim_time_s": None,
                "sim_clock": "discrete_control_ticks",
                "environment_end_reason": (state["last_obs"].get("fail_reason") or
                                           ("control_budget" if state["truncated"] else "stop")),
                "upstream_step_count": reference_metrics.get("steps"),
            }
            evidence = {
                "schema": EVIDENCE_SCHEMA,
                "reference": {
                    "scene_id": raw["scene_id"], "scan": raw["scan"],
                    "goal_xyz_m": list(goal),
                    "reference_path_xyz_m": [list(p) for p in reference],
                    "geodesic_start_to_goal_m": float(geodesic),
                    "success_radius_m": self.success_distance_m,
                    "start_xyz_m": list(positions[0]),
                },
                "trajectory": {
                    "positions_xyz_m": [list(p) for p in positions],
                    "metric_positions_xyz_m": metric_positions,
                    "measurement_source": "upstream_metric_state",
                    "control_ticks": list(range(len(positions))),
                    "path_length_m": path_length,
                },
                "benchmark_reference_metrics": reference_metrics,
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

    def prepare(self):
        self.call("prepare", {})

    def asset_files(self):
        files = {"dataset/" + self._dataset_path.name: self._dataset_path}
        roots = [self.challenge_repo / "data" / "scene_data" / "vlnverse" / scan
                 for scan in sorted({episode["scan"] for episode in self._episode_manifest()})]
        roots.append(self.challenge_repo / "data" / "Embodiments" / "vln-pe" / "h1")
        for root in roots:
            if not root.is_dir():
                raise ContractError(f"missing scene or robot assets: {root}")
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    files[str(path.relative_to(self.challenge_repo))] = path
        return files

    def runtime_identity(self):
        version_path = Path(os.environ["ISAAC_PATH"]) / "VERSION"
        return {"isaac_build": version_path.read_text().strip()}

    def close(self):
        self.call("shutdown", {})


def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]
    backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"])
    settings.pop("enable_oracle", None)
    observation = config["observation"]
    if observation.get("hfov", 90) != 90:
        raise ContractError("the VLNVerse H1 binding requires hfov 90")
    settings["sensors"] = observation["sensors"]
    settings["camera_resolution"] = (observation.get("width", 640), observation.get("height", 480))
    return VLNVerseBenchmarkService(backend=backend, **settings)
