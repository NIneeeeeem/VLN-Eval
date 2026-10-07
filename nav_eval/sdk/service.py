"""Worker facade enforcing readiness, identity, and method session contracts."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

from nav_eval.contracts import (
    ContractError,
    EpisodeContext,
    Goal,
    PolicyObservation,
    PolicyViolation,
    SensorSpec,
    validate_observation,
)
from nav_eval.sdk.method import MethodBoundaryAdapter, RuntimeReply


def public_observation(raw, geometry=None):
    validate_observation(raw)
    specs = {}
    for name, metadata in raw["sensor_specs"].items():
        spec = specs[name] = SensorSpec(**metadata)
        if spec.source not in {"rendered", "measured", "estimated"}:
            raise ContractError(f"privileged sensor: {name}")
        if spec.modality == "depth" and spec.unit != "m":
            raise ContractError("depth must be in meters")
        shape = list(spec.shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ContractError("invalid sensor shape")
        payload = raw["sensors"][name]
        if isinstance(payload, dict):
            from nav_eval.tensorcode import validate_tensor
            validate_tensor(payload)
            if payload["shape"] != shape or payload["dtype"] != spec.dtype:
                raise ContractError("tensor metadata differs from sensor specification")
        else:
            def check(values, dims):
                if dims:
                    if not isinstance(values, (list, tuple)) or len(values) != dims[0]:
                        raise ContractError("sensor payload shape mismatch")
                    for value in values:
                        check(value, dims[1:])
                elif not isinstance(values, (int, float)) or isinstance(values, bool) or not math.isfinite(values):
                    raise ContractError("sensor payload must contain finite numbers")
                elif spec.dtype == "uint8" and (type(values) is not int or not 0 <= values <= 255):
                    raise ContractError("uint8 sensor value out of range")
            check(payload, shape)
        if spec.modality in {"rgb", "depth"} and geometry:
            for field, actual in (("width", shape[1]), ("height", shape[0]),
                                  ("hfov", spec.calibration.get("hfov_deg"))):
                if field in geometry and actual != geometry[field]:
                    raise ContractError(f"live sensor {field} differs from resolved observation")
    return PolicyObservation(raw["episode_id"], raw["sequence"], raw["sim_time_s"], raw["sensors"],
                             specs, raw.get("control_tick"))


class ServiceRuntime:
    """Adapt a method plugin service to the common worker boundary."""
    def __init__(self, service):
        self.service = service
        self.accepts_transition = service.call("describe", {}).get("accepts_transition", False)
        self.generation = None

    def reset(self, context, generation_id):
        self.generation = generation_id
        self.service.call("reset", {"session_id": generation_id, "context": asdict(context)})

    def act(self, observation, generation_id):
        result = self.service.call("act", {"session_id": generation_id, "observation": asdict(observation)})
        if result.get("episode_id") != observation.episode_id:
            raise PolicyViolation("method replied for a different episode")
        return RuntimeReply(generation_id, result.get("observation_sequence"), result.get("actions"))

    def observe_transition(self, transition, generation_id):
        self.service.call("observe_transition", {"session_id": generation_id, "transition": transition})

    def close_episode(self):
        if self.generation is not None:
            generation, self.generation = self.generation, None
            self.service.call("close_episode", {"session_id": generation})


class WorkerService:
    """Single active session, loaded readiness, and plugin identity for every worker."""
    def __init__(self, service, config):
        self.service, self.config = service, config
        self.ready = False
        self.session = None
        self.startup_s = 0.0
        self.boundary = None
        self.timing = {}

    def prepare(self):
        if self.ready:
            return
        start = time.perf_counter()
        if hasattr(self.service, "prepare"):
            self.service.prepare()
        caps = self.service.call("describe", {})
        if caps.get("role") != self.config["role"]:
            raise ContractError("worker role differs from manifest")
        if self.config["role"] == "method":
            expected = self.config["capabilities"]
            for key in ("requires_sensors", "emits_actions"):
                if set(expected[key]) != set(caps[key]):
                    raise ContractError(f"live method {key} differs from manifest")
            self.boundary = MethodBoundaryAdapter(ServiceRuntime(self.service),
                required_sensors=caps["requires_sensors"], emitted_actions=caps["emits_actions"])
        else:
            required = set(self.config.get("capabilities", {}).get("requires_sensors", []))
            if required - set(caps.get("offers_sensors", [])):
                raise ContractError("live benchmark does not offer required task sensors")
        self.startup_s = time.perf_counter() - start
        self.ready = True

    def call(self, operation, payload):
        if operation == "prepare":
            self.prepare()
            return self.call("describe", {})
        if operation == "describe":
            return {**self.service.call("describe", {}), "ready": self.ready,
                    "plugin_identity": self.config["plugin"], "startup_s": self.startup_s,
                    "session_capacity": 1, "transport_codecs": ["json", "binary"]}
        if operation == "timing":
            phases = self.service.timing_snapshot() if hasattr(self.service, "timing_snapshot") else {}
            return {**self.timing, "phases": phases}
        if operation == "attest":
            import importlib.metadata
            import platform

            from nav_eval.storage import file_digest
            files = self.service.asset_files() if hasattr(self.service, "asset_files") else []
            if not isinstance(files, dict):
                files = {str(path): path for path in files}
            return {"plugin": self.config["plugin"], "python": platform.python_version(),
                    "packages": sorted([d.metadata.get("Name", ""), d.version] for d in importlib.metadata.distributions()),
                    "runtime": self.service.runtime_identity() if hasattr(self.service, "runtime_identity") else {},
                    "assets": {name: file_digest(path) for name, path in files.items()},
                    "asset_paths": {name: str(path) for name, path in files.items()}}
        if not self.ready:
            raise ContractError("worker has not completed preparation")
        session = payload.get("session_id")
        if operation == "close_episode":
            if self.session is None:
                return {}
            if session != self.session:
                raise ContractError("cannot close another session")
            try:
                if self.boundary:
                    self.boundary.close_episode()
                else:
                    self.service.call(operation, payload)
            finally:
                self.session = None
            return {}
        if operation == "reset":
            if self.session is not None or not isinstance(session, str) or not session:
                raise ContractError("worker permits one active session")
            self.session = session
        elif operation not in {"episodes"} and session != self.session:
            raise ContractError("inactive or stale worker session")
        start = time.perf_counter()
        try:
            if not self.boundary:
                result = self.service.call(operation, payload)
                required = set(self.config.get("capabilities", {}).get("requires_sensors", []))
                if required and operation in {"reset", "step"}:
                    observation = result["observation"]
                    public_observation(observation, self.config.get("observation"))
                    if required - set(observation["sensors"]):
                        raise ContractError("live observation is missing required task sensors")
                    selected = set(self.config.get("observation", {}).get("sensors", []))
                    if set(observation["sensors"]) - selected:
                        raise ContractError("live observation includes undeclared sensors")
                return result
            if operation == "reset":
                context = payload["context"]
                if set(context) != {"episode_id", "goal", "embodiment", "seed"}:
                    raise ContractError("private or invalid episode context")
                accepted = self.config.get("capabilities", {}).get("accepts_goals")
                if accepted is not None and context["goal"].get("kind") not in accepted:
                    raise ContractError("method does not accept the episode goal kind")
                import random
                import sys
                random.seed(context["seed"])
                if "numpy" in sys.modules:
                    sys.modules["numpy"].random.seed(context["seed"])
                if "torch" in sys.modules:
                    sys.modules["torch"].manual_seed(context["seed"])
                self.boundary.reset(EpisodeContext(context["episode_id"], Goal(**context["goal"]),
                                                   context["embodiment"], context["seed"]))
                return {}
            if operation == "act":
                obs = public_observation(payload["observation"], self.config.get("observation"))
                try:
                    return asdict(self.boundary.act(obs))
                except ContractError as error:
                    raise PolicyViolation(str(error)) from error
            if operation == "observe_transition":
                public_observation(payload["transition"]["observation"], self.config.get("observation"))
                self.boundary.observe_transition(payload["transition"])
                return {}
            raise ContractError(f"unsupported method operation: {operation}")
        finally:
            self.timing[operation + "_s"] = self.timing.get(operation + "_s", 0.0) + time.perf_counter() - start

    def close(self):
        if self.session is not None:
            self.call("close_episode", {"session_id": self.session})
        if hasattr(self.service, "close"):
            self.service.close()
