"""v0.2 wire contracts. Simulator-private state must not enter PolicyObservation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

SCHEMA_VERSION = "nav-eval/0.2"


class ContractError(ValueError):
    pass


class PolicyViolation(ContractError):
    """An invalid action or protocol misuse attributable to the method under test.

    Benchmark workers raise this for method-caused violations so the runner can
    attribute the failure to the policy instead of infrastructure. Transport
    preserves the class across process boundaries via an error kind tag.
    """


@dataclass(frozen=True)
class Goal:
    kind: str  # language | object_category | image_reference; future kinds are negotiated
    value: str
    language: str = "en"


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    goal: Goal
    embodiment: str
    seed: int
    # No reference path, goal coordinates, scene mesh, or metrics in this type.


@dataclass(frozen=True)
class SensorSpec:
    modality: str
    dtype: str
    shape: tuple[int, ...]
    unit: str
    frame: str
    source: str  # rendered | measured | estimated | simulator_ground_truth
    calibration: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyObservation:
    episode_id: str
    sequence: int
    sim_time_s: float | None
    sensors: dict[str, Any]
    sensor_specs: dict[str, SensorSpec]
    control_tick: int | None = None
    # Deliberately no generic `info` escape hatch.


@dataclass(frozen=True)
class Action:
    kind: str  # stop | primitive | twist | camera_tilt | wait; trajectory_se2 reserved
    values: dict[str, Any] = field(default_factory=dict)
    frame: str = "base_link"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionBatch:
    episode_id: str
    observation_sequence: int
    actions: list[Action]
    # First version executes each action in order and checks termination after each.


class MethodAdapter(Protocol):
    """Owns all method state, including map, KV cache, and both systems of N1."""

    def describe(self) -> dict[str, Any]: ...
    def reset(self, context: EpisodeContext) -> None: ...
    def act(self, observation: PolicyObservation) -> ActionBatch: ...
    def observe_transition(self, transition: dict[str, Any]) -> None: ...
    def close_episode(self) -> None: ...


class SimulatorBackend(Protocol):
    """Internal to the benchmark worker, never sent to the method process."""

    def initialize(self, task_config: Any) -> Any: ...
    def reset(self, private_episode: Any, seed: int) -> Any: ...
    def step(self, native_action: Any) -> Any: ...
    def close(self) -> None: ...


class BenchmarkAdapter(Protocol):
    """Owns dataset, observation allowlist, termination and evidence capture; metrics are separate."""

    def describe(self) -> dict[str, Any]: ...
    def episode_ids(self) -> list[str]: ...
    def reset(self, episode_id: str, seed: int) -> tuple[EpisodeContext, PolicyObservation]: ...
    def step(self, action: Action) -> tuple[PolicyObservation, bool, bool]: ...
    def finish(self) -> dict[str, Any]: ...  # execution facts and evaluator-only evidence
    def close(self) -> None: ...


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)


def validate_action(raw: dict[str, Any]) -> Action:
    if not isinstance(raw, dict) or set(raw) - {"kind", "values", "frame"}:
        raise ContractError("invalid action envelope")
    kind, values = raw.get("kind"), raw.get("values", {})
    if not isinstance(values, dict) or raw.get("frame", "base_link") != "base_link":
        raise ContractError("action values must be an object in base_link")
    if kind == "stop":
        if values:
            raise ContractError("stop has no payload")
    elif kind == "primitive":
        if set(values) != {"name", "amount"} or values["name"] not in {"forward", "left", "right"}:
            raise ContractError("primitive requires name and amount (meters or radians)")
        amount = finite_number(values["amount"], "amount")
        if amount <= 0:
            raise ContractError("primitive amount must be positive")
    elif kind == "camera_tilt":
        if set(values) != {"delta_rad"}:
            raise ContractError("camera_tilt requires relative delta_rad (positive looks up)")
        angle = finite_number(values["delta_rad"], "delta_rad")
        if not 0 < abs(angle) <= math.pi:
            raise ContractError("camera tilt must be nonzero and at most pi radians")
    elif kind == "wait":
        if set(values) != {"duration_s"} or finite_number(values["duration_s"], "duration_s") <= 0:
            raise ContractError("wait requires positive simulation duration")
    elif kind == "twist":
        if set(values) != {"linear_mps", "angular_rps", "duration_s"}:
            raise ContractError("twist requires linear_mps, angular_rps, duration_s")
        for key, value in values.items():
            finite_number(value, key)
        if values["duration_s"] <= 0:
            raise ContractError("duration_s must be positive")
    else:
        raise ContractError(f"unsupported action codec: {kind}")
    return Action(kind, dict(values))


def validate_observation(raw: dict[str, Any]) -> None:
    required = {"episode_id", "sequence", "sim_time_s", "sensors", "sensor_specs"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {"control_tick"}:
        raise ContractError("policy observation must contain only public contract fields")
    if not isinstance(raw["sequence"], int) or isinstance(raw["sequence"], bool) or raw["sequence"] < 0:
        raise ContractError("invalid observation sequence")
    tick = raw.get("control_tick")
    if tick is not None and (type(tick) is not int or tick < 0):
        raise ContractError("invalid control_tick")
    if raw["sim_time_s"] is None:
        if tick is None:
            raise ContractError("observation requires simulation time or an explicit control tick")
    elif finite_number(raw["sim_time_s"], "sim_time_s") < 0:
        raise ContractError("simulation time must be nonnegative")
    if not isinstance(raw["sensors"], dict) or set(raw["sensors"]) != set(raw["sensor_specs"]):
        raise ContractError("sensor payload and metadata names must match")
