"""Method lifecycle and public-observation boundary used by every worker."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Protocol

from nav_eval.contracts import (
    Action,
    ActionBatch,
    ContractError,
    EpisodeContext,
    PolicyObservation,
    validate_action,
    validate_observation,
)


@dataclass(frozen=True)
class RuntimeReply:
    generation_id: str
    observation_sequence: int
    actions: list[Action | dict]


class UpstreamRuntime(Protocol):
    """Worker-local model runtime wrapped by the common method boundary."""

    accepts_transition: bool

    def reset(self, context: EpisodeContext, generation_id: str) -> None: ...
    def act(self, observation: PolicyObservation, generation_id: str) -> RuntimeReply: ...
    def observe_transition(self, transition: dict, generation_id: str) -> None: ...
    def close_episode(self) -> None: ...


class MethodBoundaryAdapter:
    """Validate method inputs, outputs, ordering, and episode ownership."""

    def __init__(self, runtime: UpstreamRuntime, *, sensor_names=None,
                 required_sensors=(), emitted_actions=()):
        self.required_sensors = tuple(required_sensors)
        self.emitted_actions = tuple(emitted_actions)
        self.runtime = runtime
        self.sensor_names = tuple(sensor_names or self.required_sensors)
        if set(self.required_sensors) - set(self.sensor_names):
            raise ContractError("cannot omit required method sensors")
        self.context = None
        self.generation_id = None
        self.last_decision = -1
        self.last_feedback = 0
        self.ended = False

    def reset(self, context: EpisodeContext):
        # Goal-kind acceptance is method policy, not a platform rule: the wire
        # contract negotiates language | object_category | image_reference, and
        # each method runtime rejects kinds it cannot consume.
        self.close_episode()
        generation = uuid.uuid4().hex
        try:
            self.runtime.reset(context, generation)
        except Exception:
            self.runtime.close_episode()
            raise
        self.context, self.generation_id = context, generation
        self.last_decision, self.last_feedback, self.ended = -1, 0, False

    def _public_observation(self, observation):
        validate_observation(vars(observation))
        if self.context is None or observation.episode_id != self.context.episode_id:
            raise ContractError("observation belongs to an inactive or different episode")
        missing = set(self.sensor_names) - set(observation.sensors)
        if missing:
            raise ContractError(f"missing method sensors: {sorted(missing)}")
        for name in self.sensor_names:
            spec = observation.sensor_specs[name]
            if spec.source not in {"rendered", "measured", "estimated"}:
                raise ContractError(f"privileged or unknown sensor source: {name}")
            if spec.modality == "depth" and spec.unit != "m":
                raise ContractError("canonical depth must be meters; convert at the simulator boundary")
        return PolicyObservation(
            observation.episode_id,
            observation.sequence,
            observation.sim_time_s,
            {name: observation.sensors[name] for name in self.sensor_names},
            {name: observation.sensor_specs[name] for name in self.sensor_names},
            observation.control_tick,
        )

    def act(self, observation: PolicyObservation) -> ActionBatch:
        public = self._public_observation(observation)
        if self.ended or public.sequence <= self.last_decision:
            raise ContractError("ended episode or stale observation")
        if self.runtime.accepts_transition and public.sequence != self.last_feedback:
            raise ContractError("missing transition feedback before decision")
        self.last_decision = public.sequence
        reply = self.runtime.act(public, self.generation_id)
        if reply.generation_id != self.generation_id or reply.observation_sequence != public.sequence:
            raise ContractError("stale runtime reply: generation or observation mismatch")
        if not isinstance(reply.actions, list) or not 1 <= len(reply.actions) <= 16:
            raise ContractError("runtime must return 1..16 explicit actions; empty is not STOP")
        actions = [validate_action(action) for action in reply.actions]
        if any(action.kind not in self.emitted_actions for action in actions):
            raise ContractError("runtime action not declared by this method variant")
        return ActionBatch(public.episode_id, public.sequence, actions)

    def observe_transition(self, transition: dict):
        if not self.runtime.accepts_transition:
            raise ContractError("runtime does not accept transition feedback")
        allowed = {"episode_id", "sequence", "executed_action", "observation", "terminated", "truncated"}
        if set(transition) != allowed or self.context is None or self.ended or self.last_decision < 0:
            raise ContractError("invalid public transition or inactive decision")
        raw = transition["observation"]
        validate_observation(raw)
        from nav_eval.contracts import SensorSpec

        observation = PolicyObservation(
            raw["episode_id"],
            raw["sequence"],
            raw["sim_time_s"],
            raw["sensors"],
            {name: SensorSpec(**spec) for name, spec in raw["sensor_specs"].items()},
            raw.get("control_tick"),
        )
        public = self._public_observation(observation)
        if (transition["episode_id"] != self.context.episode_id
                or transition["sequence"] != self.last_feedback + 1
                or public.sequence != transition["sequence"]):
            raise ContractError("missing, duplicate or misrouted transition")
        if any(type(transition[key]) is not bool for key in ("terminated", "truncated")):
            raise ContractError("termination fields must be booleans")
        action = validate_action(transition["executed_action"])
        self.runtime.observe_transition(
            {**transition, "observation": asdict(public), "executed_action": action.to_dict()},
            self.generation_id,
        )
        self.last_feedback = public.sequence
        self.ended = transition["terminated"] or transition["truncated"]

    def close_episode(self):
        active = self.context is not None
        self.context, self.generation_id = None, None
        self.ended = True
        if active:
            self.runtime.close_episode()
