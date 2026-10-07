"""Lifecycle wrapper around the bundled NaVid and Uni-NaVid inference agents."""
from __future__ import annotations

import tempfile
import time
from numbers import Integral

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

FORWARD_STEP_M = 0.25
TURN_RAD = 0.5235987755982988  # upstream R2R/RxR task configs: 30 degrees
VARIANTS = {
    "navid": 90,
    "uni_navid": 120,
}


class NaVidMethodService:
    """Delegate history, generation, parsing, and action queues to upstream."""

    def __init__(self, variant, checkpoint, vision_tower=None):
        if variant not in VARIANTS:
            raise ContractError(f"unknown NaVid variant: {variant}")
        self.variant = variant
        self.checkpoint = checkpoint
        from pathlib import Path
        self.vision_tower = vision_tower or str(Path(checkpoint) / "eva_vit_g.pth")
        self.agent = None
        self.sessions = {}
        self._scratch = None
        self.phases = {"upstream_decision_s": 0.0, "generation_calls": 0, "raw_rgb_frames": 0}

    def _ensure_model(self):
        if self.agent is not None:
            return
        self._scratch = tempfile.TemporaryDirectory(prefix=f"nav-eval-{self.variant}-")
        if self.variant == "navid":
            from .runtime.agent_navid import NaVid_Agent
            self.agent = NaVid_Agent(self.checkpoint, self._scratch.name, require_map=False,
                                    vision_tower=self.vision_tower)
        else:
            # run.py uses this module. The older same-named class in
            # agent_navid lacks online feature caching and its reset protocol.
            from .runtime.agent_uninavid import UniNaVid_Agent
            self.agent = UniNaVid_Agent(self.checkpoint, self._scratch.name, exp_save="",
                                       vision_tower=self.vision_tower)

    def prepare(self):
        self._ensure_model()

    @staticmethod
    def _decode_rgb(payload):
        rgb = decode_tensor(payload)
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        return rgb

    def call(self, operation, payload):
        if operation == "describe":
            hfov = VARIANTS[self.variant]
            return {
                "schema_version": SCHEMA_VERSION,
                "role": "method",
                "real": True,
                "id": self.variant,
                "requires_sensors": ["rgb"],
                "emits_actions": ["primitive", "stop"],
                "accepts_transition": False,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": hfov},
                "checkpoint": self.checkpoint,
                "note": "Direct upstream agent wrapper; upstream owns video history and pending actions.",
            }

        session_id = payload.get("session_id")
        if operation == "reset":
            self._ensure_model()
            if self.sessions:
                raise ContractError("active session already exists")
            context = payload["context"]
            if set(context) != {"episode_id", "goal", "embodiment", "seed"}:
                raise ContractError("unexpected public episode fields")
            goal = context["goal"]
            if goal.get("kind") == "object_category":
                # The method owns its prompt policy; category goals (ObjectNav)
                # are rendered into this fixed template, part of the bundle identity.
                instruction = (f"Navigate to the {goal['value']} in the environment. "
                               f"When you get close to the {goal['value']}, stop.")
            elif goal.get("kind") == "language":
                instruction = goal["value"]
            else:
                raise ContractError(f"{self.variant} requires a language or object_category goal")
            self.agent.reset()
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": instruction,
                "last_sequence": -1,
            }
            return {}
        if operation == "close_episode":
            if self.sessions.pop(session_id, None) is not None and self.agent is not None:
                self.agent.reset()
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")

        observation = payload["observation"]
        validate_observation(observation)
        state = self.sessions[session_id]
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError(f"{self.variant} consumes exactly the rgb sensor")
        state["last_sequence"] = observation["sequence"]
        generates = not self.agent.pending_action_list
        started = time.perf_counter()
        try:
            result = self.agent.act(
                {
                    "rgb": self._decode_rgb(observation["sensors"]["rgb"]),
                    "instruction": {"text": state["instruction"]},
                },
                {},
                state["episode_id"],
            )
        except ValueError as error:
            raise ContractError(f"upstream {self.variant} could not parse generated actions") from error
        self.phases["raw_rgb_frames"] += 1
        if generates:
            self.phases["generation_calls"] += 1
            self.phases["upstream_decision_s"] += time.perf_counter() - started
        try:
            action = self._action_dict(result["action"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"upstream {self.variant} returned an invalid action") from error
        return {
            "episode_id": state["episode_id"],
            "observation_sequence": observation["sequence"],
            "actions": [action],
        }

    @staticmethod
    def _action_dict(action_id):
        if isinstance(action_id, bool) or not isinstance(action_id, Integral):
            raise ContractError(f"upstream NaVid action id must be an integer: {action_id!r}")
        try:
            return {
                0: {"kind": "stop", "values": {}},
                1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
                2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
                3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
            }[action_id]
        except KeyError as error:
            raise ContractError(f"upstream NaVid action id is out of range: {action_id}") from error

    def close(self):
        self.sessions.clear()
        self.agent = None
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def asset_files(self):
        from pathlib import Path

        return {"vision_tower": Path(self.vision_tower)}

    def runtime_identity(self):
        return {"agent_class": f"{type(self.agent).__module__}.{type(self.agent).__name__}",
                "run_type": getattr(self.agent.model.config, "run_type", None)}

    def timing_snapshot(self):
        return dict(self.phases)
