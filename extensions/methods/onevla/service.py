"""OneVLA method service: Qwen2.5-VL-3B + flow-matching action head VLA.

Wraps the OneVLA checkpoint (HF llxs/OneVLA steps_8000, framework
QwenGR00T_with_Language, 11-dim action space) behind the nav-eval method
contract. The model is imported from the bundled runtime; the agent
loop is ported from local/OneVLA/examples/R2R/eval_files/agent.py
(QwenOFT_R2R_Agent) verbatim:

- history window: all frames up to 8, then linspace(0, len-1, 8) indices;
  every frame resized to 224x224;
- per step: model.predict_action(batch_images, instructions, state=None),
  first element of the action chunk, R2R dims sliced as act[3:7] when the
  action dimension is >= 14 else act[:4] (the released checkpoint is 11-dim),
  argmax with the dominant-value < 0.1 random fallback and the
  prediction-failure random fallback;
- stop threshold envs (ONEVLA_STOP_THRESHOLD / ONEVLA_CHUNK_STOP_RATIO)
  default disabled, as upstream.

Known divergence (documented, diagnostic claim only): the flow-matching head
starts from randn, so actions are stochastic; the platform reseeds torch per
episode, which upstream (unseeded per episode) does not. Base VLM location and
attention implementation follow the upstream workbench patch
(ONEVLA_BASE_VLM / ONEVLA_ATTN_IMPL, sdpa as run in the workspace).
"""
from __future__ import annotations

import os
import random
import time

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

R2R_ACTION_START, R2R_ACTION_END = 3, 7
MAX_HISTORY_FRAMES = 8
STOP_IDX = 3

FORWARD_STEP_M = 0.25
TURN_RAD = 0.5235987755982988  # 30 degrees (upstream task config TURN_ANGLE 30)


class OneVLAMethodService:
    """requires rgb (640x480 hfov 90); emits one primitive/stop per decision."""

    def __init__(self, checkpoint, base_vlm=None):
        self.checkpoint = checkpoint
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "generation_calls": 0}
        # upstream workbench patch envs; set before the model import reads them
        if base_vlm:
            os.environ["ONEVLA_BASE_VLM"] = base_vlm
        os.environ.setdefault("ONEVLA_ATTN_IMPL", "sdpa")
        self.stop_threshold = float(os.environ.get("ONEVLA_STOP_THRESHOLD", "0"))
        self.chunk_stop_ratio = float(os.environ.get("ONEVLA_CHUNK_STOP_RATIO", "0.3"))

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return

        from .runtime.OneVLA.model.framework.base_framework import baseframework
        import torch

        self.torch = torch
        self.model = baseframework.from_pretrained(self.checkpoint).to("cuda")
        self.model = self.model.eval()

    def prepare(self):
        self._ensure_model()

    # -- upstream agent helpers (verbatim) ------------------------------------
    def _prepare_images(self, rgb_list):
        import numpy as np
        from PIL import Image

        images = list(rgb_list)
        if len(images) > MAX_HISTORY_FRAMES:
            indices = np.linspace(0, len(images) - 1, MAX_HISTORY_FRAMES,
                                  dtype=int).tolist()
            images = [images[i] for i in indices]
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_images.append(Image.fromarray(img).resize((224, 224)))
        return pil_images

    @staticmethod
    def _slice_r2r(act):
        if len(act) >= 14:
            return act[R2R_ACTION_START:R2R_ACTION_END]
        return act[:4]

    def _predict_chunk(self, images, instruction):
        started = time.perf_counter()
        with self.torch.no_grad():
            result = self.model.predict_action(batch_images=[images],
                                               instructions=[instruction],
                                               state=None)
        self.phases["preprocess_s"] += 0.0
        self.phases["generate_s"] += time.perf_counter() - started
        self.phases["generation_calls"] += 1
        normalized = result.get("normalized_actions")
        if normalized is None or normalized.shape[1] == 0:
            return None
        return normalized[0]  # [chunk_size, action_dim]

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "onevla_qwen2_5_vl_3b",
                "requires_sensors": ["rgb"], "emits_actions": ["primitive", "stop"],
                "accepts_transition": False, "batching": None,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 90},
                "checkpoint": self.checkpoint,
                "note": "Upstream QwenOFT_R2R_Agent from OneVLA "
                        "examples/R2R/eval_files/agent.py; 11-dim release slices "
                        "act[:4]; flow-matching head is stochastic (platform "
                        "reseeds per episode); 30-degree turn semantics.",
            }
        session_id = payload.get("session_id")
        if operation == "reset":
            self._ensure_model()
            if session_id in self.sessions:
                raise ContractError("active session already exists")
            context = payload["context"]
            if set(context) != {"episode_id", "goal", "embodiment", "seed"}:
                raise ContractError("unexpected public episode fields")
            if context["goal"].get("kind") != "language":
                raise ContractError("OneVLA requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "rgb_list": [], "step": 0, "last_sequence": -1,
            }
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        return self._act(payload)

    def _act(self, payload):
        import numpy as np

        session_id = payload["session_id"]
        state = self.sessions[session_id]
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("OneVLA consumes exactly the rgb sensor")
        state["last_sequence"] = observation["sequence"]

        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        state["rgb_list"].append(rgb)

        images = self._prepare_images(state["rgb_list"])
        action_chunk = self._predict_chunk(images, state["instruction"])

        action_id = None
        if action_chunk is not None and action_chunk.shape[0] > 0:
            first_action = action_chunk[0]
            state.setdefault("action_dim", len(first_action))
            r2r_action = self._slice_r2r(first_action)
            chunk_stop = False
            if self.stop_threshold > 0:
                if self.chunk_stop_ratio > 0:
                    stop_count = sum(1 for t in range(action_chunk.shape[0])
                                     if self._slice_r2r(action_chunk[t])[STOP_IDX] > self.stop_threshold)
                    chunk_stop = stop_count / action_chunk.shape[0] >= self.chunk_stop_ratio
                elif r2r_action[STOP_IDX] > self.stop_threshold:
                    chunk_stop = True
            if chunk_stop:
                action_id = 3  # onehot stop index
            else:
                vector = np.asarray(r2r_action, dtype=np.float32)
                dominant_idx = int(np.argmax(vector))
                if float(vector[dominant_idx]) >= 0.1:
                    action_id = dominant_idx
        if action_id is None:
            # upstream: onehot conversion or prediction failure -> random move
            action_id = random.choice((0, 1, 2))
        state["step"] += 1
        return {"episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"],
                "actions": [self._action_dict(action_id)]}

    def timing_snapshot(self):
        return dict(self.phases)

    @staticmethod
    def _action_dict(onehot_idx):
        # onehot layout [forward, left, right, stop] (upstream ACTION_IDX_TO_ENV)
        return {
            0: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            1: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            2: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
            3: {"kind": "stop", "values": {}},
        }[onehot_idx]
