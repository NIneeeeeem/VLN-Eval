"""NaVILA policy service using the upstream model and inference semantics.

The prompt, frame sampling, generation settings, parser, and action expansion
follow ``evaluation/vlnce_baselines/navila_trainer.py`` from the configured
NaVILA checkout. The Habitat rollout remains owned by Nav-Eval.
"""
from __future__ import annotations

import re
import time

from nav_eval.contracts import ContractError, SCHEMA_VERSION, validate_observation
from nav_eval.tensorcode import decode_tensor

NUM_VIDEO_FRAMES = 8
FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494

ACTION_PATTERNS = (
    (0, re.compile(r"\bstop\b", re.IGNORECASE)),
    (1, re.compile(r"\bis move forward\b", re.IGNORECASE)),
    (2, re.compile(r"\bis turn left\b", re.IGNORECASE)),
    (3, re.compile(r"\bis turn right\b", re.IGNORECASE)),
)


def sample_and_pad_images(images, num_frames=NUM_VIDEO_FRAMES, width=512, height=512):
    """Mirror the upstream left-padding and history sampling policy."""
    import copy

    import numpy as np
    from PIL import Image

    frames = list(images)
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    latest_frame = frames[-1]
    indices = np.linspace(
        0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int
    )
    # Only selected images reach the processor. Retain upstream independent
    # image ownership without copying an entire growing episode on every call.
    return copy.deepcopy([frames[index] for index in indices] + [latest_frame])


def parse_action_sequence(output):
    """Convert one upstream text decision into explicit 25 cm / 15 deg steps."""
    action_id = None
    for candidate, pattern in ACTION_PATTERNS:
        if pattern.search(output):
            action_id = candidate
            break
    if action_id is None and re.search(r"\bis turn\b", output, re.IGNORECASE):
        action_id = 3
    if action_id in {None, 0}:
        return [0]

    if action_id == 1:
        match = re.search(r"move forward (\d+) cm", output)
        magnitude, quantum, supported = (int(match.group(1)) if match else 25), 25, (25, 50, 75)
    else:
        direction = "left" if action_id == 2 else "right"
        match = re.search(rf"turn {direction} (\d+) degree", output)
        magnitude, quantum, supported = (int(match.group(1)) if match else 15), 15, (15, 30, 45)
    if magnitude % quantum:
        magnitude = min(supported, key=lambda value: abs(value - magnitude))
    return [action_id] * max(1, magnitude // quantum)


class NaVILAMethodService:
    """RGB-only NaVILA policy with worker-local model and episode state."""

    def __init__(self, checkpoint, repo_path):
        self.checkpoint = checkpoint
        self.repo_path = repo_path
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "generation_calls": 0}

    def _ensure_model(self):
        if self.model is not None:
            return
        import os
        import sys

        if not os.path.isdir(self.checkpoint):
            raise ContractError(f"NaVILA checkpoint directory not found: {self.checkpoint}")
        if not os.path.isdir(self.repo_path):
            raise ContractError(f"NaVILA repository not found: {self.repo_path}")
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        model_name = os.path.basename(os.path.normpath(self.checkpoint))
        tokenizer, model, image_processor, _ = load_pretrained_model(self.checkpoint, model_name)
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model.cuda().eval()
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

    def prepare(self):
        self._ensure_model()

    def _decode_rgb(self, payload):
        from PIL import Image

        rgb = decode_tensor(payload)
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        return Image.fromarray(rgb.astype("uint8")).convert("RGB")

    def _generate(self, frames, instruction):
        started = time.perf_counter()
        interleaved_images = "<image>\n" * (len(frames) - 1)
        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations {interleaved_images}, and current observation <image>\n. "
            f'Your assigned task is: "{instruction}" Analyze this series of images to decide your '
            "next action, which could be turning left or right by a specific degree, moving forward "
            "a certain distance, or stop if the task is completed."
        )
        conv = self.conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images = self.process_images(frames, self.image_processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16
        )
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        before_generate = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images.half(),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        self.phases["preprocess_s"] += before_generate - started
        self.phases["generate_s"] += time.perf_counter() - before_generate
        self.phases["generation_calls"] += 1
        if stop_str and output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output

    def timing_snapshot(self):
        return dict(self.phases)

    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION,
                "role": "method",
                "real": True,
                "id": "navila_llama3_8b",
                "requires_sensors": ["rgb"],
                "emits_actions": ["primitive", "stop"],
                "accepts_transition": False,
                "embodiment_expectation": {"width": 512, "height": 512, "hfov_deg": 90},
                "checkpoint": self.checkpoint,
                "note": "Upstream NaVILA prompt, frame sampling, parser, and queued action semantics.",
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
                raise ContractError("NaVILA requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "history": [],
                "pending": [],
                "last_sequence": -1,
            }
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")

        observation = payload["observation"]
        validate_observation(observation)
        state = self.sessions[session_id]
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("NaVILA consumes exactly the rgb sensor")

        state["history"].append(self._decode_rgb(observation["sensors"]["rgb"]))
        state["last_sequence"] = observation["sequence"]
        if state["pending"]:
            action_id = state["pending"].pop(0)
        else:
            frame_count = getattr(getattr(self.model, "config", None), "num_video_frames", NUM_VIDEO_FRAMES)
            frames = sample_and_pad_images(state["history"], num_frames=frame_count)
            sequence = parse_action_sequence(self._generate(frames, state["instruction"]))
            action_id, state["pending"] = sequence[0], sequence[1:]
        return {
            "episode_id": state["episode_id"],
            "observation_sequence": observation["sequence"],
            "actions": [self._action_dict(action_id)],
        }

    @staticmethod
    def _action_dict(action_id):
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action_id]
