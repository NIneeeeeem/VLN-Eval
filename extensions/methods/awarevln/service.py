"""AwareVLN method service: reasoning/action dual-mode VLM policy.

Wraps an AwareVLN checkpoint (https://github.com/GWxuan/AwareVLN)
using the bundled inference implementation behind the nav-eval method contract.

Adapter parity: model loading (llava.model.builder.load_pretrained_model),
sample_and_pad_images (8 frames, 512x512, black left-pad), llama_3 conversation
template, interleaved <image> prompt with optional reasoning context, greedy
generation (max_new_tokens 256), <BEGIN_OF_REASONING>/<BEGIN_OF_ACTION> prefix
handling, the reason-stuck force-stop rule and the action regex mapping are
replicated from evaluation/vlnce_baselines/awarevln_trainer.py.

Differences (recorded, not silent): the optional StopVLN stop-gate subprocess is
disabled (STOPVLN.ENABLE=false path of upstream: AwareVLN's own STOP decides);
reasoning rounds execute inside act() without environment steps, exactly as in
the upstream loop. Trained at RGB 512x512 hfov 90, forward 0.25 m / turn 15 deg.
"""
from __future__ import annotations

import re
import time

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

REASON_TOKEN_STR = "<BEGIN_OF_REASONING>"
ACT_TOKEN_STR = "<BEGIN_OF_ACTION>"
NUM_VIDEO_FRAMES = 8
MAX_REASON_ROUNDS = 3  # upstream: reason_stuck_num > 3 forces STOP

ACTION_PATTERNS = (
    ("stop", re.compile(r"\bstop\b", re.IGNORECASE)),
    ("forward", re.compile(r"move forward", re.IGNORECASE)),
    ("left", re.compile(r"turn left", re.IGNORECASE)),
    ("right", re.compile(r"turn right", re.IGNORECASE)),
)

FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494


def sample_and_pad_images(images, num_frames=NUM_VIDEO_FRAMES, width=512, height=512):
    """Upstream sampling/padding with copies restricted to selected frames."""
    import copy

    import numpy as np
    from PIL import Image

    frames = list(images)
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1,
                                  endpoint=False, dtype=int)
    return copy.deepcopy([frames[i] for i in sampled_indices] + [latest_frame])


def map_string_to_action(text):
    """Upstream regex mapping; unknown outputs default to forward."""
    for name, pattern in ACTION_PATTERNS:
        if pattern.search(text):
            return name
    return "forward"


def extract_reasoning(text):
    """Match upstream marker removal without truncating the reasoning context."""
    if not text:
        return ""
    for pattern in (r"(new\s*reasoning\s*is[:\s]*)", r"\[?ph_reasoning_token_?\]?",
                    r"<\|?end_of_text\|?>", r"<begin_of_reasoning>",
                    r"<begin_of_action>", r"<end_of_reasoning>"):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip(" '\"\n\t")


def expand_action(name, text):
    """Upstream executes one step immediately, then queues the remaining steps."""
    if name == "stop":
        return [name]
    if name == "forward":
        pattern, quantum, supported = r"move forward (\d+) cm", 25, (25, 50, 75)
    else:
        pattern, quantum, supported = rf"turn {name} (\d+) degree", 15, (15, 30, 45)
    match = re.search(pattern, text)
    amount = int(match.group(1)) if match else quantum
    if amount % quantum:
        amount = min(supported, key=lambda value: abs(value - amount))
    return [name] * max(1, amount // quantum)


class AwareVLNMethodService:
    """requires rgb (512x512 hfov 90); emits single primitive/stop per decision."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "generation_calls": 0}

    def _ensure_model(self):
        if self.model is not None:
            return
        import os
        import torch
        from .runtime.llava.constants import IMAGE_TOKEN_INDEX
        from .runtime.llava.conversation import SeparatorStyle, conv_templates
        from .runtime.llava.mm_utils import (
            KeywordsStoppingCriteria,
            process_images,
            tokenizer_image_token,
        )
        from .runtime.llava.model.builder import load_pretrained_model

        model_name = os.path.basename(os.path.normpath(self.checkpoint))
        tokenizer, model, image_processor, _ = load_pretrained_model(self.checkpoint, model_name)
        reason_id = tokenizer.convert_tokens_to_ids(REASON_TOKEN_STR)
        act_id = tokenizer.convert_tokens_to_ids(ACT_TOKEN_STR)
        invalid = {None, -1, tokenizer.unk_token_id}
        if reason_id in invalid or act_id in invalid or reason_id == act_id:
            raise ContractError("tokenizer missing AwareVLN special tokens")
        model = model.cuda().eval()
        self.torch = torch
        self.tokenizer, self.model, self.image_processor = tokenizer, model, image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

    def prepare(self):
        self._ensure_model()

    # -- prompt construction (upstream parity) -------------------------------
    def _question(self, instruction, reasoning_context, related_step, frames=NUM_VIDEO_FRAMES):
        interleaved = "<image>\n" * (frames - 1)
        if reasoning_context:
            return (
                f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f"of historical observations {interleaved}, and current observation <image>\n. "
                f'Your assigned task is: "{instruction}". '
                f'The reasoning from {related_step} steps ago was: "{reasoning_context}". '
                f"Analyze this series of images to decide whether to predict the next action or to perform reasoning. "
                f"If action prediction, decide your next action, which could be turning left or right by a specific degree, "
                f"moving forward a certain distance, or stop if the task is completed. "
                f"If reasoning, describe your current observations, assess task progress, and provide a high-level plan for the next steps."
            )
        return (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations {interleaved}, and current observation <image>\n. "
            f'Your assigned task is: "{instruction}". '
            f"Analyze this series of images to decide whether to predict the next action or to perform reasoning. "
            f"If action prediction, decide your next action, which could be turning left or right by a specific degree, "
            f"moving forward a certain distance, or stop if the task is completed. "
            f"If reasoning, describe your current observations, assess task progress, and provide a high-level plan for the next steps."
        )

    def _generate(self, question, frames):
        started = time.perf_counter()
        conv = self.conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images_tensor = self.process_images(frames, self.image_processor, self.model.config).to(
            self.model.device, dtype=self.torch.float16)
        input_ids = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        before_generate = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=images_tensor.half().cuda(),
                do_sample=False, temperature=0.0, max_new_tokens=256, use_cache=True,
                stopping_criteria=[stopping], pad_token_id=self.tokenizer.eos_token_id,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        self.phases["preprocess_s"] += before_generate - started
        self.phases["generate_s"] += time.perf_counter() - before_generate
        self.phases["generation_calls"] += 1
        for suffix in (stop_str, conv.sep2):
            if suffix and outputs.endswith(suffix):
                outputs = outputs[: -len(suffix)].strip()
        if outputs.startswith("<|begin_of_text|>"):
            outputs = outputs[len("<|begin_of_text|>"):].strip()
        return outputs

    def timing_snapshot(self):
        return dict(self.phases)

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "awarevln_internvl2_8b",
                "requires_sensors": ["rgb"], "emits_actions": ["primitive", "stop"],
                "accepts_transition": False,
                "embodiment_expectation": {"width": 512, "height": 512, "hfov_deg": 90},
                "checkpoint": self.checkpoint,
                "note": "Upstream prompt/frames/parser from awarevln_trainer.py; "
                        "StopVLN gate disabled (upstream ENABLE=false path).",
            }
        session_id = payload.get("session_id")
        if operation == "reset":
            self._ensure_model()
            if session_id in self.sessions:
                raise ContractError("active session already exists")
            context = payload["context"]
            if context["goal"].get("kind") != "language":
                raise ContractError("AwareVLN requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "history": [], "last_reasoning": "", "last_reason_step": 0,
                "steps": 0, "last_sequence": -1, "pending": [], "last_action_output": "",
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
            raise ContractError("AwareVLN consumes exactly the rgb sensor")
        state["last_sequence"] = observation["sequence"]

        from PIL import Image

        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        current = Image.fromarray(rgb.astype("uint8")).convert("RGB")

        # upstream: reasoning rounds re-decide without stepping the environment
        state["steps"] += 1
        if state["pending"]:
            action = state["pending"].pop(0)
            state["history"].append(current)
            return {"episode_id": state["episode_id"], "observation_sequence": observation["sequence"],
                    "actions": [self._action_dict(action)], "awarevln_raw_output": state["last_action_output"]}
        frame_count = getattr(getattr(self.model, "config", None), "num_video_frames", NUM_VIDEO_FRAMES)
        frames = sample_and_pad_images(state["history"] + [current], num_frames=frame_count)
        for _ in range(MAX_REASON_ROUNDS + 1):
            related_step = state["steps"] - state["last_reason_step"]
            question = self._question(state["instruction"], state["last_reasoning"], related_step, len(frames))
            outputs = self._generate(question, frames)
            if outputs.startswith(REASON_TOKEN_STR):
                state["last_reasoning"] = extract_reasoning(outputs[len(REASON_TOKEN_STR):].strip())
                state["last_reason_step"] = state["steps"]
                continue
            body = outputs[len(ACT_TOKEN_STR):].strip() if outputs.startswith(ACT_TOKEN_STR) else outputs
            action = map_string_to_action(body)
            sequence = expand_action(action, body)
            action, state["pending"] = sequence[0], sequence[1:]
            state["last_action_output"] = body[:256]
            state["history"].append(current)
            return {
                "episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"],
                "actions": [self._action_dict(action)],
                "awarevln_raw_output": body[:256],
            }
        # reason-stuck: upstream forces STOP after more than 3 consecutive rounds
        return {
            "episode_id": state["episode_id"],
            "observation_sequence": observation["sequence"],
            "actions": [{"kind": "stop", "values": {}}],
            "awarevln_raw_output": "reason_stuck_force_stop",
        }

    @staticmethod
    def _action_dict(name):
        return {
            "stop": {"kind": "stop", "values": {}},
            "forward": {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            "left": {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            "right": {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[name]
