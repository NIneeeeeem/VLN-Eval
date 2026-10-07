"""ActiveVLN method service: Qwen2.5-VL conversational VLN policy.

Wraps the ActiveVLN checkpoints (variant rl_/sft_ x r2r/rxr under weights_root)
behind the nav-eval method contract.

Reference behavior (upstream local/ActiveVLN/eval/vlnce/eval_vlnce.py,
ActiveVlnAgent): the system prompts (r2r: forward 25/50/75 cm, turns 15/30/45
degrees; rxr: turns 30/60/90 degrees), the growing multi-turn conversation that
resends the full history each model call, the client-side smart_resize to
76800 pixels (factor 28) with PNG bytes in a jpeg-labelled data URL, sampling
n=1 temperature 0.2 top_p 0.8 max_tokens 512, the action parser and the
pending-action expansion (min(3, int(numeric/unit))) including the random
fallback follow upstream verbatim.

Known divergences (documented, diagnostic claim only): the upstream vllm-serve
backend is replaced by in-process transformers generation; upstream's
EARLY_STOP_ROTATION / EARLY_STOP_STEPS harness stops stay upstream (the method
never receives distance-to-goal feedback); the rxr 30-degree turn unit is
executed as two 15-degree platform primitives (the manifest freezes 15-degree
turn semantics). Generation failures propagate to the worker failure record;
they must not become normal STOP actions without upstream's inference_error flag.
"""
from __future__ import annotations

import base64
import io
import random
import re
import time

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, "
    "turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
    "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

SYSTEM_PROMPT_RxR = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, "
    "turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, "
    "turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

# first_turn_user_prompt and normal_user_prompt are identical upstream; the
# missing space after "{}" and the trailing space are verbatim.
USER_PROMPT = (
    "Instruction: {}"
    "Decide your next action. "
    "You can take up to 3 actions at a time, separated by ','. "
)

VARIANTS = ("rl_r2r", "rl_rxr", "sft_r2r", "sft_rxr")
FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494  # 15 degrees
CLIENT_MAX_PIXELS = 76800
PROCESSOR_MAX_PIXELS = 80000  # vllm serve --mm-processor-kwargs max_pixels
MAX_TURNS = 120


class ActiveVLNMethodService:
    """requires rgb (640x480 hfov 90); emits primitive/stop; sampling decode."""

    def __init__(self, weights_root, variant="rl_r2r"):
        if variant not in VARIANTS:
            raise ContractError(f"ActiveVLN variant must be one of {VARIANTS}")
        self.weights_root = weights_root
        self.variant = variant
        self.checkpoint = f"{weights_root.rstrip('/')}/{variant}"
        self.action_space = variant.split("_", 1)[1]  # r2r | rxr
        self.forward_distance = 25
        self.turn_angle = 15 if self.action_space == "r2r" else 30
        self.system_prompt = SYSTEM_PROMPT_R2R if self.action_space == "r2r" else SYSTEM_PROMPT_RxR
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "decode_s": 0.0,
                       "generation_calls": 0}

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, attn_implementation="flash_attention_2",
            use_cache=True, torch_dtype=torch.bfloat16,
        )
        self.model.to("cuda")
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, use_fast=False,
                                                       trust_remote_code=False)
        if getattr(self.processor, "image_processor", None) is None:
            raise ContractError("ActiveVLN checkpoint did not load a multimodal processor")
        self.processor.image_processor.max_pixels = PROCESSOR_MAX_PIXELS

    def prepare(self):
        self._ensure_model()

    # -- upstream parser (verbatim) ------------------------------------------
    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        output = output.strip().lower()
        if "stop" in output:
            return 0, None
        if "forward" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return 1, self.forward_distance
            return 1, float(match.group())
        if "left" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return 2, self.turn_angle
            return 2, float(match.group())
        if "right" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return 3, self.turn_angle
            return 3, float(match.group())
        return None, None

    def extract_multi_result(self, output):
        sub_actions = output.split(", ")
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    # -- image path (verbatim, incl. the jpeg-labelled PNG data URL) ---------
    @staticmethod
    def _frame_content(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"}

    def _prepare_frame(self, observation):
        from PIL import Image
        from qwen_vl_utils.vision_process import smart_resize

        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        image = Image.fromarray(rgb.astype("uint8")).convert("RGB")
        resized_height, resized_width = smart_resize(
            image.height, image.width, max_pixels=CLIENT_MAX_PIXELS, factor=28)
        return image.resize((resized_width, resized_height))

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": f"activevln_{self.variant}",
                "requires_sensors": ["rgb"], "emits_actions": ["primitive", "stop"],
                "accepts_transition": False, "batching": None,
                "variant": self.variant, "action_space": self.action_space,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 90},
                "checkpoint": self.checkpoint,
                "note": "Upstream prompts/parser/pending expansion from ActiveVLN "
                        "eval_vlnce.py; in-process transformers replaces vllm serve; "
                        "rxr 30-degree units execute as two 15-degree primitives.",
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
                raise ContractError("ActiveVLN requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "conversations": [{"role": "system",
                                   "content": [{"type": "text", "text": self.system_prompt}]}],
                "pending": [], "count_turn": 0, "last_sequence": -1,
            }
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        return self._act(payload)

    def _act(self, payload):
        session_id = payload["session_id"]
        state = self.sessions[session_id]
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("ActiveVLN consumes exactly the rgb sensor")
        state["last_sequence"] = observation["sequence"]

        if state["pending"]:
            return self._reply(state, observation, [self._action_dict(state["pending"].pop(0))])

        state["count_turn"] += 1
        if state["count_turn"] > MAX_TURNS:
            return self._reply(state, observation, [self._action_dict(0)])

        image = self._prepare_frame(observation)
        header = "[Initial Observation]:" if len(state["conversations"]) == 1 \
            else "After that, the observation is:"
        state["conversations"].append({
            "role": "user",
            "content": [
                {"type": "text", "text": header},
                self._frame_content(image),
                {"type": "text", "text": USER_PROMPT.format(state["instruction"])},
            ],
        })
        navigation = self._generate(state["conversations"])
        state["conversations"].append(
            {"role": "assistant", "content": [{"type": "text", "text": navigation}]})

        # upstream expansion loop (verbatim semantics)
        parsed = self.extract_multi_result(navigation)
        for action_index, numeric in parsed:
            if action_index == 0:
                state["pending"].append(0)
            elif action_index == 1:
                for _ in range(min(3, int(numeric / self.forward_distance))):
                    state["pending"].append(1)
            elif action_index == 2:
                for _ in range(min(3, int(numeric / self.turn_angle))):
                    state["pending"].append(2)
            elif action_index == 3:
                for _ in range(min(3, int(numeric / self.turn_angle))):
                    state["pending"].append(3)
            if action_index is None or len(state["pending"]) == 0:
                action_index = random.randint(1, 3)
                state["pending"].append(action_index)
        # rxr 30-degree turn units become two 15-degree platform primitives
        if self.action_space == "rxr":
            expanded = []
            for unit in state["pending"]:
                expanded.extend([unit, unit] if unit in (2, 3) else [unit])
            state["pending"] = expanded
        return self._reply(state, observation, [self._action_dict(state["pending"].pop(0))])

    def _generate(self, conversations):
        from qwen_vl_utils import process_vision_info

        started = time.perf_counter()
        text = self.processor.apply_chat_template(conversations, tokenize=False,
                                                  add_generation_prompt=True)
        images = process_vision_info(conversations)[0]
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs.to("cuda")
        before_generate = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=True, temperature=0.2, top_p=0.8,
                max_new_tokens=512, num_return_sequences=1, use_cache=True)
        self.torch.cuda.synchronize()
        before_decode = time.perf_counter()
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        reply = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        self.phases["preprocess_s"] += before_generate - started
        self.phases["generate_s"] += before_decode - before_generate
        self.phases["decode_s"] += time.perf_counter() - before_decode
        self.phases["generation_calls"] += 1
        return reply

    def timing_snapshot(self):
        return dict(self.phases)

    @staticmethod
    def _action_dict(action_id):
        # ids follow upstream: 0 stop, 1 forward, 2 left, 3 right
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action_id]

    def _reply(self, state, observation, actions):
        return {"episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"], "actions": actions}
