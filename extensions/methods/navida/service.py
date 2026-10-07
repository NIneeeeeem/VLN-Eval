"""NaVIDA method service: Qwen2.5-VL next-action VLN policy.

Wraps the NaVIDA checkpoint (NaVIDA, arXiv 2601.18188; weights are passed
via the resource map) behind the nav-eval method contract using Transformers.

Reference behavior: the prompt template, system prompt, RGB resize (308x252),
uniformly-sampled 8-frame history, generation config (including checkpoint
defaults, max_new_tokens 128, repetition_penalty 1.05) and the action parser
follow upstream src/eval/eval.py (NaVIDA_Agent). This preserves upstream action
semantics without claiming model-numeric or full-benchmark reproduction parity.
"""
from __future__ import annotations

import io
import re
import time

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

SYSTEM_PROMPT = "You are a helpful assistant."
VLN_PROMPT_PREFIX = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations"
)
VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation."
    " Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if the task is completed."
)
DEFAULT_HISTORY_IMAGE_COUNT = 8

FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494  # 15 degrees


def uniform_sample_with_ends(data, n):
    # verbatim from upstream eval.py
    if len(data) <= n:
        return data
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def extract_result(output, forward_distance=25, turn_angle=15):
    """Upstream action parser: returns (action_id, numeric) with ids 0..3."""
    output_match = re.search(r"<answer>(.*?)</answer>", output)
    output = output_match.group(1).strip() if output_match else output.strip()
    output = output.lower()
    if "stop" in output:
        return 0, None
    if "forward" in output:
        match = re.search(r"-?\d+", output)
        if match is None:
            return 1, forward_distance
        return 1, float(match.group())
    if "left" in output:
        match = re.search(r"-?\d+", output)
        if match is None:
            return 2, turn_angle
        return 2, float(match.group())
    if "right" in output:
        match = re.search(r"-?\d+", output)
        if match is None:
            return 3, turn_angle
        return 3, float(match.group())
    return None, None


def extract_multi_result(output):
    # verbatim from upstream eval.py (comma-separated action chunks)
    result = []
    for sub_action in output.split(", "):
        action_index, numeric = extract_result(sub_action)
        result.append([action_index, numeric])
    return result


class NaVIDAMethodService:
    """requires rgb (640x480 hfov 90); emits single primitive/stop per decision."""

    def __init__(self, checkpoint, decoding="upstream", isolate_rng=False):
        if decoding not in {"upstream", "greedy"}:
            raise ContractError("NaVIDA decoding must be upstream or greedy")
        self.decoding = decoding
        self.isolate_rng = isolate_rng
        self.checkpoint = checkpoint
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "decode_s": 0.0,
                       "generation_calls": 0, "generation_batches": 0, "generation_batch_sizes": {},
                       "input_tokens": 0, "generated_tokens": 0}

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import GenerationConfig, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, attn_implementation="flash_attention_2",
            use_cache=True, torch_dtype=torch.bfloat16,
        )
        self.model.to("cuda")
        self.model = self.model.eval()
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.checkpoint, use_fast=False, trust_remote_code=False)
        if not hasattr(self.processor, "tokenizer") or getattr(self.processor, "image_processor", None) is None:
            raise ContractError("NaVIDA checkpoint did not load a multimodal processor")
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "max_pixels"):
            self.processor.image_processor.max_pixels = 501760
        self.generation_config = GenerationConfig(
            do_sample=False, temperature=0.2, max_new_tokens=128, top_p=1.0,
            use_cache=True, repetition_penalty=1.05, num_return_sequences=1,
        )

    def prepare(self):
        self._ensure_model()

    # -- prompt construction (upstream parity) -------------------------------
    def _history_images(self, state):
        history = state["rgb_history"]
        if len(history) <= 1:
            return [history[-1]]
        return uniform_sample_with_ends(history[:-1], DEFAULT_HISTORY_IMAGE_COUNT)

    def _build_content(self, instruction, state):
        previous = state.get("jpeg_cache", {})
        selected = {}
        def as_b64(image):
            key = id(image)  # rgb_history owns these immutable images until reset
            if key in selected:
                return selected[key]
            if key in previous:
                selected[key] = previous[key]
                return selected[key]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            import base64

            selected[key] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return selected[key]

        content = [{"type": "text", "text": VLN_PROMPT_PREFIX}]
        history_images = self._history_images(state)
        if history_images:
            content.extend({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{as_b64(item)}",
            } for item in history_images)
        else:
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{as_b64(state['rgb_history'][-1])}",
            })
        content.append({"type": "text", "text": "and an image of the current observation"})
        content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{as_b64(state['rgb_history'][-1])}",
        })
        tail = self.prompt_template.format(instruction).split("current observation.", 1)
        content.append({"type": "text", "text": tail[1]})
        # Bound encoded storage to the last selected history plus current frame.
        state["jpeg_cache"] = selected
        return content

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "navida_qwen2_5_vl_3b",
                "requires_sensors": ["rgb"], "emits_actions": ["primitive", "stop"],
                "accepts_transition": False,
                "batching": "independent_greedy" if self.decoding == "greedy" else "independent_sessions" if self.isolate_rng else None,
                "decoding": self.decoding,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 90},
                "checkpoint": self.checkpoint,
                "note": "Upstream prompt/history/parser from NaVIDA src/eval/eval.py; "
                        "one action chunk per decision; upstream preserves model-default merging, "
                        "greedy disables checkpoint sampling overrides.",
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
                raise ContractError("NaVIDA requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "rgb_history": [], "pending": [], "last_sequence": -1,
            }
            if self.isolate_rng and self.decoding == "upstream":
                self.sessions[session_id]["rng"] = self._rng_state()
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        return self.act_batch([payload])[0]

    def _prepare_action(self, payload):
        session_id = payload["session_id"]
        if session_id not in self.sessions:
            raise ContractError("session not reset")
        observation = payload["observation"]
        validate_observation(observation)
        state = self.sessions[session_id]
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("NaVIDA consumes exactly the rgb sensor")

        # every observed frame enters the history, including pending replays (upstream order)
        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        from PIL import Image

        image = Image.fromarray(rgb.astype("uint8")).convert("RGB").resize((308, 252))
        state["rgb_history"].append(image)
        state["last_sequence"] = observation["sequence"]

        # pending actions are replayed before a new model call (upstream behaviour)
        if state["pending"]:
            action_id = state["pending"].pop(0)
            return state, observation, self._reply(state, observation, [self._action_dict(action_id)]), None

        conversations = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": self._build_content(state["instruction"], state)},
        ]
        return state, observation, None, conversations

    def act_batch(self, payloads):
        if len(payloads) > 1 and self.decoding != "greedy":
            raise ContractError("multi-input batching requires explicit greedy decoding")
        if len({p["session_id"] for p in payloads}) != len(payloads):
            raise ContractError("one request per session is allowed in a batch")
        prepared = [self._prepare_action(payload) for payload in payloads]
        pending = [item for item in prepared if item[3] is not None]
        if len(pending) == 1:
            outputs = [self._generate_session(pending[0][0], pending[0][3])]
        elif pending:
            outputs = self._generate_batch([item[3] for item in pending])
        else:
            outputs = []
        if len(outputs) != len(pending):
            raise RuntimeError("generation result count differs from request batch")
        generated = iter(outputs)
        return [reply if reply is not None else self._action_from_text(state, observation, next(generated))
                for state, observation, reply, _ in prepared]

    def _rng_state(self):
        return {"cpu": self.torch.get_rng_state(), "cuda": self.torch.cuda.get_rng_state()}

    def _generate_session(self, state, conversations):
        if "rng" not in state:
            return self._generate(conversations)
        # Checkpoint defaults can enable sampling even with top_k=1. Restore
        # this episode's CPU/CUDA generators before each singleton generation;
        # scheduling or another session's reset must not change its sequence.
        with self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
            self.torch.set_rng_state(state["rng"]["cpu"])
            self.torch.cuda.set_rng_state(state["rng"]["cuda"])
            try:
                return self._generate(conversations)
            finally:
                state["rng"] = self._rng_state()

    def _action_from_text(self, state, observation, output):
        actions = []
        parsed = extract_multi_result(output)
        # Pinned upstream eval.py selects the first text action, then expands
        # its physical magnitude into <=3 primitive steps. Invalid text/zero
        # repeats become STOP in that algorithm, not in the platform codec.
        action_id, numeric = parsed[0]
        if action_id == 0:
            actions = [0]
        elif action_id is not None:
            increment = 25 if action_id == 1 else 15
            actions = [action_id] * max(0, min(3, round(numeric / increment)))
        if not actions:
            actions = [0]
        state["pending"] = actions[1:]
        return self._reply(state, observation, [self._action_dict(actions[0])])

    @property
    def prompt_template(self):
        return VLN_PROMPT_TEMPLATE

    def _generate(self, conversations):
        return self._generate_batch([conversations])[0]

    def _generate_batch(self, conversations):
        from qwen_vl_utils import process_vision_info

        started = time.perf_counter()
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 for messages in conversations]
        images = [image for messages in conversations for image in process_vision_info(messages)[0]]
        # Keep singleton preprocessing unchanged; unequal decoder-only prompts
        # need left padding when multiple independent sessions are combined.
        padding = {"padding_side": "left"} if len(conversations) > 1 else {}
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, **padding)
        inputs.to("cuda")
        before_generate = time.perf_counter()
        with self.torch.inference_mode():
            # HF >=4.50 may replace explicit False/1.0 values with checkpoint
            # sampling defaults when use_model_defaults=True. Preserve that
            # upstream path by default; explicit greedy is a frozen method setting.
            outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                          use_model_defaults=self.decoding == "upstream")
        self.torch.cuda.synchronize()
        before_decode = time.perf_counter()
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        self.phases["preprocess_s"] += before_generate - started
        self.phases["generate_s"] += before_decode - before_generate
        self.phases["decode_s"] += time.perf_counter() - before_decode
        size = len(conversations)
        self.phases["generation_calls"] += size
        self.phases["generation_batches"] += 1
        histogram = self.phases["generation_batch_sizes"]
        histogram[str(size)] = histogram.get(str(size), 0) + 1
        self.phases["input_tokens"] += int(inputs["attention_mask"].sum().item())
        # Count through the first EOS, excluding padding after a shorter reply.
        eos = self.generation_config.eos_token_id or self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for row in generated.tolist():
            self.phases["generated_tokens"] += next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
        return [text.strip() for text in texts]

    def timing_snapshot(self):
        return dict(self.phases)

    @staticmethod
    def _action_dict(action_id):
        # ids follow upstream: 0 stop, 1 forward, 2 left, 3 right; discrete embodiment
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action_id]

    def _reply(self, state, observation, actions):
        return {"episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"], "actions": actions}
