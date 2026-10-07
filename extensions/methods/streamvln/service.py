"""StreamVLN's R2R evaluation policy, exposed through the method boundary.

The upstream model, image processor, prompt builder and action parser are reused.
Every observation is retained; generation consumes a variable-length action queue
and resets the streaming KV context every 32 executed steps, as streamvln_eval.py
does. The real-world agent's separate step() interface is not the R2R policy.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494
MODEL_FUTURE_STEPS = 4
MODEL_NUM_FRAMES = 32
MODEL_NUM_HISTORY = 8


class ReusablePromptTokenizer:
    """Give upstream preprocess_qwen a worker-private tokenizer copy.

    That helper deep-copies before setting a fixed template/adding fixed tokens.
    These idempotent edits can use one dedicated copy for this exclusive worker;
    the original tokenizer used for decoding is never modified. The upstream
    helper (including its random conjunction draw) still executes on every call.
    """
    def __init__(self, tokenizer):
        self.tokenizer = copy.deepcopy(tokenizer)

    def __deepcopy__(self, memo):
        return self.tokenizer


class StreamVLNMethodService:
    def __init__(self, checkpoint=None, preprocess_mode="lazy", tokenizer_mode="reuse"):
        if preprocess_mode not in {"eager", "lazy"}:
            raise ContractError("StreamVLN preprocess_mode must be eager or lazy")
        if tokenizer_mode not in {"upstream", "reuse"}:
            raise ContractError("StreamVLN tokenizer_mode must be upstream or reuse")
        self.checkpoint = checkpoint
        self.preprocess_mode = preprocess_mode
        self.tokenizer_mode = tokenizer_mode
        self.model = self.evaluator = None
        self.sessions = {}
        self.loading_info = None
        self.phases = {"preprocess_s": 0.0, "tokenize_s": 0.0, "generate_s": 0.0,
                       "decode_s": 0.0, "generation_calls": 0, "generation_batches": 0,
                       "generation_batch_sizes": {}, "input_tokens": 0, "generated_tokens": 0,
                       "observed_frames": 0, "preprocessed_frames": 0}

    def _checkpoint_available(self):
        return bool(self.checkpoint) and Path(self.checkpoint).is_dir()

    def _ensure_model(self):
        if self.model is not None:
            return
        if not self._checkpoint_available():
            raise ContractError("StreamVLN checkpoint missing: set checkpoint to a local release directory")
        import torch
        import transformers
        from .runtime.model.stream_video_vln import StreamVLNForCausalLM
        from .runtime.policy import StreamPolicy

        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.checkpoint, model_max_length=4096, padding_side="right")
        self.prompt_tokenizer = (ReusablePromptTokenizer(self.tokenizer)
                                 if self.tokenizer_mode == "reuse" else self.tokenizer)
        config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        model, info = StreamVLNForCausalLM.from_pretrained(
            self.checkpoint, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
            config=config, low_cpu_mem_usage=False, output_loading_info=True)
        self.loading_info = info
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ContractError(f"StreamVLN checkpoint did not load exactly: {info}")
        model.model.num_history = MODEL_NUM_HISTORY  # upstream July 2025 accuracy fix
        model.requires_grad_(False)
        model.to("cuda")
        model.eval()
        model.reset(1)
        self.evaluator = StreamPolicy(model.get_vision_tower().image_processor)
        self.model = model

    def prepare(self):
        self._ensure_model()

    def asset_files(self):
        files = {}
        for directory, pattern, label in ((self.checkpoint, "*", "checkpoint"),):
            if directory:
                root = Path(directory)
                for path in root.glob(pattern):
                    if path.is_file():
                        files[f"{label}/{path.relative_to(root)}"] = path
        return files

    def runtime_identity(self):
        return {"policy": "streamvln_eval", "dtype": "bfloat16", "attention": "flash_attention_2",
                "num_frames": MODEL_NUM_FRAMES, "num_history": MODEL_NUM_HISTORY,
                "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 10000},
                "checkpoint_loading": self.loading_info}

    def timing_snapshot(self):
        return copy.deepcopy(self.phases)

    def _preprocess_rgb(self, rgb):
        from PIL import Image
        started = time.perf_counter()
        image = self.evaluator.image_processor.preprocess(
            images=Image.fromarray(rgb).convert("RGB"), return_tensors="pt")["pixel_values"][0]
        self.phases["preprocess_s"] += time.perf_counter() - started
        self.phases["preprocessed_frames"] += 1
        return image

    def _selected_images(self, state):
        step = len(state["frames"]) - 1
        indices = [step]
        if step != 0 and step % MODEL_NUM_FRAMES == 0:
            start = state["time_ids"][0]
            indices = list(range(0, start, start // MODEL_NUM_HISTORY)) + indices
        images = []
        for index in indices:
            if index not in state["processed"]:
                state["processed"][index] = self._preprocess_rgb(state["frames"][index])
            images.append(state["processed"][index])
        return images

    def _generate(self, state):
        torch = self.torch
        started = time.perf_counter()
        if state["output_ids"] is None:
            sources = copy.deepcopy(self.evaluator.conversation)
            if len(state["frames"]) > 1:
                sources[0]["value"] += " These are your historical observations <memory>."
            sources[0]["value"] = sources[0]["value"].replace("<video>\n", "")
            sources[0]["value"] = sources[0]["value"].replace("<instruction>.", state["instruction"])
            add_system = True
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        input_ids, _ = self.evaluator.preprocess_qwen(
            [sources], self.prompt_tokenizer, True, add_system=add_system)
        if state["output_ids"] is not None:
            input_ids = torch.cat([state["output_ids"], input_ids.to(state["output_ids"].device)], dim=1)
        self.phases["tokenize_s"] += time.perf_counter() - started
        images = torch.stack(self._selected_images(state)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        input_ids = input_ids.to("cuda")
        before_generate = time.perf_counter()
        # Upstream encode_rgbd consumes only images/time_ids. Depth/pose/intrinsics
        # are unused arguments; no ground-truth pose crosses the policy boundary.
        with torch.inference_mode():
            outputs = self.model.generate(
                images=images, inputs=input_ids, env_id=0, time_ids=[state["time_ids"]], task_type=[0],
                do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True,
                return_dict_in_generate=True, past_key_values=state["past_key_values"])
        before_decode = time.perf_counter()
        state["output_ids"] = outputs.sequences
        state["past_key_values"] = outputs.past_key_values
        text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].strip()
        actions = self.evaluator.parse_actions(text)
        self.phases["generate_s"] += before_decode - before_generate
        self.phases["decode_s"] += time.perf_counter() - before_decode
        self.phases["generation_calls"] += 1
        self.phases["generation_batches"] += 1
        self.phases["generation_batch_sizes"]["1"] = self.phases["generation_calls"]
        self.phases["input_tokens"] += input_ids.numel()
        self.phases["generated_tokens"] += outputs.sequences.numel()
        return actions

    def call(self, operation, payload):
        if operation == "describe":
            return {"schema_version": SCHEMA_VERSION, "role": "method", "id": "streamvln_video_qwen_1_5",
                    "status": "boundary_bound_unverified" if self._checkpoint_available() else "checkpoint_missing",
                    "real": self._checkpoint_available(), "requires_sensors": ["rgb"],
                    "emits_actions": ["primitive", "stop"], "accepts_transition": False,
                    "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 79},
                    "checkpoint": self.checkpoint, "preprocess_mode": self.preprocess_mode,
                    "tokenizer_mode": self.tokenizer_mode}
        session_id = payload.get("session_id")
        if operation == "reset":
            if self.sessions:
                raise ContractError("StreamVLN supports one active session per worker")
            self._ensure_model()
            context = payload["context"]
            self.model.reset_for_env(0)
            self.sessions[session_id] = {"episode_id": context["episode_id"], "instruction": context["goal"]["value"],
                "last_sequence": -1, "frames": [], "processed": {}, "time_ids": [], "pending": [],
                "output_ids": None, "past_key_values": None}
            return {}
        if operation == "close_episode":
            if self.sessions.pop(session_id, None) is not None:
                self.model.reset_for_env(0)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        observation = payload["observation"]
        validate_observation(observation)
        state = self.sessions[session_id]
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("StreamVLN consumes exactly the rgb sensor")
        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or str(rgb.dtype) != "uint8":
            raise ContractError("StreamVLN requires uint8 HWC RGB")
        state["last_sequence"] = observation["sequence"]
        step = len(state["frames"])
        state["frames"].append(rgb.copy())
        state["time_ids"].append(step)
        self.phases["observed_frames"] += 1
        if self.preprocess_mode == "eager":
            state["processed"][step] = self._preprocess_rgb(rgb)
        if not state["pending"]:
            state["pending"] = self._generate(state) or [0]
        action = state["pending"].pop(0)
        if (step + 1) % MODEL_NUM_FRAMES == 0:
            self.model.reset_for_env(0)
            state["output_ids"] = state["past_key_values"] = None
            state["time_ids"] = []
        return {"episode_id": state["episode_id"], "observation_sequence": observation["sequence"],
                "actions": [self._action_dict(action)]}

    @staticmethod
    def _action_dict(action):
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action]
