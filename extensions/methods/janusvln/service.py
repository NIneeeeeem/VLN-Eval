"""JanusVLN method service: Qwen2.5-VL + VGGT navigation policy.

Wraps the self-contained JanusVLN checkpoint (ModelScope misstl/JanusVLN_Base,
VGGT embedded) behind the nav-eval method contract. Only the model classes are
imported from the bundled runtime (qwen_vl/...); the inference logic is
ported from local/JanusVLN/src/evaluation.py verbatim:

- JanusVLN_Inference.__init__ model loading (bf16, flash_attention_2,
  mode='evaluation', left-padded tokenizer/processor, min_pixels 784,
  max_pixels 1605632);
- call_model: the system prompt and action-list user prompt, VGGT-preprocessed
  image tensors (load_and_preprocess_images) cropped to patch/merge multiples,
  the current frame forwarded as images_vggt (the model caches VGGT KV across
  steps; reset clears past_key_values_vggt), greedy generation
  (max_new_tokens 24, temperature 0, num_beams 1);
- eval_action: history selection (all frames up to 8, then
  linspace(0, history_len, 9) indices), exact-string action match with a stop
  fallback, max_steps 400 guard.

Known divergence (documented, diagnostic claim only): upstream runs under
torchrun with a fixed seed 42; the platform reseeds per episode instead.
"""
from __future__ import annotations

import time

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

SYSTEM_PROMPT = (
    "You are a visual language navigation model, and your should go to the locations "
    "to complete the given task. Compare the observation and instruction to infer "
    "your current progress, and then select the correct direction from the candidates "
    "to go to the target location and finish the task."
)
CONTEXT_TEMPLATE = (
    "These images are your historical observations and your current observation.\n "
    "Your task is to {task} \n You should take one of the following actions:\n "
    "MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
)
ACTIONS2IDX = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1605632
NUM_HISTORY = 8
MAX_STEPS = 400

FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494  # 15 degrees


class JanusVLNMethodService:
    """requires rgb (640x480 hfov 79); emits one primitive/stop per decision."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = None
        self.sessions = {}
        self.phases = {"preprocess_s": 0.0, "generate_s": 0.0, "decode_s": 0.0,
                       "generation_calls": 0}

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        from .runtime.qwen_vl.model.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGenerationForJanusVLN,
        )
        from .runtime.qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        self.torch = torch
        self.load_and_preprocess_images = load_and_preprocess_images
        config = AutoConfig.from_pretrained(self.checkpoint)
        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            self.checkpoint, config=config, torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}, attn_implementation="flash_attention_2",
            mode="evaluation",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
            padding_side="left")

    def prepare(self):
        self._ensure_model()

    # -- upstream call_model (verbatim) ---------------------------------------
    def _call_model(self, images, task):
        import copy
        from io import BytesIO

        import torch
        from PIL import Image
        from qwen_vl_utils import extract_vision_info

        started = time.perf_counter()
        message = [{"role": "system", "content": SYSTEM_PROMPT}]
        image_content = []
        for visual in images:
            image_content.append({"type": "image", "image": visual})
        message.append({"role": "user", "content": image_content + [{"type": "text",
                                                                      "text": CONTEXT_TEMPLATE.format(task=task)}]})
        messages = [message]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

        images_vggt = []
        image_inputs = []
        for entry in messages:
            vision_info = extract_vision_info(entry)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" not in ele:
                    raise NotImplementedError("Unsupported vision info type")
                image = ele["image"]
                if isinstance(image, Image.Image):
                    pass
                elif isinstance(image, str) and "base64," in image:
                    import base64

                    _, base64_data = image.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    with BytesIO(data) as bio:
                        image = copy.deepcopy(Image.open(bio))
                else:
                    raise NotImplementedError("Unsupported image type")
                image = self.load_and_preprocess_images([image])[0]
                if i == len(vision_info) - 1:
                    cur_images_vggt.append(image)
                _, height, width = image.shape
                patch_size = self.processor.image_processor.patch_size
                merge_size = self.processor.image_processor.merge_size
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
            images_vggt.append(torch.stack(cur_images_vggt))

        inputs = self.processor(text=text, images=image_inputs, videos=None,
                                padding=True, return_tensors="pt", do_rescale=False)
        device = self.model.device
        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)

        before_generate = time.perf_counter()
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False, temperature=0, top_p=None, num_beams=1,
            max_new_tokens=24,
        )
        self.torch.cuda.synchronize()
        before_decode = time.perf_counter()
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids
                                 in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        self.phases["preprocess_s"] += before_generate - started
        self.phases["generate_s"] += before_decode - before_generate
        self.phases["decode_s"] += time.perf_counter() - before_decode
        self.phases["generation_calls"] += 1
        return answers

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "janusvln_qwen2_5_vl",
                "requires_sensors": ["rgb"], "emits_actions": ["primitive", "stop"],
                "accepts_transition": False, "batching": None,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 79},
                "checkpoint": self.checkpoint,
                "note": "Upstream call_model/eval_action from JanusVLN "
                        "src/evaluation.py; VGGT KV cache cleared per episode; "
                        "exact-string action parse with stop fallback.",
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
                raise ContractError("JanusVLN requires a language goal")
            self.model.model.past_key_values_vggt = None  # per-episode VGGT cache reset
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "rgb_list": [], "step_id": 0, "last_sequence": -1,
            }
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        return self._act(payload)

    def _act(self, payload):
        from PIL import Image

        session_id = payload["session_id"]
        state = self.sessions[session_id]
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("JanusVLN consumes exactly the rgb sensor")
        state["last_sequence"] = observation["sequence"]

        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = rgb[:, :, 0]
        state["rgb_list"].append(Image.fromarray(rgb).convert("RGB"))

        history_len = len(state["rgb_list"]) - 1
        if history_len <= NUM_HISTORY:
            history_images = state["rgb_list"][:history_len]
            images = history_images + [state["rgb_list"][-1]]
        else:
            import numpy as np

            indices = np.linspace(0, history_len, NUM_HISTORY + 1, dtype=int)
            images = [state["rgb_list"][i] for i in indices]

        answer = self._call_model(images, state["instruction"])[0]
        action_id = ACTIONS2IDX.get(answer, 0)  # upstream: exact match else stop
        if state["step_id"] >= MAX_STEPS:
            action_id = 0
        state["step_id"] += 1
        return {"episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"],
                "actions": [self._action_dict(action_id)]}

    def timing_snapshot(self):
        return dict(self.phases)

    @staticmethod
    def _action_dict(action_id):
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action_id]
