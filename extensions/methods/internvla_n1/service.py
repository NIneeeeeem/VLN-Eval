"""InternVLA-N1 method service: dual-system VLM waypoint + NavDP local policy.

Wraps the InternVLA-N1 checkpoint (w-NavDP) behind the nav-eval method
contract. The model classes are imported from the bundled runtime; the
dual-system decision loop is ported from
local/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py
(_run_eval_dual_system) with one structural change: upstream steps its own
simulator inside one iteration (look-down x2 capture, look-up x2 restore, then
one action), while this service spreads those simulator steps across act()
calls as camera_tilt actions executed by the platform environment.

Faithful mechanics per upstream:
- every iteration probes the ground view (down, down[capture], up, up) and
  then runs one navigation action; the rgb history frame (384x384) is taken
  at iteration start;
- decision (when no queued actions and no active pixel goal): instruction
  prompt with <image> history placeholders (np.unique(linspace(0, step_id-1,
  8))), random conjunction, greedy generation (max_new_tokens 128);
- digit output -> pixel goal: generate_latents from the text call, then
  generate_traj on the captured look-down rgb/depth (224x224; depth filtered
  -> mm uint16 -> 5 m clamp), traj_to_actions padded to 8 / truncated to 4
  steps; a STOP as the first DP action triggers the upstream replan nudge
  (LEFT); forward_action > 8 resets the goal;
- non-digit output -> regex parse over STOP/up/left/right/down-arrow; the
  down-arrow executes as a double 15-degree look-down whose look-down view is
  reused by the next decision turn (assistant history continuation).

Camera probes consume control ticks, separately from the binding's navigation
budget. Both budgets are finite and recorded in evidence; explicit smoke
control limits still apply. NavDP sampling starts from randn and is seeded per
episode. Isaac embodiment/control differences remain diagnostic.
"""
from __future__ import annotations

import copy
import itertools
import random
import re
import time
from collections import OrderedDict

from nav_eval.contracts import SCHEMA_VERSION, ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor

PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next "
    "waypoint's coordinates in the image. Please output STOP when you have "
    "successfully completed the task."
)
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
ACTIONS2IDX = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})

STOP, FORWARD, LEFT, RIGHT, LOOKDOWN = 0, 1, 2, 3, 5
MAX_STEPS = 8          # upstream forward_action cap per pixel goal
MAX_LOCAL_STEPS = 4    # DP chunk execution cap
NUM_HISTORY = 8
RESIZE_W = RESIZE_H = 384
MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 10.0
TILT_RAD = 0.2617993877991494  # 15 degrees
FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494  # 15 degrees

# iteration probe: (tilt direction, capture the arriving look-down view)
ITERATION_PROBE = [(-1, False), (-1, False), (1, True), (1, False)]
RESTORE_PROBE = [(1, False), (1, False)]

# These prefixes cover every learned tensor in the released N1 safetensors
# layout.  Transformers may report mismatches as either key strings or tuples
# of ``(key, checkpoint_shape, model_shape)`` depending on its version.
TRAINED_WEIGHT_PREFIXES = (
    "visual.",
    "model.embed_tokens.",
    "model.layers.",
    "model.norm.",
    "model.latent_queries",
    "model.navdp.",
    "lm_head.",
)


def _loading_issue_key(issue):
    if isinstance(issue, str):
        return issue
    if isinstance(issue, (tuple, list)) and issue:
        return str(issue[0])
    return str(issue)


def trained_weight_issues(loading_info):
    """Return load diagnostics that affect released InternVLA-N1 parameters."""
    issues = {}
    for category in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        trained = [issue for issue in loading_info.get(category, ())
                   if _loading_issue_key(issue).startswith(TRAINED_WEIGHT_PREFIXES)]
        if trained:
            issues[category] = trained
    return issues


class InternVLAN1MethodService:
    """requires rgb+depth (640x480 hfov 79); emits primitive/stop/camera_tilt."""

    def __init__(self, checkpoint, mode="dual_system"):
        if mode != "dual_system":
            raise ContractError("only the dual_system mode is ported")
        self.checkpoint = checkpoint
        self.model = None
        self.sessions = {}
        self.phases = {"llm_s": 0.0, "dp_s": 0.0, "llm_calls": 0, "dp_calls": 0}

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        from depth_camera_filtering import filter_depth
        from .runtime.depth import preprocess_depth_image_v2
        from .runtime.internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
        )
        from .runtime.internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.split_and_clean = split_and_clean
        self.traj_to_actions = traj_to_actions
        self.preprocess_depth_image_v2 = preprocess_depth_image_v2
        self.filter_depth = filter_depth
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.processor.tokenizer.padding_side = "left"
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(self.checkpoint)
        model_config.auxiliary_checkpoint_dir = self.checkpoint
        # InternVLAN1Model consumes text_config directly for the legacy flat
        # model.* state-dict layout, so propagate the local auxiliary assets.
        model_config.text_config.auxiliary_checkpoint_dir = self.checkpoint
        self.model, loading_info = InternVLAN1ForCausalLM.from_pretrained(
            self.checkpoint, torch_dtype=torch.bfloat16,
            config=model_config,
            attn_implementation="flash_attention_2", device_map={"": "cuda"},
            output_loading_info=True,
        )
        core_issues = trained_weight_issues(loading_info)
        if core_issues:
            raise RuntimeError(
                "InternVLA-N1 checkpoint layout did not load trained core weights: "
                + repr(core_issues)
            )
        self.model.eval()

    def prepare(self):
        self._ensure_model()

    # -- upstream helpers (verbatim) ------------------------------------------
    def parse_actions(self, output):
        action_patterns = "|".join(re.escape(action) for action in ACTIONS2IDX)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [ACTIONS2IDX[match] for match in matches]
        return list(itertools.chain.from_iterable(actions))

    def _capture_look_down(self, rgb, depth_raw):
        import numpy as np
        from PIL import Image

        # The platform transports metres; upstream filters normalized [0, 1]
        # depth before converting to millimetres for NavDP preprocessing.
        normalized = (depth_raw.reshape(depth_raw.shape[:2]) - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)
        depth = self.filter_depth(normalized, blur_type=None)
        depth = depth * (MAX_DEPTH_M - MIN_DEPTH_M) + MIN_DEPTH_M
        depth = depth * 1000
        look_down_depth, _ = self.preprocess_depth_image_v2(
            Image.fromarray(depth.astype(np.uint16), mode="I;16"),
            do_depth_scale=True, depth_scale=1000,
            target_height=224, target_width=224)
        look_down_depth = self.torch.as_tensor(
            np.ascontiguousarray(look_down_depth)).float()
        look_down_depth[look_down_depth > 5.0] = 5.0
        return Image.fromarray(rgb).convert("RGB"), look_down_depth

    # -- decisions -------------------------------------------------------------
    def _user_turn(self, source_value, input_images, start_index):
        """Build one user turn; image placeholders consume input_images
        sequentially (normal turns start at 0, continuations at -1 = the
        newest look-down frame, exactly as upstream indexes)."""
        prompt = random.choice(CONJUNCTIONS) + "<image>"
        value = f"{source_value} {prompt}." if source_value else f"{prompt}."
        content = []
        img_id = start_index
        for part in self.split_and_clean(value):
            if part == "<image>":
                content.append({"type": "image", "image": input_images[img_id]})
                img_id += 1
            else:
                content.append({"type": "text", "text": part})
        return {"role": "user", "content": content}

    def _generate(self, state):
        started = time.perf_counter()
        text = self.processor.apply_chat_template(state["messages"], tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=state["input_images"],
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                past_key_values=None, return_dict_in_generate=True).sequences
        llm_outputs = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        state["llm_outputs"] = llm_outputs
        self.phases["llm_s"] += time.perf_counter() - started
        self.phases["llm_calls"] += 1
        return inputs, output_ids, llm_outputs

    def _decide(self, state):
        """Normal decision turn (upstream lines 383-433)."""
        import numpy as np

        image = state["rgb_list"][-1]
        sources_value = PROMPT_TEMPLATE.replace("<instruction>.",
                                                state["instruction"][:-1])
        if state["step_id"] == 0:
            history_id = []
        else:
            history_id = np.unique(np.linspace(
                0, state["step_id"] - 1, NUM_HISTORY, dtype=np.int32)).tolist()
            placeholder = "<image>\n" * len(history_id)
            sources_value += f" These are your historical observations: {placeholder}."
        history_id = sorted(history_id)
        state["input_images"] = [state["rgb_list"][i] for i in history_id] + [image]
        state["messages"].append(self._user_turn(sources_value, state["input_images"], 0))
        inputs, output_ids, llm_outputs = self._generate(state)
        self._consume_decision(state, inputs, output_ids, llm_outputs)

    def _decide_continuation(self, state):
        """Decision after a down-arrow command (upstream lines 375-415)."""
        state["input_images"] = state["input_images"] + [state["look_down_image"]]
        state["messages"].append({"role": "assistant",
                                  "content": [{"type": "text", "text": state["llm_outputs"]}]})
        state["messages"].append(self._user_turn("", state["input_images"], -1))
        inputs, output_ids, llm_outputs = self._generate(state)
        self._consume_decision(state, inputs, output_ids, llm_outputs, restore_first=True)

    def _consume_decision(self, state, inputs, output_ids, llm_outputs,
                          restore_first=False):
        if not re.search(r"\d", llm_outputs):
            state["action_seq"] = self.parse_actions(llm_outputs)
            return
        # digit output: pixel goal + latent + first DP chunk (upstream 436-482)
        state["forward_action"] = 0
        coord = [int(c) for c in re.findall(r"\d+", llm_outputs)]
        state["pixel_goal"] = [coord[1], coord[0]]
        state["local_actions"] = []
        pixel_values = inputs.pixel_values
        image_grid_thw = self.torch.cat([thw.unsqueeze(0)
                                         for thw in inputs.image_grid_thw], dim=0)
        with self.torch.no_grad():
            state["traj_latents"] = self.model.generate_latents(
                output_ids, pixel_values, image_grid_thw)
        state["local_actions"] = self._dp_chunk(state, state["traj_latents"])
        if restore_first:
            # still looking down: restore horizontal before executing (444-445)
            state["probe"] = list(RESTORE_PROBE)
            state["phase"] = "probe"
            state["post_probe"] = "nudge_or_select"
            return
        if state["local_actions"][0] == STOP:
            # upstream replan nudge: drop the goal and step LEFT
            self._reset_goal(state)
            state["pending_nudge"] = True

    def _dp_chunk(self, state, traj_latents):
        """One NavDP chunk on the freshest look-down capture (verbatim math)."""
        import numpy as np

        started = time.perf_counter()
        image_dp = self.torch.tensor(
            np.array(state["look_down_image"].resize((224, 224)))).to(self.torch.bfloat16) / 255
        pix_goal_image = copy.copy(image_dp)
        images_dp = self.torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.model.device)
        depth_dp = state["look_down_depth"].unsqueeze(-1).to(self.torch.bfloat16)
        pix_goal_depth = copy.copy(depth_dp)
        depths_dp = self.torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.model.device)
        with self.torch.no_grad():
            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
        action_list = self.traj_to_actions(dp_actions)
        if len(action_list) < MAX_STEPS:
            action_list = action_list + [0] * (MAX_STEPS - len(action_list))
        self.phases["dp_s"] += time.perf_counter() - started
        self.phases["dp_calls"] += 1
        return action_list[:MAX_LOCAL_STEPS]

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "internvla_n1_dual_system",
                "requires_sensors": ["rgb", "depth"],
                "emits_actions": ["camera_tilt", "primitive", "stop"],
                "accepts_transition": False, "batching": None,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 79},
                "checkpoint": self.checkpoint,
                "note": "Upstream _run_eval_dual_system state machine; ground-view "
                        "probes execute as camera_tilt actions; needs "
                        "benchmark_settings.allow_tilt on habitat030 or Isaac VLNVerse.",
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
                raise ContractError("InternVLA-N1 requires a language goal")
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "rgb_list": [], "action_seq": [], "local_actions": [],
                "input_images": [], "messages": [], "llm_outputs": "",
                "pixel_goal": None, "forward_action": 0, "traj_latents": None,
                "step_id": 0, "last_sequence": -1, "phase": "iterate",
                "look_down_image": None, "look_down_depth": None,
                "lookdown_pending": 0, "after_lookdown": False,
                "pending_nudge": False,
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
        from PIL import Image

        state = self.sessions[payload["session_id"]]
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb", "depth"}:
            raise ContractError("InternVLA-N1 consumes exactly the rgb and depth sensors")
        state["last_sequence"] = observation["sequence"]
        rgb = decode_tensor(observation["sensors"]["rgb"])
        depth = decode_tensor(observation["sensors"]["depth"])
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        frame = Image.fromarray(rgb.astype("uint8"))

        # second tilt of a main down-arrow command
        if state["lookdown_pending"] > 0:
            state["lookdown_pending"] -= 1
            return self._reply(state, observation, [self._tilt_action(-1)])

        # drain probe steps; the capture op records the arriving look-down view
        if state["phase"] == "probe":
            direction, capture = state["probe"].pop(0)
            if capture:
                state["look_down_image"], state["look_down_depth"] = \
                    self._capture_look_down(rgb, depth)
            if not state["probe"]:
                state["phase"] = "post_probe"
            return self._reply(state, observation, [self._tilt_action(direction)])

        # decision that reuses the current look-down view (after a down-arrow)
        if state["after_lookdown"]:
            state["after_lookdown"] = False
            state["look_down_image"], state["look_down_depth"] = \
                self._capture_look_down(rgb, depth)
            self._decide_continuation(state)
            if state["phase"] == "probe":  # restore probe queued by digits
                state["probe"].pop(0)  # this first restore tilt is emitted now
                return self._reply(state, observation, [self._tilt_action(1)])
            return self._select(state, observation, frame)

        if state["phase"] == "post_probe":
            state["phase"] = "idle"
            if state["post_probe"] == "nudge_or_select":
                state["post_probe"] = None
                if state["local_actions"][0] == STOP:
                    self._reset_goal(state)
                    return self._execute(state, observation, LEFT)
                return self._select(state, observation, frame)
            # normal iteration: decision (when due) then action selection
            if not state["action_seq"] and state["pixel_goal"] is None:
                self._decide(state)
                if state.get("pending_nudge"):
                    state["pending_nudge"] = False
                    return self._execute(state, observation, LEFT)
            return self._select(state, observation, frame)

        # iterate: start the next upstream iteration on this horizontal frame;
        # the resized frame enters rgb_list here (upstream appends at every
        # iteration top, matching the step_id counter one-for-one)
        state["rgb_list"].append(frame.resize((RESIZE_W, RESIZE_H)))
        state["probe"] = list(ITERATION_PROBE)
        state["probe"].pop(0)  # this first probe tilt is emitted below
        state["phase"] = "probe"
        state["post_probe"] = None
        return self._reply(state, observation, [self._tilt_action(-1)])

    def _select(self, state, observation, frame):
        if state["action_seq"]:
            action = state["action_seq"].pop(0)
            return self._execute(state, observation, action)
        if state["pixel_goal"] is not None:
            if not state["local_actions"]:
                state["local_actions"] = self._dp_chunk(state, state["traj_latents"])
            action = state["local_actions"].pop(0)
            state["forward_action"] += 1
            if state["forward_action"] > MAX_STEPS or action == STOP:
                # upstream resets and continues without executing; the platform
                # needs an action now, so the next iteration's first probe tilt
                # takes this slot (same tick order as upstream's next loop pass)
                self._reset_goal(state)
                return self._begin_iteration(state, observation, frame)
            return self._execute(state, observation, action)
        return self._execute(state, observation, STOP)

    @staticmethod
    def _reset_goal(state):
        state["pixel_goal"] = None
        state["traj_latents"] = None
        state["messages"] = []
        state["local_actions"] = []
        state["forward_action"] = 0
        state["step_id"] += 1

    def _begin_iteration(self, state, observation, frame):
        # after a reset-continue no action executed, so the arriving horizontal
        # observation is the same view upstream would append again at the next
        # iteration top
        state["rgb_list"].append(frame.resize((RESIZE_W, RESIZE_H)))
        state["probe"] = list(ITERATION_PROBE)
        state["probe"].pop(0)
        state["phase"] = "probe"
        state["post_probe"] = None
        return self._reply(state, observation, [self._tilt_action(-1)])

    def _execute(self, state, observation, action):
        if action == LOOKDOWN:
            # upstream executes the down-arrow as a double look-down; the
            # following decision reuses the look-down view
            state["lookdown_pending"] = 1
            state["after_lookdown"] = True
            return self._reply(state, observation, [self._tilt_action(-1)])
        state["step_id"] += 1
        state["messages"] = []
        state["phase"] = "iterate"
        return self._reply(state, observation, [self._action_dict(action)])

    def _reply(self, state, observation, actions):
        return {"episode_id": state["episode_id"],
                "observation_sequence": observation["sequence"], "actions": actions}

    def timing_snapshot(self):
        return dict(self.phases)

    @staticmethod
    def _tilt_action(direction):
        return {"kind": "camera_tilt",
                "values": {"delta_rad": TILT_RAD if direction > 0 else -TILT_RAD}}

    @staticmethod
    def _action_dict(action_id):
        return {
            0: {"kind": "stop", "values": {}},
            1: {"kind": "primitive", "values": {"name": "forward", "amount": FORWARD_STEP_M}},
            2: {"kind": "primitive", "values": {"name": "left", "amount": TURN_RAD}},
            3: {"kind": "primitive", "values": {"name": "right", "amount": TURN_RAD}},
        }[action_id]
