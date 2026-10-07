"""GA-VLN method service: Qwen2-7B + SigLIP + VGGT-1B with BEV memory.

Wraps the GA-VLN checkpoint (gavln_official) behind the nav-eval method
contract. The model classes are imported from the bundled runtime; the
decision block, prompt preprocessing and action parsing are ported from
local/GA-VLN/gavln/gavln_eval.py (VLNEvaluator.eval_action + preprocess_qwen)
verbatim:

- eight-step dialogue windows (dia_round 2 x bev_num_steps 4): the first
  decision of each window builds the full prompt with the <memory> BEV token,
  later decisions continue the KV cache (input_ids = cat(output_ids, turn));
  the model is reset_for_env at window boundaries;
- per-decision inputs: SigLIP front view, VGGT rgb (518 width bicubic), VGGT
  depth (filtered -> scaled to mm uint16 -> 518x392 -> 28x37 grid), BEV
  world-xy backprojection from a 27x27 depth patch grid using the per-step
  agent pose (note the upstream -rotation_matrix sign in the camera-to-world
  transform) and pinhole intrinsics derived from the sensor hfov;
- greedy generation (max_new_tokens 10000) decoded with special tokens kept,
  regex action parse over STOP/up/left/right arrows, empty parse -> stop;
- upstream argument defaults: bev_grid_size 0.25, bev_range 10, bev_pos_temp
  10000, num_front_view 0 (no <history> token).

The pose comes from the platform pose sensor ([x,y,z,qw,qx,qy,qz], habitat
world frame) instead of env.sim.get_agent_state(); depth arrives unnormalised
in metres (upstream min/max depth 0/10). R2R only (no upstream RxR weights).
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
    "Devise an action sequence to follow the instruction using the four actions: "
    "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 "
    "centimeters, or STOP."
)
ACTIONS2IDX = OrderedDict({"STOP": [0], "↑": [1], "←": [2], "→": [3]})
CONJUNCTIONS = ["you can see "]

NUM_STEPS = 4        # bev_num_steps
MAX_STEPS = 32       # bev_max_steps
DIA_ROUND = 2        # window = dia_round * num_steps = 8 executed steps
NUM_FRONT_VIEW = 0
PATCH_GRID_SIZE = 27
PATCH_SIZE_PIXEL = 14
MIN_DEPTH_M = 0.0
MAX_DEPTH_M = 10.0
MODEL_MAX_LENGTH = 4096

FORWARD_STEP_M = 0.25
TURN_RAD = 0.2617993877991494  # 15 degrees


class GAVLNMethodService:
    """requires rgb+depth+pose (640x480 hfov 79); emits primitive/stop."""

    def __init__(self, model_path, vision_tower, vggt_path):
        self.model_path = model_path
        self.vision_tower = vision_tower
        self.vggt_path = vggt_path
        self.model = None
        self.sessions = {}
        self.phases = {"decide_s": 0.0, "decisions": 0}

    # -- model lifecycle ----------------------------------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        import os
        from .runtime.model.model_gavln import GAVLNForCausalLM
        from .runtime.utils.utils import (
            DEFAULT_HIS_TOKEN,
            DEFAULT_MEMORY_TOKEN,
            HIS_TOKEN_INDEX,
            IMAGE_TOKEN_INDEX,
            MEMORY_TOKEN_INDEX,
            dict_to_cuda,
        )
        import torch
        import transformers

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        self.DEFAULT_MEMORY_TOKEN = DEFAULT_MEMORY_TOKEN
        self.DEFAULT_HIS_TOKEN = DEFAULT_HIS_TOKEN
        self.IMAGE_TOKEN_INDEX, self.MEMORY_TOKEN_INDEX, self.HIS_TOKEN_INDEX = (
            IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX, HIS_TOKEN_INDEX)
        self.dict_to_cuda = dict_to_cuda

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, model_max_length=MODEL_MAX_LENGTH, padding_side="right")
        config = transformers.AutoConfig.from_pretrained(self.model_path)
        config.mm_vision_tower = self.vision_tower
        config.vggt_model_path = self.vggt_path
        self.model = GAVLNForCausalLM.from_pretrained(
            self.model_path, attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, config=config, low_cpu_mem_usage=False,
        )
        self.model.model.bev_grid_size = 0.25
        self.model.model.bev_range = 10.0
        self.model.model.bev_pos_temp = 10000
        self.model.requires_grad_(False)
        self.model.to("cuda")
        self.model.eval()
        self.model.reset(1)  # world_size 1; per-episode reset_for_env in reset()
        self.image_processor = self.model.get_vision_tower().image_processor
        self.image_height_width = self.image_processor.crop_size["height"]
        try:
            from depth_camera_filtering import filter_depth

            self.filter_depth = filter_depth
        except ImportError as error:
            raise ContractError(f"GA-VLN needs the depth_camera_filtering package: {error}") from error

    def prepare(self):
        self._ensure_model()

    # -- upstream preprocessing (verbatim) ------------------------------------
    def preprocess_qwen(self, sources, tokenizer, has_image=False, max_len=2048,
                        system_message="You are a helpful assistant.", add_system=False):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)
            tokenizer.add_tokens(["<history>"], special_tokens=True)
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        history_token_index = tokenizer.convert_tokens_to_ids("<history>")
        chat_template = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + "
                         "'\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(CONJUNCTIONS) + self.DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role": "system",
                                                            "content": system_message}])
            for conv in source:
                try:
                    role, content = conv["role"], conv["content"]
                except KeyError:
                    role, content = conv["from"], conv["value"]
                role = roles.get(role, role)
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template([{"role": role, "content": content}])
                input_id += encode_id
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = self.IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = self.MEMORY_TOKEN_INDEX
                if encode_id == history_token_index:
                    input_id[idx] = self.HIS_TOKEN_INDEX
            input_ids.append(input_id)
        input_ids = self.torch.tensor(input_ids, dtype=self.torch.long)
        return input_ids, conversations

    def parse_actions(self, output):
        action_patterns = "|".join(re.escape(action) for action in ACTIONS2IDX)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [ACTIONS2IDX[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def _depth_image(self, depth_meters):
        import numpy as np
        from PIL import Image

        # Habitat transports metres; upstream filtering expects normalized depth.
        normalized = (depth_meters.reshape(depth_meters.shape[:2]) - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)
        depth = self.filter_depth(normalized, blur_type=None)
        depth_mm = (depth * (MAX_DEPTH_M - MIN_DEPTH_M) + MIN_DEPTH_M) * 1000
        # Pillow 9 cannot bicubic-resize I;16. I keeps the millimetre values
        # while supporting both the BEV nearest and VGGT bicubic resizes.
        return Image.fromarray(depth_mm.astype(np.uint16)).convert("I")

    # -- upstream decision block (pose from the pose sensor) -----------------
    def _decide(self, state, rgb, depth_meters, pose):
        import numpy as np
        import quaternion
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms as TF

        device = "cuda"
        past_key_values = state["past_key_values"]
        output_ids = state["output_ids"]

        if output_ids is None:
            sources = copy.deepcopy([{"from": "human", "value": PROMPT_TEMPLATE},
                                     {"from": "gpt", "value": ""}])
            sources[0]["value"] += (f" These are your bird eye view feature map of "
                                    f"historical observations: {self.DEFAULT_MEMORY_TOKEN}.")
            if NUM_FRONT_VIEW != 0 and state["step_id"] >= DIA_ROUND * NUM_STEPS:
                sources[0]["value"] += (f" You have been given a video of historical "
                                        f"observations: {self.DEFAULT_HIS_TOKEN}.")
            sources[0]["value"] = sources[0]["value"].replace(
                "<instruction>.", state["instruction"])
            input_ids, _ = self.preprocess_qwen([sources], self.tokenizer, True,
                                                add_system=True)
            frame_ids = np.arange(max(int(np.floor((state["step_id"] - MAX_STEPS) / NUM_STEPS)
                                       * NUM_STEPS), 0), state["step_id"] + 1, NUM_STEPS) // NUM_STEPS
            frame_ids_bev = torch.from_numpy(frame_ids)
            frame_ids_front = torch.tensor([state["step_id"] // NUM_STEPS])
            frame_ids_his = frame_ids_bev[-(1 + NUM_FRONT_VIEW):-1]
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            input_ids, _ = self.preprocess_qwen([sources], self.tokenizer, True,
                                                add_system=False)
            input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)
            frame_ids = torch.tensor([state["step_id"] // NUM_STEPS])
            frame_ids_bev = frame_ids
            frame_ids_front = frame_ids
            frame_ids_his = frame_ids_bev[-1:-1]

        frame_ids_bev_idx = frame_ids_bev - frame_ids_bev[0]
        frame_ids_front_idx = frame_ids_front - frame_ids_bev[0]
        frame_ids_his_idx = frame_ids_his - frame_ids_bev[0]
        frames_id_dict = {"frame_ids_bev": frame_ids_bev_idx,
                          "frame_ids_front": frame_ids_front_idx,
                          "frame_ids_his": frame_ids_his_idx}

        height, width = rgb.shape[:2]
        hfov = state["hfov_deg"]
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        image = Image.fromarray(rgb).convert("RGB")

        # rgb for VGGT: 518-wide bicubic, height rounded to a multiple of 14
        to_tensor = TF.ToTensor()
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        image_vggt = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        state["images_vggt"].append(to_tensor(image_vggt))
        bev_images_vggt = torch.stack(state["images_vggt"])[frame_ids_bev].to(torch.bfloat16)

        # SigLIP front view
        front = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
        state["vis_frames"].append(front)
        all_images = torch.stack(state["vis_frames"])[frame_ids].to(torch.bfloat16)

        depth_image = self._depth_image(depth_meters)

        # 27x27 BEV patch grid backprojection
        resized_depth_image = depth_image.resize(
            (self.image_height_width, self.image_height_width), Image.NEAREST)
        from transformers.image_utils import to_numpy_array

        depth_img = to_numpy_array(resized_depth_image) / 1000
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).float()
        depth_grid = F.interpolate(depth_tensor, size=(PATCH_GRID_SIZE, PATCH_GRID_SIZE),
                                   scale_factor=None, mode="nearest").squeeze().numpy()

        patch_centers = torch.arange(PATCH_GRID_SIZE, dtype=torch.bfloat16).to(device)
        patch_centers = patch_centers * PATCH_SIZE_PIXEL + PATCH_SIZE_PIXEL / 2
        v_coords, u_coords = torch.meshgrid(patch_centers, patch_centers, indexing="ij")
        u_coords_original = u_coords * width // self.image_height_width
        v_coords_original = v_coords * height // self.image_height_width
        patch_depth = torch.from_numpy(depth_grid).to(device)
        cam_x = (u_coords_original - cx) * patch_depth / fx
        cam_y = (v_coords_original - cy) * patch_depth / fy
        cam_z = patch_depth
        cam_coords_homo = torch.stack([cam_x, cam_y, cam_z,
                                       torch.ones_like(cam_x)], dim=-1).to(device)

        position, (qw, qx, qy, qz) = self._pose_components(pose)
        rotation = np.quaternion(qw, qx, qy, qz)
        rotation_matrix = quaternion.as_rotation_matrix(rotation)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = -rotation_matrix  # upstream sign convention
        transformation_matrix[:3, 3] = np.asarray(position)

        camera_pose = torch.from_numpy(transformation_matrix).to(torch.float32).to(device)
        world_coords = torch.matmul(cam_coords_homo, camera_pose.T)

        rotation_out = torch.from_numpy(rotation_matrix)[[0, 2], :][:, [0, 2]].to(torch.bfloat16)
        state["rotation_list"].append(rotation_out)
        translation_out = torch.from_numpy(np.asarray(position))[[0, 2]].to(torch.bfloat16)
        state["position_list"].append(translation_out)
        world_xy = world_coords[..., [0, 2]].cpu().to(torch.bfloat16)
        state["world_xy_list"].append(world_xy)

        latest_rotation = torch.stack(state["rotation_list"])[frame_ids_front[0]]
        latest_position = torch.stack(state["position_list"])[frame_ids_front[0]]
        world_xy_stack = torch.stack(state["world_xy_list"])[frame_ids_bev]

        # VGGT depth: 518x392 bicubic -> 28x37 grid
        resized_depth_image_vggt = depth_image.resize((518, 392), Image.Resampling.BICUBIC)
        depth_img_vggt = to_numpy_array(resized_depth_image_vggt) / 1000
        depth_tensor_vggt = torch.from_numpy(depth_img_vggt).unsqueeze(0).unsqueeze(0).float()
        depth_grid_vggt = F.interpolate(depth_tensor_vggt, size=(28, 37),
                                        scale_factor=None, mode="nearest").squeeze().numpy()

        patch_centers_w = torch.arange(37, dtype=torch.bfloat16).to(device) * 14 + 14 / 2
        patch_centers_h = torch.arange(28, dtype=torch.bfloat16).to(device) * 14 + 14 / 2
        v_coords_v, u_coords_v = torch.meshgrid(patch_centers_h, patch_centers_w, indexing="ij")
        u_coords_original_v = u_coords_v * width // 518
        v_coords_original_v = v_coords_v * height // 392
        patch_depth_v = torch.from_numpy(depth_grid_vggt).to(device)
        cam_x_v = (u_coords_original_v - cx) * patch_depth_v / fx
        cam_y_v = (v_coords_original_v - cy) * patch_depth_v / fy
        cam_z_v = patch_depth_v
        cam_coords_homo_v = torch.stack([cam_x_v, cam_y_v, cam_z_v,
                                         torch.ones_like(cam_x_v)], dim=-1).to(device)
        world_coords_v = torch.matmul(cam_coords_homo_v, camera_pose.T)
        world_xy_vggt = world_coords_v[..., [0, 2]].cpu().to(torch.bfloat16)
        state["world_xy_vggt_list"].append(world_xy_vggt)
        world_xy_vggt_stack = torch.stack(state["world_xy_vggt_list"])[frame_ids_bev]

        input_dict = {
            "input_ids": input_ids, "position_ids": None, "attention_mask": None,
            "labels": None, "all_images": all_images.unsqueeze(0), "image_sizes": None,
            "img_lens": [len(frame_ids)],
            "positions": latest_position.unsqueeze(0),
            "rotations": latest_rotation.unsqueeze(0),
            "world_xy": world_xy_stack.unsqueeze(0),
            "bev_images_vggt": bev_images_vggt.unsqueeze(0),
            "img_lens_vggt": [bev_images_vggt.shape[0]],
            "positions_vggt": latest_position.unsqueeze(0),
            "rotations_vggt": latest_rotation.unsqueeze(0),
            "world_xy_vggt": world_xy_vggt_stack.unsqueeze(0),
            "frames_id_dict": [frames_id_dict],
            "env_id": 0,
        }
        input_dict = self.dict_to_cuda(input_dict, torch.device(device))

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1,
                                          max_new_tokens=10000, use_cache=True,
                                          return_dict_in_generate=True,
                                          past_key_values=past_key_values)
        state["output_ids"] = outputs.sequences
        state["past_key_values"] = outputs.past_key_values
        llm_outputs = self.tokenizer.batch_decode(state["output_ids"],
                                                  skip_special_tokens=False)[0].strip()
        action_seq = self.parse_actions(llm_outputs)
        if len(action_seq) == 0:
            action_seq = [0]
        return action_seq

    # -- service surface ------------------------------------------------------
    def call(self, operation, payload):
        if operation == "describe":
            return {
                "schema_version": SCHEMA_VERSION, "role": "method", "real": True,
                "id": "gavln_qwen2_7b",
                "requires_sensors": ["depth", "pose", "rgb"],
                "emits_actions": ["primitive", "stop"],
                "accepts_transition": False, "batching": None,
                "embodiment_expectation": {"width": 640, "height": 480, "hfov_deg": 79},
                "model_path": self.model_path,
                "note": "Upstream decision block from GA-VLN gavln/gavln_eval.py; "
                        "8-step KV windows with model reset_for_env at boundaries; "
                        "pose from the measured pose sensor; R2R only.",
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
                raise ContractError("GA-VLN requires a language goal")
            self.model.reset_for_env(0)
            self.sessions[session_id] = {
                "episode_id": context["episode_id"],
                "instruction": context["goal"]["value"],
                "step_id": 0, "action_seq": [],
                "rotation_list": [], "position_list": [],
                "world_xy_list": [], "world_xy_vggt_list": [],
                "vis_frames": [], "images_vggt": [],
                "output_ids": None, "past_key_values": None,
                "last_sequence": -1, "hfov_deg": 79,
            }
            return {}
        if operation == "close_episode":
            self.sessions.pop(session_id, None)
            return {}
        if operation != "act" or session_id not in self.sessions:
            raise ContractError("unknown operation or session not reset")
        return self._act(payload)

    @staticmethod
    def _pose_components(pose):
        import math

        # Habitat's public pose is flat [x, y, z, qw, qx, qy, qz], not
        # the upstream evaluator's separate position/quaternion objects.
        if len(pose) != 7 or not all(math.isfinite(float(value)) for value in pose):
            raise ValueError("GA-VLN pose must contain seven finite xyz/quaternion values")
        return pose[:3], pose[3:]

    def _act(self, payload):
        session_id = payload["session_id"]
        state = self.sessions[session_id]
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb", "depth", "pose"}:
            raise ContractError("GA-VLN consumes exactly the rgb, depth and pose sensors")
        state["last_sequence"] = observation["sequence"]

        if not state["action_seq"]:
            rgb = decode_tensor(observation["sensors"]["rgb"])
            depth = decode_tensor(observation["sensors"]["depth"])
            pose = decode_tensor(observation["sensors"]["pose"])
            if rgb.ndim == 3 and rgb.shape[-1] == 1:
                rgb = rgb[:, :, 0]
            state["hfov_deg"] = observation["sensor_specs"]["rgb"]["calibration"]["hfov_deg"]
            started = time.perf_counter()
            state["action_seq"] = self._decide(state, rgb, depth, pose)
            self.phases["decide_s"] += time.perf_counter() - started
            self.phases["decisions"] += 1

        action_id = state["action_seq"].pop(0)
        state["step_id"] += 1
        if state["step_id"] % (DIA_ROUND * NUM_STEPS) == 0:
            self.model.reset_for_env(0)
            state["output_ids"] = None
            state["past_key_values"] = None
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
