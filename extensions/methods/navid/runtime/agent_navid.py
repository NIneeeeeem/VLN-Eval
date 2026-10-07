import json
import numpy as np
from tqdm import trange
import os
import re
import torch
import random
from extensions.methods.navid.runtime.navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from extensions.methods.navid.runtime.navid.conversation import conv_templates, SeparatorStyle
from extensions.methods.navid.runtime.navid.model.builder import load_pretrained_model
from extensions.methods.navid.runtime.navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class NaVid_Agent:

    def __init__(self, model_path, result_path, require_map=False, vision_tower=None):
        print('Initialize NaVid')
        self.result_path = result_path
        self.require_map = require_map
        self.conv_mode = 'vicuna_v1'
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, 'log'), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, 'video'), exist_ok=True)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path), vision_tower=vision_tower)
        print('Initialization Complete')
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
        self.history_rgb_tensor = None
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):
        start_img_index = 0
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim=0)
        return [self.history_rgb_tensor]

    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt
        VIDEO_START_SPECIAL_TOKEN = '<video_special>'
        VIDEO_END_SPECIAL_TOKEN = '</video_special>'
        IMAGE_START_TOKEN = '<image_special>'
        IMAGE_END_TOKEN = '</image_special>'
        NAVIGATION_SPECIAL_TOKEN = '[Navigation]'
        IAMGE_SEPARATOR = '<image_sep>'
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors='pt').input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors='pt').input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors='pt').input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors='pt').input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors='pt').input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors='pt').input_ids[0][1:].cuda()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        imgs = self.process_images(self.rgb_list)
        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(input_ids, images=imgs, do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def extract_result(self, output):
        if 'stop' in output:
            return (0, None)
        elif 'forward' in output:
            match = re.search('-?\\d+', output)
            if match is None:
                return (None, None)
            match = match.group()
            return (1, float(match))
        elif 'left' in output:
            match = re.search('-?\\d+', output)
            if match is None:
                return (None, None)
            match = match.group()
            return (2, float(match))
        elif 'right' in output:
            match = re.search('-?\\d+', output)
            if match is None:
                return (None, None)
            match = match.group()
            return (3, float(match))
        return (None, None)

    def reset(self):
        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.first_forward = False

    def act(self, observations, info, episode_id):
        self.episode_id = episode_id
        rgb = observations['rgb']
        self.rgb_list.append(rgb)
        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)
            return {'action': temp_action}
        navigation_qs = self.promt_template.format(observations['instruction']['text'])
        navigation = self.predict_inference(navigation_qs)
        action_index, num = self.extract_result(navigation[:-1])
        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num / 25))):
                self.pending_action_list.append(1)
        elif action_index == 2:
            for _ in range(min(3, int(num / 30))):
                self.pending_action_list.append(2)
        elif action_index == 3:
            for _ in range(min(3, int(num / 30))):
                self.pending_action_list.append(3)
        if action_index is None or len(self.pending_action_list) == 0:
            self.pending_action_list.append(random.randint(1, 3))
        return {'action': self.pending_action_list.pop(0)}
