"""StreamVLN inference helpers, extracted from streamvln_eval.py.

Source revision and license are recorded in provenance.json.
"""
from __future__ import annotations
import copy
import itertools
import random
import re
from collections import OrderedDict
from .utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX

class StreamPolicy:

    def __init__(self, image_processor):
        self.image_processor = image_processor
        prompt = f'<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP.'
        answer = ''
        self.conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]
        self.actions2idx = OrderedDict({'STOP': [0], '↑': [1], '←': [2], '→': [3]})
        self.conjunctions = ['you can see ', 'in front of you is ', 'there is ', 'you can spot ', 'you are toward the ', 'ahead of you is ', 'in your sight is ']

    def parse_actions(self, output):
        action_patterns = '|'.join((re.escape(action) for action in self.actions2idx))
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwen(self, sources, tokenizer, has_image: bool=False, max_len=2048, system_message: str='You are a helpful assistant.', add_system: bool=False):
        import torch
        roles = {'human': 'user', 'gpt': 'assistant'}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(['<image>'], special_tokens=True)
            tokenizer.add_tokens(['<memory>'], special_tokens=True)
        image_token_index = tokenizer.convert_tokens_to_ids('<image>')
        memory_token_index = tokenizer.convert_tokens_to_ids('<memory>')
        im_start, im_end = tokenizer.additional_special_tokens_ids
        unmask_tokens_idx = [198, im_start, im_end]
        nl_tokens = tokenizer('\n').input_ids
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]['value']) != 0:
                source[0]['value'] += f' {prompt}.'
            else:
                source[0]['value'] = f'{prompt}.'
            if roles[source[0]['from']] != roles['human']:
                source = source[1:]
            input_id, target = ([], [])
            if add_system:
                input_id += tokenizer.apply_chat_template([{'role': 'system', 'content': system_message}])
            for conv in source:
                try:
                    role = conv['role']
                    content = conv['content']
                except:
                    role = conv['from']
                    content = conv['value']
                role = roles.get(role, role)
                conv = [{'role': role, 'content': content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return (input_ids, conversations)
