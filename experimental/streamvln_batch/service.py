"""Opt-in StreamVLN tensor batching; independent, unpadded caches per episode.

The production plugin and upstream checkout remain unchanged. Model generation
uses the installed HF implementation, including checkpoint logits processors.
Only cache packing, position handling and episode state ownership differ.
"""
from __future__ import annotations

import copy
import random
import time
import traceback
from contextlib import contextmanager
from types import MethodType

from extensions.methods.streamvln.service import StreamVLNMethodService, MODEL_NUM_FRAMES
from nav_eval.contracts import ContractError, validate_observation
from nav_eval.tensorcode import decode_tensor


def legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def cache_length(cache):
    cache = legacy(cache)
    return cache[0][0].shape[-2] if cache else 0


def pack_inputs(torch, embeddings, caches):
    """Pad past and uncached suffix separately; mask both padding regions.

    HF slices embeddings at the common cache length. Padding the full prompt
    alone would incorrectly skip real suffix tokens of shorter cached rows.
    """
    lengths = [cache_length(cache) for cache in caches]
    suffixes = [embed[:, length:] for embed, length in zip(embeddings, lengths)]
    if any(suffix.shape[1] <= 0 for suffix in suffixes):
        raise ContractError("streaming prefix must extend its KV cache")
    past_max = max(lengths)
    suffix_max = max(s.shape[1] for s in suffixes)
    size = len(embeddings)
    packed = embeddings[0].new_zeros((size, past_max + suffix_max, embeddings[0].shape[-1]))
    mask = torch.zeros((size, past_max + suffix_max), dtype=torch.long, device=packed.device)
    positions = []
    for i, (past, suffix) in enumerate(zip(lengths, suffixes)):
        start = past_max + suffix_max - suffix.shape[1]
        packed[i, start:] = suffix[0]
        mask[i, past_max - past:past_max] = 1
        mask[i, start:] = 1
        positions.append((past_max - past, past_max, start, past_max + suffix_max))
    prototype = next((legacy(cache) for cache in caches if cache_length(cache)), None)
    merged = None
    if prototype:
        merged = []
        for layer_index, layer in enumerate(prototype):
            tensors = []
            for component, tensor in enumerate(layer):
                # The release has 28 decoder layers but text_config declares 32.
                # HF therefore returns four unused, empty cache slots. Preserve
                # their indices exactly, as the upstream singleton path does.
                if isinstance(tensor, list) and not tensor:
                    tensors.append([])
                    continue
                combined = tensor.new_zeros((size, tensor.shape[1], past_max, tensor.shape[-1]))
                for i, (cache, length) in enumerate(zip(caches, lengths)):
                    if length:
                        combined[i:i + 1, :, past_max - length:] = legacy(cache)[layer_index][component]
                tensors.append(combined)
            merged.append(tuple(tensors))
        merged = tuple(merged)
    return packed, mask, merged, positions


def unpack_cache(torch, cache, row, positions, generated_length):
    """Remove both padding regions and tokens computed after this row's EOS.

    generate returns KV for all output tokens except the final sampled token.
    Clone so one session never holds a view into another session's full batch.
    """
    a, b, c, d = positions
    indices = torch.cat((torch.arange(a, b), torch.arange(c, d + generated_length - 1)))
    return tuple(tuple([] if isinstance(tensor, list) and not tensor else
                       tensor[row:row + 1].index_select(-2, indices.to(tensor.device)).contiguous()
                       for tensor in layer) for layer in legacy(cache))


@contextmanager
def flash_unpad_compat():
    """Bridge FA's optional fifth return value to this pinned HF runtime.

    FA adds used sequence lengths after the original four outputs. HF 4.45.1
    only consumes those four; this branch is first reached by padded cached
    prefill, so production unpadded singleton runs never expose the mismatch.
    Restore the module immediately after the exclusive model call.
    """
    from transformers import modeling_flash_attention_utils as flash
    original = getattr(flash, "unpad_input", None)
    if original is None:
        yield
        return
    def unpad(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(result) not in (4, 5):
            raise ContractError("unsupported FlashAttention unpad_input return shape")
        return result[:4]
    flash.unpad_input = unpad
    try:
        yield
    finally:
        flash.unpad_input = original


@contextmanager
def padded_positions(model):
    """Upstream discards position_ids; preserve per-row positions for padding."""
    original = model.prepare_inputs_for_generation
    had_override = "prepare_inputs_for_generation" in model.__dict__
    # HF inspects this signature before accepting inputs_embeds; **kwargs alone
    # would incorrectly advertise that embedding-based generation is unsupported.
    def prepare(self, input_ids, past_key_values=None, attention_mask=None,
                inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True, **kwargs):
        values = original(input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
                          inputs_embeds=inputs_embeds, cache_position=cache_position,
                          position_ids=position_ids, use_cache=use_cache, **kwargs)
        mask = values.get("attention_mask")
        if mask is not None:
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            data = values["inputs_embeds"] if values.get("inputs_embeds") is not None else values["input_ids"]
            values["position_ids"] = positions[:, -data.shape[1]:].contiguous()
        return values
    model.prepare_inputs_for_generation = MethodType(prepare, model)
    try:
        yield
    finally:
        if had_override:
            model.prepare_inputs_for_generation = original
        else:
            del model.prepare_inputs_for_generation


class BatchedStreamVLN(StreamVLNMethodService):
    def __init__(self, *args, capacity=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity
        self.phases.update(vision_encode_s=0.0, cache_pack_s=0.0, cache_unpack_s=0.0)

    def _ensure_model(self):
        if self.model is None:
            import transformers
            if transformers.__version__ != "4.45.1":
                raise ContractError("experimental StreamVLN batching requires the validated Transformers 4.45.1 runtime")
        super()._ensure_model()

    def runtime_identity(self):
        return {**super().runtime_identity(), "batch_implementation": "unpadded_session_cache_v1"}

    def timing_snapshot(self):
        result = super().timing_snapshot()
        if self.model is not None and hasattr(self, "torch") and self.torch.cuda.is_available():
            result["peak_cuda_allocated_mib"] = self.torch.cuda.max_memory_allocated() / 1024**2
            result["peak_cuda_reserved_mib"] = self.torch.cuda.max_memory_reserved() / 1024**2
        return result

    def call(self, operation, payload):
        if operation == "describe":
            return {**super().call(operation, payload), "batching": "independent_greedy"}
        session = payload.get("session_id")
        if operation == "reset":
            if session in self.sessions or len(self.sessions) >= self.capacity:
                raise ContractError("duplicate session or capacity exceeded")
            context = payload["context"]
            if set(context) != {"episode_id", "goal", "embodiment", "seed"} or context["goal"].get("kind") != "language":
                raise ContractError("invalid public episode context")
            self._ensure_model()
            self.sessions[session] = {
                "episode_id": context["episode_id"], "instruction": context["goal"]["value"],
                "last_sequence": -1, "frames": [], "processed": {}, "time_ids": [], "pending": [],
                "output_ids": None, "past_key_values": None, "inputs_embeds": None,
                "rng": random.Random(context["seed"]).getstate()}
            return {}
        if operation == "close_episode":
            self.sessions.pop(session, None)
            return {}
        if operation == "act":
            return self.act_batch([payload])[0]
        raise ContractError("unsupported operation")

    def _prepare_observation(self, payload):
        state = self.sessions.get(payload.get("session_id"))
        if state is None:
            raise ContractError("inactive session")
        observation = payload["observation"]
        validate_observation(observation)
        if observation["episode_id"] != state["episode_id"] or observation["sequence"] <= state["last_sequence"]:
            raise ContractError("stale observation or wrong episode")
        if set(observation["sensors"]) != {"rgb"}:
            raise ContractError("StreamVLN consumes exactly RGB")
        rgb = decode_tensor(observation["sensors"]["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or str(rgb.dtype) != "uint8":
            raise ContractError("StreamVLN requires uint8 HWC RGB")
        step = len(state["frames"])
        state["last_sequence"] = observation["sequence"]
        state["frames"].append(rgb.copy())
        state["time_ids"].append(step)
        self.phases["observed_frames"] += 1
        if self.preprocess_mode == "eager":
            state["processed"][step] = self._preprocess_rgb(rgb)
        return state, observation

    def act_batch(self, payloads):
        if not payloads or len({p["session_id"] for p in payloads}) != len(payloads):
            raise ContractError("batch requires distinct sessions")
        prepared = [self._prepare_observation(payload) for payload in payloads]
        pending = [state for state, _ in prepared if not state["pending"]]
        if pending:
            try:
                actions = self._generate_many(pending)
            except Exception:
                traceback.print_exc()
                raise
            if len(actions) != len(pending):
                raise ContractError("generation batch size mismatch")
            for state, generated in zip(pending, actions):
                state["pending"] = generated or [0]
        replies = []
        for state, observation in prepared:
            action = state["pending"].pop(0)
            if len(state["frames"]) % MODEL_NUM_FRAMES == 0:
                state["output_ids"] = state["past_key_values"] = state["inputs_embeds"] = None
                state["time_ids"] = []
            replies.append({"episode_id": state["episode_id"], "observation_sequence": observation["sequence"],
                            "actions": [self._action_dict(action)]})
        return replies

    def _embeddings(self, state):
        torch = self.torch
        started = time.perf_counter()
        if state["output_ids"] is None:
            sources = copy.deepcopy(self.evaluator.conversation)
            if len(state["frames"]) > 1:
                sources[0]["value"] += " These are your historical observations <memory>."
            sources[0]["value"] = sources[0]["value"].replace("<video>\n", "")
            sources[0]["value"] = sources[0]["value"].replace("<instruction>.", state["instruction"])
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
        previous_rng = random.getstate()
        random.setstate(state["rng"])
        try:
            input_ids, _ = self.evaluator.preprocess_qwen(
                [sources], self.prompt_tokenizer, True, add_system=state["output_ids"] is None)
            state["rng"] = random.getstate()
        finally:
            random.setstate(previous_rng)
        if state["output_ids"] is not None:
            input_ids = torch.cat([state["output_ids"], input_ids.to(state["output_ids"].device)], dim=1)
        self.phases["tokenize_s"] += time.perf_counter() - started
        self.phases["input_tokens"] += input_ids.numel()
        images = torch.stack(self._selected_images(state)).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        started = time.perf_counter()
        result = self.model.prepare_inputs_labels_for_multimodal(
            input_ids.to("cuda"), None, None, None, None, images, None, None, None, None,
            [state["time_ids"]], [0])
        self.phases["vision_encode_s"] += time.perf_counter() - started
        embeds = result[4]
        state["inputs_embeds"] = (embeds if state["inputs_embeds"] is None else
                                  torch.cat([state["inputs_embeds"], embeds], dim=1))
        return state["inputs_embeds"]

    def _generate_many(self, states):
        from transformers import Qwen2ForCausalLM
        torch = self.torch
        with torch.inference_mode():
            embeddings = [self._embeddings(state) for state in states]
            started = time.perf_counter()
            embeds, mask, cache, positions = pack_inputs(torch, embeddings, [s["past_key_values"] for s in states])
            self.phases["cache_pack_s"] += time.perf_counter() - started
            started = time.perf_counter()
            with padded_positions(self.model), flash_unpad_compat():
                outputs = Qwen2ForCausalLM.generate(
                    self.model, inputs_embeds=embeds, attention_mask=mask, past_key_values=cache,
                    do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True,
                    return_dict_in_generate=True, return_legacy_cache=True)
            torch.cuda.synchronize()
            self.phases["generate_s"] += time.perf_counter() - started
            started = time.perf_counter()
            eos = self.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
            sequences = outputs.sequences.tolist()
            actions = []
            for row, (state, tokens) in enumerate(zip(states, sequences)):
                length = next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
                state["output_ids"] = outputs.sequences[row:row + 1, :length].clone()
                state["past_key_values"] = unpack_cache(torch, outputs.past_key_values, row, positions[row], length)
                before_decode = time.perf_counter()
                text = self.tokenizer.batch_decode(state["output_ids"], skip_special_tokens=False)[0].strip()
                actions.append(self.evaluator.parse_actions(text))
                self.phases["decode_s"] += time.perf_counter() - before_decode
                self.phases["generated_tokens"] += length
            self.phases["cache_unpack_s"] += time.perf_counter() - started
        self.phases["generation_calls"] += len(states)
        self.phases["generation_batches"] += 1
        histogram = self.phases["generation_batch_sizes"]
        histogram[str(len(states))] = histogram.get(str(len(states)), 0) + 1
        return actions


def create(config):
    settings = config["settings"]
    return BatchedStreamVLN(checkpoint=settings["checkpoint"], repo_path=settings["repo_path"],
                           preprocess_mode=settings.get("preprocess_mode", "lazy"),
                           tokenizer_mode=settings.get("tokenizer_mode", "reuse"),
                           capacity=config.get("inference", {}).get("session_capacity", 1))
