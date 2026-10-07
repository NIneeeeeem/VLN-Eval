#!/usr/bin/env python3
"""Run an exact InternVLA-N1 checkpoint, VLM, and NavDP smoke on one GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--image", required=True, help="real simulator RGB frame")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    from extensions.methods.internvla_n1.service import trained_weight_issues
    from extensions.methods.internvla_n1.runtime.internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM,
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint is not a directory: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the InternVLA-N1 verification")

    config = AutoConfig.from_pretrained(checkpoint)
    config.auxiliary_checkpoint_dir = str(checkpoint)
    config.text_config.auxiliary_checkpoint_dir = str(checkpoint)
    model, loading_info = InternVLAN1ForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
        output_loading_info=True,
    )
    issues = trained_weight_issues(loading_info)
    if issues:
        raise RuntimeError(f"trained checkpoint weights failed to load: {issues!r}")
    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise RuntimeError(f"checkpoint left unmaterialized parameters: {meta_parameters!r}")
    model.eval()

    processor = AutoProcessor.from_pretrained(checkpoint)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise ValueError(f"image is not a file: {image_path}")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Where should I go next?"},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            min_new_tokens=2,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        ).sequences
        latents = model.generate_latents(generated, inputs.pixel_values, inputs.image_grid_thw)
        trajectory = model.generate_traj(
            latents,
            torch.zeros((1, 2, 224, 224, 3), device=model.device, dtype=torch.bfloat16),
            torch.zeros((1, 2, 224, 224, 1), device=model.device, dtype=torch.bfloat16),
        )
        if not torch.isfinite(latents).all() or not torch.isfinite(trajectory).all():
            raise RuntimeError("nonfinite latent or NavDP trajectory output")

    report = {
        "checkpoint": str(checkpoint),
        "image": str(image_path),
        "image_size": list(image.size),
        "cuda_device": torch.cuda.get_device_name(model.device),
        "loading_info": loading_info,
        "trained_weight_issues": issues,
        "meta_parameters": meta_parameters,
        "generated_tokens": generated.shape[-1] - inputs.input_ids.shape[-1],
        "navdp_input": "synthetic zero RGB/depth tensors; shape/finite-output check only",
        "outputs_finite": True,
        "generation_sequence_shape": list(generated.shape),
        "latent_shape": list(latents.shape),
        "navdp_trajectory_shape": list(trajectory.shape),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
