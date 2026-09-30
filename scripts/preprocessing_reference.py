"""Validation-only baseline: retain fixed algorithms, disable CPU optimizations.

Not discovered as a production plugin. The harness copies this file into a
hashed run artifact; each isolated worker loads its own reference implementation.
"""
import ast
import copy
from pathlib import Path


def create(config):
    from nav_eval.plugins import Registry
    method = config["settings"]["reference_method"]
    if method not in {"navila", "awarevln", "navida"}:
        raise ValueError("unsupported preprocessing reference")
    adjusted = copy.deepcopy(config)
    del adjusted["settings"]["reference_method"]
    adjusted["plugin"]["id"] = method
    service = Registry().get("method", method).load()(adjusted)
    if method == "navida":
        build = service._build_content
        def uncached_content(instruction, state):
            state.pop("jpeg_cache", None)
            return build(instruction, state)
        service._build_content = uncached_content
    else:
        import importlib
        import numpy as np
        from PIL import Image
        source = Path(config["settings"]["repo_path"]) / f"evaluation/vlnce_baselines/{method}_trainer.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_and_pad_images")
        namespace = {"copy": copy, "np": np, "Image": Image}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        module = importlib.import_module(f"extensions.methods.{method}.service")
        module.sample_and_pad_images = namespace["sample_and_pad_images"]
    return service
