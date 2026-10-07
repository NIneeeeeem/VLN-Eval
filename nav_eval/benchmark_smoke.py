"""One real environment episode, without a model or a leaderboard claim."""
from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from nav_eval.plugins import Registry, configured_binding, resource_path_keys, resource_setting_keys
from nav_eval.sdk.service import public_observation


@contextmanager
def _runtime(cwd, pythonpath):
    original_cwd, original_path = Path.cwd(), sys.path[:]
    try:
        sys.path[:0] = pythonpath
        os.chdir(cwd)
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def _configuration(config, registry):
    unknown = set(config) - {"benchmark", "simulator", "benchmark_settings", "environment", "observation", "seed"}
    if unknown:
        raise ValueError(f"unknown smoke configuration fields: {sorted(unknown)}")
    bm = registry.get("benchmark", config["benchmark"])
    sm = registry.get("simulator", config.get("simulator", bm.manifest.get("default_simulator")))
    binding = bm.manifest.get("bindings", {}).get(sm.manifest["id"])
    if not binding:
        raise ValueError("benchmark has no binding for this simulator; SCAND uses evaluate-offline")
    environment = deepcopy(config.get("environment", {}))
    if set(environment) - {"cwd", "pythonpath", "settings"}:
        raise ValueError("smoke runs in the current interpreter; environment supports cwd, pythonpath, settings")
    settings = {**bm.manifest.get("defaults", {}), **config.get("benchmark_settings", {})}
    unknown = set(settings) - set(bm.manifest.get("settings", settings))
    if unknown:
        raise ValueError(f"unknown benchmark settings: {sorted(unknown)}")
    resources = environment.get("settings", {})
    allowed = resource_setting_keys(sm.manifest)
    if set(resources) - allowed:
        raise ValueError(f"unknown environment resource settings: {sorted(set(resources) - allowed)}")
    settings.update(resources)
    settings.update(binding.get("settings", {}))
    binding = configured_binding(binding, settings)
    settings["episode_limit"] = 1
    path_keys = resource_path_keys(sm.manifest) | {"repo_root"}
    paths = {}
    for key in path_keys:
        if settings.get(key):
            settings[key] = str(Path(settings[key]).resolve())
            paths[key] = settings[key]
    if settings.get("repo_root"):
        for key in ("task_config", "dataset_path"):
            value = Path(settings.get(key, ""))
            if not settings.get(key) or value.is_absolute() or ".." in value.parts:
                raise ValueError(f"{key} must be a nonempty path relative to repo_root")
            paths[key] = str(Path(settings["repo_root"]) / value)
        for index, value in enumerate(settings.get("asset_paths", [])):
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError("asset_paths must contain paths relative to repo_root")
            paths[f"asset_paths[{index}]"] = str(Path(settings["repo_root"]) / path)
    observation = {**bm.manifest.get("observation", {}), **config.get("observation", {})}
    offered = set(sm.manifest.get("capabilities", {}).get("offers_sensors", [])) | set(binding.get("offers_sensors", []))
    observation.setdefault("sensors", settings.get("sensors", sorted(offered)))
    selected = set(observation["sensors"])
    if selected - offered or set(binding.get("requires_sensors", [])) - selected:
        raise ValueError("smoke observation must select supported sensors and all required task sensors")
    if any(observation.get(key) != value for key, value in binding.get("fixed_geometry", {}).items()):
        raise ValueError("observation geometry unsupported by the selected binding")
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    worker = {"role": "benchmark", "plugin": bm.identity(), "root": str(bm.root),
              "entrypoint": binding["entrypoint"], "settings": settings,
              "observation": observation, "capabilities": deepcopy(binding),
              "simulator": {**sm.identity(), "entrypoint": sm.manifest["entrypoint"],
                            "root": str(sm.root), "settings": deepcopy(sm.manifest.get("defaults", {}))}}
    cwd = str(Path(environment.get("cwd", settings.get("repo_root", Path.cwd()))).resolve())
    pythonpath = [str(Path(path).resolve()) for path in environment.get("pythonpath", [])]
    missing = [key for key in sm.manifest.get("requires", {}).get("paths", []) if not settings.get(key)]
    paths.update({"cwd": cwd, **{f"pythonpath[{i}]": path for i, path in enumerate(pythonpath)}})
    missing.extend(key for key, value in paths.items() if not Path(value).exists())
    if not Path(cwd).is_dir() and "cwd" not in missing:
        missing.append("cwd")
    return worker, cwd, pythonpath, paths, missing, seed


def smoke_benchmark(config, *, check_only=False, registry=None):
    registry = registry or Registry()
    worker, cwd, pythonpath, paths, missing, seed = _configuration(config, registry)
    benchmark_id, simulator_id = worker["plugin"]["id"], worker["simulator"]["id"]
    result = {"benchmark": benchmark_id, "simulator": simulator_id, "claim": "diagnostic",
              "smoke_only": True, "runtime_verified": False, "paths": paths,
              "missing_paths": sorted(missing), "python": sys.executable}
    simulator = registry.get("simulator", simulator_id).manifest
    modules = list(dict.fromkeys([
        *simulator.get("requires", {}).get("modules", []),
        *worker["capabilities"].get("requires", {}).get("modules", []),
    ]))
    # Import discovery does not initialize native libraries, GPUs or scenes.
    with _runtime(cwd if Path(cwd).is_dir() else Path.cwd(), pythonpath):
        unavailable = []
        for name in modules:
            try:
                spec = importlib.util.find_spec(name)
            except ModuleNotFoundError as error:
                # A missing parent of a dotted module is a missing dependency.
                # Preserve unrelated import failures inside an installed parent.
                if not error.name or not (name == error.name or name.startswith(error.name + ".")):
                    raise
                spec = None
            if spec is None:
                unavailable.append(name)
        result["missing_modules"] = unavailable
        registry.get("benchmark", benchmark_id).load(worker["entrypoint"])
        result["status"] = "blocked" if missing or unavailable else "preflight_passed"
        if check_only or missing or unavailable:
            return result
        if simulator.get("requires", {}).get("main_thread"):
            raise ValueError("this simulator requires a main-thread worker; use nav_eval run "
                             "with isolated resources instead of in-process benchmark smoke")
        from nav_eval.execution.worker import create_service
        service = create_service(worker)
        # Main-thread simulator loops must be driven by the isolated worker.
        # Calling prepare() inline would wait on an unserviced mailbox forever.
        if hasattr(service.service, "run_main"):
            raise ValueError("this simulator requires a main-thread worker; use nav_eval run "
                             "with isolated resources instead of in-process benchmark smoke")
        prepared = False
        try:
            service.prepare()
            prepared = True
            episodes = service.call("episodes", {})
            if not episodes:
                raise ValueError("selected dataset has no episodes")
            payload = {"session_id": "benchmark-smoke", "episode_id": episodes[0], "seed": seed}
            reset = service.call("reset", payload)
            observation = reset["observation"]
            public_observation(observation, worker["observation"])
            result["episode_id"] = episodes[0]
            result["sensor_specs"] = observation["sensor_specs"]
            action = {"kind": "stop", "values": {}}
            if "primitive" in worker["capabilities"]["accepts_actions"]:
                action = {"kind": "primitive", "values": {"name": "left", "amount": worker["capabilities"]["action_semantics"]["turn_rad"]}}
            # One movement/turn, then STOP/FOUND for each active subgoal.
            actions = []
            for index in range(64):
                previous_sequence = observation["sequence"]
                step = service.call("step", {**payload, "expected_sequence": previous_sequence,
                                               "action": action if index == 0 else {"kind": "stop", "values": {}}})
                actions.append(action if index == 0 else {"kind": "stop", "values": {}})
                observation = step["observation"]
                public_observation(observation, worker["observation"])
                if observation["episode_id"] != episodes[0] or observation["sequence"] != previous_sequence + 1:
                    raise ValueError("invalid environment step identity or sequence")
                if step["terminated"] or step["truncated"]:
                    break
            else:
                raise ValueError("smoke STOP/FOUND sequence did not terminate within 64 actions")
            finished = service.call("finish", payload)
            from nav_eval.evaluation.metrics import load_metric
            metrics = {}
            for metric_id in registry.get("benchmark", benchmark_id).manifest.get("metrics", []):
                plugin = registry.get("metric", metric_id)
                for spec in plugin.manifest["metric_set"]["metrics"]:
                    metric = load_metric(spec, plugin.root)
                    metrics[metric.name] = metric.compute(finished["record"], finished["evidence"], spec.get("parameters", {}))
            result.update(status="passed", runtime_verified=True, actions=actions,
                          record={**finished["record"], "smoke_only": True}, metrics=metrics,
                          evidence_schema=worker["capabilities"]["capture_schema"],
                          note="One scripted episode only; no model quality or upstream parity claim.")
        finally:
            failed = sys.exc_info()[0] is not None
            try:
                try:
                    if prepared:
                        service.call("close_episode", {"session_id": "benchmark-smoke"})
                finally:
                    service.close()
            except Exception:
                # Cleanup must not replace the actionable initialization/step error.
                if not failed:
                    raise
    return result
