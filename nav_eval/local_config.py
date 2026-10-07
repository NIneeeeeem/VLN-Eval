"""Machine installation state, registered once and reused by evaluations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from nav_eval.plugins import Registry, resource_path_keys
from nav_eval.storage import read_json, run_lock, write_json

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "configs" / "local.json"


def _validate_resources(resources):
    if not isinstance(resources, dict):
        raise ValueError("installation configuration must contain an object")
    unknown = set(resources) - {"runtimes", "method", "environment", "replicas"}
    if unknown:
        raise ValueError(f"unknown installation fields: {sorted(unknown)}; keep experiments in configs/experiments")
    runtimes = resources.get("runtimes", {})
    if not isinstance(runtimes, dict):
        raise ValueError("runtimes must contain an object")
    for runtime in [*runtimes.values(), *(resources[key] for key in ("method", "environment") if key in resources)]:
        if not isinstance(runtime, dict) or not isinstance(runtime.get("settings", {}), dict):
            raise ValueError("each runtime and its settings must contain an object")


def _merge_runtime(base, override):
    return {**base, **override, "settings": {**base.get("settings", {}), **override.get("settings", {})}}


def selected_resources(resources, config):
    """Freeze only this experiment's deployment, not the machine inventory."""
    result = deepcopy(resources)
    if "runtimes" in result:
        ids = {config["method"], config["simulator"]}
        result["runtimes"] = {key: value for key, value in result["runtimes"].items() if key in ids}
    return result


def load_resources(config=None):
    if not LOCAL_CONFIG.is_file():
        raise ValueError("Local installation is not configured. After downloading assets, "
                         "run nav_eval configure (see scripts/register_assets.sh).")
    resources = read_json(LOCAL_CONFIG)
    _validate_resources(resources)
    registry = Registry(config.get("plugin_dirs", []) if config else [])
    if config:
        resources = selected_resources(resources, config)
        entries = [(resources["runtimes"][config[kind]], registry.get(kind, config[kind]).manifest)
                   for kind in ("method", "simulator") if config[kind] in resources.get("runtimes", {})]
        entries.extend((resources[role], registry.get(kind, config[key]).manifest)
                       for role, kind, key in (("method", "method", "method"),
                                               ("environment", "simulator", "simulator"))
                       if role in resources)
    else:
        # Inventory inspection can retain external registrations even when the
        # corresponding bundle is not part of this invocation's discovery set.
        entries = [(value, plugin.manifest) for key, value in resources.get("runtimes", {}).items()
                   for kind in ("method", "simulator")
                   if (plugin := registry.entries.get((kind, key))) is not None]
    for runtime, manifest in entries:
        for key in ("python", "cwd"):
            if runtime.get(key):
                runtime[key] = _absolute_path(runtime[key])
        for key in ("pythonpath", "library_paths"):
            if key in runtime:
                runtime[key] = [_absolute_path(path) for path in runtime[key]]
        if not config or config.get("launcher") != "docker":
            settings = runtime.get("settings", {})
            for key in resource_path_keys(manifest):
                if settings.get(key):
                    settings[key] = _absolute_path(settings[key])
    return resources


def _absolute_path(value):
    path = Path(value).expanduser()
    # Keep interpreter symlinks: resolving envs/foo/bin/python loses venv identity.
    return str(path if path.is_absolute() else ROOT / path)


def _existing_path(value, *, directory=False):
    path = Path(_absolute_path(value))
    if not path.exists() or (directory and not path.is_dir()):
        raise ValueError(f"download/install before registering this path: {path}")
    return str(path)


def configure(args):
    registry = Registry(args.plugin_dir)
    if not any((args.source, args.method, args.simulator)):
        raise ValueError("configure needs --from, --method or --simulator")
    if (args.checkpoint or args.method_python or args.method_path) and not args.method:
        raise ValueError("method paths require --method")
    if (args.data_root or args.environment_python) and not args.simulator:
        raise ValueError("environment paths require --simulator")
    # Reuse the repository's lock/atomic writer to preserve other registrations.
    with run_lock(LOCAL_CONFIG.parent):
        resources = read_json(LOCAL_CONFIG) if LOCAL_CONFIG.is_file() else {}
        _validate_resources(resources)
        if args.source:
            imported = read_json(args.source)
            _validate_resources(imported)
            for key, value in imported.items():
                if key == "runtimes":
                    runtimes = resources.setdefault(key, {})
                    for plugin_id, runtime in value.items():
                        runtimes[plugin_id] = _merge_runtime(runtimes.get(plugin_id, {}), runtime)
                elif key in ("method", "environment"):
                    resources[key] = _merge_runtime(resources.get(key, {}), value)
                else:
                    resources[key] = value
        for kind, plugin_id, python in (("method", args.method, args.method_python),
                                        ("simulator", args.simulator, args.environment_python)):
            if not plugin_id:
                continue
            # Keep exact installed ids (InternVLA-N1 uses a hyphen); accept the
            # script spelling uni-navid only as an alias for uni_navid.
            if (kind, plugin_id) not in registry.entries:
                plugin_id = {"uni-navid": "uni_navid"}.get(plugin_id, plugin_id) if kind == "method" else plugin_id
            manifest = registry.get(kind, plugin_id).manifest
            role = "method" if kind == "method" else "environment"
            runtimes = resources.setdefault("runtimes", {})
            runtime = deepcopy(runtimes.get(plugin_id, {}))
            # Migrate an imported single-pair map to plugin-specific entries.
            legacy = resources.pop(role, {})
            settings = {**runtime.get("settings", {}), **legacy.get("settings", {})}
            runtime.update(legacy)
            runtime["settings"] = settings
            if python:
                interpreter = Path(_existing_path(python))
                if not interpreter.is_file():
                    raise ValueError(f"Python interpreter must be a file: {interpreter}")
                runtime["python"] = str(interpreter)
            else:
                runtime.setdefault("python", sys.executable)
            paths = {}
            if kind == "method":
                if args.checkpoint:
                    paths["checkpoint"] = args.checkpoint
                for item in args.method_path:
                    key, separator, value = item.partition("=")
                    if not separator or not key or not value:
                        raise ValueError("--method-path expects KEY=PATH")
                    paths[key] = value
            elif args.data_root:
                paths["data_root"] = args.data_root
            allowed = resource_path_keys(manifest)
            if set(paths) - allowed:
                raise ValueError(f"unknown asset paths for {plugin_id}: {sorted(set(paths) - allowed)}")
            for key, value in paths.items():
                settings[key] = _existing_path(value, directory=key == "data_root")
            runtimes[plugin_id] = runtime
        write_json(LOCAL_CONFIG, resources)
    return {"installation_config": str(LOCAL_CONFIG), "runtimes": sorted(resources.get("runtimes", {}))}
