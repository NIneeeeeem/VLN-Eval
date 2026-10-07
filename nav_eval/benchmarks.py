"""Benchmark discovery, separate from runnable plugin registration."""
from __future__ import annotations

import json
from pathlib import Path

from nav_eval.plugins import Registry


def _normalize(name):
    return name.strip().lower().replace("-", "_")


def catalog(registry=None):
    import extensions.benchmarks

    registry = registry or Registry()
    path = Path(extensions.benchmarks.__file__).with_name("catalog.json")
    entries = json.loads(path.read_text(encoding="utf-8"))
    for entry in entries:
        plugin = registry.entries.get(("benchmark", entry["id"]))
        bindings = plugin.manifest.get("bindings", {}) if plugin else {}
        offline = bool(plugin and plugin.manifest.get("execution_mode") == "offline"
                       and plugin.manifest.get("entrypoint"))
        entry.update(
            integration_status="implemented" if bindings or offline else "catalog_only",
            execution_mode="offline" if offline or entry["id"] == "scand" else "closed_loop",
            bindings=sorted(bindings),
            validation=plugin.manifest.get("validation", {"level": "unverified"})
            if plugin else {"level": "not_implemented"},
            runtime_verified=False,
        )
    return entries


def inspect_benchmark(name, registry=None):
    normalized = _normalize(name)
    for entry in catalog(registry):
        if normalized in {_normalize(alias) for alias in [entry["id"], *entry["aliases"]]}:
            return entry
    raise ValueError(f"unknown benchmark catalog entry: {name}")
