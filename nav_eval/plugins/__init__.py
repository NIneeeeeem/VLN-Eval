"""Manifest discovery is dependency-free; implementation loading happens in workers."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

KINDS = {"simulator", "benchmark", "method", "metric", "controller"}
SCHEMA = "nav-eval-plugin/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_entrypoint(entrypoint, root):
    module, symbol = entrypoint.split(":", 1)
    if module.endswith(".py"):
        path = (Path(root) / module).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("plugin entrypoint escapes its bundle")
        source = path.read_bytes()
        name = "_nav_plugin_" + hashlib.sha256(str(path).encode() + source).hexdigest()
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ValueError(f"cannot load plugin: {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded
            try:
                # Avoid timestamp-based .pyc reuse after a same-size fast edit.
                exec(compile(source, str(path), "exec"), loaded.__dict__)
            except BaseException:
                del sys.modules[name]
                raise
        loaded = sys.modules[name]
    else:
        loaded = importlib.import_module(module)
    return getattr(loaded, symbol)


@dataclass(frozen=True)
class Plugin:
    manifest: dict
    root: Path

    def load(self, entrypoint=None):
        entrypoint = entrypoint or self.manifest.get("entrypoint")
        if not entrypoint:
            raise ValueError(f"unbound implementation: {self.manifest['id']}")
        return load_entrypoint(entrypoint, self.root)

    def identity(self):
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
                files[str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"kind": self.manifest["kind"], "id": self.manifest["id"],
                "version": self.manifest["version"], "bundle_sha256": digest(files)}


class Registry:
    def __init__(self, extra_dirs=()):
        import extensions
        roots = [Path(extensions.__file__).parent, *(Path(p).resolve() for p in extra_dirs)]
        self.entries = {}
        for root in roots:
            if not root.is_dir():
                raise ValueError(f"plugin directory does not exist: {root}")
            manifests = set(root.rglob("manifest.json"))
            manifests.update(root.rglob("*.manifest.json"))
            for path in sorted(manifests):
                item = json.loads(path.read_text(encoding="utf-8"))
                if (item.get("schema_version") != SCHEMA or item.get("kind") not in KINDS
                        or not isinstance(item.get("version"), str)
                        or not re.fullmatch(r"[a-zA-Z0-9_.-]+", item.get("id", ""))):
                    raise ValueError(f"invalid plugin manifest: {path}")
                if path.name != "manifest.json" and item["kind"] != "method":
                    raise ValueError(f"named manifests are reserved for method variants: {path}")
                if not isinstance(item.get("capabilities", {}), dict):
                    raise ValueError(f"invalid capabilities: {path}")
                key = item["kind"], item["id"]
                if key in self.entries:
                    raise ValueError(f"duplicate plugin: {key}")
                self.entries[key] = Plugin(item, path.parent)

    def get(self, kind, plugin_id):
        try:
            return self.entries[kind, plugin_id]
        except KeyError as error:
            raise ValueError(f"unknown {kind} plugin: {plugin_id}") from error

    def list(self, kind=None):
        return [p.manifest for (k, _), p in sorted(self.entries.items()) if kind is None or k == kind]
