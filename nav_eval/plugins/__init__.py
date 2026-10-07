"""Manifest discovery is dependency-free; implementation loading happens in workers."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

KINDS = {"simulator", "benchmark", "method", "metric", "controller"}
SCHEMA = "nav-eval-plugin/1"


def resource_path_keys(manifest):
    """Required and optional asset paths share one deployment contract."""
    return set(manifest.get("requires", {}).get("paths", [])) | set(manifest.get("resource_paths", []))


def resource_setting_keys(manifest):
    return resource_path_keys(manifest) | set(manifest.get("resource_settings", []))


def configured_binding(binding, settings):
    """Freeze optional actions from declared task settings before negotiation."""
    result = deepcopy(binding)
    gates = result.get("action_requirements", {})
    result["accepts_actions"] = [action for action in result["accepts_actions"]
        if all(settings.get(key) == value for key, value in gates.get(action, {}).items())]
    return result


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
        trees = {"bundle": {}, "dependencies": {}}
        dependencies = self.code_dependencies()
        initializer = self.root.parent / "shared/__init__.py"
        if dependencies:
            trees["shared_initializer"] = hashlib.sha256(initializer.read_bytes()).hexdigest() if initializer.is_file() else None
        roots = [(None, self.root)] + [(path.name, path) for path in dependencies]
        for name, root in roots:
            files = trees["bundle"] if name is None else trees["dependencies"].setdefault(name, {})
            for path in sorted(root.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    if name is not None or path.suffix in {".py", ".json", ".yaml", ".yml", ".jinja"}:
                        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"kind": self.manifest["kind"], "id": self.manifest["id"],
                "version": self.manifest["version"], "bundle_sha256": digest(trees if dependencies else trees["bundle"])}

    def code_dependencies(self):
        """Declared sibling helper directories; inspect bytes without importing code."""
        names = self.manifest.get("code_dependencies", [])
        if not isinstance(names, list) or any(not isinstance(name, str) or
                not re.fullmatch(r"\.\./shared/[a-zA-Z0-9_][a-zA-Z0-9_.-]*", name) for name in names):
            raise ValueError("code_dependencies must be a list of ../shared/<name> paths")
        if len(set(names)) != len(names):
            raise ValueError("duplicate code_dependencies")
        shared = self.root.parent / "shared"
        paths = []
        for name in names:
            path = self.root / name
            if (shared.is_symlink() or (shared / "__init__.py").is_symlink() or path.is_symlink() or not path.is_dir()
                    or path.resolve().parent != shared.resolve()
                    or any(item.is_symlink() for item in path.rglob("*"))):
                raise ValueError(f"invalid shared code directory: {name}")
            paths.append(path.resolve())
        return paths


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
                plugin = Plugin(item, path.parent)
                plugin.code_dependencies()
                self.entries[key] = plugin

    def get(self, kind, plugin_id):
        try:
            return self.entries[kind, plugin_id]
        except KeyError as error:
            raise ValueError(f"unknown {kind} plugin: {plugin_id}") from error

    def list(self, kind=None):
        return [p.manifest for (k, _), p in sorted(self.entries.items()) if kind is None or k == kind]
