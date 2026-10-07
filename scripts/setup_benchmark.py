#!/usr/bin/env python3
"""Install benchmark dependency snapshots locally without a Git checkout."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def configure_registration(runtime, spec):
    """Register benchmark tasks without importing unused training policies."""
    for name, content in spec.get("environment_registration", {}).items():
        target = runtime / name
        original = target.with_name(target.name + ".upstream")
        if not original.exists():
            shutil.copy2(target, original)
        target.write_text("# Nav-Eval environment-only registration; original in .upstream.\n" + content)


def configure_task_configs(runtime, spec):
    """Install one immutable task entrypoint per supported Habitat API."""
    task_configs = spec.get("task_configs")
    if task_configs:
        for simulator, name in task_configs.items():
            source = ROOT / f"configs/benchmarks/tasks/{name}.yaml"
            if not source.is_file():
                raise ValueError(f"missing {simulator} task config: {source}")
            shutil.copy2(source, runtime / f"nav_eval_{name}.yaml")
        return
    if spec.get("task_config"):
        name = spec["task_config"]
        shutil.copy2(ROOT / f"configs/benchmarks/tasks/{name}.yaml", runtime / f"nav_eval_{name}.yaml")


def configure_compatibility(runtime, spec):
    """Apply narrow, recorded patches needed only for task-only execution."""
    for patch in spec.get("compatibility_patches", []):
        target = runtime / patch["path"]
        original = target.with_name(target.name + ".upstream")
        if not original.exists():
            shutil.copy2(target, original)
        before, after = patch["replace"]
        content = target.read_text()
        if before not in content:
            if after in content:
                continue
            raise ValueError(f"compatibility patch no longer matches upstream: {target}")
        target.write_text(content.replace(before, after, 1))


def extract_snapshot(archive, destination):
    """GitHub archives have one top directory; reject links and path escapes."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        roots = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe archive entry: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported archive entry: {member.name}")
            roots.add(path.parts[0])
        if len(roots) != 1:
            raise ValueError("expected an archive with one top-level directory")
        for member in members:
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
                target.chmod(member.mode & 0o777)


def install(benchmark, archive=None):
    spec = json.loads((ROOT / "configs/environments/benchmark-runtimes.json").read_text())[benchmark]
    destination = ROOT / spec["directory"]
    identity = destination / "nav-eval-install.json"
    if destination.exists():
        if not identity.is_file():
            raise ValueError(f"refusing to overwrite existing runtime: {destination}")
        saved = json.loads(identity.read_text())
        if saved["repository"] != spec["repository"] or saved["revision"] != spec["revision"]:
            raise ValueError(f"installed snapshot differs from recipe: {destination}")
        if (saved.get("environment_registration") != spec.get("environment_registration")
                or saved.get("task_configs") != spec.get("task_configs")
                or saved.get("compatibility_patches") != spec.get("compatibility_patches")):
            configure_registration(destination, spec)
            configure_compatibility(destination, spec)
            configure_task_configs(destination, spec)
            saved.update(spec)
            identity.write_text(json.dumps(saved, indent=2) + "\n")
        configure_task_configs(destination, spec)
        return saved
    url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['revision']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nav-eval-benchmark-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive_path = Path(archive).resolve() if archive else temporary / "snapshot.tar.gz"
        if not archive:
            with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        runtime = temporary / "runtime"
        runtime.mkdir()
        extract_snapshot(archive_path, runtime)
        configure_registration(runtime, spec)
        configure_compatibility(runtime, spec)
        # Keep all assets in this repository's shared data/ tree. Never overwrite
        # downloaded assets with files bundled in a dependency snapshot.
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        snapshot_data = runtime / "data"
        if snapshot_data.exists():
            for item in snapshot_data.rglob("*"):
                target = data / item.relative_to(snapshot_data)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
            shutil.rmtree(snapshot_data)
        snapshot_data.symlink_to(data, target_is_directory=True)
        configure_task_configs(runtime, spec)
        digest = hashlib.sha256()
        with archive_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        saved = {**spec, "archive_url": url, "archive_sha256": digest.hexdigest()}
        (runtime / "nav-eval-install.json").write_text(json.dumps(saved, indent=2) + "\n")
        runtime.rename(destination)
    return saved


def main():
    recipes = json.loads((ROOT / "configs/environments/benchmark-runtimes.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(recipes))
    parser.add_argument("--archive", help="use an already downloaded official tar.gz snapshot")
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.benchmark, args.archive), indent=2))
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(2, f"setup-benchmark: {error}\n")


if __name__ == "__main__":
    main()
