#!/usr/bin/env python3
"""Download official evaluation episodes, preserving existing datasets."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def selected_path(name, recipe):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    if recipe.get("strip_prefix"):
        prefix = PurePosixPath(recipe["strip_prefix"])
        return Path(*path.relative_to(prefix).parts) if path.is_relative_to(prefix) else None
    parent = recipe.get("required_parent")
    if parent and parent not in path.parts:
        return None
    for index, part in enumerate(path.parts):
        if part in recipe["splits"]:
            return Path(*path.parts[index:])
    return None


def extract(archive, target, recipe):
    """Extract evaluation splits only; never replace a pre-existing asset."""
    files = []
    source = zipfile.ZipFile(archive) if recipe["format"] == "zip" else tarfile.open(archive)
    with source:
        members = source.infolist() if recipe["format"] == "zip" else source.getmembers()
        for member in members:
            name = member.filename if recipe["format"] == "zip" else member.name
            relative = selected_path(name, recipe)
            if recipe["format"] == "zip":
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"archive symlink: {name}")
                directory = member.is_dir()
            else:
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsupported archive entry: {name}")
                directory = member.isdir()
            if relative is None or directory:
                continue
            destination = target / relative
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f"target leaves repository through a symlink: {destination}")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = source.open(member) if recipe["format"] == "zip" else source.extractfile(member)
                with stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
            files.append(str(destination.relative_to(ROOT)))
    if not files:
        raise ValueError("archive contains no requested evaluation split")
    return files


def download(name, recipe):
    cache = ROOT / "data/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (name + (".zip" if recipe["format"] == "zip" else ".tar.gz"))
    receipt = cache / (name + ".json")
    url = recipe["url"]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if url.startswith("https://huggingface.co/"):
        url = endpoint + url.removeprefix("https://huggingface.co")
    if not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20",
                        "--max-time", "900", "--continue-at", "-", "--output", str(partial), url], check=True)
        # Validate archive type before promoting a Google Drive confirmation/error page.
        if recipe["format"] == "zip" and not zipfile.is_zipfile(partial):
            raise ValueError("download is not a ZIP; Google Drive may require account access or confirmation")
        if recipe["format"] == "tar" and not tarfile.is_tarfile(partial):
            raise ValueError("download is not a tar archive")
        partial.rename(archive)
    files = extract(archive, ROOT / recipe["target"], recipe)
    if name == "multion_objects":
        for filename in list(files):
            legacy = ROOT / filename
            if legacy.name.endswith(".phys_properties.json"):
                config = legacy.with_name(legacy.name.replace(".phys_properties.json", ".object_config.json"))
                if not config.exists():
                    mesh = json.loads(legacy.read_text())["render mesh"]
                    config.write_text(json.dumps({"render_asset": mesh, "collision_asset": mesh}) + "\n")
                files.append(str(config.relative_to(ROOT)))
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    result = {"dataset": name, "official_url": recipe["url"], "download_url": url,
              "archive_sha256": digest.hexdigest(), "bytes": archive.stat().st_size,
              "files": files, "scene_download": "separate; see data/README.md"}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"dataset": name, "files": len(files), "receipt": str(receipt)}, indent=2), flush=True)
    return result


def main():
    recipes = json.loads((ROOT / "configs/environments/dataset-assets.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(recipes))
    args = parser.parse_args()
    failed = []
    for name in args.datasets:
        try:
            download(name, recipes[name])
        except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
            print(f"{name}: {error}", flush=True)
            failed.append(name)
    if failed:
        parser.exit(1, "Downloads incomplete: " + ", ".join(failed) + "\n")


if __name__ == "__main__":
    main()
