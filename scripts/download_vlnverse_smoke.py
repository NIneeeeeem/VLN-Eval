#!/usr/bin/env python3
"""Download public VLNVerse validation episodes, one scene and H1 assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not args.endpoint.startswith("https://"):
        parser.error("jobs must be 1..8 and endpoint must use HTTPS")
    endpoint = args.endpoint.rstrip("/")
    downloads = ROOT / "data/downloads/vlnverse-web"
    downloads.mkdir(parents=True, exist_ok=True)
    receipt_path = downloads / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"sources": {}, "files": {}}
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        token = os.environ.get("HF_TOKEN")

    def repo_endpoint(repo):
        # Gated assets use the official endpoint; never send credentials to mirrors.
        return "https://huggingface.co" if repo == "InternRobotics/Embodiments" else endpoint

    def curl(url, path):
        command = ["curl", "--ipv4", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                        "--connect-timeout", "15", "--max-time", "900", "--continue-at", "-",
                        "--output", str(path)]
        if token and urlparse(url).hostname == "huggingface.co":
            # NamedTemporaryFile is private; the token never appears in argv/logs.
            with tempfile.NamedTemporaryFile(mode="w", prefix="nav-hf-auth-") as header:
                header.write(f"Authorization: Bearer {token}\n")
                header.flush()
                subprocess.run(command + ["--header", "@" + header.name, url], check=True)
        else:
            subprocess.run(command + [url], check=True)

    def index(repo):
        path = downloads / (repo.replace("/", "-") + "-index.json")
        if not path.exists():
            curl(f"{repo_endpoint(repo)}/api/datasets/{repo}/revision/main", path)
        value = json.loads(path.read_text())
        revision = value["sha"]
        receipt["sources"][repo] = {"revision": revision, "official_source": f"https://huggingface.co/datasets/{repo}", "endpoint": repo_endpoint(repo)}
        return revision, [item["rfilename"] for item in value["siblings"]]

    def fetch(repo, revision, name, directory):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe remote path: {name}")
        path = directory / name
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"asset target leaves repository: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path.relative_to(ROOT))
        saved = receipt["files"].get(key)
        if path.exists():
            if not saved or saved["sha256"] != digest(path):
                raise ValueError(f"existing asset has no matching download receipt: {path}")
            return path
        # Avoid stale mirror redirects whose signed CDN URLs have expired.
        url = f"{repo_endpoint(repo)}/datasets/{repo}/resolve/{revision}/{quote(name)}?download=true&nav_eval_ts={time.time_ns()}"
        partial = path.with_name(path.name + ".part")
        curl(url, partial)
        partial.rename(path)
        receipt["files"][key] = {"url": url, "sha256": digest(path), "bytes": path.stat().st_size}
        return path

    repo = "Eyz/VLNVerse_data"
    revision, names = index(repo)
    splits = ROOT / "data/datasets/vlnverse"
    for granularity in ("fine", "coarse"):
        name = f"raw_data/final_splits/{granularity}_val_unseen.json.gz"
        if name not in names:
            raise ValueError(f"upstream split missing: {name}")
        raw = fetch(repo, revision, name, downloads)
        data = json.load(gzip.open(raw, "rt"))
        for episode in data["episodes"]:
            text = episode["instruction"]["instruction_text"]
            if isinstance(text, dict):
                episode["instruction"]["instruction_text"] = text["formal"]
        splits.mkdir(parents=True, exist_ok=True)
        destination = splits / f"{granularity}_val_unseen.json.gz"
        # Never replace user-owned episodes without a receipt from this script.
        key = str(destination.relative_to(ROOT))
        if destination.exists() and (key not in receipt["files"] or digest(destination) != receipt["files"][key]["sha256"]):
            raise ValueError(f"refusing to overwrite existing split: {destination}")
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            json.dump(data, handle)
        receipt["files"][key] = {"source": str(raw.relative_to(ROOT)), "transform": "select formal instruction if variants exist; retain every episode", "sha256": digest(destination), "bytes": destination.stat().st_size}
        if granularity == "fine":
            first = sorted(data["episodes"], key=lambda item: str(item["episode_id"]))[0]
    # Save partial provenance so interrupted downloads can safely resume.
    receipt["episode_id"], receipt["scene"] = str(first["episode_id"]), first["scan"]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for repo, directory, prefix in (
        ("Eyz/VLNVerse_scene", ROOT / "data/scene_data/vlnverse", first["scan"] + "/"),
        ("InternRobotics/Embodiments", ROOT / "data/Embodiments", "vln-pe/h1/"),
    ):
        if repo == "InternRobotics/Embodiments" and not token:
            raise RuntimeError("H1 assets require web approval at https://huggingface.co/datasets/InternRobotics/Embodiments; then run 'hf auth login' or set HF_TOKEN locally and rerun with the logged-in environment's Python. No Hugging Face credential is available.")
        revision, names = index(repo)
        selected = [name for name in names if name.startswith(prefix)]
        if not selected:
            raise ValueError(f"no public assets under {repo}/{prefix}")
        print(f"Downloading {repo}: {len(selected)} files", flush=True)
        try:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(lambda name: fetch(repo, revision, name, directory), selected))
        finally:
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    for path in (ROOT / "data/Embodiments/vln-pe/h1/h1_internvla.usd",
                 ROOT / "data/Embodiments/vln-pe/h1/policy/move_by_speed/h1_loco_jit_policy.pt"):
        if not path.is_file():
            raise ValueError(f"required robot asset missing: {path}")
    print(json.dumps({"episode_id": receipt["episode_id"], "scene": receipt["scene"], "receipt": str(receipt_path),
                      "files": len(receipt["files"]), "bytes": sum(item["bytes"] for item in receipt["files"].values())}, indent=2))


if __name__ == "__main__":
    main()
