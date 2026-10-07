#!/usr/bin/env python3
"""Download only the official GOAT val_seen CLIP goal caches."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6df4daf962da14d5a57315e9c80f2969f814e6d9"


def main():
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    prefix = f"{endpoint}/datasets/axel81/goat-bench/resolve/{REVISION}/"
    cache = ROOT / "data/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "goat-cache-index.json"
    if not index.exists():
        subprocess.run(["curl", "-fL", "--retry", "2", "--max-time", "90", "-o", str(index),
                        f"{endpoint}/api/datasets/axel81/goat-bench/revision/{REVISION}"], check=True)
    files = [item["rfilename"] for item in json.loads(index.read_text())["siblings"]]
    selected = [name for name in files if name in {
        "goal_cache/ovon/category_name_clip_embeddings.pkl",
        "goal_cache/language_nav/val_seen_instruction_clip_embeddings.pkl"}
        or name.startswith("goal_cache/iin/val_seen_embeddings/") and name.endswith("_CLIP_goat_embedding.pkl")]
    if len(selected) < 3:
        raise ValueError("upstream index has no complete val_seen cache set")

    def fetch(name):
        target = ROOT / "data/goat-assets" / name
        if not target.resolve().is_relative_to(ROOT):
            raise ValueError("cache target leaves repository")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_name(target.name + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                            "--connect-timeout", "15", "--max-time", "300", "--continue-at", "-",
                            "--output", str(partial), prefix + name], check=True)
            partial.rename(target)
        return name, {"bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = dict(pool.map(fetch, selected))
    receipt = cache / "goat-cache.json"
    receipt.write_text(json.dumps({"repository": "axel81/goat-bench", "revision": REVISION,
                                  "official_source": "https://huggingface.co/datasets/axel81/goat-bench",
                                  "endpoint": endpoint, "files": receipts}, indent=2) + "\n")
    print(json.dumps({"files": len(receipts), "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
