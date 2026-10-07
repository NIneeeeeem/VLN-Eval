#!/usr/bin/env python3
"""Download one complete public HSSD scene and its referenced assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "hssd/hssd-hab"
# Fixed upstream revision; retains the rigid scenes used by MultiON.
REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"


def references(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)
    elif isinstance(value, str) and value.endswith((".glb", ".ply", ".navmesh")):
        yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="102816036")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not args.scene.isdigit() or not 1 <= args.jobs <= 16:
        parser.error("scene must be numeric; jobs must be in 1..16")
    target = ROOT / "data/scene_datasets/fphab"
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    receipts = {}
    index = ROOT / f"data/downloads/hssd-index-{REVISION}.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    if not index.exists():
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                        "--max-time", "90", "--output", str(index),
                        f"{endpoint}/api/datasets/{REPOSITORY}/revision/{REVISION}"], check=True)
    filenames = [item["rfilename"] for item in json.loads(index.read_text())["siblings"]]
    object_configs = {PurePosixPath(name).name.removesuffix(".object_config.json"): name
                      for name in filenames if name.endswith(".object_config.json")}

    def fetch(name):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe asset path: {name}")
        destination = target / name
        if not destination.resolve().is_relative_to(ROOT):
            raise ValueError("HSSD target must remain inside the repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
        if not destination.exists():
            partial = destination.with_name(destination.name + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                            "--connect-timeout", "15", "--max-time", "120", "--continue-at", "-",
                            "--output", str(partial), url], check=True)
            partial.rename(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        receipts[name] = {"sha256": digest, "bytes": destination.stat().st_size}
        return destination

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        basic = ["hssd-hab.scene_dataset_config.json", f"scenes/{args.scene}.scene_instance.json",
                 f"stages/{args.scene}.stage_config.json", "semantics/hssd-hab_semantic_lexicon.json"]
        list(pool.map(fetch, basic))
        scene = json.loads((target / basic[1]).read_text())
        handles = sorted({item["template_name"] for item in scene["object_instances"]})
        configs = [object_configs[handle] for handle in handles]
        list(pool.map(fetch, configs))
        assets = {f"stages/{args.scene}.glb"}
        for name in configs:
            for reference in references(json.loads((target / name).read_text())):
                assets.add(str(PurePosixPath(name).parent / reference))
        list(pool.map(fetch, sorted(assets)))
    receipt = ROOT / f"data/downloads/hssd-{args.scene}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION,
                                  "official_source": f"https://huggingface.co/datasets/{REPOSITORY}",
                                  "endpoint": endpoint, "scene": args.scene, "files": receipts}, indent=2) + "\n")
    print(json.dumps({"scene": args.scene, "files": len(receipts), "receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
