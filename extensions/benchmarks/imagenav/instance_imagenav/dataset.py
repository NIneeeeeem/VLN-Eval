"""Frozen selection for Habitat 0.3 InstanceImageNav episodes.

Unlike ObjectNav, InstanceImageNav preserves upstream episode ids.  Those ids
are only scene-local, so Nav-Eval exposes ``scene-relative-path::episode_id``
to make a requested episode unambiguous.
"""
import gzip
import json
from pathlib import Path, PurePosixPath

from nav_eval.contracts import ContractError

SCENE_PREFIX = "data/scene_datasets/"


DEFAULT_CONTENT_SCENES_PATH = "{data_path}/content/{scene}.json.gz"


def scene_relative(scene_id):
    value = str(scene_id).replace("\\", "/")
    value = value.removeprefix(SCENE_PREFIX)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ContractError("scene_id must be relative to scene_datasets")
    return str(path)


def public_episode_id(episode):
    return f"{scene_relative(episode['scene_id'])}::{episode['episode_id']}"


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    content_scenes_path = data.get("content_scenes_path")
    if content_scenes_path is not None and content_scenes_path != DEFAULT_CONTENT_SCENES_PATH:
        raise ContractError("nondefault content_scenes_path is unsupported by Habitat 0.3 loader")
    return data["episodes"]


def select_episodes(path, data_root, limit=None, requested=None):
    """Return raw upstream episodes annotated with an unambiguous public id."""
    path, data_root = Path(path), Path(data_root)
    episodes = list(_load(path))
    content_dir = path.parent / "content"
    if content_dir.is_dir():
        for scene_file in sorted(content_dir.glob("*.json.gz")):
            episodes.extend(_load(scene_file))

    by_public_id = {}
    for episode in episodes:
        public_id = public_episode_id(episode)
        if public_id in by_public_id:
            raise ContractError(f"duplicate scene-qualified episode id: {public_id}")
        copied = dict(episode)
        copied["_nav_eval_public_id"] = public_id
        by_public_id[public_id] = copied

    if requested is not None:
        requested = set(requested)
        missing = requested - set(by_public_id)
        if missing:
            raise ContractError(f"requested episodes absent from split: {sorted(missing)}")
        selected = [episode for episode in by_public_id.values()
                    if episode["_nav_eval_public_id"] in requested]
    else:
        selected = list(by_public_id.values())
    if limit is not None:
        selected = selected[:limit]

    missing_scenes = sorted({episode["scene_id"] for episode in selected if not (
        data_root / "scene_datasets" / scene_relative(episode["scene_id"])
    ).is_file()})
    if missing_scenes:
        raise ContractError("missing scenes in selected episodes: "
                            f"{missing_scenes[:10]}; select an explicit available subset")
    return selected
