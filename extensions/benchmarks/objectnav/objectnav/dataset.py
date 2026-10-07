"""Frozen episode selection that mirrors habitat's ObjectNavDatasetV1 order.

Habitat renumbers ObjectNav episode ids positionally: the main split file is
loaded first, then every ``content/{scene}.json.gz`` in sorted scene order, and
``episode_id = str(index)`` over that concatenation. The frozen selection below
reproduces the same order and ids so that ids requested in an experiment map to
exactly the episodes habitat will load.
"""
import gzip
import json

from nav_eval.contracts import ContractError

SCENE_PREFIX = "data/scene_datasets/"


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["episodes"]


def select_episodes(path, data_root, limit=None, requested=None):
    """Return the frozen episode list in habitat load order with habitat ids."""
    episodes = list(_load(path))
    content_dir = path.parent / "content"
    if content_dir.is_dir():
        for scene_file in sorted(content_dir.glob("*.json.gz")):
            episodes.extend(_load(scene_file))
    for index, episode in enumerate(episodes):
        episode["episode_id"] = str(index)

    if requested is not None:
        requested = set(requested)
        episodes = [e for e in episodes if str(e["episode_id"]) in requested]
        missing = requested - {str(e["episode_id"]) for e in episodes}
        if missing:
            raise ContractError(f"requested episodes absent from split: {sorted(missing)}")
    if limit is not None:
        episodes = episodes[:limit]

    missing = sorted({e["scene_id"] for e in episodes
                      if not (data_root / "scene_datasets"
                              / e["scene_id"].removeprefix(SCENE_PREFIX)).is_file()})
    if missing:
        raise ContractError(f"missing scenes in selected episodes: {missing[:10]}; select an explicit available subset")
    return episodes
