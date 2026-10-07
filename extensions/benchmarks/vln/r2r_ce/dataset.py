"""Frozen episode selection is independent of local asset availability."""
import gzip
import json

from nav_eval.contracts import ContractError


def select_episodes(path, data_root, languages=(), limit=None, requested=None):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    episodes = [e for e in episodes if not languages or e.get("instruction", {}).get("language") in languages]
    def order(episode):
        value = str(episode["episode_id"])
        return (0, int(value)) if value.isdigit() else (1, value)
    episodes.sort(key=order)
    if requested is not None:
        requested = set(requested)
        episodes = [e for e in episodes if str(e["episode_id"]) in requested]
        missing = requested - {str(e["episode_id"]) for e in episodes}
        if missing:
            raise ContractError(f"requested episodes absent from split: {sorted(missing)}")
    if limit is not None:
        episodes = episodes[:limit]
    missing = sorted({e["scene_id"] for e in episodes
                      if not (data_root / "scene_datasets" / e["scene_id"]).is_file()})
    if missing:
        raise ContractError(f"missing scenes in selected episodes: {missing[:10]}; select an explicit available subset")
    return episodes
