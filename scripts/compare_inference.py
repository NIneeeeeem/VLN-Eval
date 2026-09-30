"""Compare completed inference diagnostics; reject speed claims when trajectories differ.

Run from the repository root: python -B scripts/compare_inference.py BASE CANDIDATE
The comparison reads artifacts only and optionally writes a separate JSON report.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def actions(root):
    result = {}
    fields = ("decision_index", "observation_sequence", "proposed_action", "executed_action",
              "policy_source", "sim_time_s", "control_tick")
    for event in rows(root / "events.jsonl"):
        result.setdefault(event["episode_id"], []).append({key: event[key] for key in fields})
    return result


def metrics(root):
    result = {}
    for evaluation in sorted((root / "evaluations").glob("*/summary.json")):
        summary = read(evaluation)
        result[summary["metric_set_id"]] = {r["episode_id"]: r["metrics"] for r in rows(evaluation.parent / "episodes.jsonl")}
    if not result:
        raise ValueError(f"no scored evaluations found in {root}")
    return result


def timings(root):
    timing = read(root / "timing.json")
    histograms, counts = {}, {"generation_calls": 0, "generation_batches": 0, "input_tokens": 0, "generated_tokens": 0}
    for session in timing["sessions"]:
        workers = session["method_workers"]
        for worker in workers:
            for snapshot in [*worker["retired"], worker["current"]]:
                phases = snapshot["phases"]
                for key in counts:
                    counts[key] += phases[key]
                for size, count in phases["generation_batch_sizes"].items():
                    histograms[size] = histograms.get(size, 0) + count
    decisions = sum(s["decision_count"] for s in timing["sessions"])
    return {"rollout_s": timing["rollout_wall_time_s"], "collection_s": timing["total_wall_time_s"],
            "decisions": decisions, "decisions_per_second": decisions / timing["rollout_wall_time_s"],
            "model_workers": timing["method_worker_count"], "generation_batch_sizes": histograms, **counts}


def compare(baseline, candidate, *, streamvln_preprocessing=False):
    first, second = (read(root / "run.json") for root in (baseline, candidate))
    if first["episode_ids"] != second["episode_ids"] or first["compare_key"] != second["compare_key"]:
        raise ValueError("episode selection or benchmark comparison key differs")
    for root in (baseline, candidate):
        if read(root / "collection.json")["status"] != "complete":
            raise ValueError(f"incomplete collection: {root}")
    locks = {name: read(baseline / name) == read(candidate / name)
             for name in ("resource.lock.json", "agent.lock.json", "environment.lock.json", "evaluation.lock.json")}
    methods = [read(root / "resolved-experiment.json")["method"] for root in (baseline, candidate)]
    settings = [method["settings"] for method in methods]
    changed = {key: [settings[0].get(key), settings[1].get(key)] for key in set(settings[0]) | set(settings[1])
               if key not in settings[0] or key not in settings[1] or settings[0][key] != settings[1][key]}
    settings_equal = settings[0] == settings[1]
    # This exception permits only the two verified CPU preprocessing choices.
    # Decoding, model resources, source/runtime identities and trajectories must
    # still match. It must never permit NaVIDA's upstream -> greedy change.
    compatible_preprocessing = (streamvln_preprocessing and
        all(m["plugin"]["id"] == "streamvln" for m in methods) and
        set(changed) <= {"preprocess_mode", "tokenizer_mode"} and
        all(s.get("preprocess_mode") in {"eager", "lazy"} and
            s.get("tokenizer_mode") in {"upstream", "reuse"} for s in settings))
    before, after = actions(baseline), actions(candidate)
    evidence = [{r["episode_id"]: r["evidence"] for r in rows(root / "evidence/episodes.jsonl")}
                for root in (baseline, candidate)]
    per_episode = {episode: {"baseline_actions": len(before[episode]), "candidate_actions": len(after[episode]),
        "actions_equal": before[episode] == after[episode], "evidence_equal": evidence[0][episode] == evidence[1][episode],
        "first_action_difference": next((i for i in range(max(len(before[episode]), len(after[episode])))
            if before[episode][i:i + 1] != after[episode][i:i + 1]), None)} for episode in first["episode_ids"]}
    metric_before, metric_after = metrics(baseline), metrics(candidate)
    equivalent = (settings_equal or compatible_preprocessing) and all(locks.values()) and all(e["actions_equal"] and e["evidence_equal"] for e in per_episode.values()) and metric_before == metric_after
    base_time, candidate_time = timings(baseline), timings(candidate)
    return {"claim": "diagnostic", "baseline": str(baseline), "candidate": str(candidate),
        "identity_locks_equal": locks, "method_settings_equal": settings_equal,
        "method_settings_changes": changed, "streamvln_preprocessing_comparison": bool(compatible_preprocessing),
        "episodes": per_episode, "all_episode_metrics_equal": metric_before == metric_after,
        "compared_metric_values": sum(len(values) for episodes in metric_before.values() for values in episodes.values()),
        "equivalent": equivalent, "baseline_timing": base_time, "candidate_timing": candidate_time,
        "equivalent_rollout_speedup": base_time["rollout_s"] / candidate_time["rollout_s"] if equivalent else None,
        "equivalent_collection_speedup": base_time["collection_s"] / candidate_time["collection_s"] if equivalent else None,
        "scope": "One diagnostic comparison. Shared GPU load and startup caches affect wall time; collection excludes shutdown and offline scoring."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--streamvln-preprocessing", action="store_true",
                        help="allow only StreamVLN eager/lazy and upstream/reuse preprocessing differences")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, streamvln_preprocessing=args.streamvln_preprocessing)
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end="")
    raise SystemExit(0 if report["equivalent"] else 1)
