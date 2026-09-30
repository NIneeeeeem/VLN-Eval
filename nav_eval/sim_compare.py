"""Cross-simulator-version comparison of rollout runs.

Consumes run directories produced by the generic rollout (any habitat arm);
all arms share one evidence schema, so episodes align by id. The report
separates:
- per-episode metric deltas (offline evaluator vs benchmark reference);
- trajectory-level deltas (path length, step counts, final/start geometry);
- environment identity differences recorded in run.json describes
  (simulator version, sensor geometry, split).

It reports facts and flags divergences; causal interpretation lives in
docs/simulator-version-comparison.zh-CN.md, not in this module.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


def _load_run(run_dir):
    root = Path(run_dir)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    records = {r["episode_id"]: r for r in
               (json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines() if line.strip())}
    evidence = {}
    for line in (root / "evidence/episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["evidence"] is not None:
            evidence[row["episode_id"]] = row["evidence"]
    evaluation = None
    evaluations = sorted((root / "evaluations").glob("*")) if (root / "evaluations").is_dir() else []
    if evaluations:
        per_ep = {}
        for line in (evaluations[-1] / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            per_ep[row["episode_id"]] = row["metrics"]
        evaluation = per_ep
    return run, records, evidence, evaluation


def _simulator_label(run):
    bench = run.get("benchmark", {})
    return (bench.get("simulator_version") or bench.get("simulator") or "?").replace(" ", "_")


def _number(value, digits=3):
    return round(value, digits) if isinstance(value, (int, float)) and math.isfinite(value) else None


def compare_runs(run_dirs, output=None):
    if len(run_dirs) < 2:
        raise ValueError("sim-compare needs at least two run directories")
    loaded = []
    for run_dir in run_dirs:
        run, records, evidence, evaluation = _load_run(run_dir)
        loaded.append({"label": _simulator_label(run), "dir": str(run_dir),
                       "run": run, "records": records, "evidence": evidence,
                       "evaluation": evaluation})
    labels = Counter(arm["label"] for arm in loaded)
    for index, arm in enumerate(loaded):
        if labels[arm["label"]] > 1:
            arm["label"] += f"#{index + 1}"
    base, *others = loaded
    shared = sorted(set.intersection(*(set(arm["evidence"]) & set(arm["records"]) for arm in loaded)))
    if not shared:
        raise ValueError("no shared episode ids across runs; check episode selection")

    rows = []
    for episode_id in shared:
        row = {"episode_id": episode_id}
        base_ev = base["evidence"][episode_id]
        row["scene"] = base_ev.get("reference", {}).get("scene_id")
        for arm in [base] + others:
            ev = arm["evidence"][episode_id]
            rec = arm["records"][episode_id]
            metrics = (arm["evaluation"] or {}).get(episode_id, {})
            trajectory, reference = ev.get("trajectory", {}), ev.get("reference", {})
            distances = trajectory.get("goal_distances_m", [])
            row[arm["label"]] = {
                "status": rec["status"],
                "steps": rec.get("steps"),
                "path_length_m": _number(trajectory.get("path_length_m")),
                "start_geodesic_m": _number(reference.get("geodesic_start_to_goal_m")),
                "final_geodesic_m": _number(distances[-1]) if distances else None,
                "start_xyz": [_number(v, 4) for v in reference.get("start_xyz_m", [])],
                "success": metrics.get("r2r_ce.success", {}).get("value"),
                "spl": metrics.get("r2r_ce.spl", {}).get("value"),
                "ne_m": metrics.get("r2r_ce.ne_m", {}).get("value"),
            }
        rows.append(row)

    divergent = [r for r in rows
                 if len({r[a["label"]]["final_geodesic_m"] for a in loaded
                         if r[a["label"]]["final_geodesic_m"] is not None}) > 1]

    environments = {}
    for arm in loaded:
        bench = arm["run"].get("benchmark", {})
        environments[arm["label"]] = {
            "simulator": bench.get("simulator"),
            "simulator_version": bench.get("simulator_version"),
            "sensor_geometry": bench.get("sensor_geometry"),
            "benchmark_id": bench.get("benchmark_id"),
            "clock": bench.get("clock"),
        }

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {}
    for arm in loaded:
        label = arm["label"]
        ne = [r[label]["final_geodesic_m"] for r in rows]
        summary[label] = {
            "episodes": len(rows),
            "mean_final_geodesic_m": mean(ne),
            "mean_path_length_m": mean([r[label]["path_length_m"] for r in rows]),
            "mean_steps": mean([r[label]["steps"] for r in rows]),
            "mean_success": mean([r[label]["success"] for r in rows]),
            "mean_spl": mean([r[label]["spl"] for r in rows]),
        }

    groups = {}
    for arm in loaded:
        key = arm["run"].get("compare_key") or ("unverified:" + arm["label"])
        groups.setdefault(key, []).append(arm["label"])
    same_conditions = len(groups) == 1 and all(arm["run"].get("compare_key") for arm in loaded)
    report = {
        "runs": [{"label": arm["label"], "dir": arm["dir"]} for arm in loaded],
        "environments": environments,
        "comparison_groups": groups,
        "same_conditions": bool(same_conditions),
        "comparison_scope": "same_protocol" if same_conditions else "cross_condition_diagnostic",
        "formal_result_eligible": bool(same_conditions and all(arm["run"].get("formal_result_eligible") for arm in loaded)),
        "episodes_compared": len(rows),
        "episodes_divergent_final_distance": len(divergent),
        "summary": summary,
        "per_episode": rows,
        "note": "final_geodesic_m is the geodesic distance to goal at the last captured "
                "step, computed by each arm's own sim+navmesh; start_geodesic differences "
                "reflect NavMesh/geodesic disagreements before any action is taken.",
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report
