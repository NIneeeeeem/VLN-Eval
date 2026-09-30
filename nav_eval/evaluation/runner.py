"""Read frozen rollout artifacts, compute a metric set, write a separate evaluation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nav_eval.artifacts import ARTIFACTS, digest_bytes
from nav_eval.evaluation.metrics import evaluate_metric, load_metric
from nav_eval.rollout.generic import GENERIC_ROLLOUT_SCHEMA
from nav_eval.storage import ROLLOUT_SCHEMA as COMMITTED_ROLLOUT_SCHEMA


def evaluate_run(run_dir, metric_set=None, *, plugin_dirs=()):
    root = Path(run_dir)
    collection = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    if set(collection["artifacts"]) != set(ARTIFACTS):
        raise ValueError("unsupported rollout artifact manifest")
    raw = {name: (root / name).read_bytes() for name in ARTIFACTS}
    for name, data in raw.items():
        if digest_bytes(data) != collection["artifacts"][name]:
            raise ValueError(f"rollout integrity check failed: {name}")
    run = json.loads(raw["run.json"])
    if run.get("rollout_schema") == GENERIC_ROLLOUT_SCHEMA:
        if run.get("smoke_only") is not False:
            raise ValueError("generic runs must be explicitly marked non-smoke")
    elif run.get("rollout_schema") == COMMITTED_ROLLOUT_SCHEMA:
        if type(run.get("smoke_only")) is not bool:
            raise ValueError("committed runs must declare smoke status")
    else:
        raise ValueError("unsupported rollout schema; old metric-only runs must be recollected")
    records = [json.loads(line) for line in raw["episodes.jsonl"].splitlines() if line.strip()]
    evidence_rows = [json.loads(line) for line in raw["evidence/episodes.jsonl"].splitlines() if line.strip()]
    expected = run["episode_ids"]
    ids = [r["episode_id"] for r in records]
    if not expected or len(set(expected)) != len(expected) or sorted(ids) != sorted(expected):
        raise ValueError("episode coverage/uniqueness differs from the frozen selection")
    keys = {(r["episode_id"], r["attempt_id"]) for r in records}
    evidence = {}
    for row in evidence_rows:
        key = (row["episode_id"], row["attempt_id"])
        if key not in keys or key in evidence:
            raise ValueError("duplicate or orphaned evidence")
        if row["evidence"] is not None and row["evidence"].get("schema") != run["capture_schema"]:
            raise ValueError("evidence row schema does not match the rollout")
        evidence[key] = row["evidence"]
    if set(evidence) != keys:
        raise ValueError("missing evidence record; an unavailable capture must be explicit null")
    frozen_metric = None
    if metric_set is None and (root / "evaluation.lock.json").is_file():
        locks = json.loads((root / "evaluation.lock.json").read_text())
        if not locks:
            raise ValueError("run has no default metric set; select --metric-set")
        frozen_metric = locks[0]
    metric_path = Path(metric_set) if metric_set else None
    if frozen_metric:
        config = {**frozen_metric["config"], "plugin_root": frozen_metric["root"]}
        metric_bytes = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    elif metric_path is not None and not metric_path.is_file():
        from nav_eval.plugins import Registry
        plugin = Registry(plugin_dirs).get("metric", str(metric_set))
        config = {**plugin.manifest["metric_set"], "plugin_root": str(plugin.root)}
        metric_path = plugin.root / "manifest.json"
        metric_bytes = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    else:
        if metric_path is None:
            raise ValueError("run has no default metric set; select --metric-set")
        metric_bytes = metric_path.read_bytes()
        config = json.loads(metric_bytes)
    if config.get("schema_version") != "nav-eval-metric-set/0.1":
        raise ValueError("unsupported metric-set schema")
    if config["evidence_schema"] != run["capture_schema"]:
        raise ValueError("metric set is incompatible with the captured evidence schema")
    specs = config["metrics"]
    if not specs:
        raise ValueError("metric set must not be empty")
    plugin_root = config.get("plugin_root")
    if plugin_root is None:
        plugin_root = str(metric_path.parent)
    plugins = [load_metric(spec, plugin_root) for spec in specs]
    if len({plugin.name for plugin in plugins}) != len(plugins):
        raise ValueError("duplicate metric output names")
    eval_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = root / "evaluations" / eval_id
    destination.mkdir(parents=True, exist_ok=False)
    scored = []
    for record in records:
        blob = evidence[(record["episode_id"], record["attempt_id"])]
        scored.append({
            "episode_id": record["episode_id"], "attempt_id": record["attempt_id"],
            "metrics": {plugin.name: evaluate_metric(plugin, record, blob, spec.get("parameters", {}))
                        for plugin, spec in zip(plugins, specs)},
        })
    completed = sum(record["status"] == "completed" for record in records)
    policy_errors = sum(record["status"] == "policy_error" for record in records)
    pending = sum(record["status"] == "pending" for record in records)
    aggregates = {}
    metric_failures = 0
    for plugin in plugins:
        values = [s["metrics"][plugin.name]["value"] for s in scored if s["metrics"][plugin.name]["status"] == "ok"]
        unavailable = sum(s["metrics"][plugin.name]["status"] == "unavailable" for s in scored)
        errors = sum(s["metrics"][plugin.name]["status"] == "error" for s in scored)
        metric_failures += unavailable + errors
        aggregates[plugin.name] = {
            "mean": sum(values) / len(values) if values else None, "scored_episodes": len(values),
            "eligible_episodes": completed, "expected_episodes": len(expected),
            "coverage": len(values) / len(expected) if expected else 0.0,
            "unavailable": unavailable, "errors": errors,
        }
    implementation = {}
    for spec, plugin in zip(specs, plugins):
        import inspect
        source = inspect.getsourcefile(type(plugin))
        implementation[spec["entrypoint"]] = digest_bytes(Path(source).read_bytes()) if source else None
    smoke_free = run.get("smoke_only", True)
    manifest = {
        "evaluation_id": eval_id, "rollout_id": run["run_id"], "smoke_only": smoke_free,
        "input_sha256": collection["artifacts"], "metric_set_sha256": digest_bytes(metric_bytes),
        "metric_set": config, "metric_implementation_source_sha256": implementation,
        "note": "Source-file hashes are not full transitive dependency locks.",
    }
    summary = {
        "evaluation_id": eval_id, "run_id": run["run_id"], "smoke_only": smoke_free,
        "metric_set_id": config["id"], "expected_episodes": len(expected),
        "completed_episodes": completed,
        "infrastructure_errors": len(expected) - completed - policy_errors - pending,
        "pending_episodes": pending,
        "policy_errors": policy_errors,
        "coverage": completed / len(expected) if expected else 0.0,
        "metrics": aggregates,
        "status": "complete" if completed == len(expected) and metric_failures == 0 else "incomplete",
        "output_dir": str(destination.resolve()),
        "track": run.get("track", "legacy_unspecified"), "compare_key": run.get("compare_key"),
        "formal_result_eligible": bool(run.get("formal_result_eligible") and completed == len(expected) and metric_failures == 0),
        "aggregation_note": "Means describe scored episodes only; incomplete or uncertified runs are diagnostic.",
    }
    for aggregate in aggregates.values():
        aggregate["official_mean"] = aggregate["mean"] if summary["formal_result_eligible"] else None
    (destination / "evaluation.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "episodes.jsonl").write_text("".join(json.dumps(s, allow_nan=False) + "\n" for s in scored), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
