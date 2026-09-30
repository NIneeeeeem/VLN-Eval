"""Prepare and compare diagnostic rollouts with CPU preprocessing optimizations disabled."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav_eval.plugins import Registry  # noqa: E402
from nav_eval.storage import read_json, write_json  # noqa: E402


def prepare(args):
    config, resources = read_json(args.config), read_json(args.resources)
    method = config["method"]
    if method not in {"navila", "awarevln", "navida"}:
        raise ValueError("only image preprocessing baselines are supported")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "plugin"
    bundle.mkdir()
    manifest = copy.deepcopy(Registry().get("method", method).manifest)
    manifest["id"] = method + "_preprocessing_reference"
    manifest["entrypoint"] = "reference.py:create"
    manifest["settings"].append("reference_method")
    manifest["defaults"]["reference_method"] = method
    shutil.copyfile(ROOT / "scripts/preprocessing_reference.py", bundle / "reference.py")
    write_json(bundle / "manifest.json", manifest)
    config["method"] = manifest["id"]
    config["plugin_dirs"] = [str(bundle.resolve())]
    config["claim"] = "diagnostic"
    resources["runtimes"][manifest["id"]] = copy.deepcopy(resources["runtimes"][method])
    write_json(args.output / "config.json", config)
    write_json(args.output / "resources.json", resources)
    print(json.dumps({"config": str(args.output / "config.json"), "resources": str(args.output / "resources.json")}))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(args):
    baseline, candidate = args.baseline, args.candidate
    checks = {}
    plans = [read_json(p / "resolved-experiment.json") for p in (baseline, candidate)]
    reference_id, candidate_id = [p["method"]["plugin"]["id"] for p in plans]
    checks["reference_pair"] = (candidate_id in {"navila", "awarevln", "navida"}
                                and reference_id == candidate_id + "_preprocessing_reference")
    checks["reference_entrypoint"] = plans[0]["method"]["entrypoint"] == "reference.py:create"
    for part in ("protocol", "compare_key", "environment", "metrics", "capture_plan", "execution", "replicas"):
        checks[part] = plans[0][part] == plans[1][part]
    reference_settings = dict(plans[0]["method"]["settings"])
    checks["reference_method"] = reference_settings.pop("reference_method", None) == candidate_id
    checks["method_settings"] = reference_settings == plans[1]["method"]["settings"]
    checks["method_resource"] = plans[0]["method"]["resource"] == plans[1]["method"]["resource"]
    fixture = Path(plans[0]["method"]["root"]) / "reference.py"
    checks["known_reference_code"] = fixture.read_bytes() == (ROOT / "scripts/preprocessing_reference.py").read_bytes()
    for filename in ("agent.lock.json", "resource.lock.json", "environment.lock.json", "evaluation.lock.json"):
        left, right = [read_json(p / filename) for p in (baseline, candidate)]
        if filename == "agent.lock.json":
            # The reference plugin has a different identity by construction;
            # package versions, assets and runtime must still be identical.
            left.pop("plugin", None)
            right.pop("plugin", None)
        checks[filename] = left == right
    runs = [read_json(p / "run.json") for p in (baseline, candidate)]
    for part in ("episode_ids", "seed"):
        checks[part] = runs[0][part] == runs[1][part]
    collections = [read_json(p / "collection.json") for p in (baseline, candidate)]
    checks["complete"] = all(r["status"] == "complete" and not r["policy_errors"] and not r["infrastructure_errors"] for r in collections)
    evidence = [{r["episode_id"]: r["evidence"] for r in rows(p / "evidence/episodes.jsonl")} for p in (baseline, candidate)]
    checks["full_private_evidence"] = evidence[0] == evidence[1]
    checks["evidence_coverage"] = bool(runs[0]["episode_ids"]) and all(set(e) == set(runs[0]["episode_ids"]) for e in evidence)
    events = [[{k: e[k] for k in ("episode_id", "decision_index", "observation_sequence", "proposed_action",
                                "executed_action", "policy_source", "sim_time_s", "control_tick")}
               for e in rows(p / "events.jsonl")] for p in (baseline, candidate)]
    checks["actions"] = events[0] == events[1]
    checks["action_coverage"] = all({e["episode_id"] for e in group} == set(runs[0]["episode_ids"]) for group in events)
    metrics = []
    for path in (baseline, candidate):
        files = list((path / "evaluations").glob("*/episodes.jsonl"))
        if len(files) != 1:
            raise ValueError("requires exactly one evaluation per run")
        metrics.append({r["episode_id"]: r["metrics"] for r in rows(files[0])})
    checks["episode_metrics"] = metrics[0] == metrics[1]
    checks["metric_coverage"] = all(set(m) == set(runs[0]["episode_ids"]) for m in metrics)
    equivalent = all(checks.values())
    result = {"claim": "diagnostic", "baseline": str(baseline), "candidate": str(candidate), "checks": checks,
              "equivalent": equivalent, "decisions": [len(e) for e in events], "metrics": metrics,
              "rollout_wall_s": [c["rollout_wall_time_s"] for c in collections],
              "scope": "CPU preprocessing baseline after correctness fixes; not full upstream evaluator or full-split accuracy"}
    if equivalent:
        result["rollout_speedup"] = collections[0]["rollout_wall_time_s"] / collections[1]["rollout_wall_time_s"]
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return equivalent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--resources", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    else:
        raise SystemExit(0 if compare(arguments) else 1)
