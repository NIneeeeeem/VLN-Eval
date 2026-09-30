"""Resolved-plan execution, independent workers, episode commits and safe resume."""
from __future__ import annotations

import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from nav_eval.artifacts import artifact_digests
from nav_eval.contracts import SCHEMA_VERSION, ContractError
from nav_eval.execution.launchers import preflight
from nav_eval.execution.pool import WorkerPool
from nav_eval.planning import resolve
from nav_eval.plugins import digest
from nav_eval.storage import (AttemptStore, ROLLOUT_SCHEMA, file_digest, read_json,
                              run_lock, write_json)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    return digest({str(path.relative_to(root)): file_digest(path)
                   for folder in ("nav_eval", "extensions")
                   for path in sorted((root / folder).rglob("*.py"))})


def resource_identity(resolved):
    identities = {}
    for role in ("method", "environment"):
        settings = resolved[role]["resource"].get("settings", {})
        assets = {}
        # Dataset/scene fingerprints come from the environment's selected-asset attestation.
        for key in ("checkpoint", "repo_path"):
            if not settings.get(key):
                continue
            root = Path(settings[key])
            files = [root] if root.is_file() else sorted(p for p in root.rglob("*")
                if p.is_file() and p.suffix in ({".json", ".safetensors", ".bin", ".model", ".pth", ".pt", ".txt", ".jinja"}
                                               if key == "checkpoint" else {".py", ".json"})
                and ".git" not in p.parts and "__pycache__" not in p.parts)
            assets[key] = digest({str(p.relative_to(root)) if root.is_dir() else p.name: file_digest(p) for p in files})
        identities[role] = assets
    return identities


@contextmanager
def _record_failure(root):
    try:
        yield
    except BaseException as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "error": str(error),
                   "time": datetime.now(timezone.utc).isoformat(), "status": "failed",
                   "logs": [str(p.relative_to(root)) for p in sorted(root.glob("*.log")) + sorted(root.glob("workers/*/*.log"))]})
        raise


def _summarize(root, store, run, sessions=()):
    records = store.export(run["episode_ids"])
    complete = sum(r["status"] == "completed" for r in records)
    policy = sum(r["status"] == "policy_error" for r in records)
    infrastructure = sum(r["status"] == "infrastructure_error" for r in records)
    episode_wall = sum(r.get("wall_time_s", 0) for r in records)
    rollout_wall = sum(s["rollout_wall_time_s"] for s in sessions)
    collection_wall = sum(s["total_wall_time_s"] for s in sessions)
    timed_complete = sum(s["completed_episodes"] for s in sessions)
    summary = {"run_id": run["run_id"], "config_sha256": run["config_sha256"],
               "smoke_only": run["smoke_only"], "track": run["track"], "compare_key": run["compare_key"],
               "expected_episodes": len(records), "completed_episodes": complete,
               "policy_errors": policy, "infrastructure_errors": infrastructure,
               "pending_episodes": len(records) - complete - policy - infrastructure,
               "coverage": complete / len(records), "status": "complete" if complete == len(records) else "incomplete",
               "parallelism": run.get("parallelism", 1), "episode_wall_time_sum_s": episode_wall,
               "inference": run.get("inference", {"mode": "replicas"}),
               "method_worker_count": run.get("method_worker_count", run.get("parallelism", 1)),
               "rollout_wall_time_s": rollout_wall, "collection_wall_time_s": collection_wall,
               "episodes_per_hour": complete * 3600 / rollout_wall if rollout_wall and timed_complete == complete else None,
               "collection_episodes_per_hour": complete * 3600 / collection_wall if collection_wall and timed_complete == complete else None,
               "formal_result_eligible": False, "claim": "diagnostic",
               "output_dir": str(root.resolve()), "artifacts": artifact_digests(root)}
    write_json(root / "collection.json", summary)
    return summary


def _collect(pool, selected, store, run, execution, root, stats):
    """Only the controller mutates the store; at most one future per replica."""
    pending = deque()
    attempt_counts = {}
    for episode_id in selected:
        previous = store.attempts(episode_id)
        if any(a["record"]["status"] in {"completed", "policy_error"} for a in previous):
            continue
        attempt_counts[episode_id] = len(previous)
        if len(previous) <= execution["max_infrastructure_retries"]:
            pending.append(episode_id)
    active, idle = {}, list(pool.pairs)
    while pending or active:
        restarting = pool.shared and any(pair.needs_restart for pair in pool.pairs)
        if restarting and not active and pending:
            pool.restart_shared()
            restarting = False
        while pending and idle and not restarting:
            pair, episode_id = idle.pop(0), pending.popleft()
            active[pool.submit(pair, episode_id)] = (pair, episode_id)
        done, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in sorted(done, key=lambda f: active[f][0].index):
            pair, episode_id = active.pop(future)
            record, evidence, events = future.result()
            if pool.cancelled.is_set():
                # A fatal startup/restart cancels other in-flight RPCs. Those
                # cancellations must not consume their episode retry budgets.
                raise pool.first_error or RuntimeError("worker pool cancelled")
            record["smoke_only"] = run["smoke_only"]
            store.commit(record, evidence, events)
            attempt_counts[episode_id] += 1
            replica_stats = stats["replicas"][pair.index]
            replica_stats["attempts"] += 1
            replica_stats["episode_wall_time_s"] += record["wall_time_s"]
            replica_stats["decision_count"] += record["decision_count"]
            stats["decision_count"] += record["decision_count"]
            if record["status"] == "completed":
                stats["completed_episodes"] += 1
                replica_stats["completed_episodes"] += 1
            write_json(root / "progress.json", {"episode_id": episode_id, "replica_index": pair.index,
                       "attempts": attempt_counts[episode_id], "status": record["status"],
                       "expected_episodes": len(selected), "completed_this_invocation": stats["completed_episodes"],
                       "active_episodes": [episode for _, episode in active.values()], "queued_episodes": len(pending)})
            if record["status"] == "infrastructure_error" and attempt_counts[episode_id] <= execution["max_infrastructure_retries"]:
                pending.appendleft(episode_id)
            idle.append(pair)


def run_experiment(config=None, resources=None, *, resume=None, output=None):
    started = time.perf_counter()
    if resume:
        root = Path(resume).resolve()
        saved = read_json(root / "resolved-experiment.json")
        resolved = resolve(saved["config"], saved["resources"])
        if resolved["plan_sha256"] != saved["plan_sha256"]:
            raise ValueError("resume refused: plugins or frozen plan changed")
        run_id = root.name
    else:
        resolved = resolve(config, resources)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        root = Path(output or config.get("output", "runs")) / run_id
        root.mkdir(parents=True, exist_ok=False)
    with run_lock(root), _record_failure(root):
        if not resume:
            write_json(root / "resolved-experiment.json", resolved)
        checks = preflight(resolved, root)
        write_json(root / "preflight.json", checks)
        identity = {"source_sha256": source_identity(), "assets": resource_identity(resolved)}
        if resume:
            if read_json(root / "resource.lock.json") != identity:
                raise ValueError("resume refused: source or model resources changed")
        else:
            write_json(root / "resource.lock.json", identity)
        store = AttemptStore(root)
        with WorkerPool(resolved, root) as pool:
            first = pool.pairs[0]
            method_caps, env_caps, attestations = first.caps["method"], first.caps["environment"], first.attestations
            for role in ("method", "environment"):
                lock_path = root / ("agent.lock.json" if role == "method" else "environment.lock.json")
                if resume:
                    if read_json(lock_path) != attestations[role]:
                        raise ValueError(f"resume refused: {role} runtime or assets changed")
                else:
                    write_json(lock_path, attestations[role])
            replica_locks = [{"replica_index": pair.index, "attestations": pair.attestations} for pair in pool.pairs]
            if resume and (root / "replicas.lock.json").is_file():
                if read_json(root / "replicas.lock.json") != replica_locks:
                    raise ValueError("resume refused: replica runtime or assets changed")
            else:
                write_json(root / "replicas.lock.json", replica_locks)
            available = first.available
            requested = resolved["config"].get("episodes")
            if requested and set(requested) - set(available):
                raise ContractError("frozen requested episodes missing from worker")
            selected = [e for e in available if requested is None or e in requested]
            limit = resolved["config"].get("episode_limit")
            if limit:
                selected = selected[:limit]
            full_selection = selected
            execution = resolved["execution"]
            selected = selected[execution["shard_index"]::execution["shards"]]
            if not selected:
                raise ValueError("selected shard is empty")
            if resume:
                run = read_json(root / "run.json")
                if run["episode_ids"] != selected:
                    raise ValueError("resume refused: episode selection changed")
            else:
                run = {"run_id": run_id, "schema_version": SCHEMA_VERSION, "rollout_schema": ROLLOUT_SCHEMA,
                       "capture_schema": env_caps["capture_schema"], "benchmark": env_caps, "method": method_caps,
                       "episode_ids": selected, "full_episode_ids": full_selection, "seed": execution["seed"],
                       "smoke_only": bool(env_caps.get("smoke_only")), "clock": env_caps.get("clock", "simulation_seconds"),
                       "track": resolved["protocol"]["track"], "claim": "diagnostic", "formal_result_eligible": False,
                       "capture_plan": resolved["capture_plan"], "config_sha256": resolved["plan_sha256"],
                       "compare_key": digest({"protocol": resolved["compare_key"], "episodes": full_selection,
                                              "runtime": {k: v for k, v in attestations["environment"].items()
                                                          if k != "asset_paths"}}),
                       "shard_index": execution["shard_index"], "shards": execution["shards"],
                       "parallelism": execution["parallelism"], "scheduling": "dynamic_episode_queue"}
                run["inference"] = execution["inference"]
                run["method_worker_count"] = 1 if pool.shared else len(pool.pairs)
                write_json(root / "run.json", run)
                write_json(root / "evaluation.lock.json", resolved["metrics"])
            sessions = read_json(root / "timing.json").get("sessions", []) if (root / "timing.json").is_file() else []
            summary = _summarize(root, store, run, sessions)
            stats = {"completed_episodes": 0, "decision_count": 0,
                     "startup_wall_time_s": time.perf_counter() - started,
                     "replicas": [{"replica_index": pair.index, "attempts": 0, "completed_episodes": 0,
                                   "decision_count": 0, "episode_wall_time_s": 0.0} for pair in pool.pairs]}
            rollout_started = time.perf_counter()
            try:
                _collect(pool, selected, store, run, execution, root, stats)
            except BaseException as error:
                pool.cancelled.set()
                if pool.first_error is not None and isinstance(error, Exception):
                    raise pool.first_error
                raise
            finally:
                pool.finish()
                stats["rollout_wall_time_s"] = time.perf_counter() - rollout_started
                for replica, pair in zip(stats["replicas"], pool.pairs):
                    replica.update(pair.timing())
                    replica["busy_fraction"] = replica["episode_wall_time_s"] / stats["rollout_wall_time_s"]
                # Shared endpoint counters occur in every replica view; provide
                # a separate, unique-worker view for correct aggregation.
                stats["method_workers"] = [{"worker_index": r["method_worker_index"],
                    "startup_s": r["method_startup_s"], "current": r.get("method"),
                    "retired": [old.get("method") for old in r["retired_workers"]]}
                    for r in stats["replicas"][:1 if pool.shared else len(pool.pairs)]]
                stats["decisions_per_second"] = stats["decision_count"] / stats["rollout_wall_time_s"]
                stats["total_wall_time_s"] = time.perf_counter() - started
                sessions.append(stats)
                summary = _summarize(root, store, run, sessions)
                write_json(root / "timing.json", {"total_wall_time_s": sum(s["total_wall_time_s"] for s in sessions),
                           "inference": execution["inference"], "method_worker_count": 1 if pool.shared else len(pool.pairs),
                           "rollout_wall_time_s": sum(s["rollout_wall_time_s"] for s in sessions),
                           "parallelism": execution["parallelism"], "sessions": sessions,
                           "method_startup_s": method_caps.get("startup_s"), "environment_startup_s": env_caps.get("startup_s"),
                           "retired_workers": stats["replicas"][0]["retired_workers"],
                           "method": stats["replicas"][0].get("method"), "environment": stats["replicas"][0].get("environment")})
        from nav_eval.evaluation.runner import evaluate_run
        evaluations = []
        for metric in resolved["metrics"]:
            path = root / "metric-sets" / (metric["id"] + ".json")
            metric_config = dict(metric["config"])
            metric_config["plugin_root"] = metric["root"]
            write_json(path, metric_config)
            evaluations.append(evaluate_run(root, path))
        return {**summary, "evaluations": evaluations}
