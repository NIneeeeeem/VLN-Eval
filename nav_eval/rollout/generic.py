"""Closed-loop episode execution shared by all manifest-driven experiments."""
from __future__ import annotations

import json
import time
import uuid

from nav_eval.contracts import ContractError, PolicyViolation, validate_action, validate_observation

GENERIC_ROLLOUT_SCHEMA = "nav-eval-rollout/0.3"
MAX_DECISIONS_PER_EPISODE = 1000  # outer watchdog; the task budget is owned by the benchmark


def _episode_attempt(benchmark, method, episode_id, seed, bridge, wants_transition, events):
    """Run one episode attempt and return its record and evidence."""
    session_id = uuid.uuid4().hex
    start = time.perf_counter()
    decision_count = 0
    total_decision_s = 0.0
    environment_step_s = 0.0
    reset_s = 0.0
    failure = None  # (status, attribution, error_type, message)
    evidence = None
    termination = None
    ended = False
    try:
        initial = benchmark.call("reset", session_id=session_id, episode_id=episode_id, seed=seed)
        method.call("reset", session_id=session_id, context=initial["context"])
        obs = initial["observation"]
        reset_s = time.perf_counter() - start
        validate_observation(obs)
        if obs["episode_id"] != episode_id:
            raise ContractError("initial observation belongs to a different episode")
        for _ in range(MAX_DECISIONS_PER_EPISODE):
            before = time.perf_counter()
            batch = method.call("act", session_id=session_id, observation=obs)
            decision_s = time.perf_counter() - before
            total_decision_s += decision_s
            decision_count += 1
            if batch["episode_id"] != episode_id or batch["observation_sequence"] != obs["sequence"]:
                raise ContractError("stale or misrouted action batch")
            if not isinstance(batch["actions"], list) or not 1 <= len(batch["actions"]) <= 16:
                raise PolicyViolation("invalid action batch length")
            for index, raw in enumerate(batch["actions"]):
                try:
                    proposed = validate_action(raw)
                except ContractError as error:
                    raise PolicyViolation(f"illegal action from method: {error}") from error
                executed = bridge(proposed) if bridge else proposed
                previous_sequence = obs["sequence"]
                before_step = time.perf_counter()
                step = benchmark.call("step", session_id=session_id, expected_sequence=previous_sequence,
                                      action=executed.to_dict())
                step_s = time.perf_counter() - before_step
                environment_step_s += step_s
                obs = step["observation"]
                validate_observation(obs)
                if obs["episode_id"] != episode_id or obs["sequence"] != previous_sequence + 1:
                    raise ContractError("invalid environment step sequence")
                events.write(json.dumps({
                    "episode_id": episode_id, "session_id": session_id,
                    "decision_index": decision_count, "observation_sequence": previous_sequence,
                    "policy_source": batch.get("policy_source", "method"),
                    "proposed_action": proposed.to_dict(), "executed_action": executed.to_dict(),
                    "agent_decision_wall_time_s": decision_s if index == 0 else 0.0,
                    "inference_rpc_time_s": decision_s if index == 0 else 0.0,
                    "environment_step_wall_time_s": step_s,
                    "sim_time_s": obs["sim_time_s"],
                    "control_tick": obs.get("control_tick"),
                }, allow_nan=False) + "\n")
                if wants_transition:
                    method.call("observe_transition", session_id=session_id, transition={
                        "episode_id": episode_id, "sequence": obs["sequence"],
                        "executed_action": executed.to_dict(), "observation": obs,
                        "terminated": step["terminated"], "truncated": step["truncated"],
                    })
                if step["terminated"]:
                    termination = "stop"
                elif step["truncated"]:
                    termination = "budget"
                ended = step["terminated"] or step["truncated"]
                if ended:
                    break
            if ended:
                break
        if not ended:
            failure = ("infrastructure_error", "watchdog_timeout",
                       "TimeoutError", "outer decision watchdog expired")
        else:
            finished = benchmark.call("finish", session_id=session_id)
            record, evidence = finished["record"], finished["evidence"]
            if record["episode_id"] != episode_id or "metrics" in record:
                raise ContractError("finish must return execution facts without metric scores")
    except PolicyViolation as error:
        failure = ("policy_error", "method_policy", type(error).__name__, str(error))
    except TimeoutError as error:
        failure = ("infrastructure_error", "timeout", type(error).__name__, str(error))
    except ContractError as error:
        # A contract failure outside method action validation is a worker/protocol fault.
        failure = ("infrastructure_error", "worker_contract", type(error).__name__, str(error))
    except Exception as error:  # noqa: BLE001 - attribution must survive arbitrary worker faults
        failure = ("infrastructure_error", "unresolved", type(error).__name__, str(error))

    # Cleanup always runs for both workers; its failures are warnings and never
    # demote an already completed execution record.
    cleanup_warnings = []
    for name, client in (("method", method), ("benchmark", benchmark)):
        try:
            client.call("close_episode", session_id=session_id)
        except Exception as error:  # noqa: BLE001
            cleanup_warnings.append(f"{name}: {type(error).__name__}: {error}")

    if failure is not None:
        status, attribution, error_type, message = failure
        record = {"episode_id": episode_id, "status": status,
                  "failure_attribution": attribution, "error_type": error_type,
                  "error": message}
    else:
        record = dict(record)
        record["status"] = "completed"
        record["termination"] = termination or record.get("termination")
    record.update(attempt_id=session_id, wall_time_s=time.perf_counter() - start,
                  decision_count=decision_count, agent_decision_wall_time_s=total_decision_s,
                  inference_rpc_time_s=total_decision_s, environment_step_wall_time_s=environment_step_s,
                  reset_wall_time_s=reset_s,
                  smoke_only=False)
    if cleanup_warnings:
        record["cleanup_warnings"] = cleanup_warnings
    return record, evidence
