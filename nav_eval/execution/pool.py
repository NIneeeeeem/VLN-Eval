"""Resident environment slots with isolated or explicitly shared model workers."""
from __future__ import annotations

import io
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from nav_eval.contracts import SCHEMA_VERSION, ContractError
from nav_eval.execution.launchers import launch, replica_plans
from nav_eval.plugins import load_entrypoint
from nav_eval.rollout.generic import _episode_attempt
from nav_eval.transport import HTTPClient


def worker_timing(client):
    timeout = getattr(client, "timeout", None)
    try:
        if timeout is not None:
            client.timeout = min(timeout, 5)
        return client.call("timing")
    except Exception as error:
        # Optional telemetry must never discard a committed episode.
        return {"unavailable": type(error).__name__, "error": str(error)}
    finally:
        if timeout is not None:
            client.timeout = timeout


def validate_live(resolved, environment, method):
    for description, role in ((environment, "benchmark"), (method, "method")):
        if description.get("schema_version") != SCHEMA_VERSION or description.get("role") != role:
            raise ContractError("live worker protocol mismatch")
        capacity = resolved["execution"]["parallelism"] if role == "method" and resolved["execution"]["inference"]["mode"] == "shared" else 1
        if description.get("ready") is not True or description.get("session_capacity") != capacity:
            raise ContractError("worker readiness or session capacity differs from plan")
    if set(method["requires_sensors"]) - set(environment["offers_sensors"]):
        raise ContractError("live environment lacks method sensors")
    expected = resolved["environment"]["capabilities"]
    if environment.get("capture_schema") != expected["capture_schema"]:
        raise ContractError("live capture schema differs from plan")
    if set(environment["accepts_actions"]) != set(expected["accepts_actions"]):
        raise ContractError("live actions differ from binding")
    geometry = environment.get("sensor_geometry", {})
    for field, wire_name in (("width", "width"), ("height", "height"), ("hfov", "hfov_deg")):
        if field in resolved["environment"]["observation"] and geometry.get(wire_name) != resolved["environment"]["observation"][field]:
            raise ContractError(f"live sensor geometry differs: {field}")


class WorkerPair:
    def __init__(self, index, resolved, root, cancelled, abort):
        self.index, self.resolved, self.root = index, resolved, Path(root)
        self.cancelled = cancelled
        self.abort = abort
        self.shared = resolved["execution"]["inference"]["mode"] == "shared"
        self.stacks = {role: ExitStack() for role in ("method", "environment")}
        self.clients, self.caps, self.attestations = {}, {}, {}
        self.available = []
        self.needs_restart = False
        self.retired = []
        control = resolved["controller"]
        self.bridge = load_entrypoint(control["entrypoint"], control["root"]) if control else None

    def _start_role(self, role):
        try:
            if self.cancelled.is_set():
                raise RuntimeError("worker startup cancelled")
            client = self.stacks[role].enter_context(launch(role, self.resolved, self.root, cancel_event=self.cancelled))
            self.clients[role] = client
            self.caps[role] = client.call("describe")
            self.attestations[role] = client.call("attest")
        except BaseException as error:
            self.abort(error)
            raise

    def start(self):
        if self.resolved["execution"]["launcher"] == "local":
            for role in self.stacks:
                self._start_role(role)
        else:
            executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nav-start")
            try:
                futures = [executor.submit(self._start_role, role) for role in self.stacks]
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                self.cancelled.set()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        self.inspect()

    def inspect(self):
        validate_live(self.resolved, self.caps["environment"], self.caps["method"])
        self.available = self.clients["environment"].call("episodes")
        if (not isinstance(self.available, list) or not self.available
                or any(not isinstance(e, str) for e in self.available)
                or len(set(self.available)) != len(self.available)):
            raise ContractError("empty, invalid or duplicate episode selection")

    def timing(self):
        if self.cancelled.is_set():
            current = {role: {"unavailable": "collection cancelled"} for role in self.clients}
        else:
            current = {role: worker_timing(client) for role, client in self.clients.items()}
        return {"replica_index": self.index, "method_startup_s": self.caps.get("method", {}).get("startup_s"),
                "method_worker_index": 0 if self.shared else self.index,
                "environment_startup_s": self.caps.get("environment", {}).get("startup_s"),
                "retired_workers": self.retired, **current}

    def close(self):
        # Attempt every cleanup even if a plugin's local close fails.
        with ExitStack() as closing:
            for stack in self.stacks.values():
                closing.callback(stack.close)

    def attempt(self, episode_id):
        if self.needs_restart and not self.shared:
            self.retired.append({role: worker_timing(client) for role, client in self.clients.items()})
            expected, available = self.attestations.copy(), self.available
            self.close()
            self.start()
            if self.attestations != expected or self.available != available:
                raise ValueError(f"replica {self.index} restart refused: runtime, assets or episodes changed")
            self.needs_restart = False
        events = io.StringIO()
        record, evidence = _episode_attempt(
            self.clients["environment"], self.clients["method"], episode_id,
            self.resolved["execution"]["seed"], self.bridge,
            self.caps["method"].get("accepts_transition", False), events)
        self.needs_restart = record["status"] == "infrastructure_error" or bool(record.get("cleanup_warnings"))
        record["replica_index"] = self.index
        return record, evidence, [json.loads(line) for line in events.getvalue().splitlines()]


class WorkerPool:
    def __init__(self, resolved, root):
        self.cancelled = threading.Event()
        self.error_lock, self.first_error = threading.Lock(), None
        self.local = resolved["execution"]["launcher"] == "local"
        self.shared = resolved["execution"]["inference"]["mode"] == "shared"
        self.shared_stack = ExitStack()
        self.root = Path(root)
        plans = replica_plans(resolved)
        self.pairs = [WorkerPair(i, plan, root if len(plans) == 1 else Path(root) / "workers" / f"{i:03d}",
                                 self.cancelled, self.abort) for i, plan in enumerate(plans)]
        self.executor = None if self.local else ThreadPoolExecutor(max_workers=len(plans) + int(self.shared), thread_name_prefix="nav-replica")

    def abort(self, error):
        with self.error_lock:
            if self.first_error is None:
                self.first_error = error
            self.cancelled.set()

    def __enter__(self):
        try:
            self._start_all()
            first = self.pairs[0]
            for pair in self.pairs[1:]:
                if pair.available != first.available or pair.attestations != first.attestations:
                    raise ValueError(f"replica {pair.index}: runtime, assets or episode selection differs from replica 0")
            return self
        except BaseException as error:
            self.cancelled.set()
            self.__exit__(None, None, None)
            if self.first_error is not None and isinstance(error, Exception):
                raise self.first_error
            raise

    def _start_shared_method(self):
        try:
            self.shared_client = self.shared_stack.enter_context(launch(
                "method", self.pairs[0].resolved, self.root / "workers" / "shared-method", cancel_event=self.cancelled))
            self.shared_caps = self.shared_client.call("describe")
            self.shared_attestation = self.shared_client.call("attest")
        except BaseException as error:
            self.abort(error)
            raise

    def _start_all(self):
        if self.local:
            self.pairs[0].start()
            return
        if self.shared:
            futures = [self.executor.submit(self._start_shared_method)]
            futures += [self.executor.submit(pair._start_role, "environment") for pair in self.pairs]
        else:
            futures = [self.executor.submit(pair.start) for pair in self.pairs]
        for future in as_completed(futures):
            future.result()
        if self.shared:
            for pair in self.pairs:
                # One ordered client per environment/session, one shared endpoint.
                pair.clients["method"] = HTTPClient(self.shared_client.url, timeout=self.shared_client.timeout,
                                                     codec=self.shared_client.codec)
                pair.caps["method"] = self.shared_caps
                pair.attestations["method"] = self.shared_attestation
                pair.inspect()

    def restart_shared(self):
        """Called only after all in-flight attempts have drained and committed."""
        expected = [(pair.attestations.copy(), pair.available) for pair in self.pairs]
        for pair in self.pairs:
            pair.retired.append({role: worker_timing(client) for role, client in pair.clients.items()})
            pair.close()
        self.shared_stack.close()
        self._start_all()
        for pair, (attestations, available) in zip(self.pairs, expected):
            if pair.attestations != attestations or pair.available != available:
                raise ValueError("shared worker restart refused: runtime, assets or episodes changed")
            pair.needs_restart = False

    def submit(self, pair, episode_id):
        if self.executor is not None:
            return self.executor.submit(pair.attempt, episode_id)
        future = Future()
        try:
            future.set_result(pair.attempt(episode_id))
        except Exception as error:
            future.set_exception(error)
        return future

    def finish(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.cancelled.set()
        self.finish()
        with ExitStack() as closing:
            closing.callback(self.shared_stack.close)
            for pair in self.pairs:
                closing.callback(pair.close)
