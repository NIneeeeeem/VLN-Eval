"""Bounded inference sharing for methods declaring independent session state."""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

from nav_eval.contracts import ContractError
from nav_eval.sdk.service import WorkerService


class _BatchCalls:
    """Keep model computation and state mutation on a mutually exclusive path."""
    def __init__(self, service, options):
        self.service = service
        self.size, self.wait_s = options["max_batch_size"], options["max_wait_ms"] / 1000
        self.lock = threading.RLock()
        self.admission_lock = threading.Lock()
        self.queue = queue.Queue(maxsize=options["session_capacity"])
        self.stopped = threading.Event()
        self.stats = {"batches": 0, "requests": 0, "batch_sizes": {}, "queue_wait_s": 0.0}
        self.thread = threading.Thread(target=self._run, name="nav-microbatch", daemon=True)
        self.thread.start()

    def call(self, operation, payload):
        if operation != "act":
            with self.lock:
                return self.service.call(operation, payload)
        future = Future()
        with self.admission_lock:
            if self.stopped.is_set():
                raise RuntimeError("batch worker closed")
            try:
                self.queue.put_nowait((payload, future, time.perf_counter()))
            except queue.Full as error:
                raise RuntimeError("batch request capacity exceeded") from error
        return future.result()

    def _run(self):
        while not self.stopped.is_set():
            try:
                first = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + self.wait_s
            while len(batch) < self.size:
                try:
                    batch.append(self.queue.get(timeout=max(0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            try:
                with self.lock:
                    if self.stopped.is_set():
                        raise RuntimeError("batch worker closed")
                    now = time.perf_counter()
                    self.stats["batches"] += 1
                    self.stats["requests"] += len(batch)
                    key = str(len(batch))
                    self.stats["batch_sizes"][key] = self.stats["batch_sizes"].get(key, 0) + 1
                    self.stats["queue_wait_s"] += sum(now - queued for _, _, queued in batch)
                    replies = self.service.act_batch([payload for payload, _, _ in batch])
                    if not isinstance(replies, list) or len(replies) != len(batch):
                        raise RuntimeError("act_batch must return exactly one reply per request")
                for (_, future, _), reply in zip(batch, replies):
                    future.set_result(reply)
            except Exception as error:
                # A batch failure is an infrastructure fault, not N policy errors.
                for _, future, _ in batch:
                    future.set_exception(RuntimeError(f"batch inference failed: {type(error).__name__}: {error}"))

    def close(self):
        with self.admission_lock:
            self.stopped.set()
            while True:
                try:
                    _, future, _ = self.queue.get_nowait()
                except queue.Empty:
                    break
                future.set_exception(RuntimeError("batch worker closed"))
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("batch inference still running during close")


class SharedMethodWorker:
    """Reuse the single-session boundary; share only the method computation."""
    concurrent_requests = True

    def __init__(self, service, config):
        self.service, self.config = service, config
        self.metadata = WorkerService(service, config)
        self.capacity = config["inference"]["session_capacity"]
        self.sessions = {}
        self.registry_lock = threading.Lock()
        self.prepare_lock = threading.Lock()
        self.calls = None
        self.finished_timing = {}

    @property
    def ready(self):
        return self.metadata.ready and self.calls is not None

    def prepare(self):
        with self.prepare_lock:
            if self.ready:
                return
            supported = {"independent_greedy"} if self.config["inference"]["max_batch_size"] > 1 else {"independent_greedy", "independent_sessions"}
            if (self.config["capabilities"].get("batching") != "independent_greedy"
                    or self.service.call("describe", {}).get("batching") not in supported
                    or not callable(getattr(self.service, "act_batch", None))):
                raise ContractError("method has not implemented the requested independent inference capability")
            self.metadata.prepare()
            self.calls = _BatchCalls(self.service, self.config["inference"])

    def call(self, operation, payload):
        if operation == "prepare":
            self.prepare()
            return self.call("describe", {})
        if operation == "describe":
            return {**self.metadata.call(operation, payload), "ready": self.ready,
                    "session_capacity": self.capacity, "inference": self.config["inference"]}
        if operation == "attest":
            return self.metadata.call(operation, payload)
        if not self.ready:
            raise ContractError("worker has not completed preparation")
        if operation == "timing":
            with self.calls.lock, self.registry_lock:
                totals = dict(self.finished_timing)
                for worker, _ in self.sessions.values():
                    for key, value in worker.timing.items():
                        totals[key] = totals.get(key, 0.0) + value
                phases = self.service.timing_snapshot() if hasattr(self.service, "timing_snapshot") else {}
                return {**totals, "phases": phases, "batcher": dict(self.calls.stats)}
        session = payload.get("session_id")
        if not isinstance(session, str) or not session:
            raise ContractError("invalid session ID")
        with self.registry_lock:
            if operation == "reset":
                if session in self.sessions or len(self.sessions) >= self.capacity:
                    raise ContractError("duplicate session or shared method capacity exceeded")
                worker = WorkerService(self.calls, self.config)
                self.sessions[session] = (worker, threading.Lock())
            if session not in self.sessions:
                if operation == "close_episode":
                    return {}
                raise ContractError("inactive or stale shared session")
            worker, lock = self.sessions[session]
            if not lock.acquire(blocking=False):
                raise ContractError("concurrent requests for one session are forbidden")
        try:
            if operation == "act":
                return worker.call(operation, payload)
            with self.calls.lock:
                if operation == "reset":
                    worker.prepare()
                return worker.call(operation, payload)
        finally:
            if operation == "close_episode":
                with self.registry_lock:
                    self.sessions.pop(session)
                    for key, value in worker.timing.items():
                        self.finished_timing[key] = self.finished_timing.get(key, 0.0) + value
            lock.release()

    def close(self):
        if self.calls is not None:
            self.calls.close()
        self.metadata.close()
