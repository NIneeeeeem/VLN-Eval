"""Keep native simulator operations and teardown on one owning thread."""
from concurrent.futures import Future
from queue import Queue
from threading import Lock, Thread
from time import monotonic

from nav_eval.contracts import ContractError


class SimulatorDispatch:
    def __init__(self, handler, name="habitat-sim-dispatch", close_timeout=30):
        self.handler, self.close_timeout = handler, close_timeout
        self._queue, self._lock, self._shutdown = Queue(), Lock(), None
        self.thread = Thread(target=self._serve, daemon=True, name=name)
        self.thread.start()

    def _serve(self):
        while True:
            operation, payload, result = self._queue.get()
            try:
                result.set_result(self.handler(operation, payload))
            except BaseException as error:
                result.set_exception(error)
            if operation == "shutdown":
                return

    def call(self, operation, payload):
        if operation == "shutdown":
            self.close()
            return {}
        with self._lock:
            if self._shutdown is not None:
                raise ContractError("benchmark service is closed")
            result = Future()
            self._queue.put((operation, payload, result))
        return result.result()

    def close(self):
        with self._lock:
            if self._shutdown is None:
                self._shutdown = Future()
                self._queue.put(("shutdown", {}, self._shutdown))
            result = self._shutdown
        deadline = monotonic() + self.close_timeout
        try:
            result.result(timeout=self.close_timeout)
        finally:
            self.thread.join(timeout=max(0, deadline - monotonic()))
            if self.thread.is_alive():
                raise ContractError("simulator thread did not stop before close timeout")
