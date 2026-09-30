"""Versioned local JSON/binary worker transport; not for public network exposure."""
from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from nav_eval.contracts import SCHEMA_VERSION, ContractError, PolicyViolation

MAX_BODY_BYTES = 8 * 1024 * 1024


class LocalClient:
    def __init__(self, service):
        self.service = service

    def call(self, operation, **payload):
        return self.service.call(operation, payload)


class HTTPClient:
    def __init__(self, url, timeout=10.0, codec="json"):
        if not url.startswith("http://"):
            raise ValueError("development transport expects http://; deploy on a private container network")
        self.url = url.rstrip("/")
        self.timeout = timeout
        if codec not in {"json", "binary"}:
            raise ValueError("unknown transport codec")
        self.codec = codec
        self.negotiated = codec == "json"
        self.opener = build_opener(ProxyHandler({}))

    def call(self, operation, **payload):
        from nav_eval.sdk import wire
        if self.codec == "binary" and not self.negotiated:
            self.negotiated = True
            self.codec = "json"
            try:
                capabilities = self.call("describe")
            finally:
                self.codec = "binary"
            if "binary" not in capabilities.get("transport_codecs", []):
                self.negotiated = False
                raise ContractError("worker does not support requested binary transport")
        envelope = {"schema_version": SCHEMA_VERSION, "operation": operation, "payload": payload}
        binary = self.codec == "binary"
        body = wire.encode(envelope) if binary else json.dumps(envelope, allow_nan=False).encode()
        limit = wire.MAX_BINARY_BYTES if binary else MAX_BODY_BYTES
        if len(body) > limit:
            raise ContractError("request exceeds development transport size limit")
        request = Request(self.url + "/rpc", body, {"Content-Type": wire.CONTENT_TYPE if binary else "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(limit + 1)
                response_binary = response.headers.get("Content-Type") == wire.CONTENT_TYPE
        except HTTPError as error:
            detail = error.read(MAX_BODY_BYTES + 1).decode(errors="replace")
            kind = None
            try:
                kind = json.loads(detail).get("error_kind")
            except ValueError:
                pass
            if kind == "policy_violation":
                raise PolicyViolation(f"worker rejected {operation}: {detail[:4096]}") from error
            raise ContractError(f"worker rejected {operation}: {detail[:4096]}") from error
        if len(raw) > limit:
            raise ContractError("oversize response")
        result = wire.decode(raw) if response_binary else json.loads(raw)
        if result.get("schema_version") != SCHEMA_VERSION or not result.get("ok"):
            raise ContractError("invalid worker response")
        return result["result"]

    def wait_ready(self, seconds=15):
        deadline = time.monotonic() + seconds
        while True:
            try:
                return self.call("describe")
            except (URLError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"worker not ready after {seconds}s: {self.url}")
                time.sleep(0.1)


def make_server(service, host="127.0.0.1", port=0):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, binary=False):
            from nav_eval.sdk import wire
            data = wire.encode(body) if binary else json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", wire.CONTENT_TYPE if binary else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                ready = getattr(service, "ready", False)
                self.reply(200 if ready else 503, {"alive": True, "ready": ready, "schema_version": SCHEMA_VERSION})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/rpc":
                self.reply(404, {"error": "not found"})
                return
            try:
                from nav_eval.sdk import wire
                binary = self.headers.get("Content-Type") == wire.CONTENT_TYPE
                limit = wire.MAX_BINARY_BYTES if binary else MAX_BODY_BYTES
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= limit:
                    raise ContractError("invalid body size")
                data = self.rfile.read(length)
                request = wire.decode(data) if binary else json.loads(data)
                if request.get("schema_version") != SCHEMA_VERSION:
                    raise ContractError("incompatible schema version")
                if set(request) != {"schema_version", "operation", "payload"}:
                    raise ContractError("invalid request envelope")
                with nullcontext() if getattr(service, "concurrent_requests", False) else lock:
                    result = service.call(request["operation"], request["payload"])
                if request["operation"] == "describe":
                    result = {**result, "transport_codecs": ["json", "binary"]}
                self.reply(200, {"ok": True, "schema_version": SCHEMA_VERSION, "result": result}, binary=binary)
            except PolicyViolation as error:
                self.reply(400, {"ok": False, "schema_version": SCHEMA_VERSION,
                                 "error": str(error), "error_kind": "policy_violation"})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"ok": False, "schema_version": SCHEMA_VERSION, "error": str(error)})
            except Exception:
                import traceback
                traceback.print_exc()
                self.reply(500, {"ok": False, "schema_version": SCHEMA_VERSION, "error": "worker internal error"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
