"""Generic worker entrypoint; the plugin owns heavy imports and thread affinity."""
from __future__ import annotations

import argparse
import json
import threading

from nav_eval.plugins import load_entrypoint
from nav_eval.sdk.service import WorkerService
from nav_eval.storage import read_json
from nav_eval.transport import make_server


def create_service(config):
    factory = load_entrypoint(config["entrypoint"], config["root"])
    service = factory(config)
    if config.get("inference", {}).get("mode") == "shared":
        from nav_eval.sdk.batching import SharedMethodWorker
        return SharedMethodWorker(service, config)
    return WorkerService(service, config)


def serve(config, host="127.0.0.1", port=0):
    service = create_service(config)
    server = make_server(service, host, port)
    address, actual_port = server.server_address
    print(json.dumps({"nav_eval_ready": True, "url": f"http://{address}:{actual_port}"}), flush=True)
    try:
        if hasattr(service.service, "run_main"):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            service.service.run_main()
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", required=True)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=0)
    args = cli.parse_args()
    serve(read_json(args.config), args.host, args.port)


if __name__ == "__main__":
    main()
