"""Local, isolated Python and single-host Docker share one worker contract."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from nav_eval.contracts import ContractError
from nav_eval.plugins import Plugin
from nav_eval.storage import write_json
from nav_eval.transport import HTTPClient, LocalClient

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def replica_plans(resolved):
    """Scientific configuration stays fixed; only deployment allocations vary."""
    return [{**resolved, **{role: {**resolved[role], "resource": allocation[role]}
                           for role in ("method", "environment")}}
            for allocation in resolved["replicas"]]


def role_config(role, resolved):
    result = deepcopy({key: value for key, value in resolved[role].items() if key not in {"resource", "requires"}})
    resource = resolved[role]["resource"]
    result["settings"] = {**result["settings"], **resource.get("settings", {})}
    if role == "method" and resolved["execution"]["inference"]["mode"] == "shared":
        result["inference"] = {**resolved["execution"]["inference"], "session_capacity": resolved["execution"]["parallelism"]}
    if role == "environment":
        if resolved["config"].get("episodes") is not None:
            result["settings"]["episode_ids"] = resolved["config"]["episodes"]
        if resolved["config"].get("episode_limit") is not None:
            result["settings"]["episode_limit"] = resolved["config"]["episode_limit"]
        if "gpu" in resource:
            # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to worker index 0.
            result["settings"]["gpu_device_id"] = 0
    return result


def _preflight_replica(resolved, gpu_devices, reservations, index):
    launcher = resolved["execution"]["launcher"]
    findings = []
    for role in ("method", "environment"):
        if role == "method" and index > 0 and resolved["execution"]["inference"]["mode"] == "shared":
            continue  # The identical shared model allocation is counted once.
        settings = role_config(role, resolved)["settings"]
        required = resolved[role]["requires"]
        resource = resolved[role]["resource"]
        for key in required.get("paths", []):
            if not settings.get(key) or not Path(settings[key]).exists():
                raise ValueError(f"{role}: missing registered asset {key}; update configs/local.json using nav_eval configure")
        if launcher == "python":
            python = resource.get("python", sys.executable)
            if not Path(python).is_file() and not shutil.which(python):
                raise ValueError(f"missing worker interpreter: {python}")
        if launcher == "local" and required.get("gpu"):
            raise ValueError("heavy plugins require isolated python or docker launchers")
        if launcher == "docker":
            image = resource.get("image", "")
            if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
                raise ValueError(f"{role}: docker image must be pinned by SHA256 digest")
        if required.get("gpu"):
            if "gpu" not in resource:
                raise ValueError(f"{role}: select a physical GPU with --gpu or --gpus")
            gpu = str(resource["gpu"])
            if not gpu or "," in gpu or isinstance(resource["gpu"], bool):
                raise ValueError(f"{role}: allocate exactly one physical GPU per worker")
            budget = resource.get("min_free_memory_mib", 0)
            if type(budget) is not int or budget < 0 or (len(resolved["replicas"]) > 1 and budget == 0):
                raise ValueError(f"{role}: parallel GPU workers require a positive min_free_memory_mib budget")
            if gpu not in gpu_devices:
                output = subprocess.run(["nvidia-smi", f"--id={gpu}",
                    "--query-gpu=uuid,name,memory.free,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15)
                if output.returncode:
                    raise ValueError(f"GPU preflight failed: {output.stderr.strip() or output.stdout.strip()}")
                fields = [field.strip() for field in output.stdout.strip().split(",")]
                if len(fields) != 4 or "\n" in output.stdout.strip():
                    raise ValueError(f"{role}: GPU allocation must identify one device")
                gpu_devices[gpu] = {"uuid": fields[0], "name": fields[1],
                                    "free_mib": int(fields[2]), "driver": fields[3]}
            device = gpu_devices[gpu]
            reservation = reservations.setdefault(device["uuid"], {"budget_mib": 0, "free_mib": device["free_mib"]})
            reservation["budget_mib"] += budget
            findings.append({"replica_index": index, "role": role, "gpu": resource["gpu"],
                             "device": device, "budget_mib": budget})
    return findings


def preflight(resolved, root):
    launcher = resolved["execution"]["launcher"]
    if resolved["claim"] == "validated":
        raise ValueError("validated result claims require upstream parity certificates; current bundles are unverified")
    devices, reservations, findings = {}, {}, []
    for index, plan in enumerate(replica_plans(resolved)):
        findings.extend(_preflight_replica(plan, devices, reservations, index))
    for gpu, reservation in reservations.items():
        if reservation["budget_mib"] > reservation["free_mib"]:
            raise ValueError(f"GPU {gpu}: combined worker budgets require {reservation['budget_mib']} MiB; "
                             f"only {reservation['free_mib']} MiB free")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 128 * 1024 * 1024:
        raise ValueError("less than 128 MiB free in output filesystem")
    if launcher == "docker":
        subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True,
                       capture_output=True, text=True, timeout=15)
    return {"checks": findings, "gpu_reservations": reservations, "launcher": launcher,
            "parallelism": len(resolved["replicas"]), "renderer": "verified during worker prepare"}


def _stop_process(process):
    # Every worker has its own session. Wrapper interpreters can exit before
    # their child, so cleanup must also work when process.poll() is not None.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def docker_command(config_path, resource, name, port, source):
    command = ["docker", "run", "--rm", "--name", name, "--init", "--user", f"{os.getuid()}:{os.getgid()}",
               "-p", f"127.0.0.1:{port}:8000", "--shm-size", resource.get("shm_size", "2g"),
               "-v", f"{source}:/opt/nav-eval:ro", "-v", f"{config_path}:/config/worker.json:ro",
               "-e", "PYTHONPATH=/opt/nav-eval", "-w", "/opt/nav-eval"]
    if "gpu" in resource:
        command += ["--gpus", f"device={resource['gpu']}"]
    for mount in resource.get("mounts", []):
        command += ["-v", f"{mount['source']}:{mount['target']}:{'ro' if mount.get('read_only', True) else 'rw'}"]
    for key, value in resource.get("env", {}).items():
        command += ["-e", f"{key}={value}"]
    command += ["--entrypoint", resource.get("container_python", "python"), resource["image"], "-u", "-B", "-m",
                "nav_eval.execution.worker", "--config", "/config/worker.json", "--host", "0.0.0.0", "--port", "8000"]
    return command


def _docker_source(config, root, role):
    """Mount code and this role's config, without exposing other roles' assets."""
    source = Path(root).resolve() / "worker-source" / role
    for package in ("nav_eval", "extensions"):
        shutil.copytree(SOURCE_ROOT / package, source / package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for label, spec in (("plugin", config), ("simulator", config.get("simulator"))):
        if spec is None:
            continue
        bundle = Path(spec["root"]).resolve()
        if bundle.is_relative_to(SOURCE_ROOT / "extensions"):
            spec["root"] = str(Path("/opt/nav-eval") / bundle.relative_to(SOURCE_ROOT))
        else:
            dependencies = Plugin(spec, bundle).code_dependencies()
            destination = source / "external" / label
            if dependencies:
                destination /= "bundle"
            shutil.copytree(bundle, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for dependency in dependencies:
                shutil.copytree(dependency, destination.parent / "shared" / dependency.name,
                                dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            initializer = bundle.parent / "shared/__init__.py"
            if dependencies and initializer.is_file():
                shutil.copy2(initializer, destination.parent / "shared/__init__.py")
            spec["root"] = str(Path("/opt/nav-eval") / destination.relative_to(source))
    return source


@contextmanager
def launch(role, resolved, root, *, cancel_event=None):
    config = role_config(role, resolved)
    resource = resolved[role]["resource"]
    mode = resolved["execution"]["launcher"]
    config_path = Path(root).resolve() / "role-configs" / f"{role}.json"
    source = _docker_source(config, root, role) if mode == "docker" else None
    write_json(config_path, config)
    timeout = float(resource.get("timeout_s", 1800))
    if mode == "local":
        from nav_eval.execution.worker import create_service
        service = create_service(config)
        try:
            service.prepare()
            yield LocalClient(service)
        finally:
            service.close()
        return
    env = {**os.environ, **{k: str(v) for k, v in resource.get("env", {}).items()}}
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE_ROOT), *resource.get("pythonpath", []), env.get("PYTHONPATH", "")])
    if resource.get("library_paths"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*resource["library_paths"], env.get("LD_LIBRARY_PATH", "")])
    if "gpu" in resource:
        env["CUDA_VISIBLE_DEVICES"] = str(resource["gpu"])
    name = "nav-eval-" + uuid.uuid4().hex
    if mode == "python":
        command = [resource.get("python", sys.executable), "-u", "-B", "-m", "nav_eval.execution.worker", "--config", str(config_path)]
    else:
        command = docker_command(config_path, resource, name, 0, source)
    log_path = Path(root) / f"{role}.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=resource.get("cwd", str(SOURCE_ROOT)), start_new_session=True)
        stopped, stop_lock = threading.Event(), threading.Lock()
        def stop():
            with stop_lock:
                try:
                    if mode == "docker":
                        subprocess.run(["docker", "stop", "-t", "10", name], capture_output=True, timeout=20)
                finally:
                    _stop_process(process)
        def cancel_watch():
            while not stopped.wait(0.1):
                if cancel_event is not None and cancel_event.is_set():
                    stop()
                    return
        watcher = threading.Thread(target=cancel_watch, daemon=True)
        watcher.start()
        ready = queue.Queue()
        def pump():
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        message = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(message, dict) and message.get("nav_eval_ready"):
                        ready.put(message)
            finally:
                ready.put(None)
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            try:
                announcement = ready.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError(f"{role} startup timeout; see {log_path}") from error
            if announcement is None:
                raise RuntimeError(f"{role} worker exited during startup; see {log_path}")
            url = announcement["url"]
            if mode == "docker":
                port = subprocess.check_output(["docker", "port", name, "8000/tcp"], text=True).strip().rsplit(":", 1)[1]
                url = "http://127.0.0.1:" + port
            client = HTTPClient(url, timeout=timeout, codec=resolved["execution"]["transport"])
            try:
                reported = client.call("prepare")
            except (ContractError, OSError, TimeoutError) as error:
                raise RuntimeError(f"{role} preparation failed; see {log_path}: {error}") from error
            if not reported.get("ready") or reported.get("plugin_identity") != config["plugin"]:
                raise ValueError("worker readiness or plugin identity differs from launch plan")
            yield client
        finally:
            stopped.set()
            try:
                stop()
            finally:
                watcher.join(timeout=25)
                thread.join(timeout=5)
                if process.stdout:
                    process.stdout.close()
