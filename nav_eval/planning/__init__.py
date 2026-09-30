"""Resolve an experiment once; launchers never invent scientific defaults."""
from __future__ import annotations

from copy import deepcopy
import math

from nav_eval.plugins import Registry, digest

SCHEMA = "nav-eval-experiment/1"


def resolve(config, resources=None, registry=None):
    config, resources = deepcopy(config), deepcopy(resources or {})
    registry = registry or Registry(config.get("plugin_dirs", []))
    unknown = set(config) - {"schema_version", "benchmark", "simulator", "method", "track",
                             "benchmark_settings", "method_settings", "observation", "controller",
                             "metrics", "seed", "episodes", "episode_limit", "plugin_dirs",
                             "launcher", "transport", "output", "max_infrastructure_retries",
                             "shards", "shard_index", "claim", "parallelism", "inference"}
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
    if config.get("schema_version", SCHEMA) != SCHEMA:
        raise ValueError("unsupported experiment schema")
    track = config.get("track")
    if track not in {"native", "standardized"}:
        raise ValueError("track must explicitly be native or standardized")
    benchmark = registry.get("benchmark", config["benchmark"])
    simulator = registry.get("simulator", config["simulator"])
    method = registry.get("method", config["method"])
    bm, sm, mm = benchmark.manifest, simulator.manifest, method.manifest
    binding = deepcopy(bm.get("bindings", {}).get(sm["id"]))
    if not binding:
        raise ValueError(f"missing benchmark binding: {bm['id']} / {sm['id']}")
    if not mm.get("entrypoint"):
        raise ValueError(f"unbound method implementation: {mm['id']}")
    observation = deepcopy(bm.get("observation", {}))
    if track == "native":
        observation.update(mm.get("observation", {}))
    observation.update(config.get("observation", {}))
    offered = set(sm.get("capabilities", {}).get("offers_sensors", []))
    required = set(mm.get("capabilities", {}).get("requires_sensors", []))
    sensors = observation.get("sensors", sorted(required))
    if required - set(sensors) or set(sensors) - offered:
        raise ValueError("incompatible sensors in method, observation or simulator")
    observation["sensors"] = sensors
    supported_geometry = binding.get("fixed_geometry")
    if supported_geometry and any(observation.get(k) != v for k, v in supported_geometry.items()):
        raise ValueError("observation geometry unsupported by the selected binding")
    actions = set(mm["capabilities"]["emits_actions"])
    controller = None
    if config.get("controller"):
        plugin = registry.get("controller", config["controller"])
        capabilities = plugin.manifest["capabilities"]
        if actions - set(capabilities["accepts_actions"]):
            raise ValueError("controller does not handle all method actions")
        actions = set(capabilities["emits_actions"])
        controller = {**plugin.identity(), "entrypoint": plugin.manifest["entrypoint"],
                      "root": str(plugin.root)}
    if actions - set(binding["accepts_actions"]):
        raise ValueError("incompatible actions; select an explicit controller")
    for key, value in mm.get("action_semantics", {}).items():
        if key in binding.get("action_semantics", {}) and value != binding["action_semantics"][key]:
            if track == "native" and value in binding.get("supported_action_semantics", {}).get(key, []):
                # Freeze the selected physical action in both the protocol hash
                # and the environment specification. Never split one turn into
                # multiple steps, which would change history and the step budget.
                binding["action_semantics"][key] = value
            else:
                raise ValueError(f"incompatible action semantics: {key}")
    metrics = []
    for metric_id in config.get("metrics", bm.get("metrics", [])):
        metric = registry.get("metric", metric_id)
        ms = metric.manifest["metric_set"]
        if ms["evidence_schema"] != binding["capture_schema"]:
            raise ValueError(f"incompatible evidence schema for metric: {metric_id}")
        missing = set(metric.manifest.get("requires_evidence", [])) - set(binding.get("provides_evidence", []))
        if missing:
            raise ValueError(f"binding cannot capture required evidence: {sorted(missing)}")
        metrics.append({**metric.identity(), "config": ms, "root": str(metric.root)})
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for key, default, minimum in (("shards", 1, 1), ("shard_index", 0, 0), ("parallelism", 1, 1),
                                   ("max_infrastructure_retries", 0, 0)):
        value = config.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {key}")
    shards, shard_index = config.get("shards", 1), config.get("shard_index", 0)
    if shard_index >= shards:
        raise ValueError("shard_index must be smaller than shards")
    if config.get("episode_limit") is not None and (type(config["episode_limit"]) is not int or config["episode_limit"] <= 0):
        raise ValueError("episode_limit must be positive or omitted")
    episodes = config.get("episodes")
    if episodes is not None and (not isinstance(episodes, list) or not episodes
                                or any(not isinstance(e, str) for e in episodes)
                                or len(set(episodes)) != len(episodes)):
        raise ValueError("episodes must be nonempty unique string IDs")
    launcher = config.get("launcher", "python")
    transport = config.get("transport", "json")
    if launcher not in {"local", "python", "docker"} or transport not in {"json", "binary"}:
        raise ValueError("unknown launcher or transport")
    parallelism = config.get("parallelism", 1)
    if launcher == "local" and parallelism != 1:
        raise ValueError("parallel inference requires isolated python or docker workers")
    inference = config.get("inference", {"mode": "replicas"})
    if not isinstance(inference, dict) or set(inference) - {"mode", "max_batch_size", "max_wait_ms"}:
        raise ValueError("invalid inference configuration")
    mode = inference.get("mode", "replicas")
    if mode not in {"replicas", "shared"}:
        raise ValueError("inference mode must be replicas or shared")
    if mode == "shared":
        if launcher == "local" or parallelism < 2:
            raise ValueError("shared inference requires isolated workers and parallelism >= 2")
        if mm["capabilities"].get("batching") != "independent_greedy":
            raise ValueError(f"{mm['id']}: shared inference requires independent_greedy batching capability")
        # Keep singleton numerical behavior until the user explicitly chooses a
        # larger experimental batch and validates its closed-loop equivalence.
        size, delay = inference.get("max_batch_size", 1), inference.get("max_wait_ms", 5)
        if type(size) is not int or not 1 <= size <= parallelism:
            raise ValueError("max_batch_size must be between 1 and parallelism")
        if size > 1:
            batching_settings = {**mm.get("defaults", {}), **config.get("method_settings", {})}
            for key, expected in mm["capabilities"].get("batching_requires", {}).items():
                if batching_settings.get(key) != expected:
                    raise ValueError(f"{mm['id']}: batch size > 1 requires method_settings.{key}={expected!r}")
        if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 1000:
            raise ValueError("max_wait_ms must be finite and between 0 and 1000")
        inference = {"mode": mode, "max_batch_size": size, "max_wait_ms": delay}
    else:
        if set(inference) - {"mode"}:
            raise ValueError("batch parameters require shared inference")
        inference = {"mode": mode}
    benchmark_settings = {**bm.get("defaults", {}), **config.get("benchmark_settings", {}),
                          **binding.get("settings", {})}
    protocol = {"track": track, "benchmark": benchmark.identity(),
                "simulator": simulator.identity(), "binding": binding,
                "benchmark_settings": benchmark_settings, "observation": observation,
                "controller": None if controller is None else {k: v for k, v in controller.items() if k != "root"},
                "metrics": [{k: v for k, v in metric.items() if k != "root"} for metric in metrics],
                "seed": seed, "requested_episodes": episodes, "episode_limit": config.get("episode_limit")}
    method_settings = {**mm.get("defaults", {}), **config.get("method_settings", {})}
    for settings, manifest in ((method_settings, mm), (benchmark_settings, bm)):
        invalid = set(settings) - set(manifest.get("settings", settings))
        if invalid:
            raise ValueError(f"unknown settings for {manifest['id']}: {sorted(invalid)}")
    def role_resource(role, manifest):
        base, override = resources.get("runtimes", {}).get(manifest["id"], {}), resources.get(role, {})
        result = {**base, **override, "settings": {**base.get("settings", {}), **override.get("settings", {})}}
        allowed = set(manifest.get("requires", {}).get("paths", [])) | set(manifest.get("resource_settings", []))
        invalid = set(result["settings"]) - allowed
        if invalid:
            raise ValueError(f"{role}: unknown or scientific resource settings: {sorted(invalid)}; use experiment settings")
        return result
    method_role = {"role": "method", "plugin": method.identity(), "root": str(method.root),
                   "entrypoint": mm["entrypoint"], "capabilities": mm["capabilities"],
                   "settings": method_settings, "observation": observation,
                   "resource": role_resource("method", mm), "requires": mm.get("requires", {})}
    environment = {"role": "benchmark", "plugin": benchmark.identity(), "root": str(benchmark.root),
                   "entrypoint": binding["entrypoint"], "settings": benchmark_settings,
                   "simulator": {**simulator.identity(), "entrypoint": sm["entrypoint"],
                                 "root": str(simulator.root), "settings": sm.get("defaults", {})},
                   "observation": observation, "capabilities": binding,
                   "resource": role_resource("environment", sm), "requires": sm.get("requires", {})}
    replicas = resources.get("replicas")
    if replicas is None:
        if parallelism > 1:
            raise ValueError("parallelism > 1 requires explicit resources.replicas allocations")
        replicas = [{}]
    if not isinstance(replicas, list) or len(replicas) != parallelism:
        raise ValueError("resources.replicas length must equal parallelism")
    allocations = []
    deployment_fields = {"gpu", "min_free_memory_mib", "python", "pythonpath", "library_paths", "cwd",
                         "env", "timeout_s", "image", "container_python", "mounts", "shm_size"}
    for index, replica in enumerate(replicas):
        if not isinstance(replica, dict) or set(replica) - {"method", "environment"}:
            raise ValueError(f"replica {index}: expected method/environment resource overrides")
        allocation = {}
        for role, spec in (("method", method_role), ("environment", environment)):
            override = replica.get(role, {})
            if not isinstance(override, dict) or set(override) - deployment_fields:
                raise ValueError(f"replica {index} {role}: only deployment resources may vary")
            if parallelism > 1 and spec["requires"].get("gpu") and "gpu" not in override:
                raise ValueError(f"replica {index} {role}: explicitly allocate a GPU")
            allocation[role] = {**spec["resource"], **override,
                                "env": {**spec["resource"].get("env", {}), **override.get("env", {})}}
        allocations.append(allocation)
    if mode == "shared" and any(a["method"] != allocations[0]["method"] for a in allocations[1:]):
        raise ValueError("shared inference requires identical method deployment resources across replicas")
    claim = config.get("claim", "diagnostic")
    if claim not in {"diagnostic", "validated"}:
        raise ValueError("claim must be diagnostic or validated")
    # A bundle declaration is not a parity certificate. Certificates are checked by the launcher.
    resolved = {"schema_version": SCHEMA, "config": config, "resources": resources,
                "protocol": protocol, "compare_key": digest(protocol), "method": method_role,
                "environment": environment, "replicas": allocations, "controller": controller, "metrics": metrics,
                "capture_plan": {"schema": binding["capture_schema"], "fields": binding.get("provides_evidence", [])},
                "execution": {"launcher": launcher, "transport": transport, "seed": seed,
                              "shards": shards, "shard_index": shard_index, "parallelism": parallelism,
                              "inference": inference,
                              "max_infrastructure_retries": config.get("max_infrastructure_retries", 0)},
                "claim": claim, "status": "resolved", "runtime_verified": False,
                "notes": ["Manifest compatibility does not establish upstream parity."]}
    resolved["plan_sha256"] = digest(resolved)
    return resolved
