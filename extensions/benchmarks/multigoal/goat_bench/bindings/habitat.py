"""GOAT keeps one Habitat task instance and advances with ``subtask_stop``."""
from dataclasses import asdict

from extensions.benchmarks.shared.upstream_habitat.common import UpstreamHabitatService
from nav_eval.contracts import ContractError, SensorSpec
from nav_eval.tensorcode import encode_array


def _goal(episode):
    goals = []
    resolved = getattr(episode, "goals", [])
    for index, task in enumerate(getattr(episode, "tasks", [])):
        category, kind = task[0], task[1]
        if kind == "object":
            goals.append({"kind": "object_category", "value": str(category)})
        elif kind == "description":
            candidates = resolved[index] if index < len(resolved) else []
            object_id = task[2]
            matching = [goal for goal in candidates
                        if isinstance(goal, dict) and goal.get("object_id") == object_id]
            description = matching[0].get("lang_desc") if len(matching) == 1 else None
            if not isinstance(description, str) or not description:
                raise ValueError("GOAT description task lacks upstream lang_desc")
            goals.append({"kind": "language", "value": description})
        elif kind == "image":
            goals.append({"kind": "embedding_reference", "value": "goal_embedding"})
        else:
            raise ValueError(f"unsupported GOAT task type {kind!r}")
    return goals


class HabitatGOATBenchmarkService(UpstreamHabitatService):
    def _load_config(self):
        from goat_bench.config import HabitatConfigPlugin
        from habitat.config.default_structured_configs import register_hydra_plugin
        register_hydra_plugin(HabitatConfigPlugin)
        return super()._load_config()

    def __init__(self, **settings):
        settings.pop("benchmark_id", None)
        super().__init__(benchmark_id="goat_bench", upstream_modules=("goat_bench",),
                         goal_projector=_goal, sequence=True, **settings)

    def _observation(self, obs, sequence):
        import numpy as np
        result = super()._observation(obs, sequence)
        embedding = obs.get("goat_subtask_goal")
        if not isinstance(embedding, np.ndarray) or embedding.dtype != np.float32:
            raise ContractError("GOAT observation lacks GoatGoalSensor float32 embedding")
        if embedding.shape != (1024,) or not np.isfinite(embedding).all():
            raise ContractError("GoatGoalSensor must be a finite float32[1024] embedding")
        result["sensors"]["goal_embedding"] = encode_array(embedding)
        result["sensor_specs"]["goal_embedding"] = asdict(SensorSpec(
            "goal_embedding", "float32", (1024,), "embedding", "none", "estimated",
            {"upstream_sensor": "goat_subtask_goal", "representation": "upstream_goat_goal_embedding", "dimension": 1024},
        ))
        return result


def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]
    backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"]); settings.update(config.get("observation", {}))
    return HabitatGOATBenchmarkService(backend=backend, **settings)
