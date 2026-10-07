"""OVON's ``ovon`` package registers its Habitat dataset/task at import time."""
from extensions.benchmarks.shared.upstream_habitat.common import UpstreamHabitatService


def _goal(episode):
    category = getattr(episode, "object_category", None)
    if category is None and getattr(episode, "goals", None):
        category = getattr(episode.goals[0], "object_category", None)
    if not isinstance(category, str) or not category:
        raise ValueError("OVON episode lacks its free-form object category")
    return [{"kind": "language", "value": category}]


class HabitatOVONBenchmarkService(UpstreamHabitatService):
    def _load_config(self):
        from habitat.config.default_structured_configs import register_hydra_plugin
        from ovon.config import HabitatConfigPlugin
        register_hydra_plugin(HabitatConfigPlugin)
        return super()._load_config()

    def __init__(self, **settings):
        settings.pop("benchmark_id", None)
        super().__init__(benchmark_id="hm3d_ovon", upstream_modules=("ovon",),
                         goal_projector=_goal, sequence=False, **settings)


def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]
    backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"])
    settings.update(config.get("observation", {}))
    return HabitatOVONBenchmarkService(backend=backend, **settings)
