"""Portable MultiON HSSD binding for Habitat-Lab 0.2.4 and 0.3.0."""
from extensions.benchmarks.shared.upstream_habitat.common import UpstreamHabitatService


def _goal(episode):
    values = getattr(episode, "goals", None) or []
    goals = []
    for value in values:
        text = getattr(value, "language_instruction", None)
        if isinstance(text, str) and text:
            goals.append({"kind": "language", "value": text})
            continue
        category = getattr(value, "object_category", None)
        if not isinstance(category, str) or not category:
            raise ValueError("MultiON target does not contain a public language instruction or category")
        goals.append({"kind": "object_category", "value": category})
    return goals


class HabitatMultiONBenchmarkService(UpstreamHabitatService):
    def __init__(self, **settings):
        settings.pop("benchmark_id", None)
        settings.pop("upstream_module", None)
        self.variant = settings.pop("variant", "hssd")
        if self.variant not in {"hssd", "mp3d"}:
            raise ValueError("MultiON variant must be 'hssd' or 'mp3d'")
        super().__init__(benchmark_id="multion", upstream_modules=("extensions.benchmarks.multigoal.multion.portable",),
                         goal_projector=_goal, sequence=True, camera_tilt=False, **settings)

    def _load_config(self):
        import habitat
        from extensions.benchmarks.multigoal.multion.portable import register_habitat_plugins
        register_habitat_plugins()
        return habitat.get_config(str(self.task_config))

    def _configure_live_task(self, config):
        from nav_eval.contracts import ContractError
        super()._configure_live_task(config)
        from habitat.config import read_write
        with read_write(config):
            # PointNav's composed defaults evaluate the placeholder goal pose
            # and expose a pointgoal sensor. Only the MultiON sequence measures
            # and public goals belong to this task.
            config.habitat.task.lab_sensors.clear()
            for key in ("distance_to_goal", "spl", "distance_to_goal_reward"):
                config.habitat.task.measurements.pop(key, None)
            measures = config.habitat.task.measurements
            # Hydra merges preserve inherited insertion order. FOUND must read
            # this tick's distance/path, and scoring must read the new progress.
            config.habitat.task.measurements = {
                key: measures[key] for key in (
                    "multion_path_length", "multion_distance_to_currgoal", "multion_sub_success",
                    "progress", "success", "mspl", "ppl"
                )
            }
        actions = config.habitat.task.actions
        found = [key for key, value in actions.items() if value.type == "NavEvalFoundAction"]
        if len(found) != 1:
            raise ContractError("MultiON config requires exactly one NavEvalFoundAction")
        self.subtask_stop_action = found[0]

    def _upstream_active_goal(self):
        task = getattr(self.env, "task", None) or getattr(self.env, "_task", None)
        value = getattr(task, "current_goal_index", None)
        if type(value) is not int:
            from nav_eval.contracts import ContractError
            raise ContractError("MultiON task does not expose current_goal_index")
        return value

def create(config):
    from nav_eval.plugins import load_entrypoint
    simulator = config["simulator"]; backend = load_entrypoint(simulator["entrypoint"], simulator["root"])(simulator["settings"])
    settings = dict(config["settings"]); settings.update(config.get("observation", {}))
    return HabitatMultiONBenchmarkService(backend=backend, **settings)
