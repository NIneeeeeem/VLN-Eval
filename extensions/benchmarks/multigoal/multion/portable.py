"""Portable MultiON task semantics for Habitat-Lab 0.2.4 and 0.3.0.

The published MultiON repositories patch Habitat-Lab itself.  This module keeps
the sequence state, FOUND semantics, and public measurements in the benchmark
plugin instead, so a normal Habitat installation is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import json
import os
from types import SimpleNamespace


def _point(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [float(item) for item in value]
    return None


def goal_points(goal):
    """Return the published target positions without changing their order."""
    objects = getattr(goal, "goal_object", None) or []
    points = [_point(getattr(item, "centroid", None)) for item in objects]
    points = [point for point in points if point is not None]
    if points:
        return points
    value = getattr(goal, "position", None)
    point = _point(value)
    if point is not None:
        return [point]
    if isinstance(value, (list, tuple)):
        return [point for item in value if (point := _point(item)) is not None]
    return []


def success_points(goal):
    """Mirror HSSD's VIEW_POINTS semantics before falling back to centroids."""
    points = []
    for item in getattr(goal, "goal_object", None) or []:
        for point in getattr(item, "navigable_points", None) or []:
            point = _point(point)
            if point is not None:
                points.append(point)
    return points or goal_points(goal)


def distance(a, b):
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def template_handle(category, handles):
    """Resolve a legacy cylinder category without relying on template indexes."""
    category = str(category).lower()
    candidates = [handle for handle in handles if category in os.path.basename(handle).lower()]
    if not candidates:
        raise RuntimeError(f"MultiON MP3D object template is missing for {category!r}")
    return sorted(candidates, key=lambda handle: (not os.path.basename(handle).lower().startswith(category), handle))[0]


def spawn_legacy_objects(sim, goals, previous_ids=()):
    """Use Habitat-Sim's public rigid-object manager instead of removed APIs."""
    native = getattr(sim, "_sim", sim)
    manager = native.get_rigid_object_manager()
    for object_id in previous_ids:
        if manager.get_object_by_id(object_id) is not None:
            manager.remove_object_by_id(object_id)
    handles = native.get_object_template_manager().get_template_handles()
    spawned = []
    for goal in goals:
        if getattr(goal, "goal_object", None):
            continue
        category, position = getattr(goal, "object_category", None), _point(getattr(goal, "position", None))
        if not category or position is None:
            continue
        obj = manager.add_object_by_template_handle(template_handle(category, handles))
        obj.translation = position
        spawned.append(obj.object_id)
    return spawned


@dataclass
class MultiONState:
    """SDK-independent MultiON sequence and path accounting."""

    goal_count: int
    optimal_prefixes: tuple[float, ...]
    index: int = 0
    progress: float = 0.0
    successful_path_prefixes: tuple[float, ...] = ()
    terminated: bool = False

    @classmethod
    def from_goals(cls, start_position, goals, distance_fn=distance):
        previous = _point(start_position)
        total = 0.0
        prefixes = []
        for goal in goals:
            points = goal_points(goal)
            if previous is not None and points:
                target = min(points, key=lambda item: distance_fn(previous, item))
                total += distance_fn(previous, target)
                previous = target
            prefixes.append(total)
        return cls(len(goals), tuple(prefixes))

    def found(self, correct, agent_path_length):
        """Apply exactly one FOUND action and return whether it was accepted."""
        if self.terminated:
            return False
        if not correct or self.index >= self.goal_count:
            self.terminated = True
            return False
        self.index += 1
        self.progress = self.index / self.goal_count if self.goal_count else 0.0
        self.successful_path_prefixes += (float(agent_path_length),)
        if self.index == self.goal_count:
            self.terminated = True
        return True

    @property
    def success(self):
        return float(self.goal_count > 0 and self.index == self.goal_count)

    def mspl(self, agent_path_length):
        if not self.success or not self.optimal_prefixes:
            return 0.0
        optimal = self.optimal_prefixes[-1]
        return optimal / max(optimal, float(agent_path_length), 1e-12)

    def ppl(self):
        if not self.index or not self.optimal_prefixes:
            return 0.0
        optimal = self.optimal_prefixes[self.index - 1]
        path = self.successful_path_prefixes[-1]
        return self.progress * optimal / max(optimal, path, 1e-12)


def register_habitat_plugins():
    """Register task components lazily in the selected Habitat interpreter."""
    from habitat.core.embodied_task import Measure, SimulatorTaskAction
    from habitat.core.registry import registry
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, NavigationTask
    from habitat.config.default_structured_configs import MeasurementConfig
    from hydra.core.config_store import ConfigStore

    @dataclass
    class MultiONSubSuccessConfig(MeasurementConfig):
        type: str = "NavEvalMultiONSubSuccess"
        success_distance: float = 0.5

    ConfigStore.instance().store(group="habitat/task/measurements", name="nav_eval_multion_sub_success",
                                 node=MultiONSubSuccessConfig,
                                 package="habitat.task.measurements.multion_sub_success")

    def config_value(config, *names, default=None):
        for name in names:
            if hasattr(config, name):
                return getattr(config, name)
            if isinstance(config, dict) and name in config:
                return config[name]
        return default

    def metric(task, name, default=0.0):
        measures = getattr(getattr(task, "measurements", None), "measures", {})
        value = measures.get(name)
        return default if value is None else value.get_metric()

    @registry.register_dataset(name="NavEvalMultiON-v1")
    class NavEvalMultiONDataset(PointNavDatasetV1):
        """Read the public MultiON JSON without importing its Habitat fork."""

        def from_json(self, json_str, scenes_dir=None):
            raw = json.loads(json_str)
            goal_sets = raw.get("goals_by_category", {})

            def decode_goal(value):
                value = dict(value)
                objects = [SimpleNamespace(**item) for item in value.get("goal_object", [])]
                position = value.pop("position", None)
                goal = NavigationGoal(position=position or [0.0, 0.0, 0.0])
                for key, item in value.items():
                    setattr(goal, key, item)
                goal.goal_object = objects
                return goal

            for number, value in enumerate(raw.get("episodes", [])):
                value = dict(value)
                goals = value.get("goals", [])
                if not goals and goal_sets:
                    categories = value.get("object_category", [])
                    if isinstance(categories, str):
                        categories = [categories]
                    key = "_".join([os.path.basename(value["scene_id"]), *categories])
                    stored = goal_sets.get(key, [])
                    goals = stored.get("goals", []) if isinstance(stored, dict) else stored
                episode = NavigationEpisode(
                    episode_id=str(value.get("episode_id", number)),
                    scene_id=value["scene_id"],
                    scene_dataset_config=value.get("scene_dataset_config", "default"),
                    additional_obj_config_paths=value.get("additional_obj_config_paths", []),
                    start_position=value.get("start_position"),
                    start_rotation=value.get("start_rotation"),
                    info=value.get("info", {}),
                    goals=[decode_goal(goal) for goal in goals],
                    start_room=value.get("start_room"),
                    shortest_paths=value.get("shortest_paths"),
                )
                episode.distractors = [decode_goal(goal) for goal in value.get("distractors", [])]
                episode.object_category = value.get("object_category")
                episode.object_index = value.get("object_index", 0)
                episode.current_goal_index = 0
                if scenes_dir and not os.path.isabs(value["scene_id"]):
                    episode.scene_id = os.path.join(scenes_dir, value["scene_id"])
                self.episodes.append(episode)

    @registry.register_task(name="NavEvalMultiON-v1")
    class NavEvalMultiONTask(NavigationTask):
        def reset(self, episode):
            self._nav_eval_object_ids = spawn_legacy_objects(
                self._sim, episode.goals, getattr(self, "_nav_eval_object_ids", ())
            )
            self.current_goal_index = 0
            self.is_found_called = False
            # HSSD's published PPL implementation explicitly uses Euclidean
            # centroid chains.  The original MP3D task's MSPL/PSPL uses a
            # geodesic chain over the inserted cylinder positions.
            use_geodesic = all(not getattr(goal, "goal_object", None) for goal in episode.goals)
            def geodesic(start, target):
                value = self._sim.geodesic_distance(start, target)
                return float(value) if value is not None and math.isfinite(float(value)) else float("inf")
            self.multion_state = MultiONState.from_goals(
                episode.start_position, episode.goals, geodesic if use_geodesic else distance
            )
            return super().reset(episode)

    @registry.register_task_action(name="NavEvalFoundAction")
    class NavEvalFoundAction(SimulatorTaskAction):
        name = "FOUND"

        def reset(self, *args, task, **kwargs):
            task.is_found_called = False

        def step(self, *args, task, **kwargs):
            task.is_found_called = True
            return self._sim.get_observations_at()

    @registry.register_measure(name="NavEvalMultiONDistance")
    class NavEvalMultiONDistance(Measure):
        cls_uuid = "multion_distance_to_currgoal"

        def __init__(self, sim, config, *args, **kwargs):
            self._sim, self._config, self._metric = sim, config, 0.0
            super().__init__(*args, **kwargs)

        def _get_uuid(self, *args, **kwargs):
            return self.cls_uuid

        def _update(self, episode, task):
            if task.current_goal_index >= len(episode.goals):
                self._metric = 0.0
                return
            position = self._sim.get_agent_state().position
            targets = success_points(episode.goals[task.current_goal_index])
            if not targets:
                self._metric = float("inf")
                return
            value = self._sim.geodesic_distance(position, targets)
            self._metric = float(value) if value is not None and math.isfinite(float(value)) else float("inf")

        def reset_metric(self, episode, task, *args, **kwargs):
            self._update(episode, task)

        def update_metric(self, episode, task, *args, **kwargs):
            self._update(episode, task)

    @registry.register_measure(name="NavEvalMultiONPathLength")
    class NavEvalMultiONPathLength(Measure):
        cls_uuid = "multion_path_length"

        def __init__(self, sim, config, *args, **kwargs):
            self._sim, self._previous, self._metric = sim, None, 0.0
            super().__init__(*args, **kwargs)

        def _get_uuid(self, *args, **kwargs):
            return self.cls_uuid

        def reset_metric(self, episode, task, *args, **kwargs):
            self._previous = _point(self._sim.get_agent_state().position)
            self._metric = 0.0

        def update_metric(self, episode, task, *args, **kwargs):
            current = _point(self._sim.get_agent_state().position)
            if self._previous is not None and current is not None:
                self._metric += distance(self._previous, current)
            self._previous = current

    @registry.register_measure(name="NavEvalMultiONSubSuccess")
    class NavEvalMultiONSubSuccess(Measure):
        cls_uuid = "multion_sub_success"

        def __init__(self, sim, config, *args, **kwargs):
            self._sim, self._config, self._metric = sim, config, 0.0
            super().__init__(*args, **kwargs)

        def _get_uuid(self, *args, **kwargs):
            return self.cls_uuid

        def reset_metric(self, episode, task, *args, **kwargs):
            self._metric = 0.0
            task.measurements.check_measure_dependencies(
                self.uuid, [NavEvalMultiONDistance.cls_uuid, NavEvalMultiONPathLength.cls_uuid]
            )

        def update_metric(self, episode, task, *args, **kwargs):
            self._metric = 0.0
            if not getattr(task, "is_found_called", False):
                return
            task.is_found_called = False
            # `configs/mopa/orasem.yaml` overrides the parent 1.5m value to 0.5m.
            threshold = float(config_value(self._config, "success_distance", "SUCCESS_DISTANCE", default=0.5))
            current = float(metric(task, NavEvalMultiONDistance.cls_uuid, float("inf")))
            accepted = task.multion_state.found(
                current < threshold, metric(task, NavEvalMultiONPathLength.cls_uuid, 0.0)
            )
            self._metric = float(accepted)
            task.current_goal_index = task.multion_state.index
            if task.multion_state.terminated:
                task._is_episode_active = False

    class _StateMeasure(Measure):
        source = "progress"
        cls_uuid = "multion_state"

        def __init__(self, sim, config, *args, **kwargs):
            self._sim, self._config, self._metric = sim, config, 0.0
            super().__init__(*args, **kwargs)

        def _get_uuid(self, *args, **kwargs):
            return self.cls_uuid

        def reset_metric(self, episode, task, *args, **kwargs):
            self.update_metric(episode, task, *args, **kwargs)

        def update_metric(self, episode, task, *args, **kwargs):
            state = task.multion_state
            path = metric(task, NavEvalMultiONPathLength.cls_uuid, 0.0)
            self._metric = float(getattr(state, self.source)(path) if self.source == "mspl" else getattr(state, self.source)()) if self.source in {"mspl", "ppl"} else float(getattr(state, self.source))

    @registry.register_measure(name="NavEvalMultiONProgress")
    class NavEvalMultiONProgress(_StateMeasure):
        cls_uuid, source = "progress", "progress"

    @registry.register_measure(name="NavEvalMultiONSuccess")
    class NavEvalMultiONSuccess(_StateMeasure):
        cls_uuid, source = "success", "success"

    @registry.register_measure(name="NavEvalMultiONMSPL")
    class NavEvalMultiONMSPL(_StateMeasure):
        cls_uuid, source = "mspl", "mspl"

    @registry.register_measure(name="NavEvalMultiONPPL")
    class NavEvalMultiONPPL(_StateMeasure):
        cls_uuid, source = "ppl", "ppl"

    return NavEvalMultiONTask
