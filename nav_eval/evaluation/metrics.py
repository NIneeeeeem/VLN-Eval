"""Metric implementations used by the shipped benchmark extensions."""
from __future__ import annotations

import math
from typing import Any

from nav_eval.contracts import finite_number


class MissingEvidence(ValueError):
    pass


def field_at(evidence: dict, path: str) -> Any:
    value = evidence
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value or value[part] is None:
            raise MissingEvidence(f"capture lacks {path}; rerun with an appropriate capture profile")
        value = value[part]
    if path == "trajectory.goal_distances_m":
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in value):
            raise MissingEvidence("trajectory includes unavailable geodesic distances")
        if any(evidence.get("trajectory", {}).get("goal_distance_fallbacks", [])):
            raise MissingEvidence("fallback distances cannot be scored as geodesic evidence")
    return value


# --- R2R-CE metrics (evidence schema habitat-r2r-evidence/1) ----------------
# Definitions follow the official R2R-CE protocol: an episode is successful when
# the method STOPs within the success radius; NE is the final geodesic distance
# to the goal on the recorded NavMesh; SPL discounts success by the ratio of
# geodesic distance to travelled path length. All inputs come from captured
# evaluator evidence, never from policy-visible data.


def _radius_from(parameters, evidence):
    radius = parameters.get("success_radius_m")
    if radius is None:
        radius = field_at(evidence, "reference.success_radius_m")
    radius = finite_number(radius, "success_radius_m")
    if radius <= 0:
        raise ValueError("success_radius_m must be positive")
    return radius


class R2RCESuccess:
    name = "r2r_ce.success"
    required_fields = ("trajectory.goal_distances_m", "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        radius = _radius_from(parameters, evidence)
        require_stop = parameters.get("require_stop", True)
        if not isinstance(require_stop, bool):
            raise ValueError("require_stop must be boolean")
        distances = field_at(evidence, "trajectory.goal_distances_m")
        if not distances:
            raise MissingEvidence("trajectory has no distance samples")
        stopped = record.get("termination") == "stop"
        return float((not require_stop or stopped) and distances[-1] < radius)


class R2RCENE:
    name = "r2r_ce.ne_m"
    required_fields = ("trajectory.goal_distances_m",)

    def compute(self, record, evidence, parameters):
        distances = field_at(evidence, "trajectory.goal_distances_m")
        if not distances:
            raise MissingEvidence("trajectory has no distance samples")
        return finite_number(distances[-1], "final_goal_distance")


class R2RCEOracleSuccess:
    name = "r2r_ce.oracle_success"
    required_fields = ("trajectory.goal_distances_m", "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        radius = _radius_from(parameters, evidence)
        distances = field_at(evidence, "trajectory.goal_distances_m")
        if not distances:
            raise MissingEvidence("trajectory has no distance samples")
        return float(min(distances) < radius)


class R2RCESPL:
    name = "r2r_ce.spl"
    required_fields = ("trajectory.goal_distances_m", "trajectory.positions_xyz_m",
                       "reference.geodesic_start_to_goal_m", "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        success = R2RCESuccess().compute(record, evidence, parameters)
        if success <= 0:
            return 0.0
        positions = field_at(evidence, "trajectory.positions_xyz_m")
        if len(positions) < 2:
            raise ValueError("trajectory too short for path length")
        for position in positions:
            if len(position) != 3:
                raise ValueError("expected xyz positions")
            for value in position:
                finite_number(value, "position")
        path_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
        geodesic = finite_number(field_at(evidence, "reference.geodesic_start_to_goal_m"),
                                 "geodesic_start_to_goal_m")
        if geodesic <= 0:
            raise ValueError("geodesic_start_to_goal_m must be positive")
        return geodesic / max(path_length, geodesic)


# --- ObjectNav metrics (evidence schema habitat-objectnav-evidence/1) ----------
# Definitions follow the official habitat ObjectNav protocol: SR requires a
# STOP with the final view-point geodesic distance below the success radius;
# SPL discounts success by the ratio of start distance to travelled path
# length; SoftSPL replaces the boolean success with max(0, 1 - d_final/d_start)
# exactly as habitat's SoftSPL measure. All inputs come from captured
# evaluator evidence, never from policy-visible data.


class ObjectNavSuccess(R2RCESuccess):
    name = "objectnav.success"


class ObjectNavNE(R2RCENE):
    name = "objectnav.ne_m"


class ObjectNavOracleSuccess(R2RCEOracleSuccess):
    name = "objectnav.oracle_success"


class ObjectNavSPL(R2RCESPL):
    name = "objectnav.spl"


class ObjectNavSoftSPL:
    name = "objectnav.soft_spl"
    required_fields = ("trajectory.goal_distances_m", "trajectory.positions_xyz_m",
                       "reference.geodesic_start_to_goal_m")

    def compute(self, record, evidence, parameters):
        distances = field_at(evidence, "trajectory.goal_distances_m")
        if not distances:
            raise MissingEvidence("trajectory has no distance samples")
        geodesic = finite_number(field_at(evidence, "reference.geodesic_start_to_goal_m"),
                                 "geodesic_start_to_goal_m")
        if geodesic <= 0:
            raise ValueError("geodesic_start_to_goal_m must be positive")
        positions = field_at(evidence, "trajectory.positions_xyz_m")
        path_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
        soft_success = max(0.0, 1.0 - distances[-1] / geodesic)
        return soft_success * (geodesic / max(geodesic, path_length))


# --- SocialNav metrics (evidence schema habitat-socialnav-evidence/1) ----------
# Position-based protocol on the captured robot-to-human DistToGoal evidence:
# SR requires a STOP with the final distance below the success radius, SPL and
# SoftSPL follow the same habitat formulas as ObjectNav. The upstream
# nav_seek_success measure (which additionally requires facing the human) is
# recorded in the evidence parity layer, not recomputed here.


class SocialNavSuccess(ObjectNavSuccess):
    name = "socialnav.success"


class SocialNavNE(ObjectNavNE):
    name = "socialnav.ne_m"


class SocialNavOracleSuccess(ObjectNavOracleSuccess):
    name = "socialnav.oracle_success"


class SocialNavSPL(ObjectNavSPL):
    name = "socialnav.spl"


class SocialNavSoftSPL(ObjectNavSoftSPL):
    name = "socialnav.soft_spl"


# --- VLNVerse metrics (evidence schema vlnverse-evidence/1) ------------------
# Formulas replicate the challenge's VLNPEMetric (xy coordinates only):
# NE = euclidean xy distance to reference_path[-1]; SR = NE < success_distance
# at finish; OSR = min NE over trajectory; SPL = SR * geodesic / max(geodesic,
# path_length); nDTW accumulates exp(-d^2 / (2 r^2)) against the nearest
# reference point, normalised by trajectory length.


def _xy_positions(evidence):
    path = "trajectory.metric_positions_xyz_m" if evidence.get("schema") == "vlnverse-evidence/2" else "trajectory.positions_xyz_m"
    positions = field_at(evidence, path)
    if not positions:
        raise MissingEvidence("trajectory has no samples")
    return [(finite_number(p[0], "x"), finite_number(p[1], "y")) for p in positions]


def _goal_xy(evidence):
    goal = field_at(evidence, "reference.goal_xyz_m")
    return goal[0], goal[1]


class VLNVerseNE:
    name = "vlnverse.ne_m"
    required_fields = ("trajectory.positions_xyz_m", "reference.goal_xyz_m")

    def compute(self, record, evidence, parameters):
        positions = _xy_positions(evidence)
        goal = _goal_xy(evidence)
        return math.dist(positions[-1], goal)


class VLNVerseSuccess:
    name = "vlnverse.success"
    required_fields = ("trajectory.positions_xyz_m", "reference.goal_xyz_m",
                       "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        radius = _radius_from(parameters, evidence)
        ne = VLNVerseNE().compute(record, evidence, parameters)
        return float(ne < radius)


class VLNVerseOSR:
    name = "vlnverse.osr"
    required_fields = ("trajectory.positions_xyz_m", "reference.goal_xyz_m",
                       "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        radius = _radius_from(parameters, evidence)
        goal = _goal_xy(evidence)
        return float(min(math.dist(p, goal) for p in _xy_positions(evidence)) < radius)


class VLNVerseSPL:
    name = "vlnverse.spl"
    required_fields = ("trajectory.positions_xyz_m", "reference.goal_xyz_m",
                       "reference.geodesic_start_to_goal_m", "reference.success_radius_m", "trajectory.path_length_m")

    def compute(self, record, evidence, parameters):
        success = VLNVerseSuccess().compute(record, evidence, parameters)
        if success <= 0:
            return 0.0
        path_length = finite_number(field_at(evidence, "trajectory.path_length_m"), "path_length_m")
        if path_length < 0:
            raise ValueError("path_length_m must be nonnegative")
        if path_length == 0:
            return 0.0
        geodesic = finite_number(field_at(evidence, "reference.geodesic_start_to_goal_m"),
                                 "geodesic_start_to_goal_m")
        if geodesic <= 0:
            raise ValueError("geodesic_start_to_goal_m must be positive")
        return geodesic / max(path_length, geodesic)


class VLNVerseNDTW:
    name = "vlnverse.ndtw"
    required_fields = ("trajectory.positions_xyz_m", "reference.reference_path_xyz_m",
                       "reference.success_radius_m")

    def compute(self, record, evidence, parameters):
        radius = _radius_from(parameters, evidence)
        trajectory = _xy_positions(evidence)
        reference = [(p[0], p[1]) for p in field_at(evidence, "reference.reference_path_xyz_m")]
        if not reference:
            raise MissingEvidence("reference path not captured")
        dtw_distance = 0.0
        for point in trajectory:
            min_dist = min(math.dist(point, ref_point) for ref_point in reference)
            dtw_distance += math.exp(-(min_dist ** 2) / (2 * radius ** 2))
        return dtw_distance / len(trajectory) if trajectory else 0.0


def load_metric(spec, root=None):
    # Entrypoint comes from the user's selected metric set, never from recorded evidence.
    from nav_eval.plugins import load_entrypoint
    plugin = load_entrypoint(spec["entrypoint"], root or ".")()
    if not isinstance(plugin.name, str) or not hasattr(plugin, "required_fields"):
        raise ValueError("invalid metric plugin")
    return plugin


def evaluate_metric(plugin, record, evidence, parameters):
    if record["status"] != "completed":
        return {"status": "not_scored", "value": None, "reason": record["status"]}
    try:
        for field in plugin.required_fields:
            field_at(evidence or {}, field)
        value = finite_number(plugin.compute(record, evidence or {}, parameters), "metric value")
        return {"status": "ok", "value": value}
    except MissingEvidence as error:
        return {"status": "unavailable", "value": None, "reason": str(error)}
    except Exception as error:
        return {"status": "error", "value": None, "reason": f"{type(error).__name__}: {error}"}
