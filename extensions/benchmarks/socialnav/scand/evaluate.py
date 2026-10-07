"""Explicitly offline trajectory diagnostics; SCAND has no simulator.

JSONL rows: episode_id, timestamp_s, frame_id, position_xy_m. Predictions must
cover exactly the same timestamps in the same coordinate frame as the reference.
ADE/FDE are imitation diagnostics, not an official SCAND leaderboard protocol
or a test of counterfactual safety/social compliance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from nav_eval.contracts import ContractError, finite_number
from nav_eval.storage import file_digest

FIELDS = {"episode_id", "timestamp_s", "frame_id", "position_xy_m"}


def validate_rows(rows):
    episodes = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ContractError(f"row {index}: expected exactly {sorted(FIELDS)}")
        for key in ("episode_id", "frame_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ContractError(f"row {index}: {key} must be a nonempty string")
        timestamp = finite_number(row["timestamp_s"], "timestamp_s")
        if timestamp < 0:
            raise ContractError("timestamp_s must be nonnegative")
        position = row["position_xy_m"]
        if not isinstance(position, list) or len(position) != 2:
            raise ContractError("position_xy_m must contain x and y in meters")
        for component in position:
            finite_number(component, "position_xy_m")
        samples = episodes.setdefault(row["episode_id"], [])
        if samples and timestamp <= samples[-1]["timestamp_s"]:
            raise ContractError("timestamps must be strictly increasing within each episode")
        if samples and row["frame_id"] != samples[0]["frame_id"]:
            raise ContractError("coordinate frame changes within an episode")
        samples.append(row)
    if not episodes:
        raise ContractError("trajectory file is empty")
    return episodes


def load_trajectories(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_rows(json.loads(line) for line in handle if line.strip())


def evaluate_files(reference_path, predictions_path):
    reference = load_trajectories(reference_path)
    predictions = load_trajectories(predictions_path)
    if set(reference) != set(predictions):
        raise ContractError("reference and predictions must contain exactly the same episode IDs")
    episodes = []
    for episode_id, truth in reference.items():
        proposed = predictions[episode_id]
        if ([row["timestamp_s"] for row in truth] != [row["timestamp_s"] for row in proposed]
                or any(a["frame_id"] != b["frame_id"] for a, b in zip(truth, proposed))):
            raise ContractError(f"{episode_id}: timestamps and coordinate frames must match exactly")
        errors = [math.dist(a["position_xy_m"], b["position_xy_m"])
                  for a, b in zip(truth, proposed)]
        for error in errors:
            finite_number(error, "trajectory displacement")
        episodes.append({"episode_id": episode_id, "samples": len(errors),
                         "ade_m": math.fsum(error / len(errors) for error in errors), "fde_m": errors[-1]})
    count = len(episodes)
    return {
        "schema_version": "nav-eval-scand-offline/1", "benchmark": "scand",
        "execution_mode": "offline", "claim": "diagnostic", "episodes": episodes,
        "metrics": {"trajectory_ade_m": math.fsum(e["ade_m"] / count for e in episodes),
                    "trajectory_fde_m": math.fsum(e["fde_m"] / count for e in episodes)},
        "aggregation": "arithmetic mean over episodes (equal episode weight)",
        "alignment": "exact timestamps and coordinate frames; no interpolation or registration",
        "reference_sha256": file_digest(Path(reference_path)),
        "predictions_sha256": file_digest(Path(predictions_path)),
        "limitations": "Offline imitation error only; not closed-loop success, collision avoidance or social compliance.",
    }


def export_rosbag(bag_path, topic, episode_id, output_path):
    """Export a user-selected nav_msgs/Odometry topic using its header timestamps.

    Run in a ROS1 environment with rosbag installed. Topic names and coordinate
    frames vary by platform/recording, so the caller must select the topic.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ContractError("an explicit odometry topic is required")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    try:
        import rosbag
    except ImportError as error:
        raise ImportError("SCAND bag export requires the ROS1 rosbag package in the selected interpreter") from error
    rows = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, message, _ in bag.read_messages(topics=[topic]):
            if getattr(message, "_type", None) != "nav_msgs/Odometry":
                raise ContractError("selected topic must contain nav_msgs/Odometry messages")
            position = message.pose.pose.position
            rows.append({"episode_id": episode_id, "timestamp_s": message.header.stamp.to_sec(),
                         "frame_id": message.header.frame_id,
                         "position_xy_m": [float(position.x), float(position.y)]})
    validate_rows(rows)
    # Validate the entire export before creating a new artifact; never clobber.
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return {"benchmark": "scand", "episode_id": episode_id, "samples": len(rows),
            "topic": topic, "output": str(output_path), "sha256": file_digest(output_path)}
