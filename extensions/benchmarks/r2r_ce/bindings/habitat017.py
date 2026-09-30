"""Only the version-specific task configuration differs for Habitat 0.1.7."""
from __future__ import annotations

import math

from extensions.benchmarks.r2r_ce.bindings.habitat import HabitatR2RBenchmarkService
from nav_eval.contracts import ContractError

TASK_YAML = """\
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500
SIMULATOR:
  ACTION_SPACE_CONFIG: v0
  AGENT_0:
    SENSORS: [RGB_SENSOR, DEPTH_SENSOR]
  FORWARD_STEP_SIZE: 0.25
  TURN_ANGLE: 15
  HABITAT_SIM_V0:
    GPU_DEVICE_ID: 0
    ALLOW_SLIDING: True
  RGB_SENSOR:
    WIDTH: 640
    HEIGHT: 480
    HFOV: 90
    TYPE: HabitatSimRGBSensor
  DEPTH_SENSOR:
    WIDTH: 640
    HEIGHT: 480
    HFOV: 90
    NORMALIZE_DEPTH: False
TASK:
  TYPE: VLN-v0
  SUCCESS_DISTANCE: 3.0
  SENSORS: []
  POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
  MEASUREMENTS: [DISTANCE_TO_GOAL, SUCCESS, SPL]
  SUCCESS:
    SUCCESS_DISTANCE: 3.0
DATASET:
  TYPE: R2RVLN-v1
  SPLIT: val_unseen
  CONTENT_SCENES: ['*']
  DATA_PATH: data/datasets/r2r/val_unseen/val_unseen.json.gz
  SCENES_DIR: data/scene_datasets/
"""


class Habitat017BenchmarkService(HabitatR2RBenchmarkService):
    def __init__(self, data_root, benchmark_id="r2r_val_unseen", **settings):
        if benchmark_id.startswith("rxr"):
            raise ValueError("RxR requires the Habitat 0.2.4/0.3.0 task binding")
        settings.setdefault("hfov", 90)
        settings["simulator_version"] = "0.1.7"
        super().__init__(data_root, benchmark_id, **settings)

    def _ensure_env(self):
        if self.env is not None:
            return
        import tempfile
        import habitat

        self._episodes = self._episode_manifest()
        if not self._episodes:
            raise ContractError("empty selected dataset")
        self._episode_order = {str(e["episode_id"]): i for i, e in enumerate(self._episodes)}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(TASK_YAML)
            handle.flush()
            config = habitat.get_config(config_paths=handle.name,
                opts=[item for pair in self._config_overrides() for item in pair])
        self.env = self.backend.initialize(config)
        self.actual_version = str(habitat.__version__)

    def _config_overrides(self):
        pairs = [("SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID", str(self.gpu_device_id)),
                 ("SIMULATOR.TURN_ANGLE", str(round(math.degrees(self.turn_angle_rad)))),
                 ("SIMULATOR.DEPTH_SENSOR.MAX_DEPTH", str(self.max_depth_m)),
                 ("SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH", "False"),
                 ("TASK.SUCCESS_DISTANCE", str(float(self.success_distance_m))),
                 ("TASK.SUCCESS.SUCCESS_DISTANCE", str(float(self.success_distance_m))),
                 ("DATASET.SPLIT", self.split), ("DATASET.DATA_PATH", str(self._dataset_path)),
                 ("DATASET.SCENES_DIR", str(self.data_root / "scene_datasets")),
                 ("DATASET.CONTENT_SCENES", "['*']")]
        for sensor in ("RGB_SENSOR", "DEPTH_SENSOR"):
            for field, value in (("WIDTH", self.width), ("HEIGHT", self.height), ("HFOV", self.hfov)):
                pairs.append((f"SIMULATOR.{sensor}.{field}", str(value)))
        return pairs

    def _to_habitat_action(self, action):
        return super()._to_habitat_action(action).upper()
