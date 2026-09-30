"""Habitat-Lab registration for RxR VLN-CE guide episodes.

Adapted from StreamVLN's Apache-2.0 ``rxr_vln_dataset.py``. Import this module
inside the Habitat worker before loading an ``RxR-VLN-CE-v1`` dataset.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import attr

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import VLNEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@attr.s(auto_attribs=True)
class RxRInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = None
    language: Optional[str] = None
    annotator_id: Optional[str] = None
    edit_distance: Optional[float] = None
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = None
    instruction_tokens: Optional[List[str]] = None


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Load one RxR role file selected through ``config.data_path``."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path.format(split=config.split)) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []
        self.instruction_vocab = VocabDict(word_list=[])
        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        with gzip.open(dataset_filename, "rt", encoding="utf-8") as stream:
            self.from_json(stream.read(), scenes_dir=config.scenes_dir)
        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        payload = json.loads(json_str)
        for raw_episode in payload["episodes"]:
            episode = VLNEpisode(**raw_episode)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = RxRInstructionData(**episode.instruction)
            episode.goals = [NavigationGoal(**goal) for goal in episode.goals]
            self.episodes.append(episode)
