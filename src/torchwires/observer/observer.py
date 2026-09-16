import json
import os.path
from typing import Any, Dict

import torch

from ..common.logger.logger import print_log
from ..pass_state.pass_state import PassState


class Observer:
    _FILE_NAME_EXTENSION = "history.json"

    def __init__(self):
        self._tracked_features = []
        self._history_dict: list[dict[str, Any]] = []

    def track_feature(self, feature: str):
        if feature not in self._tracked_features:
            self._tracked_features.append(feature)
            print_log(
                title=f"Observer tracks",
                content=f"feature={feature}",
            )
        else:
            print_log(
                title=f"Feature already tracked",
                content=f"feature={feature}",
            )

    def track_features(self, features: list):
        for feature in features:
            self.track_feature(feature)

    def load(
            self,
            repo_name: str,
            experiment: str,
    ):
        cache_path = os.path.join(
            repo_name, experiment, Observer._FILE_NAME_EXTENSION
        )

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._history_dict = json.load(f)

            print_log(
                title=f"History loaded",
                content=f"path={cache_path}",
            )
        else:
            print_log(
                title=f"History cache not found",
                content=f"path={cache_path}",
            )

    def save(
            self,
            repo_name: str,
            experiment: str,
            silent: bool = False,
    ):
        cache_path = os.path.join(
            repo_name, experiment, Observer._FILE_NAME_EXTENSION
        )

        with open(cache_path, "w") as f:
            json.dump(self._history_dict, f)

        if not silent:
            print_log(
                title=f"History saved",
                content=f"json={cache_path}",
            )

    def observe(self, state: PassState) -> Dict[Any, Any]:
        # track main features
        new_record = state.get_base_record()

        # track custom records
        state_dict = state.__dict__

        for feature in self._tracked_features:
            value = state_dict.get(feature, None)

            if isinstance(value, torch.Tensor):
                value = value.numpy(force=True).tolist()

            new_record[feature] = value

        self._history_dict.append(new_record)

        return new_record

    @property
    def history_dict(self) -> list[dict[str, Any]]:
        return self._history_dict
