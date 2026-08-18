import json
import os
from typing import Any, Dict

import numpy as np
import torch
import pandas as pd

from ..common.logger.logger import print_log


class History:
    def __init__(
            self,
    ):
        self._tracked_features = []
        self._content: list[dict[str, Any]] = []

    def save(
            self,
            repo_name: str,
            experiment: str,
    ):
        dir_path = os.path.join(repo_name, experiment)
        cache_path = os.path.join(repo_name, experiment, "history.json")
        csv_path = os.path.join(repo_name, experiment, "history.csv")
        tracked_features_path = os.path.join(repo_name, experiment, "history_tracked.json")

        os.makedirs(dir_path, exist_ok=True)

        with open(cache_path, "w") as f:
            f.write(
                json.dumps(self._content)
            )

        pd.read_json(cache_path).to_csv(csv_path, index=False)

        with open(tracked_features_path, "w") as f:
            f.write(
                json.dumps(self._tracked_features)
            )

    def load(
            self,
            repo_name: str,
            experiment: str,
    ):
        cache_path = os.path.join(repo_name, experiment, "history.json")
        tracked_features_path = os.path.join(repo_name, experiment, "history_tracked.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._content = json.loads(f.read())
            print_log(
                title="History loaded",
                content=f"path={cache_path}",
            )
        else:
            print_log(
                title="No History cache found",
                content=f"path={cache_path}",
            )

        if os.path.exists(tracked_features_path):
            with open(tracked_features_path, "r") as f:
                self._tracked_features = json.loads(f.read())
            print_log(
                title="History Features loaded",
                content=f"path={tracked_features_path}",
            )
        else:
            print_log(
                title="No History Features found",
                content=f"path={tracked_features_path}",
            )

    def track_feature(self, feature: str):
        if feature not in self._tracked_features:
            self._tracked_features.append(feature)

    def track_features(self, features: list[str]):
        for feature in features:
            self.track_feature(feature)

    def record(
            self, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        new_line = {}

        for feature in self._tracked_features:
            val = state_dict.get(feature, None)

            if isinstance(val, torch.Tensor):
                val = val.item()

            if isinstance(val, np.generic):
                return val.item()

            new_line[feature] = val

        self._content.append(new_line)
        return new_line

    def get_feature_columns(self, feature: str) -> list:
        li = []
        for row in self._content:
            li.append(row.get(feature))
        return li

    def get_tracked_features(self) -> list:
        return self._tracked_features

    def return_as_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self._content)
        return df
