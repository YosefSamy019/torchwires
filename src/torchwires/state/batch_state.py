from typing import Dict, List

import torch

from .base_state import BaseState


class BatchState(BaseState):

    def __init__(
            self,
            epoch_no: int,
            batch_no: int,
            loader_type: str,
    ):
        self._state: Dict = {
            BatchState.KEY_EPOCH_NO: epoch_no,
            BatchState.KEY_BATCH_NO: batch_no,
            BatchState.KEY_LOADER_TYPE: loader_type,
            BatchState.KEY_TOTAL_LOSS: 0.0,
        }

    def __setitem__(self, key, value):
        self._state[key] = value

    def __getitem__(self, key):
        return self._state[key]

    def __str__(self):
        return str(self._state)

    def get(self, key: str, default=None, must_exist=False):
        if must_exist and key not in self._state:
            raise KeyError(key)
        return self._state.get(key, default)

    def get_all(self, keys: List[str], default=None, must_exist=False):
        return [self.get(key, default=default, must_exist=must_exist) for key in keys]

    def set(self, key: str, value):
        self._state[key] = value

    def set_all(self, keys: List[str], values=None):
        if isinstance(values, torch.Tensor):
            values = [values]

        if len(values) != len(values):
            raise ValueError(f"len(values)={len(values)}, len(values)={len(values)}")

        for key, value in zip(keys, values):
            self.set(key, value)

    @property
    def epoch(self) -> int:
        return int(self._state[BatchState.KEY_EPOCH_NO])

    @property
    def batch(self) -> int:
        return int(self._state[BatchState.KEY_BATCH_NO])

    @property
    def loader_type(self) -> str:
        return str(self._state[BatchState.KEY_LOADER_TYPE])

    def get_dict(self) -> Dict:
        return self._state
