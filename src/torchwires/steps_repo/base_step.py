from abc import ABC, abstractmethod
from typing import Any

import torch

from ..models_repo.models_repo import ModelsRepo
from ..state.batch_state import BatchState


class BaseStep(
    ABC,
):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def get_trackable_features(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState,
            device: torch.device | str,
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_uuid(self) -> str:
        raise NotImplementedError()

    def _safe_move_to_device(
            self,
            val: Any,
            device: torch.device | str,
    ):
        if isinstance(val, torch.Tensor):
            return val.to(device)
        else:
            return val
