from typing import Callable, Any

import torch

from ..base_step import BaseStep
from ...common.logger.logger import print_log
from ...models_repo.models_repo import ModelsRepo
from ...state.batch_state import BatchState


class MetricStep(BaseStep):
    def __init__(
            self,
            metric_name: str,
            metric_function: Callable[[BatchState], Any],
    ):
        super().__init__()

        self._metric_name = metric_name
        self._metric_function = metric_function

    def get_trackable_features(self) -> list[str]:
        return [self._metric_name]

    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState,
            device: torch.device | str,
    ):
        metric_val = self._metric_function(state)

        metric_val = self._safe_move_to_device(metric_val, device)

        state.set(
            key=self._metric_name,
            value=metric_val,
        )

    def get_uuid(self) -> str:
        return f"{self._metric_name}"
