from typing import Callable, Any

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
            state: BatchState
    ):
        metric_val = self._metric_function(state)

        state.set(
            key=self._metric_name,
            value=metric_val,
        )

    def get_uuid(self) -> str:
        return f"{self._metric_name}"
