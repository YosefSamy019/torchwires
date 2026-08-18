from typing import Callable, Any

import torch

from ..base_step import BaseStep
from ...constants.constants import WEIGHT_ANNOT_FOR_LOSS_NAME, EFF_ANNOT_FOR_LOSS_NAME, RAW_ANNOT_FOR_LOSS_NAME
from ...models_repo.models_repo import ModelsRepo
from ...state.batch_state import BatchState


class LossStep(BaseStep):

    def __init__(
            self,
            loss_name: str,
            loss_function: Callable[[BatchState], Any],
            weight_function: Callable[[BatchState], float],
    ):
        super().__init__()

        self._loss_name = loss_name
        self._loss_function = loss_function
        self._weight_function = weight_function

    def get_trackable_features(self) -> list[str]:
        return [
            f"{self._loss_name}{RAW_ANNOT_FOR_LOSS_NAME}",
            f"{self._loss_name}{WEIGHT_ANNOT_FOR_LOSS_NAME}",
            f"{self._loss_name}{EFF_ANNOT_FOR_LOSS_NAME}"
        ]

    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState,
            device: torch.device | str,
    ):
        loss_val = self._loss_function(state)
        weight_val = self._weight_function(state)
        eff_loss = loss_val * weight_val

        loss_val = self._safe_move_to_device(loss_val, device)
        weight_val = self._safe_move_to_device(weight_val, device)
        eff_loss = self._safe_move_to_device(eff_loss, device)

        state.set(
            key=f"{self._loss_name}{RAW_ANNOT_FOR_LOSS_NAME}",
            value=loss_val,
        )

        state.set(
            key=f"{self._loss_name}{WEIGHT_ANNOT_FOR_LOSS_NAME}",
            value=weight_val,
        )

        state.set(
            key=f"{self._loss_name}{EFF_ANNOT_FOR_LOSS_NAME}",
            value=eff_loss,
        )

        cur_total_loss = state[BatchState.KEY_TOTAL_LOSS]

        cur_total_loss = self._safe_move_to_device(cur_total_loss, device)

        cur_total_loss += eff_loss

        state.set(
            BatchState.KEY_TOTAL_LOSS,
            value=cur_total_loss,
        )

    def get_uuid(self) -> str:
        return f"{self._loss_name}"
