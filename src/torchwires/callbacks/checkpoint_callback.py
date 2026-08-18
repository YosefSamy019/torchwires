from typing import Literal, Callable

from .base_callback import BaseCallback
from ..common.logger.logger import print_log
from ..state.epoch_state import EpochState


class CheckpointCallback(BaseCallback):
    def __init__(
            self,
            save_function: Callable[[str], None],
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
    ):
        super().__init__()
        self._save_function = save_function
        self._split = split
        self._monitor = monitor
        self._mode = mode

        self._checkpoint_name = f"best_checkpoint_{self._split}-{self._monitor}"

        if mode == "min":
            self._best_value = float("inf")
        elif mode == "max":
            self._best_value = float("inf") * -1
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def on_epoch_end(self, epoch_state: EpochState):
        cur_value = epoch_state.aggregate_over_batches(feature=self._monitor, split=self._split, func='mean')

        if self._mode == "min" and cur_value < self._best_value:
            new_target_achieved = True
        elif self._mode == "max" and cur_value > self._best_value:
            new_target_achieved = True
        else:
            new_target_achieved = False

        if new_target_achieved:
            print_log(
                title="Checkpoint taken",
                content=
                f"{self._split}-{self._monitor} improved from {self._best_value:0.6f} to {cur_value:0.6f}"
                " | "
                f"checkpoint name: {self._checkpoint_name}",
            )

            self._best_value = cur_value

            self._save_function(self._checkpoint_name)
