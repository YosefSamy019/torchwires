from typing import Literal

from .base_callback import BaseCallback
from ..common.logger.logger import print_log
from ..state.epoch_state import EpochState


class EarlyStoppingCallback(BaseCallback):
    def __init__(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            patience: int,
    ):
        super().__init__()
        self._split = split
        self._monitor = monitor
        self._mode = mode
        self._patience = patience

        if mode == "min":
            self._best_value = float("inf")
        elif mode == "max":
            self._best_value = float("inf") * -1
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self._counter = 0

    def on_train_start(self):
        self._counter = 0

    def on_epoch_end(self, epoch_state: EpochState):
        cur_val = epoch_state.aggregate_over_batches(
            feature=self._monitor,
            split=self._split,
            func='mean'
        )

        if cur_val is None:
            print_log(
                title="Early Stopping",
                content=f"Couldn't find {self._monitor} feature in {self._split} split",
            )
            return

        if self._mode == "min":
            self._best_value = min(cur_val, self._best_value)
            if cur_val > self._best_value:
                self._counter += 1
            else:
                self._counter = 0

        if self._mode == "max":
            self._best_value = max(cur_val, self._best_value)
            if cur_val < self._best_value:
                self._counter += 1
            else:
                self._counter = 0

        if self.should_stop_training():
            print_log(
                title="Early Stopping",
                content=f'the training has been stopped',
            )
        else:
            pass
            # print_log(
            #     title="Early Stopping",
            #     content=f"counter={self._counter:d}, best={self._best_value:0.5f}",
            # )

    def should_stop_training(self) -> bool:
        return self._counter >= self._patience
