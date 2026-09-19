from typing import Literal, Any

import numpy as np

from .base_callback import BaseCallback
from ..common.logger.logger import print_log


class EarlyStoppingCallback(BaseCallback):
    def __init__(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            aggregate_mode: Literal["mean", "sum", "last"],
            patience: int,
    ):
        super().__init__()
        self._split = split
        self._monitor = monitor
        self._mode = mode
        self._aggregate_mode = aggregate_mode
        self._patience = patience

        self._stores_values = []

        if mode == "min":
            self._best_value = float("inf")
        elif mode == "max":
            self._best_value = float("inf") * -1
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self._counter = 0

    def on_train_start(self):
        self._counter = 0

    def on_train_batch(
            self,
            train_record: dict[str, Any],
    ):
        if self._split != "train":
            return

        if self._monitor in train_record.keys():
            self._stores_values.append(train_record[self._monitor])
        else:
            raise KeyError(f"Key'{self._monitor}' not found in batch record")

    def on_val_batch(
            self,
            val_record: dict[str, Any],
    ):
        if self._split != "val":
            return

        if self._monitor in val_record.keys():
            self._stores_values.append(val_record[self._monitor])
        else:
            raise KeyError(f"Key'{self._monitor}' not found in batch record")

    def on_epoch_end(self):
        # aggregate
        if self._aggregate_mode == "mean":
            cur_agg_val = np.mean(self._stores_values)
        elif self._aggregate_mode == "sum":
            cur_agg_val = np.sum(self._stores_values)
        elif self._aggregate_mode == "last":
            cur_agg_val = self._stores_values[-1]
        else:
            raise ValueError(f"Invalid aggregate_mode: {self._aggregate_mode}")

        # compare
        if self._mode == "min":
            self._best_value = min(cur_agg_val, self._best_value)
            if cur_agg_val > self._best_value:
                self._counter += 1
            else:
                self._counter = 0

        if self._mode == "max":
            self._best_value = max(cur_agg_val, self._best_value)
            if cur_agg_val < self._best_value:
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
            print_log(
                title="Early Stopping",
                content=f"counter={self._counter:d}/{self._patience}, best={self._best_value}",
            )

        # Important: start collecting values for the next epoch
        self._stores_values.clear()

    def should_stop_training(self) -> bool:
        return self._counter >= self._patience
