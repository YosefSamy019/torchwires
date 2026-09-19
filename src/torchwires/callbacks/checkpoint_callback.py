from typing import Literal, Callable, Any

import numpy as np

from .base_callback import BaseCallback
from ..common.logger.logger import print_log


class CheckpointCallback(BaseCallback):
    def __init__(
            self,
            save_function: Callable[[str], None],
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            aggregate_mode: Literal["mean", "sum", "last"],
    ):
        super().__init__()
        self._save_function = save_function
        self._split = split
        self._monitor = monitor
        self._mode = mode
        self._aggregate_mode = aggregate_mode

        self._checkpoint_name = f"checkpoint {self._split}-{self._monitor}-{self._mode}"

        self._stores_values = []

        if mode == "min":
            self._best_value = float("inf")
        elif mode == "max":
            self._best_value = float("inf") * -1
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
        if self._mode == "min" and cur_agg_val < self._best_value:
            new_target_achieved = True
        elif self._mode == "max" and cur_agg_val > self._best_value:
            new_target_achieved = True
        else:
            new_target_achieved = False

        # check
        if new_target_achieved:
            print_log(
                title="Checkpoint",
                content=
                f"{self._split}-{self._monitor} improved from {self._best_value} to {cur_agg_val}"
                " | "
                f"checkpoint name: {self._checkpoint_name}",
            )

            self._best_value = cur_agg_val

            self._save_function(self._checkpoint_name)

        # Important: start collecting values for the next epoch
        self._stores_values.clear()
