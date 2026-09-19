from typing import Literal, Any

import numpy as np

from .base_callback import BaseCallback
from ..common.logger.logger import print_log
from ..optimizers.optimizer_unit import OptimizerUnit


class ReduceLROnPlateauCallback(BaseCallback):
    def __init__(
            self,
            split: Literal["train", "val"],
            monitor: str,
            optimizer: OptimizerUnit,
            mode: Literal["min", "max"],
            aggregate_mode: Literal["mean", "sum", "last"],
            patience: int,
            factor: float = 0.1,
            min_lr: float = 1e-5,
    ):
        super().__init__()

        if factor <= 0 or factor >= 1:
            raise ValueError("factor must be between 0 and 1")

        if patience < 1:
            raise ValueError("patience must be >= 1")

        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")

        self._split = split
        self._monitor = monitor
        self._optimizer = optimizer
        self._mode = mode
        self._aggregate_mode = aggregate_mode

        self._patience = patience
        self._factor = factor
        self._min_lr = min_lr

        self._stores_values = []

        if mode == "min":
            self._best_value = float("inf")
        elif mode == "max":
            self._best_value = float("-inf")
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

        if self._monitor in train_record:
            self._stores_values.append(train_record[self._monitor])
        else:
            raise KeyError(
                f"Key '{self._monitor}' not found in batch record"
            )

    def on_val_batch(
            self,
            val_record: dict[str, Any],
    ):
        if self._split != "val":
            return

        if self._monitor in val_record:
            self._stores_values.append(val_record[self._monitor])
        else:
            raise KeyError(
                f"Key '{self._monitor}' not found in batch record"
            )

    def on_epoch_end(self):
        if not self._stores_values:
            raise RuntimeError(
                f"No values were collected for "
                f"{self._split}-{self._monitor}"
            )

        # aggregate
        if self._aggregate_mode == "mean":
            cur_agg_val = np.mean(self._stores_values)

        elif self._aggregate_mode == "sum":
            cur_agg_val = np.sum(self._stores_values)

        elif self._aggregate_mode == "last":
            cur_agg_val = self._stores_values[-1]

        else:
            raise ValueError(
                f"Invalid aggregate_mode: {self._aggregate_mode}"
            )

        # compare
        if self._mode == "min":
            improved = cur_agg_val < self._best_value
        else:
            improved = cur_agg_val > self._best_value

        if improved:
            self._best_value = cur_agg_val
            self._counter = 0

        else:
            self._counter += 1

        # reduce LR
        if self._counter >= self._patience:
            self._reduce_lr()
            self._counter = 0

        print_log(
            title=f"Reduce LR On Plateau for {self._optimizer.name}",
            content=", ".join(
                [
                    f"{self._split}-{self._monitor}={cur_agg_val} ",
                    f"best={self._best_value} ",
                    f"counter={self._counter:d}/{self._patience} ",
                    f"lr={self._optimizer.lr}",
                ]
            ),
        )

        # Important: start collecting values for the next epoch
        self._stores_values.clear()

    def _reduce_lr(self):
        new_lr = max(
            self._optimizer.lr * self._factor,
            self._min_lr,
        )

        if new_lr == self._optimizer.lr:
            print_log(
                title=f"Reduce LR On Plateau for {self._optimizer.name}",
                content=(
                    f"learning rate already reached min_lr={self._min_lr}"
                ),
            )
            return

        print_log(
            title=f"Reduce LR On Plateau for {self._optimizer.name}",
            content=(
                f"reducing learning rate "
                f"from {self._optimizer.lr} to {new_lr}"
            ),
        )

        self._optimizer.set_lr(new_lr)
