from abc import ABC, abstractmethod
from typing import Any


class BaseCallback(ABC):
    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(
            self
    ):
        pass

    def on_train_batch(
            self,
            train_record: dict[str, Any],
    ):
        pass

    def on_val_batch(
            self,
            val_record: dict[str, Any],
    ):
        pass

    def on_epoch_end(
            self,
    ):
        pass

    def should_stop_training(self):
        return False
