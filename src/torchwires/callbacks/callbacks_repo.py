from typing import List

from .base_callback import BaseCallback
from ..state.epoch_state import EpochState


class CallbacksRepo:
    def __init__(self):
        self._callbacks_list: List[BaseCallback] = []

    def register_callback(
            self,
            callback: BaseCallback
    ):
        self._callbacks_list.append(callback)

    def register_callbacks(
            self,
            callbacks: List[BaseCallback]
    ):
        for callback in callbacks:
            self.register_callback(callback)

    def notify_train_start(self) -> None:
        for callback in self._callbacks_list:
            callback.on_train_start()

    def notify_train_end(self) -> None:
        for callback in self._callbacks_list:
            callback.on_train_end()

    def notify_epoch_start(
            self
    ) -> None:
        for callback in self._callbacks_list:
            callback.on_epoch_start()

    def notify_epoch_end(
            self,
            epoch_state: EpochState
    ) -> None:
        for callback in self._callbacks_list:
            callback.on_epoch_end(epoch_state=epoch_state)

    def should_stop_training(self) -> bool:
        for callback in self._callbacks_list:
            if callback.should_stop_training():
                return True

        return False
