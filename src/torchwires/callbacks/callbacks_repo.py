from typing import List, Any

from .base_callback import BaseCallback


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
    ) -> None:
        for callback in self._callbacks_list:
            callback.on_epoch_end()

    def notify_train_batch(
            self,
            train_record: dict[str, Any],
    ):
        for callback in self._callbacks_list:
            callback.on_train_batch(train_record=train_record)

    def notify_val_batch(
            self,
            val_record: dict[str, Any],
    ):
        for callback in self._callbacks_list:
            callback.on_val_batch(val_record=val_record)

    def should_stop_training(self) -> bool:
        for callback in self._callbacks_list:
            if callback.should_stop_training():
                return True

        return False
