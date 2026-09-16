from typing import Callable, Any

from .base_callback import BaseCallback
from ..common.logger.logger import print_log


class AutoSaveCallback(BaseCallback):
    def __init__(
            self,
            save_function: Callable[[str], None],
            interval: int,
    ):
        self._save_function = save_function
        self._interval = interval

        self._last_epoch_no = None

    def on_train_batch(
            self,
            train_record: dict[str, Any],
    ):
        if 'epoch' in train_record:
            self._last_epoch_no = train_record['epoch']

    def on_epoch_end(self):
        if self._last_epoch_no % self._interval == 0:
            checkpoint_name = f"auto_save epoch {self._last_epoch_no}"

            print_log(
                title="Auto Save Checkpoint",
                content=
                f"save to checkpoint '{checkpoint_name}'",
            )

            self._save_function(checkpoint_name)
