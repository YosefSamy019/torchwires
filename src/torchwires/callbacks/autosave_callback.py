from typing import Literal, Callable

from .base_callback import BaseCallback
from ..common.logger.logger import print_log
from ..state.epoch_state import EpochState


class AutoSaveCallback(BaseCallback):
    def __init__(
            self,
            save_function: Callable[[str], None],
            interval: int,
            checkpoint_name: str,
            concat_with_epoch_no: bool,
    ):
        super().__init__()
        self._save_function = save_function
        self._interval = interval
        self._checkpoint_name = checkpoint_name
        self._concat_with_epoch_no = concat_with_epoch_no

    def on_epoch_end(self, epoch_state: EpochState):
        cur_epoch = epoch_state.aggregate_over_batches(
            split="train",
            feature=epoch_state.KEY_EPOCH_NO,
            func='max'
        )

        if cur_epoch % self._interval == 0:
            if self._concat_with_epoch_no:
                save_name = f"{self._checkpoint_name} epoch={cur_epoch}"
            else:
                save_name = f"{self._checkpoint_name}"

            print_log(
                title="Auto Save Checkpoint",
                content=
                f"save to checkpoint {save_name}",
            )

            self._save_function(save_name)
