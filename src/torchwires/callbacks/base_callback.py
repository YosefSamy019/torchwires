from abc import ABC

from ..state.epoch_state import EpochState


class BaseCallback(ABC):
    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(
            self
    ):
        pass

    def on_epoch_end(
            self,
            epoch_state: EpochState,
    ):
        pass

    def should_stop_training(self):
        return False
