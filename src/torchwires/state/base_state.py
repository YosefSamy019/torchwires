from abc import ABC


class BaseState(ABC):
    KEY_EPOCH_NO = 'epoch'
    KEY_BATCH_NO = 'batch'
    KEY_LOADER_TYPE = 'loader'
    KEY_TOTAL_LOSS = 'total_loss'

    @staticmethod
    def get_trackable_features() -> list[str]:
        return [
            BaseState.KEY_EPOCH_NO,
            BaseState.KEY_BATCH_NO,
            BaseState.KEY_LOADER_TYPE,
            BaseState.KEY_TOTAL_LOSS
        ]
