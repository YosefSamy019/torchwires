from typing import Any

from ..constants.constants import SPLIT


class PassState:
    def __init__(
            self,
            epoch: int,
            batch: int,
            split: SPLIT,
            device: str,
    ):
        self.epoch = epoch
        self.batch = batch
        self.split = split
        self.device = device

    def update(self, state: dict[str, Any]):
        for k, v in state.items():
            setattr(self, k, v)

    def get_base_record(self) -> dict[str, Any]:
        return {
            'epoch': self.epoch,
            'batch': self.batch,
            'split': self.split,
            'device': self.device,
        }

    def __getitem__(self, item):
        return getattr(self, item, None)
