from abc import ABC, abstractmethod

from ..models_repo.models_repo import ModelsRepo
from ..state.batch_state import BatchState


class BaseStep(
    ABC,
):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def get_trackable_features(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState
    ):
        raise NotImplementedError()

    @abstractmethod
    def get_uuid(self) -> str:
        raise NotImplementedError()
