from typing import List

from ..base_step import BaseStep
from ...common.logger.logger import print_log
from ...models_repo.model_node import ModelNode
from ...models_repo.models_repo import ModelsRepo
from ...state.batch_state import BatchState


class ForwardStep(BaseStep):
    def __init__(
            self,
            model_name: str,
            inputs: List[str],
            outputs: List[str],
    ):
        super().__init__()

        self._model_name: str = model_name
        self._inputs: List[str] = inputs
        self._outputs: List[str] = outputs



    def get_trackable_features(self) -> list[str]:
        return []

    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState
    ):
        model: ModelNode = models_repo[self._model_name]

        inputs_data: List = state.get_all(keys=self._inputs, must_exist=True)

        outputs_data = model(*inputs_data)

        state.set_all(
            keys=self._outputs,
            values=outputs_data,
        )

    def get_uuid(self) -> str:
        return f"{','.join(self._outputs)} = {self._model_name}({', '.join(self._inputs)})"
