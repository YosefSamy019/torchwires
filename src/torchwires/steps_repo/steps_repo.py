from typing import List

import torch

from ..common.logger.logger import print_log
from ..models_repo.models_repo import ModelsRepo
from .base_step import BaseStep
from ..state.batch_state import BatchState


class StepsRepo:
    def __init__(self):
        self._steps: List[BaseStep] = []

    def add_step(self, step: BaseStep):
        step_uuid = step.get_uuid()

        for s in self._steps:
            if step_uuid == s.get_uuid():
                raise Exception(f"Step {step.get_uuid()} already exists")

        print_log(
            title=f"{step.__class__.__name__} added",
            content=f"{step.get_uuid()}",
        )

        self._steps.append(step)

    def execute(
            self,
            models_repo: ModelsRepo,
            state: BatchState,
            device: torch.device | str,
    ):
        for step in self._steps:
            step.execute(
                models_repo=models_repo,
                state=state,
                device=device
            )
