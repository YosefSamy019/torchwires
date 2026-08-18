from typing import Dict, Optional

import torch

from .optimizer_node import OptimizerNode
from ..common.logger.logger import print_log
from ..constants.constants import DEFAULT_CHECKPOINT_NAME
from ..models_repo.model_node import ModelNode
from ..models_repo.models_repo import ModelsRepo
from ..state.batch_state import BatchState


class OptimizersRepo:
    def __init__(
            self,
            models_repo: ModelsRepo,
    ):
        self._models_repo = models_repo
        self._optimizers: Dict[str, OptimizerNode] = {}

    def register_optimizer(
            self,
            name: str,
            optimizer: torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler,
    ) -> OptimizerNode:
        if name in self._optimizers:
            raise ValueError(f"Optimizer {name} already registered")

        optimizer_node = OptimizerNode(
            name=name,
            optimizer=optimizer
        )

        self._optimizers[name] = optimizer_node

        print_log(
            title=f"Optimizer {name}",
            content=f"has been registered",
        )

        return optimizer_node

    def load(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str = DEFAULT_CHECKPOINT_NAME,
    ):
        for key, value in self._optimizers.items():
            value.load_optimizer(
                repo_name=repo_name,
                experiment=experiment,
                checkpoint=checkpoint,
            )

    def save(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str = DEFAULT_CHECKPOINT_NAME,
    ):
        for key, value in self._optimizers.items():
            value.save_optimizer(
                repo_name=repo_name,
                experiment=experiment,
                checkpoint=checkpoint,
            )

    def zero_grad(self):
        for optimizer in self._optimizers.values():
            optimizer.zero_grad()

    def step(self, state: BatchState):
        for optimizer in self._optimizers.values():
            optimizer.step(state=state)

    def scheduler_step(self):
        for optimizer in self._optimizers.values():
            optimizer.scheduler_step()

    def get_trackable_features(self) -> list[str]:
        li = []
        for optimizer in self._optimizers.values():
            li.extend(optimizer.get_trackable_features())
        return li
