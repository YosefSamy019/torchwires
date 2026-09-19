from typing import Dict, Any

import torch

from .optimizer_unit import OptimizerUnit
from ..common.logger.logger import print_log
from ..constants.constants import DEFAULT_CHECKPOINT_NAME


class OptimizersRepo:
    def __init__(
            self,
    ):
        self._optimizers: Dict[str, OptimizerUnit] = {}

    def register(
            self,
            name: str,
            optimizer: torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler,
    ) -> OptimizerUnit:
        if name in self._optimizers:
            raise KeyError(f"Optimizer {name} already registered")

        optimizer_node = OptimizerUnit(
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

    def step(self):
        optimizers_state: Dict[str, Any] = {}

        for optimizer in self._optimizers.values():
            opt_state = optimizer.step()
            optimizers_state.update(opt_state)

        return optimizers_state
