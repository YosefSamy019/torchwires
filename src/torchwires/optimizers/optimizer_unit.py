import os.path
from collections.abc import Callable
from typing import Any, Dict

import torch

from ..common.logger.logger import print_log


class OptimizerUnit:
    _OPTIMIZER_EXTENSION = ".optimizer.pth"

    def __init__(
            self,
            name: str,
            optimizer: torch.optim.Optimizer,
    ):
        self._name = name
        self._optimizer = optimizer
        self._tracked_features_dict: dict[str, Callable[[], Any]] = {
            f'{self._name}.lr': lambda: self._optimizer.param_groups[0]['lr'],
        }

    def load_optimizer(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str,
    ):
        path = str(
            os.path.join(
                repo_name, experiment, checkpoint, self._name + OptimizerUnit._OPTIMIZER_EXTENSION
            )
        )

        if os.path.isfile(path):
            self._optimizer.load_state_dict(torch.load(path))
            print_log(
                title=f"Optimizer {self._name} loaded",
                content=f"path={path}",
            )
        else:
            print_log(
                title=f"Optimizer {self._name} cache not found",
                content=f"path={path}",
            )

    def save_optimizer(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str,
    ):
        path = str(
            os.path.join(
                repo_name, experiment, checkpoint, self._name + OptimizerUnit._OPTIMIZER_EXTENSION
            )
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self._optimizer.state_dict(), path)
        print_log(
            title=f"Optimizer {self._name} weights saved",
            content=f"path={path}",
        )

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self) -> Dict[str, Any]:
        self._optimizer.step()

        optimizer_state: dict[str, Any] = {
            k: v()
            for k, v in self._tracked_features_dict.items()
        }

        return optimizer_state

    @property
    def tracked_features(self) -> list[str]:
        return list(self._tracked_features_dict.keys())
