import os.path

import torch

from ..common.logger.logger import print_log
from ..constants.constants import LR_ANNOT_FOR_OPTIMIZER_NAME
from ..state.batch_state import BatchState


class OptimizerNode:
    _OPTIMIZER_EXTENSION = ".optimizer.pth"

    def __init__(
            self,
            name: str,
            optimizer: torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler,
    ):
        self._name = name
        self._optimizer_scheduler = optimizer

    def load_optimizer(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str,
    ):
        path = str(
            os.path.join(
                repo_name, experiment, checkpoint, self._name + OptimizerNode._OPTIMIZER_EXTENSION
            )
        )

        if os.path.isfile(path):
            self._optimizer_scheduler.load_state_dict(torch.load(path))
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
                repo_name, experiment, checkpoint, self._name + OptimizerNode._OPTIMIZER_EXTENSION
            )
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self._optimizer_scheduler.state_dict(), path)
        print_log(
            title=f"Optimizer {self._name} weights saved",
            content=f"path={path}",
        )

    def zero_grad(self):
        if isinstance(self._optimizer_scheduler, torch.optim.lr_scheduler.LRScheduler):
            self._optimizer_scheduler.optimizer.zero_grad()
        else:
            self._optimizer_scheduler.zero_grad()

    def step(self, state: BatchState):
        if isinstance(self._optimizer_scheduler, torch.optim.lr_scheduler.LRScheduler):
            self._optimizer_scheduler.optimizer.step()

            cur_lr = self._optimizer_scheduler.optimizer.param_groups[0].get('lr')
        else:
            self._optimizer_scheduler.step()
            cur_lr = self._optimizer_scheduler.param_groups[0].get('lr')

        state.set(f'{self._name}{LR_ANNOT_FOR_OPTIMIZER_NAME}', cur_lr)

    def get_trackable_features(self) -> list[str]:
        return [
            f'{self._name}{LR_ANNOT_FOR_OPTIMIZER_NAME}',
        ]

    def scheduler_step(self):
        if isinstance(self._optimizer_scheduler, torch.optim.lr_scheduler.LRScheduler):
            self._optimizer_scheduler.step()
