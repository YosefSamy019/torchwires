import os
from typing import Callable, Any

import torch
from torch import nn

from ..constants.constants import DEFAULT_CHECKPOINT_NAME
from ..models.models_repo import ModelsRepo
from ..nodes.nodes_repo import NodesRepo
from ..observer.observer import Observer
from ..optimizers.optimizers_repo import OptimizersRepo


class Repo:
    def __init__(
            self,
            repo_name: str,
            auto_load: bool = True,
    ):
        self._repo_name = repo_name
        self._experiment = "exp_1"

        self._models_repo = ModelsRepo()
        self._optimizers_repo = OptimizersRepo()
        self._observer = Observer()
        self._nodes_repo = NodesRepo()

        os.makedirs(self._repo_name, exist_ok=True)
        os.makedirs(os.path.join(self._repo_name, self.experiment), exist_ok=True)

        if auto_load:
            self.load()

    @property
    def repo_name(self) -> str:
        return self._repo_name

    @property
    def experiment(self) -> str:
        return self._experiment

    @property
    def models_repo(self) -> ModelsRepo:
        return self._models_repo

    @property
    def optimizers_repo(self) -> OptimizersRepo:
        return self._optimizers_repo

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def nodes_repo(self) -> NodesRepo:
        return self._nodes_repo

    def load(
            self,
            checkpoint: str = DEFAULT_CHECKPOINT_NAME,
    ):
        self._models_repo.load(
            checkpoint=checkpoint,
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

        self._optimizers_repo.load(
            checkpoint=checkpoint,
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

        self._observer.load(
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

    def save(
            self,
            checkpoint: str = DEFAULT_CHECKPOINT_NAME,
    ):
        self._models_repo.save(
            checkpoint=checkpoint,
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

        self._optimizers_repo.save(
            checkpoint=checkpoint,
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

        self._observer.save(
            repo_name=self._repo_name,
            experiment=self._experiment,
        )

    def register(
            self,
            name: str,
            value: nn.Module | torch.optim.Optimizer,
    ):
        if isinstance(value, nn.Module):
            self._models_repo.register(
                name=name,
                model=value,
            ).load_model(
                repo_name=self._repo_name,
                experiment=self._experiment,
                checkpoint=DEFAULT_CHECKPOINT_NAME
            )

        elif isinstance(value, torch.optim.Optimizer):
            self._optimizers_repo.register(
                name=name,
                optimizer=value,
            ).load_optimizer(
                repo_name=self._repo_name,
                experiment=self._experiment,
                checkpoint=DEFAULT_CHECKPOINT_NAME
            )
        else:
            raise TypeError(
                "Argument 'value' is not Supported"
            )

    def wire(
            self,
            name: str,
            body: Callable[..., dict[str, Any]],
            condition: Callable[..., bool] = lambda: True,
            tracked_features: list[str] | None = None,
    ):
        self._nodes_repo.wire(
            name=name,
            body=body,
            condition=condition,
            models_repo=self.models_repo,
            optimizers_repo=self.optimizers_repo,
            observer=self.observer,
        )

        for f in tracked_features or []:
            self._observer.track_feature(f)
