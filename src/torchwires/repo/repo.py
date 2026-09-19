import os
from typing import Callable, Any

import torch
from torch import nn
from huggingface_hub import snapshot_download, HfApi

from ..common.logger.logger import print_log
from ..constants.constants import DEFAULT_CHECKPOINT_NAME
from ..models.models_repo import ModelsRepo
from ..nodes.nodes_repo import NodesRepo
from ..observer.observer import Observer
from ..optimizers.optimizers_repo import OptimizersRepo


class Repo:
    def __init__(
            self,
            repo_name: str,
            experiment: str,
            huggingface_repo_id: str | None = None
    ):
        self._repo_name = repo_name
        self._experiment = experiment
        self._huggingface_repo_id = huggingface_repo_id

        self._models_repo = ModelsRepo()
        self._optimizers_repo = OptimizersRepo()
        self._observer = Observer()
        self._nodes_repo = NodesRepo()

        os.makedirs(self._repo_name, exist_ok=True)
        os.makedirs(os.path.join(self._repo_name, self.experiment), exist_ok=True)

        if huggingface_repo_id is not None:
            print_log(
                title="Downloading from Huggingface",
                content=f'repo-id={self._huggingface_repo_id}'
            )

            snapshot_download(
                repo_id=self._huggingface_repo_id,
                repo_type="model",
                local_dir=self._repo_name,
            )

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
            push_to_huggingface: bool = True,
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

        if self._huggingface_repo_id is not None and push_to_huggingface:
            print_log(
                title="Pushing to Huggingface",
                content=f'repo-id={self._huggingface_repo_id}'
            )

            api = HfApi()

            api.upload_folder(
                folder_path=self._repo_name,
                repo_id=self._huggingface_repo_id,
                repo_type="model",
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
            opt = self._optimizers_repo.register(
                name=name,
                optimizer=value,
            )

            opt.load_optimizer(
                repo_name=self._repo_name,
                experiment=self._experiment,
                checkpoint=DEFAULT_CHECKPOINT_NAME
            )

            self._observer.track_features(opt.tracked_features)
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
