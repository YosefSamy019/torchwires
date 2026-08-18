from typing import Dict, Optional, List

import torch
from torch import nn

from .model_node import ModelNode
from ..common.logger.logger import print_log
from ..constants.constants import DEFAULT_CHECKPOINT_NAME


class ModelsRepo:
    def __init__(
            self,
    ):
        self._models: Dict[str, ModelNode] = {}

    def register_model(
            self,
            name: str,
            model: nn.Module,
    ) -> ModelNode:
        if name in self._models:
            raise ValueError(f"Model {name} already registered")

        model_node = ModelNode(
            name=name,
            model=model,
        )

        self._models[name] = model_node

        print_log(
            title=f"Model {name}",
            content=f"has been registered",
        )

        return model_node

    def load(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str = DEFAULT_CHECKPOINT_NAME,
    ):
        for key, value in self._models.items():
            value.load_model(
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
        for key, value in self._models.items():
            value.save_model(
                repo_name=repo_name,
                experiment=experiment,
                checkpoint=checkpoint,
            )

    def is_registered(
            self,
            model_name: str,
    ) -> bool:
        return model_name in self._models

    def to(self, device: torch.device):
        for key, value in self._models.items():
            value.to(device)

    def __getitem__(self, item) -> ModelNode:
        return self._models[item]

    def train(self):
        for key, value in self._models.items():
            value.train()

    def eval(self):
        for key, value in self._models.items():
            value.eval()

    def get_model_node(self, name: str) -> Optional[ModelNode]:
        return self._models.get(name)

    def get_all_models_names(self) -> List[str]:
        return list(self._models.keys())
