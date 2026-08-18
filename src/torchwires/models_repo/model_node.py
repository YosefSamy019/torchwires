import os.path

import torch

from ..common.logger.logger import print_log


class ModelNode:
    _WEIGHTS_EXTENSION = ".weights.pth"

    def __init__(
            self,
            name: str,
            model: torch.nn.Module,
    ):
        self._name = name
        self._model = model

    def load_model(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str,
    ):

        path = str(
            os.path.join(
                repo_name, experiment, checkpoint, self._name + ModelNode._WEIGHTS_EXTENSION
            )
        )

        if os.path.isfile(path):
            self._model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            print_log(
                title=f"Model {self._name} weights loaded",
                content=f"path={path}",
            )
        else:
            print_log(
                title=f"Model {self._name} weights not found",
                content=f"path={path}",
            )

    def save_model(
            self,
            repo_name: str,
            experiment: str,
            checkpoint: str,
    ):
        path = str(
            os.path.join(
                repo_name, experiment, checkpoint, self._name + ModelNode._WEIGHTS_EXTENSION
            )
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self._model.state_dict(), path)
        print_log(
            title=f"Model {self._name} weights saved",
            content=f"path={path}",
        )

    def to(self, device: torch.device | str):
        self._model.to(device)

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def get_model(self) -> torch.nn.Module:
        return self._model
