from typing import Callable, Any, List, Literal

import torch
from torch import nn

from ..callbacks.autosave_callback import AutoSaveCallback
from ..callbacks.checkpoint_callback import CheckpointCallback
from ..callbacks.callbacks_repo import CallbacksRepo
from ..constants.constants import DEFAULT_CHECKPOINT_NAME
from ..history.history import History
from ..models_repo.models_repo import ModelsRepo
from ..optimizers_repo.optimizers_repo import OptimizersRepo
from ..state.batch_state import BatchState
from ..steps_repo.steps.forward_step import ForwardStep
from ..steps_repo.steps.loss_step import LossStep
from ..steps_repo.steps.metric_step import MetricStep
from ..steps_repo.steps_repo import StepsRepo


class Repo:
    def __init__(
            self,
            repo_name: str,
            load: bool = True,
    ):
        self._repo_name = repo_name
        self._experiment = "exp_1"

        self._models_repo = ModelsRepo()
        self._optimizers_repo = OptimizersRepo(models_repo=self._models_repo)
        self._steps_repo = StepsRepo()
        self._history = History()

        # Expose inner methods
        self.get_model_node = self._models_repo.get_model_node
        self.get_all_models_names = self._models_repo.get_all_models_names

        if load:
            self.load()

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

        self._history.load(
            experiment=self._experiment,
            repo_name=self._repo_name,
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

        self._history.save(
            experiment=self._experiment,
            repo_name=self._repo_name,
        )

    def register_model(
            self,
            name: str,
            model: nn.Module,
    ):
        model_node = self._models_repo.register_model(
            name=name,
            model=model,
        )

        model_node.load_model(
            repo_name=self._repo_name,
            experiment=self._experiment,
            checkpoint=DEFAULT_CHECKPOINT_NAME
        )

    def register_optimizer(
            self,
            name: str,
            optimizer: torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler,
    ):
        optimizer_node = self._optimizers_repo.register_optimizer(
            name=name,
            optimizer=optimizer,
        )

        optimizer_node.load_optimizer(
            repo_name=self._repo_name,
            experiment=self._experiment,
            checkpoint=DEFAULT_CHECKPOINT_NAME
        )

    def add_loss(
            self,
            loss_name: str,
            loss_function: Callable[[BatchState], Any],
            weight_function: Callable[[BatchState], float],
    ):
        step = LossStep(
            loss_name=loss_name,
            loss_function=loss_function,
            weight_function=weight_function
        )

        self._history.track_features(step.get_trackable_features())

        self._steps_repo.add_step(step)

    def add_metric(
            self,
            metric_name: str,
            metric_function: Callable[[BatchState], Any],
    ):
        step = MetricStep(
            metric_name=metric_name,
            metric_function=metric_function,
        )

        self._history.track_features(step.get_trackable_features())

        self._steps_repo.add_step(step)

    def add_forward(
            self,
            model_name: str,
            inputs: List[str],
            outputs: List[str],
    ):
        step = ForwardStep(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        self._history.track_features(step.get_trackable_features())

        self._steps_repo.add_step(step)

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
    def steps_repo(self) -> StepsRepo:
        return self._steps_repo

    @property
    def history(self) -> History:
        return self._history
