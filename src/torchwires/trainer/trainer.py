from typing import Dict, Any, List, Literal

import torch

from ..callbacks.autosave_callback import AutoSaveCallback
from ..callbacks.callbacks_repo import CallbacksRepo
from ..callbacks.base_callback import BaseCallback
from ..callbacks.checkpoint_callback import CheckpointCallback
from ..callbacks.early_stopping_callback import EarlyStoppingCallback
from ..common.logger.logger import print_log
from ..constants.constants import LOADER_TRAIN_TYPE, LOADER_VAL_TYPE, LOADER_TEST_TYPE
from ..dataloaders_repo.dataloaders_repo import DataLoadersRepo
from ..display_widget.display_widget import DisplayWidget
from ..repo.repo import Repo
from ..state.batch_state import BatchState
from ..state.epoch_state import EpochState


class Trainer:
    def __init__(
            self,
            repo: Repo,
            device: torch.device | str
    ):
        self._repo = repo
        self._repo_name = repo.repo_name
        self._experiment = repo.experiment
        self._models_repo = repo.models_repo
        self._optimizers_repo = repo.optimizers_repo
        self._steps_repo = repo.steps_repo
        self._history = repo.history
        self._device = device

        self._base_callbacks: list[BaseCallback] = []

    def register_checkpoint_callback(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
    ):
        self._base_callbacks.append(
            CheckpointCallback(
                split=split,
                monitor=monitor,
                mode=mode,
                save_function=self._repo.save
            )
        )

    def register_auto_save_callback(
            self,
            interval: int,
            checkpoint_name: str,
            concat_with_epoch_no: bool = True,
    ):
        self._base_callbacks.append(
            AutoSaveCallback(
                interval=interval,
                checkpoint_name=checkpoint_name,
                save_function=self._repo.save,
                concat_with_epoch_no=concat_with_epoch_no
            )
        )

    def register_early_stopping_callback(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            patience: int,
    ):
        self._base_callbacks.append(
            EarlyStoppingCallback(
                split=split,
                monitor=monitor,
                mode=mode,
                patience=patience,
            )
        )

    def register_callback(
            self,
            callback: BaseCallback,
    ):
        self._base_callbacks.append(callback)

    def train(
            self,
            n_epochs: int,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            loader_output_keys: List[str],
            callbacks: List[BaseCallback] | None = None,
    ):
        data_loader_repo = DataLoadersRepo()
        data_loader_repo.attach_train_loader(train_loader)
        data_loader_repo.attach_val_loader(val_loader)
        data_loader_repo.attach_loader_output_keys(loader_output_keys)

        callbacks_repo = CallbacksRepo()
        callbacks_repo.register_callbacks(self._base_callbacks)
        if callbacks is not None:
            callbacks_repo.register_callbacks(callbacks)

        _all_completed_repos = self._history.get_feature_columns('epoch')
        start_epoch_idx = max(_all_completed_repos) if _all_completed_repos else 0

        print_log(
            title=f"Training {self._repo_name}",
            content=f"Starting training epoch {start_epoch_idx} -> {n_epochs} epochs",
        )

        self._history.track_features(BatchState.get_trackable_features())
        self._history.track_features(self._optimizers_repo.get_trackable_features())

        self._models_repo.to(self._device)

        display_widget = DisplayWidget(
            tracked_features=self._history.get_tracked_features()
        )

        callbacks_repo.notify_train_start()

        for epoch_idx in range(start_epoch_idx, n_epochs):
            epoch_state = EpochState()

            callbacks_repo.notify_epoch_start()

            self._train_epoch(
                epoch_idx=epoch_idx,
                max_epochs=n_epochs,
                display_widget=display_widget,
                epoch_state=epoch_state,
                data_loader_repo=data_loader_repo
            )

            self._eval_epoch(
                epoch_idx=epoch_idx,
                max_epochs=n_epochs,
                display_widget=display_widget,
                epoch_state=epoch_state,
                data_loader_repo=data_loader_repo
            )

            self._history.save(
                repo_name=self._repo_name,
                experiment=self._experiment,
            )

            display_widget.display_epoch_state(
                epoch_no=epoch_idx + 1,
                max_epochs=n_epochs,
                epoch_state=epoch_state
            )

            self._optimizers_repo.scheduler_step()

            # reset
            display_widget.reset()

            callbacks_repo.notify_epoch_end(epoch_state=epoch_state)

            print("\n")

            if callbacks_repo.should_stop_training():
                break

        callbacks_repo.notify_train_end()

        print_log(
            title=f"Training {self._repo_name}",
            content="Done",
        )

    def _train_epoch(
            self,
            epoch_idx: int,
            max_epochs: int,
            display_widget: DisplayWidget,
            epoch_state: EpochState,
            data_loader_repo: DataLoadersRepo
    ):
        self._models_repo.train()

        for batch_idx, batch in enumerate(data_loader_repo.get_train_loader()):
            # define state
            batch_state = BatchState(
                epoch_no=epoch_idx + 1,
                batch_no=batch_idx + 1,
                loader_type=LOADER_TRAIN_TYPE,
            )

            # move to device
            batch.to(device=self._device)

            # pass batch to state var
            batch_state.set_all(
                keys=data_loader_repo.get_loader_output_keys(),
                values=batch,
            )

            # reset optimizers
            self._optimizers_repo.zero_grad()

            # reset total loss
            batch_state.set(batch_state.KEY_TOTAL_LOSS, torch.tensor(0.0))

            # execute steps
            self._steps_repo.execute(
                models_repo=self._models_repo,
                state=batch_state,
            )

            # get final loss
            total_loss = batch_state.get(batch_state.KEY_TOTAL_LOSS)

            # back prob
            total_loss.backward()

            # optimizers step
            self._optimizers_repo.step(state=batch_state)

            # update epoch state
            epoch_state.add_batch_state(batch_state)

            # update history
            added_row = self._history.record(
                state_dict=batch_state.get_dict(),
            )

            # Display
            display_widget.display_batch_state(
                epoch_no=epoch_idx + 1,
                max_epochs=max_epochs,
                batch_no=batch_idx + 1,
                max_batch=len(data_loader_repo.get_train_loader()),
                batch_state=batch_state,
                split='train'
            )

    def _eval_epoch(
            self,
            epoch_idx: int,
            max_epochs: int,
            display_widget: DisplayWidget,
            epoch_state: EpochState,
            data_loader_repo: DataLoadersRepo
    ):
        self._models_repo.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader_repo.get_val_loader()):
                # define state
                batch_state = BatchState(
                    epoch_no=epoch_idx + 1,
                    batch_no=batch_idx + 1,
                    loader_type=LOADER_VAL_TYPE,
                )

                # move to device
                for i in range(len(batch)):
                    batch[i].to(self._device)

                # pass batch to state var
                batch_state.set_all(
                    keys=data_loader_repo.get_loader_output_keys(),
                    values=batch,
                )

                # reset total loss
                batch_state.set(batch_state.KEY_TOTAL_LOSS, torch.tensor(0.0))

                # execute steps
                self._steps_repo.execute(
                    models_repo=self._models_repo,
                    state=batch_state,
                )

                # update epoch state
                epoch_state.add_batch_state(batch_state)

                # update history
                self._history.record(
                    state_dict=batch_state.get_dict(),
                )

                # Display
                display_widget.display_batch_state(
                    epoch_no=epoch_idx + 1,
                    max_epochs=max_epochs,
                    batch_no=batch_idx + 1,
                    max_batch=len(data_loader_repo.get_val_loader()),
                    batch_state=batch_state,
                    split='val'
                )

    def predict(
            self,
            dataloader: torch.utils.data.DataLoader,
            loader_output_keys: List[str],
    ) -> List[Dict[str, Any]]:
        self._models_repo.eval()

        epoch_state = EpochState()

        display_widget = DisplayWidget(
            tracked_features=self._history.get_tracked_features()
        )

        self._models_repo.to(self._device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):

                # define state
                batch_state = BatchState(
                    epoch_no=1,  # 1 as placeholder
                    batch_no=batch_idx + 1,
                    loader_type=LOADER_TEST_TYPE,
                )

                # move to device
                for i in range(len(batch)):
                    batch[i].to(self._device)

                # pass batch to state var
                batch_state.set_all(
                    keys=loader_output_keys,
                    values=batch,
                )

                # reset total loss
                batch_state.set(batch_state.KEY_TOTAL_LOSS, torch.tensor(0.0))

                # execute steps
                self._steps_repo.execute(
                    models_repo=self._models_repo,
                    state=batch_state,
                )

                # update epoch state
                epoch_state.add_batch_state(batch_state)

                # Display
                display_widget.display_batch_state(
                    epoch_no=1,
                    max_epochs=1,
                    batch_no=batch_idx + 1,
                    max_batch=len(dataloader),
                    batch_state=batch_state,
                    split='test'
                )

            display_widget.display_epoch_state(
                epoch_state=epoch_state,
                epoch_no=1,
                max_epochs=1,
            )

        return [
            x.get_dict()
            for x in epoch_state.get_split('test')
        ]
