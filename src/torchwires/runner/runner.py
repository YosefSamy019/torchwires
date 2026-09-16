from typing import Any, Dict, Literal

import torch

from ..callbacks.autosave_callback import AutoSaveCallback
from ..callbacks.callbacks_repo import CallbacksRepo
from ..callbacks.checkpoint_callback import CheckpointCallback
from ..callbacks.early_stopping_callback import EarlyStoppingCallback
from ..common.logger.logger import print_log
from ..pass_state.pass_state import PassState
from ..repo.repo import Repo
from ..tqdm.tqdm import Tqdm


class Runner:
    def __init__(
            self,
            repo: Repo
    ):
        self._repo = repo
        self._models_repo = repo.models_repo
        self._optimizers_repo = repo.optimizers_repo
        self._observer = repo.observer

        self._callbacks_repo = CallbacksRepo()

        print_log(
            title="Runner has been initialized",
            content=f"last complete epoch={self._get_last_complete_epoch()}",
        )

    def enable_autosave(
            self,
            interval: int,
    ):
        self._callbacks_repo.register_callback(
            AutoSaveCallback(
                interval=interval,
                save_function=lambda checkpoint: self._repo.save(checkpoint=checkpoint),
            )
        )

    def enable_checkpointing(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            aggregate_mode: Literal["mean", "sum", "last"],
    ):
        self._callbacks_repo.register_callback(
            CheckpointCallback(
                save_function=lambda checkpoint: self._repo.save(checkpoint=checkpoint),
                split=split,
                monitor=monitor,
                mode=mode,
                aggregate_mode=aggregate_mode,
            )
        )

    def enable_early_stopping(
            self,
            split: Literal["train", "val"],
            monitor: str,
            mode: Literal["min", "max"],
            aggregate_mode: Literal["mean", "sum", "last"],
            patience: int,
    ):
        self._callbacks_repo.register_callback(
            EarlyStoppingCallback(
                split=split,
                monitor=monitor,
                mode=mode,
                aggregate_mode=aggregate_mode,
                patience=patience
            )
        )

    def _get_last_complete_epoch(self) -> int | None:
        last_complete_epoch = max([-1] + [x.get('epoch', -1) for x in self._observer.history_dict])

        if last_complete_epoch == -1:
            return None
        else:
            return last_complete_epoch

    def _run_batch(
            self,
            state: PassState,
            backprob_losses: list[str] | None = None,
    ) -> dict[str, Any]:
        # FLAGS
        flag_train = state.split in ['train']
        flag_inference = state.split in ['val', 'test']

        # check flags
        assert flag_train or flag_inference, ValueError(f"Unknown split {state.split}")
        assert not (len(backprob_losses or []) > 0 and flag_inference), ValueError(
            f"'backprob_losses' must be None in '{state.split}' Mode"
        )

        # set proper device
        self._models_repo.to(state.device)

        # configure models
        if flag_train:
            self._models_repo.train()

        if flag_inference:
            self._models_repo.eval()

        # reset optimizers
        if flag_train:
            self._optimizers_repo.zero_grad()

        if flag_train:
            self._flow_graph_back_prob_opt_step(
                state=state,
                backprob_losses=backprob_losses,
                flag_train=flag_train,
            )

        if flag_inference:
            with torch.no_grad():
                self._flow_graph_back_prob_opt_step(
                    state=state,
                    backprob_losses=backprob_losses,
                    flag_train=flag_train,
                )

        # observer
        new_record = self._observer.observe(state=state)

        # observer save
        self._observer.save(
            repo_name=self._repo.repo_name,
            experiment=self._repo.experiment,
            silent=True
        )

        return new_record

    def _flow_graph_back_prob_opt_step(
            self,
            state: PassState,
            backprob_losses: list[str] | None,
            flag_train: bool,
    ):
        # execute wires
        for node in self._repo.nodes_repo.nodes:
            intermediate_state = node.forward(state=state)

            state.update(intermediate_state)

        # back prob
        if flag_train:
            for backprob_loss in backprob_losses or []:
                state[backprob_loss].backward()

        # optimizer step
        if flag_train:
            self._optimizers_repo.step()

    def train(
            self,
            epochs: int,
            backprob_losses: list[str],
            device: str | torch.device,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader | None = None,
    ):
        tqdm = Tqdm()

        resume_epoch = (self._get_last_complete_epoch() or -1) + 1

        self._callbacks_repo.notify_epoch_start()

        if resume_epoch >= epochs:
            print_log(
                title="Nothing to train",
                content=f"last complete epoch={self._get_last_complete_epoch()}",
            )
            return
        else:
            print_log(
                title="Start training",
                content=f"from epoch={resume_epoch} to epoch={epochs}",
            )

        for epoch in range(resume_epoch, epochs):

            if self._callbacks_repo.should_stop_training():
                break

            tqdm.new_line()

            self._callbacks_repo.notify_epoch_start()

            for batch_idx, train_values in enumerate(train_loader):

                # TRAIN
                train_app_state = PassState(
                    epoch=epoch,
                    batch=batch_idx,
                    device=device,
                    split='train',
                )

                # inject train values
                if isinstance(train_values, dict):
                    train_app_state.update(train_values)
                else:
                    raise TypeError('train_loader must return a dict')

                new_record = self._run_batch(
                    train_app_state,
                    backprob_losses=backprob_losses
                )

                tqdm.update(new_record)

                self._callbacks_repo.notify_train_batch(
                    train_record=train_app_state.__dict__,
                )

            tqdm.new_line()

            if val_loader is not None:
                for batch_idx, val_values in enumerate(val_loader):

                    # VAL
                    val_app_state = PassState(
                        epoch=epoch,
                        batch=batch_idx,
                        device=device,
                        split='val',
                    )

                    # inject val values
                    if isinstance(val_values, dict):
                        val_app_state.update(val_values)
                    else:
                        raise TypeError('val_loader must return a dict')

                    new_record = self._run_batch(
                        val_app_state,
                        backprob_losses=None
                    )

                    tqdm.update(new_record)

                    self._callbacks_repo.notify_val_batch(
                        val_record=val_app_state.__dict__,
                    )

            # Epoch is done
            tqdm.new_line()
            self._callbacks_repo.notify_epoch_end()

        self._callbacks_repo.notify_train_end()

    def inference(
            self,
            device: str | torch.device,
            inputs: dict[str, Any],
    ) -> Dict[str, dict[str, Any]]:
        test_app_state = PassState(
            epoch=0,
            batch=0,
            device=device,
            split='test',
        )

        # inject test values
        if isinstance(inputs, dict):
            test_app_state.update(inputs)
        else:
            raise TypeError('inputs must return a dict')

        _ = self._run_batch(
            test_app_state,
            backprob_losses=None
        )

        return test_app_state.__dict__
