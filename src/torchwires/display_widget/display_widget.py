import time
from pickle import NONE
from typing import List

from ..state.batch_state import BatchState
from ..state.epoch_state import EpochState


class DisplayWidget:
    def __init__(
            self,
            tracked_features: List[str],
            n_columns: int = 4,
            columns_width: int = 35,
    ):
        self._tracked_features = tracked_features
        self._n_columns = n_columns
        self._columns_width = columns_width
        self._max_length_ever = 0

    def display_batch_state(
            self,
            split: str,
            epoch_no: int,
            max_epochs: int,
            batch_no: int,
            max_batch: int,
            batch_state: BatchState,
    ):
        # time.sleep(0.1)

        segments = [
            "\r"
            f"Epoch: {epoch_no}/{max_epochs} ({100 * epoch_no / max_epochs:5.1f} %)",
            f"Batch: {batch_no}/{max_batch} ({100 * batch_no / max_batch:5.1f} %)",
            f"Split: {split}",
            " | ",
        ]

        for k, v in batch_state.get_dict().items():
            if k not in self._tracked_features:
                continue

            if isinstance(v, float):
                segments.append(f"{k}: {v:3.5f}")
            else:
                segments.append(f"{k}: {v}")

            segments.append(f" - ")

        line_str = ' '.join(segments[:-1])
        self._max_length_ever = max(self._max_length_ever, len(line_str))
        print(line_str, end=' ' * 15)

    def display_epoch_state(
            self,
            epoch_no: int,
            max_epochs: int,
            epoch_state: EpochState,
    ):
        print(f"\rEpoch: {epoch_no}/{max_epochs}", end=' ' * self._max_length_ever)
        print()

        cell_i = 0

        for feature in self._tracked_features:
            for split in epoch_state.get_all_splits():
                if feature in [BatchState.KEY_LOADER_TYPE, BatchState.KEY_EPOCH_NO, BatchState.KEY_BATCH_NO]:
                    continue

                value = epoch_state.aggregate_over_batches(
                    feature=feature,
                    split=split,
                    func='mean',
                )

                if value is None:
                    val_str = f"{split}-{feature}: {None}"
                else:
                    val_str = f"{split}-{feature}: {value:3.5f}"

                val_str = val_str.ljust(self._columns_width)
                print(val_str, end=' ')

                if (cell_i + 1) % self._n_columns == 0:
                    print()
                else:
                    print(' | ', end='')

                cell_i += 1

    def reset(self):
        self._max_length_ever = 0
