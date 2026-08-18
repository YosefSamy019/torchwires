from typing import Dict, List, Any, Literal

from .base_state import BaseState
from .batch_state import BatchState


class EpochState(BaseState):
    def __init__(
            self,
    ):
        self._batch_states_per_split: Dict[str, List[BatchState]] = {}

    def add_batch_state(
            self,
            batch_state: BatchState
    ):
        split = batch_state.get(batch_state.KEY_LOADER_TYPE, must_exist=True)

        if split not in self._batch_states_per_split:
            self._batch_states_per_split[split] = []

        self._batch_states_per_split[split].append(batch_state)

    def get_all_splits(self) -> List[str]:
        return list(self._batch_states_per_split.keys())

    def aggregate_over_batches(
            self,
            feature: str,
            split: str,
            func: Literal["mean", "min", "max"],
    ) -> Any:
        vals = [
            row.get(feature)
            for row in self._batch_states_per_split[split]
        ]

        vals = list(filter(lambda x: x is not None, vals))

        if func == 'mean':
            if len(vals) > 0:
                return sum(vals) / (len(vals))
            else:
                return None

        if func == 'max':
            return max(vals)

        if func == 'min':
            return min(vals)

        raise ValueError(f'Unknown function `{func}`')

    def get_split(self, split:str)->List[BatchState]:
        return self._batch_states_per_split[split]