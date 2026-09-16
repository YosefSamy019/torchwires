from typing import Callable

from inspect import signature
from typing import Any, get_type_hints

from ..models.models_repo import ModelsRepo
from ..observer.observer import Observer
from ..optimizers.optimizers_repo import OptimizersRepo
from ..pass_state.pass_state import PassState


class NodeUnit:
    def __init__(
            self,
            name: str,
            models_repo: ModelsRepo,
            optimizers_repo: OptimizersRepo,
            observer: Observer,
            body: Callable[..., dict[str, Any]],
            condition: Callable[..., bool] = lambda: True,
    ):
        self._name = name
        self._condition = condition
        self._body = body
        self._models_repo: ModelsRepo = models_repo
        self._optimizers_repo: OptimizersRepo = optimizers_repo
        self._observer: Observer = observer

    @property
    def name(self) -> str:
        return self._name

    @property
    def models(self) -> ModelsRepo:
        return self._models_repo

    @property
    def optimizers(self) -> OptimizersRepo:
        return self._optimizers_repo

    @property
    def observer(self) -> Observer:
        return self._observer

    def forward(self, state: PassState) -> dict[str, Any]:
        out_condition = self._dynamic_call(
            state=state,
            fun=self._condition,
        )

        if out_condition:
            out_val = self._dynamic_call(
                state=state,
                fun=self._body,
            )

            return out_val

        else:
            return {}

    def _dynamic_call(
            self,
            state: PassState,
            fun: Callable[..., Any],
    ) -> Any:
        available = [
            state,
            self.models,
            self.optimizers,
            self.observer,
        ]

        hints = get_type_hints(fun)
        params = signature(fun).parameters

        args = []

        for name, param in params.items():
            if name not in hints:
                raise TypeError(
                    f"Parameter '{name}' must have a type annotation"
                )

            expected_type = hints[name]

            for value in available:
                if isinstance(value, expected_type) or issubclass(expected_type, type(value)):
                    args.append(value)
                    break
            else:
                raise TypeError(
                    f"No dependency found for parameter "
                    f"'{name}: {expected_type}'"
                )

        return fun(*args)
