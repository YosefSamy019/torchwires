from typing import Dict, Optional, List, Callable, Any

from ..common.logger.logger import print_log
from ..models.models_repo import ModelsRepo
from .node_unit import NodeUnit
from ..observer.observer import Observer
from ..optimizers.optimizers_repo import OptimizersRepo


class NodesRepo:
    def __init__(
            self,
    ):
        self._nodes: Dict[str, NodeUnit] = {}

    def wire(
            self,
            name: str,
            models_repo: ModelsRepo,
            optimizers_repo: OptimizersRepo,
            observer: Observer,
            body: Callable[..., dict[str, Any]],
            condition: Callable[..., bool] | None = lambda: True,
    ) -> NodeUnit:
        if name in self._nodes:
            raise KeyError(f"Node {name} already registered")

        node = NodeUnit(
            name=name,
            models_repo=models_repo,
            optimizers_repo=optimizers_repo,
            observer=observer,
            body=body,
            condition=condition,
        )

        self._nodes[name] = node

        print_log(
            title=f"Node {name}",
            content=f"has been registered",
        )

        return node

    def is_registered(
            self,
            node_name: str,
    ) -> bool:
        return node_name in self._nodes

    # def __getitem__(self, item) -> NodeUnit:
    #     print("KEY", item)
    #     return self._nodes[item]
    #

    @property
    def nodes(self) -> list[NodeUnit]:
        return [v for k, v in self._nodes.items()]
