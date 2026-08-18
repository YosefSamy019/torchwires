import torch

from src.torchwires.models_repo.models_repo import ModelsRepo
from src.torchwires.optimizers_repo.optimizers_repo import OptimizersRepo
from src.torchwires.state.state import BatchState
from src.torchwires.steps_repo.steps.forward_step import ForwardStep
from src.torchwires.steps_repo.steps.loss_step import LossStep
from src.torchwires.steps_repo.steps_repo import StepsRepo


class LinearStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def test_steps_repo():
    models_repo = ModelsRepo()
    optimizers_repo = OptimizersRepo(models_repo=models_repo)
    steps_repo = StepsRepo()
    state = BatchState()

    m1 = models_repo.register_model(
        "m1",
        LinearStack()
    )

    optimizers_repo.register_optimizer(
        "m1",
        torch.optim.Adam(params=m1.get_model().parameters()),
    )

    state.set('x', torch.randn(size=(4, 2)))
    state.set('y', torch.randn(size=(4, 2)))

    steps_repo.add_step(
        step=ForwardStep(
            model_name="m1",
            inputs=['x'],
            outputs=['y_hat'],
        )
    )

    steps_repo.add_step(
        step=LossStep(
            loss_name='sub',
            loss_function=lambda s: s['y'] - s['y_hat'],
            weight_function=lambda s: 1
        )
    )

    steps_repo.execute(
        models_repo=models_repo,
        state=state,
    )

    print(state)


if __name__ == "__main__":
    test_steps_repo()
