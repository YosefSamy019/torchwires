import torch

from src.torchwires.models_repo.models_repo import ModelsRepo
from src.torchwires.optimizers_repo.optimizers_repo import OptimizersRepo


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


def test_optimizers_repo():
    models_repo = ModelsRepo()
    optimizers_repo = OptimizersRepo(models_repo=models_repo)

    m1 = models_repo.register_model(
        "m1",
        LinearStack()
    )

    # optimizers_repo.register_optimizer(
    #     "o1",
    #     torch.optim.Adam(params=m1.get_model().parameters(), ),
    # )

    o1 = optimizers_repo.register_optimizer(
        "m1",
        torch.optim.Adam(params=m1.get_model().parameters()),
    )

    o1.load_optimizer(
        repo_name="test_3_repo",
        experiment="test_3_experiment",
        checkpoint="test_3_checkpoint",
    )

    o1.save_optimizer(
        repo_name="test_3_repo",
        experiment="test_3_experiment",
        checkpoint="test_3_checkpoint",
    )

    o1.load_optimizer(
        repo_name="test_3_repo",
        experiment="test_3_experiment",
        checkpoint="test_3_checkpoint",
    )


if __name__ == "__main__":
    test_optimizers_repo()
