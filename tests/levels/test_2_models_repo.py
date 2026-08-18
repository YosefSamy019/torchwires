import os

import torch

from src.torchwires.models_repo.models_repo import ModelsRepo


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


def test_models_repo():
    models_repo = ModelsRepo()

    m1 = models_repo.register_model(
        "m1",
        LinearStack()
    )

    # models_repo.register_model(
    #     "m1",
    #     LinearStack()
    # )

    m2 = models_repo.register_model(
        "m2",
        LinearStack()
    )

    m1.load_model(
        repo_name="test_2_repo",
        experiment="test_2_experiment",
        checkpoint="test_2_checkpoint",
    )

    m1.save_model(
        repo_name="test_2_repo",
        experiment="test_2_experiment",
        checkpoint="test_2_checkpoint",
    )

    m1.load_model(
        repo_name="test_2_repo",
        experiment="test_2_experiment",
        checkpoint="test_2_checkpoint",
    )

    m1.to("cpu")


if __name__ == "__main__":
    test_models_repo()
