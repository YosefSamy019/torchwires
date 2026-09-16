# torchwires

[![PyPI](https://img.shields.io/pypi/v/torchwires.svg)](https://pypi.org/project/torchwires/)
[![Python](https://img.shields.io/pypi/pyversions/torchwires.svg)](https://pypi.org/project/torchwires/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`torchwires` is a lightweight experiment and training orchestration library for [PyTorch](https://pytorch.org/). It keeps ordinary PyTorch modules, optimizers, datasets, dataloaders, and tensors visible while providing a small set of abstractions for connecting model operations, losses, metrics, checkpointing, and history tracking.

The library is especially useful for small and medium-sized experiments where a conventional training loop is becoming difficult to organize. The included `debug/mnist` notebooks demonstrate an end-to-end MNIST classifier built with the same public API described below.

## Features

- Register PyTorch models and optimizers in a named experiment repository.

- Connect computation steps, or **wires**, that read from and write to a shared `PassState`.

- Train with dictionary batches from standard PyTorch `DataLoader` instances.

- Track selected losses and metrics in a JSON history file and inspect them as a pandas DataFrame.

- Save and restore model weights, optimizer state, and experiment history.

- Enable periodic autosaves, metric-based checkpointing, and early stopping.

- Visualize one repository or compare multiple repositories with matplotlib.

- Run inference through the same wired computation graph without gradients.

## Installation

`torchwires` requires Python 3.12 or newer. Install the published package with:

```bash
python -m pip install torchwires
```

The package declares these dependencies:

| Dependency | Role |
| --- | --- |
| `torch` | Models, tensors, optimizers, dataloaders, and automatic differentiation |
| `numpy` | Numerical aggregation and callback calculations |
| `pandas` | Training-history tables |
| `matplotlib` | Repository visualizations |

For a specific CPU or CUDA build of PyTorch, install that build first using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install `torchwires`.

To work from the repository instead of PyPI:

```bash
git clone https://github.com/YosefSamy019/torchwires.git
cd torchwires
python -m pip install -e .
```

## How the library is organized

A `Repo` owns the experiment configuration and persisted state. Models and optimizers are registered under names, while `Repo.wire( )` adds ordered computation nodes to the graph. A `Runner` executes those nodes for each batch.

```
DataLoader returns {"x": ..., "y": ...}
                 |
                 v
       PassState: x, y, epoch, batch, split, device
                 |
       model-pass wire: x -> y_hat
                 |
       loss wire: y_hat, y -> ce_loss
                 |
       metric wire: y_hat, y -> accuracy
                 |
       Observer: selected features -> history.json
```

Each wire is a Python function. Its parameters must have type annotations so `torchwires` can inject the matching registered model, optimizer repository, observer, or `PassState`. The most common pattern is to annotate a wire with a custom `PassState` subclass and return a dictionary of new state values.

## Quick start: a small classifier

The following example follows the pattern used in `debug/mnist/mnist_cls_r1.ipynb`. A dataset item must be a dictionary because `Runner.train()` injects each dictionary into a `PassState`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchwires import PassState, Repo, Runner


class ToyDataset(Dataset):
    def __init__(self, n=512):
        self.x = torch.randn(n, 4)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


class AppState(PassState):
    x = None
    y = None
    y_hat = None
    ce_loss = None
    accuracy = None


repo = Repo(repo_name="toy_classifier", load=False)
model = Classifier()
repo.register(name="classifier", value=model)
repo.register(
    name="optimizer",
    value=torch.optim.Adam(model.parameters(), lr=1e-3),
)


def model_pass(state: AppState):
    return {"y_hat": model(state.x)}


def cross_entropy(state: AppState):
    return {"ce_loss": F.cross_entropy(state.y_hat, state.y)}


def accuracy(state: AppState):
    value = state.y_hat.argmax(dim=1).eq(state.y).float().mean()
    return {"accuracy": value}


repo.wire(name="model-pass", body=model_pass)
repo.wire(name="ce-loss", body=cross_entropy, tracked_features=["ce_loss"])
repo.wire(name="accuracy", body=accuracy, tracked_features=["accuracy"])

runner = Runner(repo=repo)
runner.enable_checkpointing(
    split="val",
    monitor="ce_loss",
    aggregate_mode="mean",
    mode="min",
)

train_loader = DataLoader(ToyDataset(512), batch_size=32, shuffle=True)
val_loader = DataLoader(ToyDataset(128), batch_size=32)
runner.train(
    epochs=10,
    backprob_losses=["ce_loss"],
    device="cpu",
    train_loader=train_loader,
    val_loader=val_loader,
)
repo.save()
```

### What happens during a batch?

1. The runner creates a `PassState` containing `epoch`, `batch`, `split`, and `device`.

1. Values from the dictionary batch are added to that state, such as `x` and `y`.

1. Wires execute in registration order and merge their returned dictionaries into the state.

1. During training, each name in `backprob_losses` is backpropagated and every registered optimizer steps.

1. During validation and inference, the graph runs under `torch.no_grad()`.

1. The observer records the base fields and any features marked with `tracked_features`.

## MNIST example

The repository includes three notebooks under [`debug/mnist`](debug/mnist):

- [`mnist_cls_r1.ipynb`](debug/mnist/mnist_cls_r1.ipynb) — current dictionary-batch MNIST classifier example.

- [`mnist_cls_r0.ipynb`](debug/mnist/mnist_cls_r0.ipynb) — earlier iteration of the classifier example.

- [`mnist_cls compare.ipynb`](debug/mnist/mnist_cls%20compare.ipynb) — comparison and visualization experiments.

The current example wraps torchvision's `(image, label)` samples into dictionaries:

```python
class DictDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {"x": x, "y": y}
```

A minimal MNIST model and its wires look like this:

```python
class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 28 * 28, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        return self.fc2(self.fc1(self.flatten(x)))


cls_model = ClassifierModel()
repo.register(name="classifier-m", value=cls_model)
repo.register(
    name="opt-cls",
    value=torch.optim.Adam(cls_model.parameters(), lr=0.01),
)


def model_pass_node(state: AppState):
    return {"y_hat": cls_model(state.x)}


def ce_node(state: AppState):
    return {"ce_loss": F.cross_entropy(state.y_hat, state.y)}


def acc_node(state: AppState):
    return {
        "accuracy": state.y_hat.argmax(dim=1).eq(state.y).float().mean()
    }


repo.wire(name="model-pass", body=model_pass_node)
repo.wire(name="ce-node", body=ce_node, tracked_features=["ce_loss"])
repo.wire(name="acc-node", body=acc_node, tracked_features=["accuracy"])
```

Run the experiment with:

```python
runner = Runner(repo=repo)
runner.enable_checkpointing(
    split="val",
    monitor="ce_loss",
    aggregate_mode="mean",
    mode="min",
)
runner.train(
    epochs=30,
    backprob_losses=["ce_loss"],
    device="cpu",
    train_loader=train_loader,
    val_loader=val_loader,
)
```

## Checkpoints, autosaving, and early stopping

Checkpoints contain registered model state, optimizer state, and the observer history. Recreate the same model, optimizer, and wires before loading a checkpoint in a new process:

```python
repo = Repo(repo_name="toy_classifier", load=False)
# Register the same model, optimizer, and wires here.
repo.load(checkpoint="best")
```

The runner provides three built-in callback helpers:

```python
runner.enable_autosave(interval=2)

runner.enable_checkpointing(
    split="val",
    monitor="ce_loss",
    aggregate_mode="mean",  # "mean", "sum", or "last"
    mode="min",             # "min" for losses, "max" for metrics
)

runner.enable_early_stopping(
    split="val",
    monitor="accuracy",
    aggregate_mode="mean",
    mode="max",
    patience=3,
)
```

A new experiment should normally be created with `load=False`. If `load=True` is used (the default), the repository attempts to restore its existing files. Training also resumes from the last complete epoch recorded in the history.

## History and visualization

Pass `tracked_features` when registering a wire to record its outputs. The observer stores history in `<repo_name>/<experiment>/history.json` and exposes it as a pandas DataFrame:

```python
from torchwires import Visualizer

history_df = repo.observer.get_pandas()
print(history_df.head())

summary_df = Visualizer.display_df(repo=repo)
print(summary_df)

Visualizer.visualize_repo(repo=repo, n_cols=2)
```

To compare several experiments, pass a list of repositories:

```python
Visualizer.visualize_comparison([repo_a, repo_b])
```

## Inference

Inference accepts a dictionary of inputs and executes the same graph without backpropagation:

```python
sample = next(iter(val_loader))
state = runner.inference(
    device="cpu",
    inputs={"x": sample["x"], "y": sample["y"]},
)

print(state["y_hat"])
print(state.keys())
```

The returned object is the inference state's `__dict__`, so it contains the original inputs, generated values, and the standard state fields.

## Project layout

```
src/torchwires/
├── callbacks/       Autosave, checkpoint, early-stopping, and callback base classes
├── common/           Logging helpers
├── models/           Registered model wrappers
├── nodes/            Wired computation nodes
├── observer/         Feature tracking and JSON/pandas history
├── optimizers/       Registered optimizer and scheduler wrappers
├── pass_state/       Per-batch state container
├── repo/             Experiment façade
├── runner/           Training and inference execution
└── visualizer/       History plotting and comparison utilities

debug/mnist/          Executable MNIST notebooks
```

## API overview

| API | Purpose |
| --- | --- |
| `Repo(repo_name, load=True)` | Create or restore an experiment repository |
| `repo.register(name, value)` | Register a PyTorch module, optimizer, or scheduler |
| `repo.wire(name, body, ...)` | Add an ordered computation node |
| `Runner(repo)` | Create the training and inference runner |
| `runner.train(...)` | Execute training and optional validation |
| `runner.inference(...)` | Execute the graph without gradients |
| `repo.save(checkpoint="default")` | Save model, optimizer, and history state |
| `repo.load(checkpoint="default")` | Restore a previously saved checkpoint |
| `repo.observer.get_pandas()` | Return recorded history as a DataFrame |
| `Visualizer.visualize_repo(repo)` | Plot one repository's tracked features |
| `Visualizer.visualize_comparison(repos)` | Compare tracked features across repositories |

## Troubleshooting

| Symptom | Likely cause and solution |
| --- | --- |
| `Runner.train()` rejects a batch | The dataloader must return a dictionary, for example `{"x": image, "y": label}`. |
| A wire cannot resolve its argument | Add a type annotation and ensure the requested `PassState` or dependency is available. |
| A later wire cannot find a value | Register the producing wire earlier, or correct the state attribute name. |
| A new run unexpectedly restores files | Use a new `repo_name` or construct `Repo(..., load=False)`. |
| Checkpointing never triggers | Ensure `monitor` is tracked and use `mode="min"` for losses or `mode="max"` for higher-is-better metrics. |
| A loaded checkpoint fails in a new script | Recreate the same model, optimizer, and wire configuration before calling `load()`. |
| Training resumes at an unexpected epoch | The runner uses the saved history; remove or rename the experiment directory for a clean run. |

## Project status

Version `1.0.0` is an early public release. The API may evolve between releases. See [CHANGELOG.md](CHANGELOG.md) for release notes and the [GitHub repository](https://github.com/YosefSamy019/torchwires) for source code, issues, and releases.

## References

[1]: # "PyTorch documentation, official reference for modules, optimizers, datasets, dataloaders, and automatic differentiation."

[2]: # "torchwires repository, source code, metadata, issues, and releases."

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.