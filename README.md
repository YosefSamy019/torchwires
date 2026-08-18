# torchwires

[![PyPI](https://img.shields.io/pypi/v/torchwires.svg)](https://pypi.org/project/torchwires/)
[![Python](https://img.shields.io/pypi/pyversions/torchwires.svg)](https://pypi.org/project/torchwires/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**torchwires** is a lightweight training framework for [PyTorch](https://pytorch.org/). It lets you define a training workflow as named, composable steps: models consume named inputs, produce named outputs, and loss or metric functions read from the same batch state.

The package is designed to make small and medium-sized PyTorch experiments easier to organize without hiding the underlying PyTorch models, datasets, dataloaders, optimizers, or tensors.

## Why torchwires?

A conventional PyTorch training loop often mixes model execution, loss computation, metric reporting, checkpointing, and logging in one function. `torchwires` separates these concerns while keeping each individual operation close to ordinary PyTorch code.

The main abstraction is a repository-backed experiment. A `Repo` stores the registered models, optimizers, computation steps, and training history. A `Trainer` executes that experiment over training and validation dataloaders. Names such as `x`, `y`, and `y_hat` connect the steps together.

| Component | Responsibility |
| --- | --- |
| `Repo` | Defines, saves, and loads an experiment. |
| `Trainer` | Runs training, validation, prediction, and callbacks. |
| `BaseCallback` | Provides hooks for custom training-time behavior. |
| Forward step | Calls a registered model with named inputs and stores named outputs. |
| Loss step | Computes a differentiable objective and contributes it to the total loss. |
| Metric step | Computes and records a value for monitoring or analysis. |

## Installation

Install the latest published package from PyPI:

```bash
python -m pip install torchwires
```

The package declares the following dependencies:

| Dependency | Purpose |
| --- | --- |
| `torch` | Models, tensors, optimizers, and automatic differentiation. |
| `numpy` | Numerical operations used by the ecosystem. |
| `pandas` | History and tabular analysis support. |
| `matplotlib` | Visualization support. |

If you need a specific CPU or CUDA build of PyTorch, install that build first using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install `torchwires`.

## Quick start

The following example trains a small neural network to approximate `sin(x)`. It demonstrates the complete workflow: create data, register a model and optimizer, connect named steps, define a loss and metric, train, and save a checkpoint.

```python
import torch
from torch.utils.data import DataLoader, Dataset

from torchwires import Repo, Trainer


class SineDataset(Dataset):
    def __init__(self, n_samples=512):
        self.x = torch.linspace(-3.14, 3.14, n_samples).unsqueeze(1)
        self.y = torch.sin(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def mse_loss(state):
    return torch.nn.functional.mse_loss(
        state.get("y_hat"), state.get("y")
    )


def mean_absolute_error(state):
    return (state.get("y_hat") - state.get("y")).abs().mean()


train_loader = DataLoader(
    SineDataset(512), batch_size=32, shuffle=True
)
val_loader = DataLoader(SineDataset(128), batch_size=32)

model = Regressor()
repo = Repo(repo_name="sine_regression", load=False)

repo.register_model(name="regressor", model=model)
repo.register_optimizer(
    name="regressor",
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
)

# x -> regressor -> y_hat
repo.add_forward(
    model_name="regressor",
    inputs=["x"],
    outputs=["y_hat"],
)

repo.add_loss(loss_name="mse", loss_function=mse_loss)
repo.add_metric(
    metric_name="mae",
    metric_function=mean_absolute_error,
)

trainer = Trainer(repo=repo, device="cpu")
trainer.train(
    n_epochs=50,
    train_loader=train_loader,
    val_loader=val_loader,
    loader_output_keys=["x", "y"],
)

repo.save("final_model")
```

The `loader_output_keys` list maps batch positions to names. Because each dataset item returns `(x, y)`, the trainer assigns the first value to `x` and the second value to `y`. The names used by `add_forward`, `add_loss`, and `add_metric` must match those values or values produced by earlier steps.

## Core concepts

### 1. Repositories

A repository represents one experiment and defaults to the experiment name `exp_1`. For a new experiment, pass `load=False`; otherwise the repository attempts to restore previously saved state.

```python
from torchwires import Repo

new_repo = Repo(repo_name="my_experiment", load=False)
```

A repository can expose its registered model names and individual model nodes:

```python
print(new_repo.get_all_models_names())
model = new_repo.get_model_node("regressor").get_model()
```

### 2. Named forward steps

A forward step connects a registered model to named values in the batch state:

```python
repo.add_forward(
    model_name="regressor",
    inputs=["x"],
    outputs=["y_hat"],
)
```

For multiple inputs or outputs, provide matching lists. The model receives the values in the same order as `inputs`, and its returned values are stored using the names in `outputs`.

### 3. Losses

A loss function receives the current batch state. Retrieve values with `state.get("name")` and return a scalar tensor suitable for backpropagation.

```python
def mse_loss(state):
    prediction = state.get("y_hat")
    target = state.get("y")
    return torch.nn.functional.mse_loss(prediction, target)


repo.add_loss(
    loss_name="mse",
    loss_function=mse_loss,
)
```

Losses support an optional weight function. The effective loss is the returned loss multiplied by the weight.

```python
repo.add_loss(
    loss_name="regularized_objective",
    loss_function=mse_loss,
    weight_function=lambda state: 0.5,
)
```

### 4. Metrics

Metrics are also functions of the batch state, but they are recorded for monitoring and analysis rather than used directly for optimization.

```python
def mae(state):
    return (state.get("y_hat") - state.get("y")).abs().mean()


repo.add_metric(metric_name="mae", metric_function=mae)
```

### 5. Training and validation

The trainer requires both a training dataloader and a validation dataloader:

```python
trainer = Trainer(repo=repo, device="cpu")
trainer.train(
    n_epochs=20,
    train_loader=train_loader,
    val_loader=val_loader,
    loader_output_keys=["x", "y"],
)
```

During training, the framework switches models between training and evaluation modes, executes the configured steps, records history, and updates registered optimizers.

## Checkpoints and callbacks

`Trainer` includes built-in callbacks for checkpoints, periodic saves, and early stopping. Use `mode="min"` for a quantity such as validation loss and `mode="max"` for a quantity where larger values are better.

```python
trainer.register_checkpoint_callback(
    split="val",
    monitor="total_loss",
    mode="min",
)

trainer.register_auto_save_callback(
    interval=5,
    checkpoint_name="periodic",
    concat_with_epoch_no=True,
)

trainer.register_early_stopping_callback(
    split="val",
    monitor="total_loss",
    mode="min",
    patience=8,
)
```

You can also create custom callbacks by subclassing `BaseCallback`:

```python
from torchwires import BaseCallback


class ValidationLogger(BaseCallback):
    def on_epoch_end(self, epoch_state):
        value = epoch_state.aggregate_over_batches(
            feature="mae",
            split="val",
            func="mean",
        )
        print(f"validation MAE: {value:.5f}")


trainer.register_callback(ValidationLogger())
```

Available callback hooks are `on_train_start`, `on_train_end`, `on_epoch_start`, `on_epoch_end(epoch_state)`, and `should_stop_training()`.

## Saving and loading

The repository saves model state, optimizer state, and training history. Use the default checkpoint or provide a name:

```python
repo.save()
repo.save("final_model")
```

To restore a named checkpoint in a new process, recreate the same model, optimizer, and computation steps, then load the checkpoint:

```python
restored = Repo(repo_name="sine_regression", load=False)

# Register the same model, optimizer, and steps here.
restored.load("final_model")
```

Checkpoints restore state; they do not recreate Python class definitions or the experiment configuration automatically.

## Prediction

Prediction uses the same positional-to-name mapping as training and runs the registered models without gradient tracking:

```python
predictions = trainer.predict(
    dataloader=val_loader,
    loader_output_keys=["x", "y"],
)

print(type(predictions))
print(len(predictions))
```

Inspect an individual record with `print(predictions[0])` when exploring the returned state values.

## API reference

| API | Description |
| --- | --- |
| `Repo(repo_name, load=True)` | Creates or restores an experiment repository. |
| `repo.register_model(name, model)` | Registers a `torch.nn.Module`. |
| `repo.register_optimizer(name, optimizer)` | Registers an optimizer or scheduler. |
| `repo.add_forward(model_name, inputs, outputs)` | Adds a model execution step. |
| `repo.add_loss(loss_name, loss_function, weight_function)` | Adds a differentiable objective. |
| `repo.add_metric(metric_name, metric_function)` | Adds a tracked metric. |
| `Trainer(repo, device)` | Creates the training and prediction runner. |
| `trainer.train(...)` | Runs training and validation. |
| `trainer.predict(...)` | Evaluates batches without gradients. |
| `repo.save(checkpoint)` | Saves model, optimizer, and history state. |
| `repo.load(checkpoint)` | Loads a previously saved checkpoint. |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| The model cannot find an input | Ensure each name in `add_forward(inputs=...)` appears in `loader_output_keys` or was produced by an earlier step. |
| The loss cannot find a prediction | Ensure the forward output name matches the key read by the loss function. |
| A new experiment tries to load files | Construct it with `Repo(..., load=False)`. |
| A callback does not improve | Use `mode="min"` for losses and `mode="max"` for higher-is-better metrics, and verify the monitor name. |
| A checkpoint does not restore a new script | Register the same model, optimizer, and steps before calling `load`. |
| Training resumes unexpectedly | `train` consults saved history; use a new `repo_name` for a clean experiment. |

## Requirements and project status

`torchwires` currently targets Python 3.12 or newer. It is an early-stage project, so APIs may evolve between releases. The package source, issue tracker, and release information are available in the [GitHub repository](https://github.com/YosefSamy019/torchwires).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

## References

[1] [PyTorch documentation](https://pytorch.org/docs/stable/), official reference for modules, optimizers, datasets, dataloaders, and automatic differentiation.

[2] [torchwires repository](https://github.com/YosefSamy019/torchwires), source code, metadata, issues, and releases.
