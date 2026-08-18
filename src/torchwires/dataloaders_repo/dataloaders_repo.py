import torch.utils.data


class DataLoadersRepo:
    def __init__(self) -> None:
        self._train_loader: torch.utils.data.DataLoader | None = None
        self._val_loader: torch.utils.data.DataLoader | None = None

        self._loader_output_keys: list[str] = []

    def attach_train_loader(
            self, train_loader: torch.utils.data.DataLoader | None) -> None:
        self._train_loader = train_loader

    def attach_val_loader(self, val_loader: torch.utils.data.DataLoader | None) -> None:
        self._val_loader = val_loader

    def attach_loader_output_keys(self, loader_output_keys: list[str]) -> None:
        self._loader_output_keys = loader_output_keys

    def get_train_loader(self) -> torch.utils.data.DataLoader | None:
        return self._train_loader

    def get_val_loader(self) -> torch.utils.data.DataLoader | None:
        return self._val_loader

    def get_loader_output_keys(self) -> list[str]:
        return self._loader_output_keys
