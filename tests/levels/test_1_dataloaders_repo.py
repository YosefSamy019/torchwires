import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.torchwires.dataloaders_repo.dataloaders_repo import DataLoadersRepo


class MyDataset(Dataset):
    def __init__(self):
        self.err = 0.1

    def __len__(self):
        return 30

    def __getitem__(self, index):
        x = torch.rand((10,))
        y = torch.sin(x) + self.err * torch.randn()
        return x, y


def test_dataloaders_repo():
    dataloader_repo = DataLoadersRepo()

    train_loader = DataLoader(
        MyDataset(),
        batch_size=2,
    )

    val_loader = DataLoader(
        MyDataset(),
        batch_size=2,
    )

    print("Attach train loader")
    dataloader_repo.attach_train_loader(train_loader)

    print("Attach val loader")
    dataloader_repo.attach_val_loader(val_loader)

    print("Get train loader")
    train_loader = dataloader_repo.get_train_loader()
    print(train_loader)

    print("Get val loader")
    val_loader = dataloader_repo.get_val_loader()
    print(val_loader)


if __name__ == "__main__":
    test_dataloaders_repo()
