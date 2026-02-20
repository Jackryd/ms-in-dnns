import os
import pathlib as pl

import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFAR10DataModule(L.LightningDataModule):
    CLASS_NAMES = CLASS_NAMES

    def __init__(self, data_root: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_root, train=True, download=True)
        datasets.CIFAR10(root=self.data_root, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=self.transform,
        )
        self.val_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=False,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()


def get_default_data_root() -> str:
    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        return str(pl.PurePosixPath("/gcs", bucket_name, "cifar10_data"))
    repo_root = pl.Path(__file__).resolve().parents[3]
    return str(repo_root / "data" / "cifar10")


def get_cifar10_dataloaders(
    data_root: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    if data_root is None:
        data_root = get_default_data_root()
    dm = CIFAR10DataModule(data_root=data_root, batch_size=batch_size, num_workers=num_workers)
    dm.prepare_data()
    dm.setup()
    return dm.train_dataloader(), dm.val_dataloader()
