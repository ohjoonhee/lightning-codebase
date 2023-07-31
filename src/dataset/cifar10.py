from typing import Union

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10


import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms


class Cifar10DataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        val_split: Union[int, float],
        # train_transform=None,
        # val_transform=None,
        # test_transform=None,
        transforms: BaseTransforms,
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.train_transform = transforms.train_transform()
        self.val_transform = transforms.val_transform()
        self.test_transform = transforms.test_transform()
        self.val_split = val_split
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        self.train_dataset = CIFAR10(
            self.root, train=True, download=True, transform=self.train_transform
        )
        self.val_dataset = CIFAR10(
            self.root, train=True, download=True, transform=self.val_transform
        )
        self.test_dataset = CIFAR10(
            self.root, train=False, download=True, transform=self.test_transform
        )

        indices = list(range(len(self.train_dataset)))
        targets = list(self.train_dataset.targets)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_split, random_state=0
        )

        train_indices, val_indices = next(sss.split(indices, targets))

        self.train_dataset = Subset(self.train_dataset, train_indices)
        self.val_dataset = Subset(self.val_dataset, val_indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, batch_size=100, shuffle=False, num_workers=8
        )
