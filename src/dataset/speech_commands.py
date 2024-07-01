from typing import Union

import numpy as np

import torch
from torch.utils.data import Subset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms


class SpeechCommandsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        transforms: BaseTransforms,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.train_transform = transforms.train_transform()
        self.val_transform = transforms.val_transform()
        self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ID2CLS = [
            "no",
            "two",
            "backward",
            "four",
            "five",
            "nine",
            "right",
            "follow",
            "visual",
            "off",
            "yes",
            "six",
            "dog",
            "learn",
            "left",
            "bird",
            "forward",
            "wow",
            "zero",
            "eight",
            "bed",
            "go",
            "house",
            "tree",
            "seven",
            "on",
            "three",
            "one",
            "down",
            "stop",
            "up",
            "happy",
            "marvin",
            "cat",
            "sheila",
        ]
        self.CLS2ID = {cls: i for i, cls in enumerate(self.ID2CLS)}

    def prepare_data(self) -> None:
        SPEECHCOMMANDS(self.root, download=True)

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = SPEECHCOMMANDS(
                self.root, download=True, subset="training"
            )
            self.val_dataset = SPEECHCOMMANDS(
                self.root, download=True, subset="validation"
            )
        else:
            self.test_dataset = SPEECHCOMMANDS(
                self.root, download=True, subset="testing"
            )

    def _collate_fn(self, batch):
        tensors = [b[0].t() for b in batch if b]
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tensors = tensors.transpose(1, -1)

        # srs = [b[1] for b in batch if b]
        # srs = torch.tensor(srs)

        targets = torch.tensor([self.CLS2ID[b[2]] for b in batch if b])

        return tensors, targets

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
