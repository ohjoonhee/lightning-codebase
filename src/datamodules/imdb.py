from typing import Optional
from transformers import PreTrainedTokenizerBase

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import lightning as L

from transformers import AutoTokenizer


class HFIMDBDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        val_split: float,
        tokenizer: str,
        tokenizer_kwargs: Optional[dict] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)

    def prepare_data(self):
        """Only called from the main process for downloading dataset"""
        load_dataset("imdb", split="train", cache_dir=self.root)
        load_dataset("imdb", split="test", cache_dir=self.root)

    def setup(self, stage: str):
        if stage == "fit":
            ds = load_dataset("imdb", split="train", cache_dir=self.root)
            ds = ds.rename_column("label", "labels")
            ds = ds.map(
                lambda x: self.tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ),
                batched=True,
            )
            ds.set_format(type="torch", columns=ds.column_names)
            split_ds = ds.train_test_split(test_size=self.val_split)
            self.train_ds = split_ds["train"]
            self.val_ds = split_ds["test"]

        if stage == "test":
            ds = load_dataset("imdb", split="test", cache_dir=self.root)
            ds = ds.rename_column("label", "labels")
            ds = ds.map(
                lambda x: self.tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ),
                batched=True,
                remove_columns=["text"],
            )
            ds.set_format(type="torch", columns=ds.column_names)
            self.test_ds = ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
