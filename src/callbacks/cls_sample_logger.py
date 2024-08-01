from typing import Mapping, Any, Union

import torch
import lightning as L

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class ClassificationSampleLogger(L.Callback):
    def __init__(
        self,
        train_epochs: int,
        train_batches: int,
        train_samples_per_batch: int,
        val_epochs: int,
        val_batches: int,
        val_samples_per_batch: int,
    ) -> None:
        super().__init__()
        self.train_epochs = train_epochs
        self.train_batches = train_batches or 0
        self.train_samples_per_batch = train_samples_per_batch
        self.val_epochs = val_epochs
        self.val_batches = val_batches or 0
        self.val_samples_per_batch = val_samples_per_batch

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if _WANDB_AVAILABLE:
            self.table = wandb.Table(columns=["input", "pred", "label"])

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: torch.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.val_batches <= batch_idx:
            return

        for i in range(min(self.val_samples_per_batch, len(batch["text"]))):
            self.table.add_data(
                batch["text"][i],
                outputs[i],
                batch["labels"][i],
            )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        trainer.logger.experiment.log({"val/samples": self.table})
