from typing import Optional

import torch
from torch import nn

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import Accuracy

try:
    import wandb
finally:
    pass


class DefaultModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        vis_per_batch: int,
        vis_batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion
        self.vis_per_batch = vis_per_batch
        self.vis_batches = vis_batches if vis_batches is not None else float("inf")
        self.num_classes = num_classes

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.net(x)

    def on_fit_start(self) -> None:
        # Note that self.logger is set by the Trainer.fit()
        # self.logger is None at self.__init__
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0
        if self.vis_per_batch:
            if hasattr(self.trainer.datamodule, "ID2CLS"):
                self.ID2CLS = self.trainer.datamodule.ID2CLS
            else:
                self.ID2CLS = list(range(self.num_classes))

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        pred = self(waveforms)

        loss = self.criterion(pred, labels)
        self.log("train_loss", loss.item())

        return loss

    def on_validation_epoch_start(self) -> None:
        if self.vis_per_batch:
            self.table = wandb.Table(columns=["audio", "label", "pred"])

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch
        pred = self(waveforms)

        loss = self.criterion(pred, labels)
        acc = self.accuracy(pred, labels)

        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_acc": acc,
            },
            on_epoch=True,
            on_step=False,
        )

        if self.vis_per_batch and batch_idx < self.vis_batches:
            self.visualize_preds(waveforms, labels, pred)

    def visualize_preds(self, waveforms, labels, pred):
        for i in range(min(len(waveforms), self.vis_per_batch)):
            self.table.add_data(
                wandb.Audio(waveforms[i].squeeze().cpu().numpy(), sample_rate=16000),
                self.ID2CLS[labels[i].item()],
                self.ID2CLS[pred[i].argmax(-1).item()],
            )

    def on_validation_epoch_end(self) -> None:
        if self.vis_per_batch:
            self.logger.experiment.log({"val_samples": self.table})

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.accuracy(pred, labels)
        self.log("test_acc", acc)
