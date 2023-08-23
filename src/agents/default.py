import torch
from torch import nn

import lightning as L

from torchmetrics import Accuracy


class LitCifar10(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_module: nn.Module,
        # metric_module: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_module", "metric_module"])

        self.net = net

        self.loss_module = loss_module
        self.metric_module = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_module(pred, labels)
        self.log("train/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.loss_module(pred, labels)
        self.log("val/loss", loss.item(), on_epoch=True, on_step=False)

        acc = self.metric_module(pred, labels)
        self.log("val/acc", acc, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.metric_module(pred, labels)
        self.log("test/acc", acc)
