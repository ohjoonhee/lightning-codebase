import torch
from torch import nn

import lightning as L

from torchmetrics import Accuracy


class DefaultModule(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net

        self.criterion = criterion
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.criterion(pred, labels)
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

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

        return

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.accuracy(pred, labels)
        self.log("test_acc", acc)
