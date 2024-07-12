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
        vis_per_batch: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion
        self.vis_per_batch = vis_per_batch

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.net(x)

    def on_fit_start(self) -> None:
        # Note that self.logger is set by the Trainer.fit()
        # self.logger is None at self.__init__
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0
        if self.vis_per_batch:
            self.ID2CLS = [
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

    def training_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.criterion(pred, labels)
        self.log("train/loss", loss.item())

        return loss

    def on_validation_epoch_start(self) -> None:
        if self.vis_per_batch:
            self.table = wandb.Table(columns=["img", "label", "pred"])

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        loss = self.criterion(pred, labels)
        acc = self.accuracy(pred, labels)

        self.log_dict(
            {
                "val/loss": loss.item(),
                "val/acc": acc,
            },
            on_epoch=True,
            on_step=False,
        )

        if self.vis_per_batch:
            self.visualize_preds(img, labels, pred)

    def visualize_preds(self, img, labels, pred):
        for i in range(min(len(img), self.vis_per_batch)):
            self.table.add_data(
                wandb.Image(img[i].permute(1, 2, 0).cpu().numpy()),
                self.ID2CLS[labels[i].item()],
                self.ID2CLS[pred[i].argmax(-1).item()],
            )

    def on_validation_epoch_end(self) -> None:
        if self.vis_per_batch:
            self.logger.experiment.log({"val/samples": self.table})

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.accuracy(pred, labels)
        self.log("test/acc", acc)
