import torch
from torch import nn

import lightning as L

from transformers import AutoModelForSequenceClassification
import evaluate

# TODO: Add various logging options or add logger checking


class HFClassificationModel(L.LightningModule):
    def __init__(
        self,
        net: str,
        net_kwargs: dict,
        criterion: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.net = AutoModelForSequenceClassification.from_pretrained(net, **net_kwargs)
        self.criterion = criterion

        self.accuracy = evaluate.load("accuracy")

    def forward(self, text, *args, **kwargs):
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        preds, loss, acc = self._get_step_output(batch)
        self.log_dict(
            {
                "train/loss": loss,
                "train/acc": acc,
            }
        )

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_step_output(batch)
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": acc,
            },
            on_epoch=True,
            on_step=False,
        )
        return preds

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_step_output(batch)
        self.log_dict(
            {
                "test/loss": loss,
                "test/acc": acc,
            },
            on_epoch=True,
            on_step=False,
        )

    def _get_step_output(self, batch):
        output = self(**batch)
        preds = torch.argmax(output.logits, dim=1)
        loss = output.loss.mean()
        acc = self.accuracy.compute(
            references=batch["labels"].data, predictions=preds.data
        )["accuracy"]
        return preds, loss, acc
