from typing import Literal

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbAlert(L.Callback):
    def __init__(self, monitor: str, mode: Literal["min", "max"]) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.monitor_op = torch.lt if mode == "min" else torch.gt

    def on_train_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, WandbLogger):
            return
        if self.monitor_op(trainer.callback_metrics[self.monitor], self.best_metric):
            self.best_metric = trainer.callback_metrics[self.monitor]
            trainer.logger.experiment.alert(
                title="Metric improved",
                text=f"{self.monitor}={self.best_metric:.4f}\nval_loss={trainer.callback_metrics['val/loss']:.6f}\nepoch={trainer.current_epoch}",
                wait_duration=1,
            )
