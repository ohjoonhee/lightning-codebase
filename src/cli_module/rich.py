import os.path as osp

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)


class RichCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/loss",
                "model_ckpt.mode": "min",
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "best",
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val/loss",
                "early_stopping.mode": "min",
                "early_stopping.strict": False,
            }
        )

        # add `-n` argument linked with trainer.logger.name for easy cmdline access
        parser.add_argument(
            "--name", "-n", dest="name", action="store", default="default_name"
        )
        parser.add_argument(
            "--version", "-v", dest="version", action="store", default="version_0"
        )

    def _update_model_ckpt_dirpath(self, logger_log_dir):
        subcommand = self.config["subcommand"]
        if subcommand is None:
            return

        ckpt_root_dirpath = self.config[subcommand]["model_ckpt"]["dirpath"]
        if ckpt_root_dirpath:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                ckpt_root_dirpath, logger_log_dir, "checkpoints"
            )
        else:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                logger_log_dir, "checkpoints"
            )
