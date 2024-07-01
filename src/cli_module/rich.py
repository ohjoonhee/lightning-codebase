import os
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
                "model_ckpt.monitor": "val_loss",
                "model_ckpt.mode": "min",
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "best-{epoch:02d}-{val_loss:.2f}",
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_loss",
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

        # add `--incremental_version` for sweep versioning, e.g. version_0, version_1, ...
        # This disables resume feature
        parser.add_argument("--increment_version", action="store_true", default=False)

    def _increment_version(self, save_dir: str, name: str) -> str:
        subcommand = self.config["subcommand"]
        if subcommand is None:
            return

        i = 0
        while osp.exists(osp.join(save_dir, name, f"version_{i}")):
            i += 1

        return f"version_{i}"

    def _update_model_ckpt_dirpath(self, logger_log_dir):
        if "subcommand" not in self.config:
            # ckpt_root_dirpath usually set with gs:// or s3://
            ckpt_root_dirpath = self.config["model_ckpt"]["dirpath"]
            if ckpt_root_dirpath:
                self.config["model_ckpt"]["dirpath"] = osp.join(
                    ckpt_root_dirpath, logger_log_dir, "checkpoints"
                )
            else:
                self.config["model_ckpt"]["dirpath"] = osp.join(
                    logger_log_dir, "checkpoints"
                )
            return

        subcommand = self.config["subcommand"]
        ckpt_root_dirpath = self.config[subcommand]["model_ckpt"]["dirpath"]
        if ckpt_root_dirpath:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                ckpt_root_dirpath, logger_log_dir, "checkpoints"
            )
        else:
            self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
                logger_log_dir, "checkpoints"
            )
