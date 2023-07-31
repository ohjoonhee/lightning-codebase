import os.path as osp

import torch

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser


class RichCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(RichProgressBar, "rich_progress")
        parser.set_defaults({"rich_progress.theme.progress_bar": "purple"})
        parser.add_lightning_class_args(ModelCheckpoint, "model_ckpt")
        parser.set_defaults(
            {
                "model_ckpt.monitor": "val/loss",
                "model_ckpt.mode": "min",
                "model_ckpt.save_last": True,
                "model_ckpt.filename": "best-{epoch:03d}",
            }
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({"lr_monitor.logging_interval": "epoch"})

        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {"save_dir": "logs"},
                },
            }
        )

        # add `-n` argument linked with trainer.logger.name for easy cmdline access
        parser.add_argument(
            "--name", "-n", dest="name", action="store", default="default_name"
        )
        parser.add_argument(
            "--version", "-v", dest="version", action="store", default="version_0"
        )

    def before_run(self):
        if hasattr(torch, "compile"):
            torch.compile(self.model)

    before_fit = before_validate = before_test = before_run

    def _check_resume(self):
        subcommand = self.config["subcommand"]
        if subcommand != "fit":
            return
        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config[subcommand]["name"]
        version = self.config[subcommand]["version"]
        sub_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["sub_dir"]

        log_dir = osp.join(save_dir, name, version, sub_dir)

        if not osp.exists(log_dir):
            return

        i = 1
        while osp.exists(osp.join(save_dir, name, version, f"{sub_dir}{i}")):
            i += 1

        prev_sub_dir = sub_dir + (str(i - 1) if (i - 1) else "")
        sub_dir = sub_dir + str(i)

        self.config[subcommand]["trainer"]["logger"]["init_args"]["sub_dir"] = sub_dir

        prev_log_dir = osp.join(save_dir, name, version, prev_sub_dir)
        self.config[subcommand]["ckpt_path"] = osp.join(
            prev_log_dir, "checkpoints", "last.ckpt"
        )

    def before_instantiate_classes(self) -> None:
        if "subcommand" not in self.config:
            return
        # Dividing directories into subcommand (e.g. fit, validate, test, etc...)
        subcommand = self.config["subcommand"]

        self.config[subcommand]["trainer"]["logger"]["init_args"]["name"] = self.config[
            subcommand
        ]["name"]
        self.config[subcommand]["trainer"]["logger"]["init_args"][
            "version"
        ] = self.config[subcommand]["version"]
        self.config[subcommand]["trainer"]["logger"]["init_args"][
            "sub_dir"
        ] = subcommand

        self._check_resume()

        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config[subcommand]["name"]
        version = self.config[subcommand]["version"]
        sub_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["sub_dir"]

        save_dir = osp.join(save_dir, name, version, sub_dir)

        self.config[subcommand]["model_ckpt"]["dirpath"] = osp.join(
            save_dir, "checkpoints"
        )
