import os.path as osp
import yaml

from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .rich import RichCLI


class RichTensorboardCLI(RichCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                    "init_args": {"save_dir": "logs"},
                },
            }
        )

    def _check_resume(self):
        subcommand = self.config["subcommand"]
        if subcommand != "fit":
            return

        if self.config[subcommand]["increment_version"]:
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
        with open(osp.join(prev_log_dir, "config.yaml"), "r") as f:
            prev_config = yaml.load(f, Loader=yaml.FullLoader)
        self.config[subcommand]["ckpt_path"] = osp.join(
            prev_config["model_ckpt"]["dirpath"], "last.ckpt"
        )

    @rank_zero_only
    def before_instantiate_classes(self) -> None:
        if "subcommand" not in self.config:
            return
        # Dividing directories into subcommand (e.g. fit, validate, test, etc...)
        subcommand = self.config["subcommand"]

        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config[subcommand]["trainer"]["logger"]["init_args"][
            "name"
        ] = self.config[subcommand]["name"]
        version = self.config[subcommand]["trainer"]["logger"]["init_args"][
            "version"
        ] = (
            self.config[subcommand]["version"]
            if not self.config[subcommand]["increment_version"]
            else self._increment_version(save_dir, name)
        )
        sub_dir = self.config[subcommand]["trainer"]["logger"]["init_args"][
            "sub_dir"
        ] = subcommand

        self._check_resume()

        log_dir = osp.join(save_dir, name, version, sub_dir)

        self._update_model_ckpt_dirpath(log_dir)
