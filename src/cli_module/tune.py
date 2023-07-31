import os.path as osp


from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser


class TuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "lightning.pytorch.loggers.CSVLogger",
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

    def before_instantiate_classes(self) -> None:
        assert "subcommand" not in self.config

        self.config["trainer"]["logger"]["init_args"]["name"] = self.config["name"]
        self.config["trainer"]["logger"]["init_args"]["version"] = self.config[
            "version"
        ]

        save_dir = self.config["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config["name"]
        version = self.config["version"]

        save_dir = osp.join(save_dir, name, version, "tune")
        self.config["trainer"]["logger"]["init_args"]["save_dir"] = save_dir
