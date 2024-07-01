from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint

import os
import os.path as osp

import yaml
import json
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .rich import RichCLI


class CleanUpWandbLogger(WandbLogger):
    def __init__(self, clean: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clean = clean
        self.api = wandb.Api()

    @rank_zero_only
    def _clean_model_artifacts(self) -> None:
        url = f"{self.experiment.entity}/{self.experiment.project}/{self.version}"
        run = self.api.run(url)
        model_artifacts = [
            artifact
            for artifact in run.logged_artifacts()
            if artifact.type == "model" and artifact.state == "COMMITTED"
        ]
        for artifact in model_artifacts:
            if not artifact.aliases:
                artifact.delete()

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().after_save_checkpoint(checkpoint_callback)
        if self._clean and self._log_model:
            self._clean_model_artifacts()

    def __del__(self):
        if self._clean and self._log_model:
            self._clean_model_artifacts()


class RichWandbCLI(RichCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.set_defaults(
            {
                "trainer.logger": {
                    "class_path": "cli_module.rich_wandb.CleanUpWandbLogger",
                    "init_args": {
                        "project": "debug",
                        "save_dir": "logs",
                        # "log_model": "all",
                        "clean": True,
                    },
                },
            }
        )

        parser.link_arguments("name", "trainer.logger.init_args.name")
        parser.link_arguments(
            "version", "trainer.logger.init_args.tags", compute_fn=lambda e: [e[:64]]
        )

    @rank_zero_only
    def before_fit(self):
        def _parse_config(s):
            try:
                return vars(s)
            except:
                if hasattr(s, "__repr__"):
                    return repr(s)
                elif hasattr(s, "__str__"):
                    return str(s)
                else:
                    return None

        if not isinstance(self.trainer.logger, WandbLogger):
            print("WandbLogger not found! Skipping config upload...")

        elif "subcommand" not in self.config:
            pass

        else:
            subcommand = self.config["subcommand"]
            dict_config = json.loads(
                json.dumps(self.config[subcommand], default=_parse_config)
            )
            self.trainer.logger.experiment.config.update(
                wandb.helper.parse_config(
                    dict_config,
                    exclude=(),  # exclude any configs (keys) for readability
                ),
                allow_val_change=True,
            )
            print("Config uploaded to Wandb!!!")

            self.trainer.logger.experiment.log_code("./src")
            print("Code artifacts uploaded to Wandb!!!")

    def _check_resume(self):
        subcommand = self.config["subcommand"]
        if subcommand != "fit":
            return subcommand

        if self.config[subcommand]["increment_version"]:
            return subcommand

        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config[subcommand]["name"]
        version = self.config[subcommand]["version"]
        sub_dir = subcommand

        log_dir = osp.join(save_dir, name, version, sub_dir)

        if not osp.exists(log_dir):
            return subcommand

        i = 1
        while osp.exists(osp.join(save_dir, name, version, f"{sub_dir}{i}")):
            i += 1

        prev_sub_dir = sub_dir + (str(i - 1) if (i - 1) else "")
        sub_dir = sub_dir + str(i)

        prev_log_dir = osp.join(save_dir, name, version, prev_sub_dir)
        with open(osp.join(prev_log_dir, "config.yaml"), "r") as f:
            prev_config = yaml.load(f, Loader=yaml.FullLoader)
        self.config[subcommand]["ckpt_path"] = osp.join(
            prev_config["model_ckpt"]["dirpath"], "last.ckpt"
        )

        wandb_run_file = [
            e
            for e in os.listdir(osp.join(prev_log_dir, "wandb", "latest-run"))
            if ".wandb" in e
        ]

        assert len(wandb_run_file) == 1

        wandb_run_file = wandb_run_file[0]
        wandb_run_id = wandb_run_file.split(".")[0][4:]

        self.config[subcommand]["trainer"]["logger"]["init_args"][
            "version"
        ] = wandb_run_id
        self.config[subcommand]["trainer"]["logger"]["init_args"]["resume"] = "allow"
        print("Resume logging to wandb run:", wandb_run_id)

        return sub_dir

    @rank_zero_only
    def before_instantiate_classes(self) -> None:
        if "subcommand" not in self.config:
            # Dividing directories into subcommand (e.g. fit, validate, test, etc...)
            save_dir = self.config["trainer"]["logger"]["init_args"]["save_dir"]
            name = self.config["name"]
            version = self.config["version"]

            # check if save_dir already contains name/version/subcommand
            if name in save_dir:
                assert version in save_dir

                paths = save_dir.split(os.sep)
                name_idx = paths.index(name)

                save_dir = osp.join(*paths[:name_idx])
                self.config["trainer"]["logger"]["init_args"]["save_dir"] = save_dir

            if self.config["increment_version"]:
                logger_tags = self.config["trainer"]["logger"]["init_args"]["tags"]
                assert version in logger_tags
                logger_tags.remove(version)
                version = self._increment_version(save_dir, name)
                logger_tags.append(version)

            # sub_dir = self._check_resume()
            sub_dir = "fit"

            log_dir = osp.join(save_dir, name, version, sub_dir)
            self.config["trainer"]["logger"]["init_args"]["save_dir"] = log_dir

            self._update_model_ckpt_dirpath(log_dir)

            # Making logger save_dir to prevent wandb using /tmp/wandb
            if not osp.exists(log_dir):
                os.makedirs(log_dir)

            # Specifying job_type of wandb.init() for quick grouping
            # self.config["trainer"]["logger"]["init_args"].tags.append(
            #     subcommand
            # )
            return

        # Dividing directories into subcommand (e.g. fit, validate, test, etc...)
        subcommand = self.config["subcommand"]
        save_dir = self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"]
        name = self.config[subcommand]["name"]
        version = self.config[subcommand]["version"]

        # check if save_dir already contains name/version/subcommand
        if name in save_dir:
            assert version in save_dir

            paths = save_dir.split(os.sep)
            name_idx = paths.index(name)

            save_dir = osp.join(*paths[:name_idx])
            self.config[subcommand]["trainer"]["logger"]["init_args"][
                "save_dir"
            ] = save_dir

        if self.config[subcommand]["increment_version"]:
            logger_tags = self.config[subcommand]["trainer"]["logger"]["init_args"][
                "tags"
            ]
            assert version in logger_tags
            logger_tags.remove(version)
            version = self._increment_version(save_dir, name)
            logger_tags.append(version)

        sub_dir = self._check_resume()

        log_dir = osp.join(save_dir, name, version, sub_dir)
        self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"] = log_dir

        self._update_model_ckpt_dirpath(log_dir)

        # Making logger save_dir to prevent wandb using /tmp/wandb
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        # Specifying job_type of wandb.init() for quick grouping
        self.config[subcommand]["trainer"]["logger"]["init_args"].tags.append(
            subcommand
        )
