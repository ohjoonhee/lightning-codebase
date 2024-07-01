import os
import os.path as osp
from pprint import pprint
from lightning.pytorch.loggers import WandbLogger
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ray.tune.integration.pytorch_lightning import TuneReportCallback

# from ray.tune.integration.wandb import WandbLoggerCallback

from main import cli_main


def convert_hparams_to_args(hparams: dict) -> list:
    args = []
    for key, value in hparams.items():
        args.append("--" + key + "=" + str(value))
    return args


def train_model(hparams: dict, metrics: dict):
    args = convert_hparams_to_args(hparams)
    config_path = "configs/config.yaml"
    trial_dir = osp.normpath(tune.get_trial_dir())
    trial_dir, version = osp.split(trial_dir)
    trial_dir, name = osp.split(trial_dir)

    args = [
        # "fit",
        f"--config={config_path}",
        f"--name={name}",
        f"--version={version}",
        "--trainer.enable_progress_bar=False",
    ] + args
    cli = cli_main(args, run=False)
    cli.model.configure_callbacks = lambda: TuneReportCallback(
        metrics=metrics, on="validation_end"
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    if isinstance(cli.trainer.logger, WandbLogger):
        cli.trainer.logger.experiment.finish()

    return


def raytune_main():
    hparams = {
        "optimizer.init_args.lr": tune.loguniform(1e-5, 1e-1),
    }

    metrics = {
        "loss": "val_loss",
        "acc": "val_acc",
    }

    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=list(hparams.keys()), metric_columns=list(metrics.keys())
    )

    resources_per_trial = {
        "cpu": 2,
        "gpu": 0.5,
    }

    train_fn = tune.with_parameters(train_model, metrics=metrics)

    result = tune.run(
        train_fn,
        resources_per_trial=resources_per_trial,
        metric="acc",
        mode="max",
        config=hparams,
        num_samples=2,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./logs",
        name="dev-raytune",
        chdir_to_trial_dir=False,
    )
    best_trial = result.get_best_trial("acc", "max", "all")
    pprint({metric: best_trial.last_result[metric] for metric in metrics.keys()})


if __name__ == "__main__":
    raytune_main()
