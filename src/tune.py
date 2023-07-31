import os.path as osp

import torch

import lightning as L
from lightning.pytorch.tuner import Tuner

from cli_module.tune import TuneCLI


import dataset
import model
import module
import transforms


def cli_tune():
    torch.set_float32_matmul_precision("medium")

    cli = TuneCLI(
        L.LightningModule,
        L.LightningDataModule,
        parser_kwargs={
            "parser_mode": "omegaconf",
        },
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
    )

    tuner = Tuner(cli.trainer)
    setattr(cli.model, "lr", 0.0)
    lr_finder = tuner.lr_find(cli.model, datamodule=cli.datamodule)

    best_lr = lr_finder.suggestion()
    fig = lr_finder.plot(suggest=True)

    max_batch_sz = tuner.scale_batch_size(cli.model, datamodule=cli.datamodule)

    logger = cli.trainer.logger

    fig.savefig(osp.join(logger.save_dir, "lr_find.png"))
    with open(osp.join(logger.save_dir, "tune_results.txt"), "w") as f:
        f.write("Optimal LR is:\n")
        f.write(f"{best_lr}\n")

        f.write("\n")
        f.write("Max Batch Size is:\n")
        f.write(f"{max_batch_sz}\n")


if __name__ == "__main__":
    cli_tune()
