import os
import torch

import lightning as L

try:
    if os.environ.get("WANDB_DISABLED", False):
        raise ImportError
    import wandb
    from cli_module.rich_wandb import RichWandbCLI

    CLI = RichWandbCLI
except:
    from cli_module.rich import RichCLI

    CLI = RichCLI


import dataset
import model
import module
import transforms


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = CLI(
        L.LightningModule,
        L.LightningDataModule,
        parser_kwargs={
            "parser_mode": "omegaconf",
        },
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
