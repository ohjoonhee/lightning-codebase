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


import policy
import model


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = CLI(
        L.LightningModule,
        parser_kwargs={
            "parser_mode": "omegaconf",
        },
        subclass_mode_model=True,
    )


if __name__ == "__main__":
    cli_main()
