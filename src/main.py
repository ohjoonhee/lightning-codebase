import logging

logger = logging.getLogger(__name__)


import os
import torch
import lightning as L
from argparse import ArgumentError

try:
    if os.environ.get("WANDB_DISABLED", False):
        raise ImportError
    import wandb
    from cli_module.rich_wandb import RichWandbCLI

    CLI = RichWandbCLI
except:
    from cli_module.rich_tensorboard import TensorboardCLI

    CLI = TensorboardCLI


import dataset
import model
import module
import transforms


def cli_main():
    torch.set_float32_matmul_precision("medium")

    try:
        cli = CLI(
            L.LightningModule,
            L.LightningDataModule,
            parser_kwargs={
                "parser_mode": "omegaconf",
                "exit_on_error": False,
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
    except ArgumentError as e:
        try:
            logger.warning(
                "DataModule is not implemented! Using only LightningModule..."
            )
            cli = CLI(
                L.LightningModule,
                parser_kwargs={
                    "parser_mode": "omegaconf",
                    "exit_on_error": False,
                },
                subclass_mode_model=True,
            )
        except Exception as e:
            raise e from None
    except Exception as e:
        raise e


if __name__ == "__main__":
    cli_main()
