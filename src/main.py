import logging

logger = logging.getLogger(__name__)


import os
import torch
import lightning as L
from lightning.pytorch.cli import ArgsType
from argparse import ArgumentError

try:
    if os.environ.get("WANDB_DISABLED", False):
        raise ImportError
    import wandb
    from cli_module.rich_wandb import RichWandbCLI

    CLI = RichWandbCLI
except:
    from cli_module.rich_tensorboard import RichTensorboardCLI

    CLI = RichTensorboardCLI


import dataset
import net
import model
import transforms


def cli_main(args: ArgsType = None, run: bool = True):
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
            args=args,
            run=run,
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
                args=args,
                run=run,
            )
        except Exception as e:
            raise e from None
    except Exception as e:
        raise e

    return cli


if __name__ == "__main__":
    cli_main()
