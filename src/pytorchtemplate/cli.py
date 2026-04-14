from typing import Annotated
from typing import Any

import torch
import typer
import wandb
from mlconfig import instantiate
from mlconfig import load
from omegaconf import OmegaConf

from pytorchtemplate.utils import manual_seed

app = typer.Typer()


def flatten(data: Any, prefix: str | None = None, sep: str = ".") -> dict[str, Any]:
    d = {}

    for key, value in data.items():
        if prefix is not None:
            key = prefix + sep + key

        if isinstance(value, dict):
            d.update(flatten(value, prefix=key))
            continue

        d[key] = value

    return d


@app.command()
def train(
    config_file: Annotated[str, typer.Option("-c", "--config")] = "configs/mnist.yaml",
    resume: Annotated[str | None, typer.Option("-r", "--resume")] = None,
) -> None:
    wandb.login()

    with wandb.init(dir="./experiments"):
        wandb.save(config_file)

        config = load(config_file)
        wandb.config.update(flatten(OmegaConf.to_object(config)))

        manual_seed()

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = instantiate(config.model).to(device)
        optimizer = instantiate(config.optimizer, model.parameters())
        scheduler = instantiate(config.scheduler, optimizer)
        train_loader = instantiate(config.dataset, train=True)
        test_loader = instantiate(config.dataset, train=False)

        trainer = instantiate(
            config.trainer,
            device,
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
        )

        if resume is not None:
            trainer.resume(resume)

        trainer.fit()


def main() -> None:
    app()
