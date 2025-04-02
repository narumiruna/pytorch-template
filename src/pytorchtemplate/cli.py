from typing import Annotated
from typing import Any

import typer
import wandb
from mlconfig import instantiate
from mlconfig import load
from omegaconf import OmegaConf


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


def run(
    config_file: Annotated[str, typer.Option("-c", "--config")] = "configs/mnist.yaml",
    resume: Annotated[str | None, typer.Option("-r", "--resume")] = None,
) -> None:
    wandb.login()

    with wandb.init(dir="./experiments"):
        wandb.save(config_file)

        config = load(config_file)
        wandb.config.update(flatten(OmegaConf.to_object(config)))

        job = instantiate(config.job)
        job.run(config, resume)


def main() -> None:
    typer.run(run)
