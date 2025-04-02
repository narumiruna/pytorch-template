import click
import wandb
from mlconfig import instantiate
from mlconfig import load
from omegaconf import OmegaConf


def flatten(data: dict, prefix=None, sep="."):
    d = {}

    for key, value in data.items():
        if prefix is not None:
            key = prefix + sep + key

        if isinstance(value, dict):
            d.update(flatten(value, prefix=key))
            continue

        d[key] = value

    return d


@click.command()
@click.option("-c", "--config-file", type=click.STRING, default="configs/mnist.yaml")
@click.option("-r", "--resume", type=click.STRING, default=None)
def main(config_file, resume) -> None:
    wandb.login()

    with wandb.init(dir="./experiments"):
        wandb.save(config_file)

        config = load(config_file)
        wandb.config.update(flatten(OmegaConf.to_object(config)))

        job = instantiate(config.job)
        job.run(config, resume)
