import click
import wandb
from mlconfig import instantiate
from mlconfig import load


@click.command()
@click.option("-c", "--config-file", type=click.STRING, default="configs/mnist.yaml")
@click.option("-r", "--resume", type=click.STRING, default=None)
def main(config_file, resume):
    wandb.login()

    with wandb.init():
        wandb.save(config_file)

        config = load(config_file)
        wandb.config.update(dict(config.log_params))

        job = instantiate(config.job)
        job.run(config, resume)
