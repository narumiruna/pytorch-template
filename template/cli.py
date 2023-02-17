import click

from .utils import instantiate
from .utils import load_config


@click.command()
@click.option('-c', '--config-file', type=click.STRING, default='configs/mnist.yaml')
@click.option('-r', '--resume', type=click.STRING, default=None)
def main(config_file, resume):
    config = load_config(config_file)

    job = instantiate(config.job)
    job.run(config, resume)
