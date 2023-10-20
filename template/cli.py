import click
from mlconfig import instantiate
from mlconfig import load


@click.command()
@click.option("-c", "--config-file", type=click.STRING, default="configs/mnist.yaml")
@click.option("-r", "--resume", type=click.STRING, default=None)
def main(config_file, resume):
    config = load(config_file)

    job = instantiate(config.job)
    job.run(config, resume)
