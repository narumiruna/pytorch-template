import click
import mlflow

from pytorch_template.utils import instantiate
from pytorch_template.utils import load


@click.command()
@click.option('-c', '--config-file', type=click.STRING, default='configs/mnist.yaml')
@click.option('-r', '--resume', type=click.STRING, default=None)
def main(config_file, resume):
    config = load(config_file)

    mlflow.log_artifact(config_file)

    job = instantiate(config.job)
    job.run(config, resume)


if __name__ == '__main__':
    main()
