import click
import mlflow
import torch

from pytorch_template.utils import instantiate
from pytorch_template.utils import load
from pytorch_template.utils import manual_seed


@click.command()
@click.option('-c', '--config-file', type=click.STRING, default='configs/mnist.yaml')
@click.option('-r', '--resume', type=click.STRING, default=None)
def main(config_file, resume):
    config = load(config_file)
    mlflow.log_artifact(config_file)
    mlflow.log_params(config.log_params)

    manual_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.scheduler, optimizer)
    train_loader = instantiate(config.dataset, train=True)
    test_loader = instantiate(config.dataset, train=False)

    trainer = instantiate(config.trainer, device, model, optimizer, scheduler, train_loader, test_loader)

    if resume is not None:
        trainer.resume(resume)

    trainer.fit()


if __name__ == '__main__':
    main()
