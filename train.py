import argparse

import mlflow
import numpy as np
import torch

from src.config import Config
from src.datasets import DatasetFactory
from src.models import ModelFactory
from src.optim import OptimFactory, SchedulerFactory
from src.trainers import TrainerFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.yaml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def log_params(config):
    mlflow.log_param('num_epochs', config.trainer.num_epochs)
    mlflow.log_param('lr', config.optimizer.lr)
    mlflow.log_param('batch_size', config.dataset.batch_size)

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    manual_seed()

    config = Config.from_yaml(args.config_file)

    log_params(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModelFactory.create(**config.model).to(device)
    optimizer = OptimFactory.create(model.parameters(), **config.optimizer)
    scheduler = SchedulerFactory.create(optimizer, **config.scheduler)
    train_loader = DatasetFactory.create(train=True, **config.dataset)
    test_loader = DatasetFactory.create(train=False, **config.dataset)

    trainer = TrainerFactory.create(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device=device,
        **config.trainer,
    )

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()




if __name__ == '__main__':
    main()
