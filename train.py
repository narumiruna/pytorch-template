import argparse

import numpy as np
import torch

from src.config import Config
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


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    manual_seed()

    config = Config.from_yaml(args.config_file)

    trainer = TrainerFactory.create(**config)

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()


if __name__ == '__main__':
    main()
