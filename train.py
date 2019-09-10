import argparse

import numpy as np
import torch

from src.config import load_config
from src.trainers import TrainerFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.json')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    manual_seed()

    config = load_config(args.config)

    trainer = TrainerFactory.create(**config)

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()


if __name__ == '__main__':
    main()
