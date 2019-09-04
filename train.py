import argparse

import gin
import numpy as np
import torch

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.yaml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


@gin.configurable
def train(trainer):
    trainer.fit()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    manual_seed()

    gin.parse_config_file('config.gin')

    train()


if __name__ == '__main__':
    main()
