import argparse
import os

import numpy as np
import torch

from src.config import Config
from src.trainers import TrainerFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.json')
    return parser.parse_args()


def set_random_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    set_random_seed()

    config = Config(args.config_file)
    config.save(os.path.join(config['output_dir'], 'training.json'))

    trainer = TrainerFactory.create(**config)
    trainer.fit()

if __name__ == '__main__':
    main()
