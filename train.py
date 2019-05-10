import argparse
import os

import numpy as np
import torch

from src.config import Config
from src.trainers import TrainerFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.json')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda
    return args


def set_random_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    set_random_seed()

    config = Config(args.config_file)
    config['output_dir'] = args.output_dir
    config['use_cuda'] = args.use_cuda
    config.save(os.path.join(args.output_dir, 'training.json'))

    trainer = TrainerFactory.create(**config)
    trainer.run()


if __name__ == '__main__':
    main()
