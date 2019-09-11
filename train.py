import argparse

import gin
import mlflow
import torch

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.ini')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


@gin.configurable
def train(trainer, resume=None):
    if resume is not None:
        trainer.resume(resume)

    trainer.fit()


@gin.configurable
def log_params(**params):
    mlflow.log_params(params)

def main():
    args = parse_args()
    gin.parse_config_file(args.config_file)

    torch.backends.cudnn.benchmark = True
    src.utils.manual_seed()
    log_params()

    train(resume=args.resume)


if __name__ == '__main__':
    main()
