import argparse
import os

from src.core.config import Config
from src.core.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config-file', type=str, default='configs/mnist.json')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(args.config_file)
    config.save_config(os.path.join(args.output_dir, 'config.json'))

    runner = Runner(config)
    runner.run(output_dir=args.output_dir, use_cuda=True)


if __name__ == '__main__':
    main()
