import argparse
import os

from .core.config import Config
from .core.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default='configs/mnist.json')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')
    return parser.parse_args()

def main():
    args = parse_args()

    config = Config()
    config.load_config(args.config_file)
    f = os.path.join(args.output_dir, config.basename)
    config.save_config(f)

    runner = Runner(config)
    runner.run(output_dir=args.output_dir, use_cuda=True)


if __name__ == '__main__':
    main()
