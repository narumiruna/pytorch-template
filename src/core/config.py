import argparse
import os

from ..trainers import TrainerFactory
from ..utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config-file', type=str, default='configs/mnist.json')
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')
    return parser.parse_args()


class ConfigLoader(object):

    def __init__(self):
        args = parse_args()
        self.config_file = args.config_file
        self.config = load_json(args.config_file)
        self.output_dir = args.output_dir

    def run(self):
        self.save_config()

        TrainerFactory.create(
            output_dir=self.output_dir, use_cuda=True, **self.config).run()

    def save_config(self):
        os.makedirs(self.output_dir, exist_ok=True)
        f = os.path.join(self.output_dir, os.path.basename(self.config_file))
        save_json(self.config, f, indent=4)
