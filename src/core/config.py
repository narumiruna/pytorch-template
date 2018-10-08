import argparse

from ..utils import load_json
from ..trainers import TrainerFactory


class ConfigLoader(object):

    def __init__(self):
        self.parse()
        self.config = load_json(self.args.config)

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c', '--config', type=str, default='configs/default.json')
        self.args = parser.parse_args()

    def run(self):
        TrainerFactory.create(**self.config)
