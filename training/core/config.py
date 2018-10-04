import argparse

from ..utils import load_json
from ..trainers import TrainerFactory


class ConfigLoader(object):

    def __init__(self):
        self._parser_args()
        self._config = load_json(self._args.config)

    def _parser_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-c', '--config', type=str, default='configs/default.json')
        self._args = parser.parse_args()

    def run(self):
        for trainer in self._config['trainers']:
            TrainerFactory.create(**trainer)
