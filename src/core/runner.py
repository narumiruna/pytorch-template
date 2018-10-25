from ..trainers import TrainerFactory
from .config import Config


class Runner(object):

    def __init__(self, config: Config):
        self.config = config

    def run(self, **kwargs):
        trainer = TrainerFactory.create(**self.config.data, **kwargs)
        trainer.run()
