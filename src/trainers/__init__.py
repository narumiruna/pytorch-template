import sys

from ..utils import get_factory
from .classification import ClassificationTrainer


class TrainerFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(sys.modules[__name__], name).from_config(*args, **kwargs)
