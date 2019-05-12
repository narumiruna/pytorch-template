import sys

from ..utils import get_factory
from .classification import ClassificationTrainer

TrainerFactory = get_factory(sys.modules[__name__])
