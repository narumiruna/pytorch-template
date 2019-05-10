import sys

from ..utils import get_factory
from .classification import ImageClassificationTrainer

TrainerFactory = get_factory(sys.modules[__name__])
