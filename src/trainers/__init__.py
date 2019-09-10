import sys

from ..utils import get_factory
from .trainer import Trainer

TrainerFactory = get_factory(sys.modules[__name__])
