import sys

from ..utils import get_factory
from .lenet import LeNet

ModelFactory = get_factory(sys.modules[__name__])
