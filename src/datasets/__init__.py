import sys

from ..utils import get_factory
from .mnist import MNISTDataloader

DatasetFactory = get_factory(sys.modules[__name__])
