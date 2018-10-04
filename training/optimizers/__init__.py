import sys
from torch import optim


class TorchOptimizerFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(optim, name)(*args, **kwargs)
