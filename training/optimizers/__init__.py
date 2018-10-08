import sys
from torch import optim
from torch.optim import lr_scheduler


class OptimFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(optim, name)(*args, **kwargs)


class SchedulerFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(lr_scheduler, name)(*args, **kwargs)
