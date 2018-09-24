from torch import optim
from torch.optim import lr_scheduler

from .. import datasets
from .. import trainers


class Factory(object):
    module = None
    module_name = None

    def create(self, name, *args, **kwargs):
        return getattr(self.module, name)(*args, **kwargs)

    def create_from_config(self, config, *args):
        if isinstance(config, dict):
            key, value = config[self.module_name].popitem()
            return self.create(key, *args, **value)
        else:
            raise TypeError


class OptimFactory(Factory):
    module = optim
    module_name = 'optim'


class LRSchedulerFactory(Factory):
    module = lr_scheduler
    module_name = 'lr_scheduler'


class DatasetFactory(Factory):
    module = datasets
    module_name = 'datasets'


class TrainerFactory(Factory):
    module = trainers
    module_name = 'trainers'
