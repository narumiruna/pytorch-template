from torch import optim
from torch.optim import lr_scheduler

from .. import datasets
from .. import trainers


class Factory(object):

    def __init__(self, module, module_name):
        self.module = module
        self.module_name = module_name

    def create(self, class_name, *args, **kwargs):
        return getattr(self.module, class_name)(*args, **kwargs)

    def create_from_config(self, config, *args):
        if isinstance(config, dict):
            key, value = config[self.module_name].popitem()
            return self.create(key, *args, **value)
        else:
            raise TypeError


class OptimFactory(Factory):
    module = optim
    module_name = 'optim'

    def __init__(self):
        pass


class LRSchedulerFactory(Factory):
    module = lr_scheduler
    module_name = 'lr_scheduler'

    def __init__(self):
        pass


class DatasetFactory(Factory):
    module = datasets
    module_name = 'datasets'

    def __init__(self):
        pass


class TrainerFactory(Factory):
    module = trainers
    module_name = 'trainers'

    def __init__(self):
        pass
