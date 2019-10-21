from mlconfig import Config

_C = Config()
_C.trainer = Config()
_C.trainer.name = 'Trainer'
_C.trainer.num_epochs = 20

_C.dataset = Config()
_C.dataset.name = 'MNISTDataloader'
_C.dataset.root = 'data'
_C.dataset.batch_size = 256
_C.dataset.num_workers = 8

_C.model = Config()
_C.model.name = 'LeNet'

_C.optimizer = Config()
_C.optimizer.name = 'Adam'
_C.optimizer.lr = 1.e-3

_C.scheduler = Config()
_C.scheduler.name = 'StepLR'
_C.scheduler.step_size = 10
_C.scheduler.gamma = 0.1


def get_default_config():
    return _C
