from torch import optim

from .utils import register

register(optim.Adam)

register(optim.lr_scheduler.StepLR)
