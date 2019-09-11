import mlconfig
from torch import optim

mlconfig.register(optim.Adam)

mlconfig.register(optim.lr_scheduler.StepLR)
