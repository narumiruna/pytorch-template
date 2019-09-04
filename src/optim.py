import gin
from torch import optim

gin.external_configurable(optim.Adam, module='Adam')
gin.external_configurable(optim.SGD, module='SGD')

gin.external_configurable(optim.lr_scheduler.StepLR, module='StepLr')
