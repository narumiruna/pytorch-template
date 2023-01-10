import mlflow
import torch

from ..utils import instantiate
from ..utils import manual_seed
from ..utils import register


class Job:

    def run(self):
        raise NotImplementedError


@register
class MNISTTrainingJob(object):

    def run(self, config, resume=None):
        mlflow.log_params(config.log_params)

        manual_seed()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = instantiate(config.model).to(device)
        optimizer = instantiate(config.optimizer, model.parameters())
        scheduler = instantiate(config.scheduler, optimizer)
        train_loader = instantiate(config.dataset, train=True)
        test_loader = instantiate(config.dataset, train=False)

        trainer = instantiate(config.trainer, device, model, optimizer, scheduler, train_loader, test_loader)

        if resume is not None:
            trainer.resume(resume)

        trainer.fit()
