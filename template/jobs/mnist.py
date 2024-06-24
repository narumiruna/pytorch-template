import torch
from mlconfig import instantiate
from mlconfig import register
from omegaconf import DictConfig

from ..utils import manual_seed
from .job import Job


@register
class MNISTTrainingJob(Job):
    def run(self, config: DictConfig, resume: str | None = None) -> None:
        manual_seed()

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = instantiate(config.model).to(device)
        optimizer = instantiate(config.optimizer, model.parameters())
        scheduler = instantiate(config.scheduler, optimizer)
        train_loader = instantiate(config.dataset, train=True)
        test_loader = instantiate(config.dataset, train=False)

        trainer = instantiate(
            config.trainer,
            device,
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
        )

        if resume is not None:
            trainer.resume(resume)

        trainer.fit()
