import torch
import torch.nn.functional as F
from mlconfig import register
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm
from tqdm import trange

import wandb

from .trainer import Trainer


@register
class MNISTTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        num_classes: int,
    ) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.best_acc = 0
        self.state = {"epoch": 1}

    def fit(self) -> None:
        for epoch in trange(self.state["epoch"], self.num_epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            self.scheduler.step()

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
            wandb.log(metrics, step=epoch)

            format_string = "Epoch: {}/{}, ".format(epoch, self.num_epochs)
            format_string += "train loss: {:.4f}, train acc: {:.4f}, ".format(train_loss, train_acc)
            format_string += "test loss: {:.4f}, test acc: {:.4f}, ".format(test_loss, test_acc)
            format_string += "best test acc: {:.4f}.".format(self.best_acc)
            tqdm.write(format_string)

            self.state["epoch"] = epoch

    def train(self) -> None:
        self.model.train()

        loss_metric = MeanMetric().to(self.device)
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        return loss_metric.compute().item(), acc_metric.compute().item()

    @torch.no_grad()
    def evaluate(self) -> None:
        self.model.eval()

        loss_metric = MeanMetric().to(self.device)
        acc_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

        for x, y in tqdm(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        test_acc = acc_metric.compute().item()
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_checkpoint("best.pth")

        return loss_metric.compute().item(), test_acc

    def save_checkpoint(self, f) -> None:
        self.model.eval()

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": self.state,
            "best_acc": self.best_acc,
        }

        torch.save(checkpoint, f)
        wandb.save(f)

    def resume(self, f) -> None:
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state = checkpoint["state"]
        self.best_acc = checkpoint["best_acc"]
