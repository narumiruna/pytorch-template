import mlflow
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average
from ..utils import get_logger
from .trainer import AbstractTrainer

LOGGER = get_logger(__name__)


class ClassificationTrainer(AbstractTrainer):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler, train_loader: data.DataLoader,
                 test_loader: data.DataLoader, num_epochs: int, output_dir: str, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = device

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.scheduler.step()
            self.train()
            self.evaluate()

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

        mlflow.log_metric('train loss', train_loss.value, step=self.epoch)
        mlflow.log_metric('train acc', train_acc.value, step=self.epoch)

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for x, y in tqdm(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                test_loss.update(loss.item(), number=x.size(0))
                test_acc.update(output, y)

        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_checkpoint()

        mlflow.log_metric('test loss', test_loss.value, step=self.epoch)
        mlflow.log_metric('test acc', test_acc.value, step=self.epoch)

    def save_checkpoint(self):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        f = 'checkpoint.pth'
        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
