import os

import torch
import torch.nn.functional as F

from ..datasets import DatasetFactory
from ..metrics import Accuracy, Average
from ..models import ModelFactory
from ..optim import OptimFactory, SchedulerFactory
from ..utils import get_logger
from .trainer import AbstractTrainer

LOGGER = get_logger(__name__)


class ClassificationTrainer(AbstractTrainer):

    def __init__(self, model: dict, optimizer: dict, dataset: dict, scheduler: dict, use_cuda: bool, epochs: int,
                 output_dir: str):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.epochs = epochs
        self.output_dir = output_dir
        self.use_cuda = use_cuda
        self.start_epoch = 1
        self.best_acc = 0

        self._prepare(model, optimizer, dataset, scheduler)

    def _prepare(self, model, optimizer, dataset, scheduler):
        self.model = ModelFactory.create(**model)
        self.optimizer = OptimFactory.create(self.model.parameters(), **optimizer)
        self.train_loader, self.test_loader = DatasetFactory.create(**dataset)
        self.scheduler = SchedulerFactory.create(self.optimizer, **scheduler)

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.model.to(self.device)
        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()

        self.fit()

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.scheduler.step()

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            if test_acc.accuracy > self.best_acc:
                self.best_acc = test_acc.accuracy
                self.save_checkpoint(epoch)

            LOGGER.info('Epoch: %d/%d, train loss: %s, train acc: %s, test loss: %s, test acc: %s, best acc: %.2f.',
                        epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, self.best_acc * 100)

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1)

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(pred, y)

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                pred = output.argmax(dim=1)

                test_loss.update(loss.item(), number=x.size(0))
                test_acc.update(pred, y)

        return test_loss, test_acc

    def save_checkpoint(self, epoch):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, self.checkpoint_path)

    def restore_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
