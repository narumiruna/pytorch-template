import os

import torch
import torch.nn.functional as F

from ..datasets import DatasetFactory
from ..metrics import Accuracy, Average
from ..models import ModelFactory
from ..optim import OptimFactory, SchedulerFactory
from ..utils import get_logger
from .trainer import Trainer

LOGGER = get_logger(__name__)


class ImageClassificationTrainer(Trainer):

    def __init__(self, epochs: int, model: dict, optimizer: dict, dataset: dict, scheduler: dict,
                 **kwargs):
        super(ImageClassificationTrainer, self).__init__(**kwargs)
        train_loader, test_loader = DatasetFactory.create(**dataset)
        self.model = ModelFactory.create(**model).to(self.device)
        self.optimizer = OptimFactory.create(self.model.parameters(), **optimizer)
        self.scheduler = SchedulerFactory.create(self.optimizer, **scheduler)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
        self.start_epoch = 1
        self.best_acc = 0

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

            LOGGER.info(
                'Epoch: %d/%d, '
                'train loss: %s, train acc: %s, '
                'test loss: %s, test acc: %s, '
                'best test acc: %.2f.',
                epoch,
                self.epochs,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                self.best_acc * 100,
            )

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

        eval_loss = Average()
        eval_acc = Accuracy()

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                pred = output.argmax(dim=1)

                eval_loss.update(loss.item(), number=x.size(0))
                eval_acc.update(pred, y)

        return eval_loss, eval_acc

    def save_checkpoint(self, epoch):
        self.model.eval()

        checkpoint = {
            'net': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, self.checkpoint_path)

    def restore_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)

        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
