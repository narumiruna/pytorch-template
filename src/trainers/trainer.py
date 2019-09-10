from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F

from ..datasets import DataFactory
from ..metrics import Accuracy, Average
from ..models import ModelFactory
from ..optim import OptimFactory, SchedulerFactory


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


class Trainer(AbstractTrainer):

    def __init__(self, model, optimizer, scheduler, dataset, num_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModelFactory.create(**model).to(self.device)
        self.optimizer = OptimFactory.create(self.model.parameters(), **optimizer)
        self.scheduler = SchedulerFactory.create(self.optimizer, **scheduler)
        self.train_loader = DataFactory.create(train=True, **dataset)
        self.test_loader = DataFactory.create(train=False, **dataset)
        self.num_epochs = num_epochs

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        for self.epoch in range(self.epoch, self.num_epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            self.scheduler.step()

            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, train acc: {}, '.format(train_loss, train_acc)
            format_string += 'test loss: {}, test acc: {}, '.format(test_loss, test_acc)
            format_string += 'best test acc: {}.'.format(self.best_acc)
            print(format_string)

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

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

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

                test_loss.update(loss.item(), number=x.size(0))
                test_acc.update(output, y)

        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_checkpoint()

        return test_loss, test_acc

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

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
