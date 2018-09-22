import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data

from metrics import Accuracy, Average


class Trainer(object):

    def __init__(self,
                 net: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: data.DataLoader,
                 valid_loader: data.DataLoader,
                 device: torch.device,
                 output_dir: str,
                 lr_scheduler: optim.lr_scheduler._LRScheduler = None):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.output_dir = output_dir
        self.lr_scheduler = lr_scheduler

        self.device = device

    def fit(self, epochs: int):

        for epoch in range(1, epochs + 1):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.validate()

            print(
                'epoch: {}/{},'.format(epoch + 1, epochs),
                'train loss: {:.4f}, train acc: {:.2f}%,'.format(
                    train_loss, train_acc * 100),
                'valid loss: {:.4f}, valid acc: {:.2f}%'.format(
                    valid_loss, valid_acc * 100))

            f = os.path.join(self.output_dir,
                             'model_{:04d}.pt'.format(epoch + 1))
            self.save_weights(f)

    def train(self):
        self.net.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.net(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1)

            train_loss.update(loss.item())
            train_acc.update(pred, y)

        return train_loss.average, train_acc.accuracy

    def validate(self):
        self.net.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.net(x)
                loss = F.cross_entropy(output, y)

                pred = output.argmax(dim=1)

                valid_loss.update(loss.item(), x.size(0))
                valid_acc.update(pred, y)

        return valid_loss.average, valid_acc.accuracy

    def save_weights(self, f):
        state_dict = self.net.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()

        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save(state_dict, f)
