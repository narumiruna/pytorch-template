import torch
import torch.nn.functional as F

import utils


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, valid_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

    def train(self):
        self.model.train()

        train_loss = utils.MovingAverageMeter()
        train_acc = utils.AccuracyMeter()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(float(loss.data))
            train_acc.update(output, y)

        return train_loss.average, train_acc.accuracy

    def validate(self):
        self.model.eval()

        valid_loss = utils.AverageMeter()
        valid_acc = utils.AccuracyMeter()

        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                valid_loss.update(float(loss.data), x.size(0))
                valid_acc.update(output, y)

        return valid_loss.average, valid_acc.accuracy
