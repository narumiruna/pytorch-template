import torch.nn.functional as F
from torch.autograd import Variable

from utils import AccuracyMeter, AverageMeter, MovingAverageMeter


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, valid_loader, use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.use_cuda = use_cuda

    def train(self, epoch):
        self.model.train()

        train_loss = MovingAverageMeter()
        train_acc = AccuracyMeter()

        for i, (x, y) in enumerate(self.train_loader):
            x = Variable(x)
            y = Variable(y)

            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(float(loss.data))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            train_acc.update(correct, x.size(0))

        return train_loss.average, train_acc.accuracy

    def validate(self):
        self.model.eval()

        valid_loss = AverageMeter()
        valid_acc = AccuracyMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x = Variable(x, volatile=True)
            y = Variable(y)

            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            valid_loss.update(float(loss.data), x.size(0))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            valid_acc.update(correct, x.size(0))

        return valid_loss.average, valid_acc.accuracy
