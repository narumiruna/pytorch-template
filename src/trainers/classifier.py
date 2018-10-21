import os

import torch
import torch.nn.functional as F

from ..datasets import DatasetFactory
from ..metrics import Accuracy, Average
from ..networks import NetFactory
from ..optimizers import OptimFactory, SchedulerFactory


class ImageClassificationTrainer(object):

    def __init__(self,
                 epochs: int,
                 net: dict,
                 optimizer: dict,
                 dataset: dict,
                 scheduler: dict,
                 use_cuda: bool = True,
                 output_dir: str = None):
        train_loader, test_loader = DatasetFactory.create(**dataset)

        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   use_cuda else 'cpu')
        self.net = NetFactory.create(**net).to(self.device)
        self.optimizer = OptimFactory.create(self.net.parameters(), **optimizer)
        self.scheduler = SchedulerFactory.create(self.optimizer, **scheduler)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.output_dir = output_dir

        self.checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')

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
            test_loss, test_acc = self.test()

            if test_acc.accuracy > self.best_acc:
                self.best_acc = test_acc.accuracy
                self.save_checkpoint(epoch)

            print(
                'Training epoch: {}/{},'.format(epoch, self.epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {},'.format(test_loss, test_acc),
                'best acc: {:.2f}%.'.format(self.best_acc * 100))

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

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(pred, y)

        return train_loss, train_acc

    def test(self):
        self.net.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.net(x)
                loss = F.cross_entropy(output, y)

                pred = output.argmax(dim=1)

                test_loss.update(loss.item(), number=x.size(0))
                test_acc.update(pred, y)

        return test_loss, test_acc

    def save_checkpoint(self, epoch):
        self.net.eval()

        checkpoint = {
            'net': {k: v.cpu() for k, v in self.net.state_dict().items()},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, self.checkpoint_path)

    def restore_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)

        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
