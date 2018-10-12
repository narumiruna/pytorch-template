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
                 network: dict,
                 optimizer: dict,
                 dataset: dict,
                 scheduler: dict,
                 use_cuda: bool = True,
                 output_dir: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   use_cuda else 'cpu')
        self.net = NetFactory.create(**network).to(self.device)
        self.optimizer = OptimFactory.create(self.net.parameters(), **optimizer)
        self.scheduler = SchedulerFactory.create(self.optimizer, **scheduler)
        self.train_loader, self.test_loader = DatasetFactory.create(**dataset)
        self.epochs = epochs
        self.output_dir = output_dir

    def run(self):
        self.fit()

    def fit(self):
        best_acc = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step()

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            print(
                'Training epoch: {}/{},'.format(epoch, self.epochs),
                'train loss: {:.6f}, train acc: {:.2f}%,'.format(
                    train_loss, train_acc * 100),
                'test loss: {:.6f}, test acc: {:.2f}%.'.format(
                    test_loss,
                    test_acc * 100), 'best acc: {:.2f}%'.format(best_acc * 100))

            if test_acc > best_acc:
                best_acc = test_acc

                f = os.path.join(self.output_dir, 'weights.pt')
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

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(pred, y)

        return train_loss.average, train_acc.accuracy

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

        return test_loss.average, test_acc.accuracy

    def save_weights(self, f):
        self.net.eval()

        state_dict = self.net.state_dict()

        for key, value in state_dict.items():
            state_dict[key] = value.cpu()

        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save(state_dict, f)
