import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

from ..datasets import DatasetFactory
from ..metrics import Accuracy, Average
from ..networks import NetworkFactory
from ..optimizers import TorchOptimizerFactory


def evaluate(network: nn.Module, dataloader: data.DataLoader,
             device: torch.device):
    network.eval()

    eval_loss = Average()
    eval_acc = Accuracy()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = network(x)
            loss = F.cross_entropy(output, y)

            pred = output.argmax(dim=1)

            eval_loss.update(loss.item(), number=x.size(0))
            eval_acc.update(pred, y)

    return eval_loss.average, eval_acc.accuracy


class ImageClassificationTrainer(object):

    def __init__(self,
                 epochs: int,
                 network: dict,
                 optimizer: dict,
                 dataset: dict,
                 use_cuda: bool = True,
                 output_dir: str = None):
        train_dataloader, valid_dataloader, test_dataloader = DatasetFactory.create(
            **dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                   use_cuda else 'cpu')
        self.network = NetworkFactory.create(**network).to(self.device)
        self.optimizer = TorchOptimizerFactory.create(self.network.parameters(),
                                                      **optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.output_dir = output_dir

        self.train()

    def train(self):

        best_valid_acc = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch()
            valid_loss, valid_acc = self._valid_epoch()

            print(
                'Training epoch: {}/{},'.format(epoch, self.epochs),
                'train loss: {:.6f}, train acc: {:.2f}%,'.format(
                    train_loss, train_acc * 100),
                'valid loss: {:.6f}, valid acc: {:.2f}%.'.format(
                    valid_loss, valid_acc * 100))

            # TODO checkpoint
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc

                f = os.path.join(self.output_dir, 'weights.pt')
                self._save_weights(f)

        eval_loss, eval_acc = evaluate(self.network, self.test_dataloader,
                                       self.device)
        print('Test loss: {:.6f}, test acc: {:.2f}%.'.format(
            eval_loss, eval_acc * 100))

    def _train_epoch(self):
        self.network.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in self.train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.network(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1)

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(pred, y)

        return train_loss.average, train_acc.accuracy

    def _valid_epoch(self):
        return evaluate(self.network, self.valid_dataloader, self.device)

    def _save_weights(self, f):
        self.network.eval()

        state_dict = self.network.state_dict()

        for k, v in state_dict.items():
            state_dict[k] = v.cpu()

        os.makedirs(os.path.dirname(f), exist_ok=True)
        torch.save(state_dict, f)
