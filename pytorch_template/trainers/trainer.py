import mlflow
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics import MeanMetric
from tqdm import tqdm
from tqdm import trange

from ..utils import register


@register
class Trainer(object):

    def __init__(self, device, model, optimizer, scheduler, train_loader, test_loader, num_epochs):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            self.scheduler.step()

            metrics = dict(train_loss=train_loss.value,
                           train_acc=train_acc.value,
                           test_loss=test_loss.value,
                           test_acc=test_acc.value)
            mlflow.log_metrics(metrics, step=self.epoch)

            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, train acc: {}, '.format(train_loss, train_acc)
            format_string += 'test loss: {}, test acc: {}, '.format(test_loss, test_acc)
            format_string += 'best test acc: {}.'.format(self.best_acc)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        loss_metric = MeanMetric()
        acc_metric = Accuracy()

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        return loss_metric.compute().item(), acc_metric.compute().item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        loss_metric = MeanMetric()
        acc_metric = Accuracy()

        for x, y in tqdm(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            loss_metric.update(loss, weight=x.size(0))
            acc_metric.update(output, y)

        test_acc = acc_metric.compute().item()
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.save_checkpoint('best.pth')

        return loss_metric.compute().item(), test_acc

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
