import gin
import mlflow
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average
from ..utils import get_logger
from .trainer import Trainer

LOGGER = get_logger(__name__)


@gin.configurable
class MNISTTrainer(Trainer):

    def __init__(self, model_cls, optimizer_cls, scheduler_cls, train_loader, test_loader, num_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_cls().to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters())
        self.scheduler = scheduler_cls(self.optimizer)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        self.epoch = 1
        self.best_acc = 0

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.scheduler.step()

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            metrics = dict(
                train_loss=train_loss.value,
                train_acc=train_acc.value,
                test_loss=test_loss.value,
                test_acc=test_acc.value,
            )
            mlflow.log_metrics(metrics, step=self.epoch)

            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, train acc: {}, '.format(train_loss, train_acc)
            format_string += 'test loss: {}, test acc: {}, '.format(test_loss, test_acc)
            format_string += 'best test acc: {}.'.format(self.best_acc)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        for x, y in tqdm(self.train_loader):
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
            for x, y in tqdm(self.test_loader):
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
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
