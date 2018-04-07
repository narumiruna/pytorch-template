import argparse

import torch
from torch import nn, optim

from datasets import mnist_loader
from models import CIFAR10Net
from trainers import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    model = CIFAR10Net()
    if args.cuda:
        if args.parallel:
            model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_loader, valid_loader = mnist_loader(args.root, args.batch_size)

    trainer = Trainer(model, optimizer, train_loader, valid_loader)

    for epoch in range(args.epochs):
        scheduler.step()

        train_loss, train_acc = trainer.train(epoch)
        valid_loss, valid_acc = trainer.validate()

        print('epoch: {}/{},'.format(epoch + 1, args.epochs),
              'train loss: {:.4f}, train acc: {:.2f}%,'.format(train_loss, train_acc * 100),
              'valid loss: {:.4f}, valid acc: {:.2f}%'.format(valid_loss, valid_acc * 100))


if __name__ == '__main__':
    main()
