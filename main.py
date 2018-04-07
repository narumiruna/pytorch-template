import argparse

import torch
from torch import optim, nn

from datasets import mnist_loader
from models import Net, KaggleNet
from trainers import Trainer
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    model = KaggleNet()
    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_loader, valid_loader = mnist_loader(args.root, args.batch_size, train_num_workers=2, valid_num_workers=2)

    trainer = Trainer(model, optimizer, train_loader, valid_loader)

    for epoch in range(args.epochs):
        scheduler.step()

        start = time()

        train_loss, train_acc = trainer.train(epoch)
        valid_loss, valid_acc = trainer.validate()

        print('epoch: {}/{},'.format(epoch + 1, args.epochs),
              'train loss: {:.4f}, train acc: {:.2f}%,'.format(train_loss, train_acc * 100),
              'valid loss: {:.4f}, valid acc: {:.2f}%,'.format(valid_loss, valid_acc * 100),
              'time: {:.2f}s'.format(time() - start))


if __name__ == '__main__':
    main()
