import argparse
import os
from datetime import datetime

import torch
from torch import nn, optim

import datasets
import networks
import trainers
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.json')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    config = utils.load_json(args.config)

    model = networks.Net()
    if args.parallel:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), **config['adam'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, **config['steplr'])

    train_loader, valid_loader = datasets.mnist_loader(**config['dataset'])

    trainer = trainers.Trainer(model, optimizer, train_loader, valid_loader,
                               device)

    output_dir = os.path.join(config['output_dir'],
                              datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)

    # save config to output dir
    utils.save_json(config, os.path.join(output_dir, 'config.json'))

    for epoch in range(config['epochs']):
        scheduler.step()

        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.validate()

        print(
            'epoch: {}/{},'.format(epoch + 1, config['epochs']),
            'train loss: {:.4f}, train acc: {:.2f}%,'.format(
                train_loss, train_acc * 100),
            'valid loss: {:.4f}, valid acc: {:.2f}%'.format(
                valid_loss, valid_acc * 100))

        f = os.path.join(output_dir, 'model_{:04d}.pt'.format(epoch + 1))
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()
