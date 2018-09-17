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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **config['steplr'])

    train_loader, valid_loader = datasets.mnist_loader(**config['dataset'])

    output_dir = os.path.join(config['output_dir'],
                              datetime.now().strftime('%Y%m%d_%H%M%S'))

    trainer = trainers.Trainer(
        model,
        optimizer,
        train_loader,
        valid_loader,
        device,
        output_dir,
        lr_scheduler=lr_scheduler)

    # save config to output dir
    utils.save_json(config, os.path.join(output_dir, 'config.json'))

    trainer.fit(config['epochs'])


if __name__ == '__main__':
    main()
