import argparse
import os
from datetime import datetime

import torch
from torch import nn

from . import networks
from . import utils
from .common import DatasetFactory, LRSchedulerFactory, OptimFactory, TrainerFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.json')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    config = utils.load_json(args.config)

    output_dir = os.path.join('mnist', datetime.now().strftime('%Y%m%d_%H%M%S'))

    # save config to output dir
    utils.save_json(config, os.path.join(output_dir, 'config.json'))

    net = networks.Net()
    if args.parallel:
        net = nn.DataParallel(net)
    net.to(device)

    optimizer = OptimFactory().create_from_config(config, net.parameters())
    lr_scheduler = LRSchedulerFactory().create_from_config(config, optimizer)
    train_loader, valid_loader = DatasetFactory().create_from_config(config)

    trainer = TrainerFactory().create_from_config(
        config, net, optimizer, train_loader, valid_loader, device, output_dir,
        lr_scheduler)

    trainer.fit()


if __name__ == '__main__':
    main()
