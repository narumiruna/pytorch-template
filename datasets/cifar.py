from torch.utils import data
from torchvision import datasets, transforms


def cifar10_loader(root,
                   train_batch_size,
                   valid_batch_size=None,
                   train_shuffle=True,
                   valid_shuffle=False,
                   train_num_workers=0,
                   valid_num_workers=0):

    if valid_batch_size is None:
        valid_batch_size = train_batch_size

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_loader = data.DataLoader(datasets.CIFAR10(root,
                                                    train=True,
                                                    transform=transform,
                                                    download=True),
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle,
                                   num_workers=train_num_workers)

    valid_loader = data.DataLoader(datasets.CIFAR10(root,
                                                    train=False,
                                                    transform=valid_transform,
                                                    download=True),
                                   batch_size=valid_batch_size,
                                   shuffle=valid_shuffle,
                                   num_workers=valid_num_workers)

    return train_loader, valid_loader


def cifar100_loader(root,
                    train_batch_size,
                    valid_batch_size=None,
                    train_shuffle=True,
                    valid_shuffle=False,
                    train_num_workers=0,
                    valid_num_workers=0):

    if valid_batch_size is None:
        valid_batch_size = train_batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_loader = data.DataLoader(datasets.CIFAR10(root,
                                                    train=True,
                                                    transform=transform,
                                                    download=True),
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle,
                                   num_workers=train_num_workers)

    valid_loader = data.DataLoader(datasets.CIFAR10(root,
                                                    train=False,
                                                    transform=transform,
                                                    download=True),
                                   batch_size=valid_batch_size,
                                   shuffle=valid_shuffle,
                                   num_workers=valid_num_workers)

    return train_loader, valid_loader
