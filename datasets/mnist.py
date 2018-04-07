from torch.utils import data
from torchvision import datasets, transforms


def mnist_loader(root,
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
        transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
    ])

    train_loader = data.DataLoader(datasets.MNIST(root,
                                                  train=True,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle,
                                   num_workers=train_num_workers)

    valid_loader = data.DataLoader(datasets.MNIST(root,
                                                  train=False,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=valid_batch_size,
                                   shuffle=valid_shuffle,
                                   num_workers=valid_num_workers)

    return train_loader, valid_loader
