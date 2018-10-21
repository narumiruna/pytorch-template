from torch.utils import data
from torchvision import datasets, transforms


def mnist(root='data', batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = data.DataLoader(
        dataset=datasets.MNIST(
            root,
            train=True,
            transform=transform,
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST(
            root,
            train=False,
            transform=transform,
            download=True,
        ),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader
