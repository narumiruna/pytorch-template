from torch.utils import data
from torchvision import datasets, transforms


class MNISTDataloader(data.DataLoader):

    def __init__(self, root: str, train: bool, batch_size: int, shuffle: bool):
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        dataset = datasets.MNIST(
            root,
            train=train,
            transform=transform,
            download=True,
        )

        super(MNISTDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def mnist(root='data', batch_size=128):
    train_loader = MNISTDataloader(root, train=True, batch_size=batch_size, shuffle=True)
    test_loader = MNISTDataloader(root, train=False, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
