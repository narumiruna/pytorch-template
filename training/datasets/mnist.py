from torch.utils import data
from torchvision import datasets, transforms


def mnist(root='data', download=False, batch_size=128, valid_ratio=0.1):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        root,
        train=True,
        transform=transform,
        download=download,
    )

    num_valid_samples = int(len(dataset) * valid_ratio)

    train_set, valid_set = data.random_split(
        dataset, [len(dataset) - num_valid_samples, num_valid_samples])

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    valid_loader = train_loader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False)

    test_loader = data.DataLoader(
        datasets.MNIST(
            root,
            train=False,
            transform=transform,
            download=download,
        ),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, valid_loader, test_loader
