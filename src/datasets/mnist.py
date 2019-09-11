import mlconfig
from torch.utils import data
from torchvision import datasets, transforms


@mlconfig.register
class MNISTDataloader(data.DataLoader):

    def __init__(self, root: str, train: bool, batch_size: int, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(root, train=train, transform=transform, download=True)

        super(MNISTDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)
