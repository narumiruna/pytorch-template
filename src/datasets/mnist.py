import gin
from torch.utils import data
from torchvision import datasets, transforms


@gin.configurable
class MNISTDataloader(data.DataLoader):

    def __init__(self, root: str, train: bool, batch_size: int, **kwargs):
        self.root = root
        self.train = train
        self.batch_size = batch_size

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(root, train=train, transform=transform, download=True)

        super(MNISTDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=train, **kwargs)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'root={}'.format(self.root)
        format_string += ', train={}'.format(self.train)
        format_string += ', batch_size={}'.format(self.batch_size)
        format_string += ')'
        return format_string
