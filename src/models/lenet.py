from torch import nn


class ConvBNReLUPool(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        ]
        super(ConvBNReLUPool, self).__init__(*layers)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLUPool(1, 6, 5),
            ConvBNReLUPool(6, 16, 5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
