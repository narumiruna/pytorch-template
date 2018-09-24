from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
