from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out
