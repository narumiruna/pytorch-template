from torch import nn


class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv3x3, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.25),
        )

    def forward(self, input):
        return self.main(input)


class Conv5x5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv5x5, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.25),
        )

    def forward(self, input):
        return self.main(input)


class KaggleNet(nn.Module):
    def __init__(self):
        super(KaggleNet, self).__init__()

        self.conv = nn.Sequential(
            Conv3x3(1, 128),
            Conv3x3(128, 128),
            Conv3x3(128, 128),

            Conv5x5(128, 128),
            Conv3x3(128, 128),
            Conv3x3(128, 128),

            Conv5x5(128, 128),
            Conv3x3(128, 128),
            Conv3x3(128, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
