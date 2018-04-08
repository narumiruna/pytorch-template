from torch import nn

import torch.nn.functional as F


class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p=0.2):
        super(Conv3x3, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p),
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch),
        )

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = F.relu(out)
        out = self.bn2(out)
        return out


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()

        self.conv = nn.Sequential(
            Conv3x3(3, 32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),

            Conv3x3(32, 64, stride=2),
            # 16x16
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),

            Conv3x3(64, 128, stride=2),
            # 8x8
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            #Conv3x3(128, 256, stride=2),
            # 4x4
            #ResidualBlock(256),
            #ResidualBlock(256),
            #ResidualBlock(256),
            #ResidualBlock(256),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            #nn.Linear(2048, 1024),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),

            nn.Linear(1024, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
