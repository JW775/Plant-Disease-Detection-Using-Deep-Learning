import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, use_1conv=False):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.ReLU(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.conv3 is not None:
            x = self.conv3(x)
        return self.ReLU(x + y)

# 你的完整模型
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(Residual(64, 64), Residual(64, 64))
        self.b3 = nn.Sequential(Residual(64, 128, stride=2, use_1conv=True), Residual(128, 128))
        self.b4 = nn.Sequential(Residual(128, 256, stride=2, use_1conv=True), Residual(256, 256))
        self.b5 = nn.Sequential(Residual(256, 512, stride=2, use_1conv=True), Residual(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
