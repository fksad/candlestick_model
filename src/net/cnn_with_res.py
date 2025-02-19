# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/19 17:15
# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/17 14:00
import torch.nn as nn


class StockCNNWithRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1),
                      dilation=(2, 1), padding=(12, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(3*60*256, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3),
                      stride=(3, 1), dilation=(2, 1), padding=(12, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 如果输入和输出的维度不一致，需要用 1x1 卷积调整维度
        self.proj_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(3, 1),
                          padding=(8, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation_layer = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        identity = self.proj_layer(x)
        out += identity
        out = self.activation_layer(out)
        return out
