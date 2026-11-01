import torch
import torch.nn as nn
import numpy as np

from modules.hsgroup import hsgconv


class SimpleNetwork(nn.Module):
    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, n_groups_luminance=1):
        super(SimpleNetwork, self).__init__()
        if n_groups_luminance != 1:
            raise ValueError("Small networks support n_groups_luminance=1 only; use a ResNet for luminance groups.")

        conv_1 = hsgconv.GroupConvHS(3, 32, 3, padding=1, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation)
        pool2d_1 = nn.MaxPool2d((2, 2))
        conv_2 = hsgconv.GroupConvHS(32, 64, 3, padding=1, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation)
        pool2d_2 = nn.MaxPool2d((2, 2))
        group_pool = hsgconv.GroupPool(n_groups_hue * n_groups_saturation)
        flatten = nn.Flatten()
        fc1 = nn.Linear(64*7*7, 128)
        fc2 = nn.Linear(128, num_classes)
        actv = nn.ReLU()
        finactv = nn.Softmax()
        self.network = nn.Sequential(conv_1, actv, pool2d_1, conv_2, actv, pool2d_2, group_pool, flatten, fc1, actv, fc2, finactv)

    def forward(self, x):
        return self.network(x)

class Z2CNN(nn.Module):
    """
    Derived from description given in https://arxiv.org/pdf/1602.07576.pdf
    """

    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, n_groups_luminance=1):
        super(Z2CNN, self).__init__()
        if n_groups_luminance != 1:
            raise ValueError("Small networks support n_groups_luminance=1 only; use a ResNet for luminance groups.")

        self.n_classes = num_classes
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        n_channels = int(20 / np.sqrt(n_groups_hue * n_groups_saturation))
        layers = [
            self.conv(3, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.max_pool((2,2)),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, self.n_classes, 4),
            self.batch_norm(self.n_classes),
            self.group_pool(),
            nn.Flatten(),
            nn.Softmax(),
        ]
        self.network = nn.Sequential(*layers)

    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return hsgconv.GroupConvHS(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            n_groups_hue=self.n_groups_hue,
            n_groups_saturation=self.n_groups_saturation
            )

    def batch_norm(self, channels):
        return hsgconv.GroupBatchNorm2d(
            channels,
            momentum=0.1,
            n_groups_hue=self.n_groups_hue,
            n_groups_saturation=self.n_groups_saturation)

    def max_pool(self, filter_size):
        return nn.MaxPool2d(filter_size)

    def group_pool(self):
        return hsgconv.GroupPool(self.n_groups_hue * self.n_groups_saturation)


    def forward(self, x):
        return self.network(x)


class HybridZ2CNN(nn.Module):
    """
    Derived from description given in https://arxiv.org/pdf/1602.07576.pdf
    """

    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, group_pool=False, n_groups_luminance=1):
        super(HybridZ2CNN, self).__init__()
        if n_groups_luminance != 1:
            raise ValueError("Small networks support n_groups_luminance=1 only; use a ResNet for luminance groups.")

        self.n_classes = num_classes
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        n_channels = int(20 / np.sqrt(n_groups_hue * n_groups_saturation))
        layers = [
            self.conv(3, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.group_pool() if group_pool else ExplodeGroupDim(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(
                n_channels if group_pool else n_channels * self.n_groups_hue * n_groups_saturation,
                n_channels,
                3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, self.n_classes, 4),
            nn.BatchNorm2d(self.n_classes),
            nn.Flatten(),
            nn.Softmax(),
        ]
        self.network = nn.Sequential(*layers)

    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return hsgconv.GroupConvHS(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            n_groups_hue=self.n_groups_hue,
            n_groups_saturation=self.n_groups_saturation
            )

    def batch_norm(self, channels):
        return hsgconv.GroupBatchNorm2d(
            channels,
            momentum=0.1,
            n_groups_hue=self.n_groups_hue,
            n_groups_saturation=self.n_groups_saturation)

    def max_pool(self, filter_size):
        return nn.MaxPool2d(filter_size)

    def group_pool(self):
        return hsgconv.GroupPool(self.n_groups_hue * self.n_groups_saturation)


    def forward(self, x):
        return self.network(x)

class ExplodeGroupDim(nn.Module):
    def __init__(self):
        super(ExplodeGroupDim, self).__init__()

    def forward(self, x):
        s = x.shape
        return x.view(s[0], -1, s[-2], s[-1])

if __name__ == "__main__":
    h = 4
    s = 3
    c = 10
    inputs = torch.rand((2, 3 * h * s, 28, 28))
    for network_class in (SimpleNetwork, Z2CNN, HybridZ2CNN):
        network = network_class(c, h, s)
        print(network_class.__name__, network(inputs).shape)
