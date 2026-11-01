'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.hsgroup import hsgconv

import math



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv, bn, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                bn(self.expansion*planes)
            )


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv, bn, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = bn(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                bn(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, shapes = [64, 128, 256, 512], n_groups_hue=1, n_groups_saturation = 1, n_groups_luminance =1):
        super(ResNet, self).__init__()
        self.in_planes = shapes[0]
        self.in_planes = int(self.in_planes/math.sqrt(n_groups_hue * n_groups_saturation * n_groups_luminance))

        def groupconv(*args, **kwargs):
            return hsgconv.GroupConvHS(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance = n_groups_luminance, *args, **kwargs)
        conv = groupconv
        def groupbn(*args, **kwargs):
            return hsgconv.GroupBatchNorm2d(*args, **kwargs, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation, n_groups_luminance = n_groups_luminance)
            
        bn = groupbn
        shapes = [int(s/math.sqrt(n_groups_hue * n_groups_saturation* n_groups_luminance)) for _, s in enumerate(shapes)]

        # self.conv1 = conv(3, shapes[0], kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.conv1 = hsgconv.GroupConvHS(3, shapes[0], kernel_size=3, stride=1, padding=1, bias=False, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation, n_groups_luminance = n_groups_luminance)
        self.bn1 = bn(shapes[0])
        self.layers = [
            self._make_layer(block, shape, num_block, conv=conv, bn=bn, stride=1 if i == 0 else 2)
            for i, shape, num_block in zip(range(len(shapes)), shapes, num_blocks)
        ]
        self.layers = nn.Sequential(*self.layers)

        self.group_pool = hsgconv.GroupPool(n_groups_hue * n_groups_saturation* n_groups_luminance)

        self.linear = nn.Linear(shapes[-1]*block.expansion, num_classes)

        self.softmax = nn.Softmax(dim=1)
        

    def _make_layer(self, block, planes, num_blocks, conv, bn, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv=conv, bn=bn, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layers(out)

        s = out.shape
        out = out.view((s[0], -1, s[-2], s[-1]))
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(s[:-2]).unsqueeze(-1).unsqueeze(-1)
        out = self.group_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], shapes=[32, 64, 128], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

def parse_network_name(name):
    if name == "resnet18":
        return ResNet18
    elif name == "resnet34":
        return ResNet34
    elif name == "resnet44":
        return ResNet44
    elif name == "resnet50":
        return ResNet50
    elif name == "resnet101":
        return ResNet101
    elif name == "resnet152":
        return ResNet152
    else:
        raise ValueError("Network name not recognized")

# test()


#if __name__ == "__main__":
