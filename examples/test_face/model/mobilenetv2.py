'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Block', 'MBlock', 'MobileNetV2', 'Modified_MobileNetV2', 'mobilenetv2']


class MBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride, modified1, modified2):
        super(MBlock, self).__init__()
        self.stride = stride
        self.modified1 = modified1
        self.modified2 = modified2
        self.in_planes, self.out_planes, self.expansion = in_planes, out_planes, expansion

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def square(self, x):
        return x**2 * 0.0001

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        out = F.relu(out)
        return out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    # m_cfg = [
    #     (1, 16, 1, 1, 1, 0),
    #     (6, 24, 1, 1, 1, 0),
    #     (6, 32, 1, 2, 0, 0),
    #     (6, 32, 1, 1, 1, 0),
    #     (6, 64, 1, 2, 1, 0),
    #     (6, 64, 1, 1, 1, 0),
    #     (6, 160, 1, 2, 0, 0),
    #     (6, 320, 1, 1, 0, 0),
    # ]
    # m_cfg = [(1,  16,  1, 1, 0, 0),
    #   (6,  24,  1, 1, 0, 0),
    #   (6,  32,  2, 2, 0, 0),
    #   (6,  64,  2, 2, 0, 0),
    #   (6, 160,  1, 2, 0, 0),
    #   (6, 320,  1, 1, 0, 0),]
    m_cfg = [(1,  16, 1, 1, 1, 0),
            #    (6,  24, 1, 1, 1, 0),  # NOTE: change stride 2 -> 1 for CIFAR10
            #    (6,  24, 1, 1, 0, 1),
               (6,  16, 1, 2, 0, 0),
            #    (4,  32, 1, 1, 0, 0),
            #    (8,  32, 1, 1, 1, 0),
               (6,  32, 1, 2, 1, 0),
            #    (6,  64, 1, 1, 0, 0),
            #    (4,  64, 1, 1, 0, 0),
            #    (8,  64, 1, 1, 1, 0),
            #    (4,  96, 1, 1, 1, 0),
            #    (6,  96, 1, 1, 0, 1),
            #    (6,  96, 1, 1, 0, 0),
               (6, 80, 1, 2, 0, 0),
            #    (4, 160, 1, 1, 1, 0),
            #    (4, 160, 1, 1, 1, 0),
               (6, 320, 1, 1, 0, 0)]

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        if width_mult != 1.0:
            raise ValueError('data/new_mvbv2.py backbone only supports width_mult=1.0')

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layers = self._make_modified_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.linear = nn.Linear(1280, num_classes)
        self.name = 'Modified_MobileNetV2'

    @property
    def classifier(self):
        return self.linear

    def _make_modified_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride, m1, m2 in self.m_cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(MBlock(in_planes, out_planes, expansion, stride, m1, m2))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layers(out)
        return out

    def conv(self, x):
        return F.relu(self.bn2(self.conv2(x)))

    def forward(self, x):
        out = self.features(x)
        out = self.conv(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


Modified_MobileNetV2 = MobileNetV2


def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)


def test():
    net = MobileNetV2()
    print(net.name)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y)


if __name__ == '__main__':
    test()
