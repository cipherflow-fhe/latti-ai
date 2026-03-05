# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent))
sys.path.append(str(script_dir.parent.parent))

from nn_tools.activations import RangeNormPoly2d


class SingleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return x


class SingleAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = RangeNormPoly2d(num_features=32)

    def forward(self, x):
        x = self.relu0(x)
        return x


class SingleAvgpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool0 = nn.AvgPool2d(kernel_size=2, padding=1)

    def forward(self, x):
        x = self.pool0(x)
        return x


class SingleMaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool0 = nn.MaxPool2d(kernel_size=2, padding=1)

    def forward(self, x):
        x = self.pool0(x)
        return x


class SingleDense(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: if bias=False, the ONNX contains a (unsupported) MatMul op instead of Gemm
        self.dense0 = nn.Linear(in_features=64, out_features=32, bias=True)

    def forward(self, x):
        x = self.dense0(x)
        return x


class SingleReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape0 = nn.Flatten(1)

    def forward(self, x):
        x = self.reshape0(x)
        return x


class SingleMultCoeff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = 5 * x
        return x


class ConvSeries(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 40
        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.convs[i](x)
        return x


class ActSeries(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 20
        self.acts = nn.ModuleList()
        for i in range(self.n_layers):
            self.acts.append(RangeNormPoly2d(num_features=32))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.acts[i](x)
        return x


class ConvSeriesWithStride(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 20
        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    bias=False,
                    stride=2 if (i % 4 == 2) else 1,
                    padding=1,
                )
            )

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.convs[i](x)
        return x


class MultCoeffSeries(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 5

    def forward(self, x):
        for i in range(self.n_layers):
            x = x * (1.1 + i * 0.1)
        return x


class ConvAndMultCoeffSeries(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 5
        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.convs[i](x)
            x = x * (1.1 + i * 0.1)
        return x


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = RangeNormPoly2d(num_features=planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = RangeNormPoly2d(num_features=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class SingleAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        return x0 + x1


class MismatchedScale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x * 5
        x = x + y
        return x


class Unit(nn.Module):
    def __init__(self, pairs: int = 2):
        super().__init__()
        self.pairs = pairs
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(pairs):
            self.convs.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1))
            self.acts.append(RangeNormPoly2d(num_features=32))

    def forward(self, x):
        for i in range(self.pairs):
            x = self.convs[i](x)
            x = self.acts[i](x)
        return x


class Intertwined(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = nn.ModuleList()
        for i in range(8):
            self.units.append(Unit(pairs=(3 if i % 2 == 0 else 2)))

    def forward(self, x):
        x0, x1 = self.units[0](x), self.units[1](x)
        x0, x1 = self.units[2](x0) + self.units[3](x1), self.units[4](x0) + self.units[5](x1)
        x = self.units[6](x0) + self.units[7](x1)
        return x


class IntertwinedWithCoeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = nn.ModuleList()
        for i in range(8):
            self.units.append(Unit(pairs=(3 if i % 2 == 0 else 2)))

    def forward(self, x):
        x0, x1 = self.units[0](x), self.units[1](x) * 1.1
        x0, x1 = self.units[2](x0) * 1.2 + self.units[3](x1) * 1.3, self.units[4](x0) * 1.4 + self.units[5](x1) * 1.5
        x = self.units[6](x0) * 1.6 + self.units[7](x1)
        return x


class MutipleInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_inputs = 3
        self.units = nn.ModuleList()
        for i in range(self.n_inputs + 1):
            self.units.append(Unit(pairs=5))

    def forward(self, xs):
        s = torch.zeros_like(x)
        for i in range(self.n_inputs):
            s += self.units[i](xs[i])
        x = self.units[self.n_inputs](s)
        return x


class MutipleOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_outputs = 3
        self.units = nn.ModuleList()
        for i in range(self.n_outputs + 1):
            self.units.append(Unit(pairs=5))

    def forward(self, x):
        x = torch.zeros_like(x)
        x = self.units[0](x)
        ys = list()
        for i in range(self.n_outputs):
            ys.append(self.units[i + 1](x))
        return ys


class WrongPadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=(0, 0))

    def forward(self, x):
        x = self.conv0(x)
        return x


class WrongDilation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, dilation=2)

    def forward(self, x):
        x = self.conv0(x)
        return x


class WrongGroups(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, groups=2)

    def forward(self, x):
        x = self.conv0(x)
        return x


class SingleRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = nn.ReLU()

    def forward(self, x):
        x = self.relu0(x)
        return x


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv0(x1)
        x3 = x + x2
        x4 = self.conv0(x3)
        return x4


class ConvAndConvTransposeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class ConvAndUpsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv0(x)
        x = self.resize(x)
        return x
