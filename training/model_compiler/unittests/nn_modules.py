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


class NN0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return x


class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = RangeNormPoly2d(num_features=32)

    def forward(self, x):
        x = self.relu0(x)
        return x


class NN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 20
        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.convs[i](x)
        return x


class NN3(nn.Module):
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


class NN4(nn.Module):
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
