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

import torch
import torch.nn as nn


class SingleConv(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=stride)

    def forward(self, x):
        x = self.conv0(x)
        return x


class SingleConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return x


class SingleAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = nn.ReLU()

    def forward(self, x):
        x = self.relu0(x)
        return x


class SingleAct1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu0 = nn.ReLU()

    def forward(self, x):
        x = self.relu0(x)
        return x


class SingleAvgpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool0 = nn.AvgPool2d(kernel_size=2, padding=0)

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


class TwoDense(nn.Module):
    """Pure FC-FC network: graph input is 0d (1-D feature vector)."""

    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.dense1 = nn.Linear(in_features=64, out_features=32, bias=True)

    def forward(self, x):
        x = self.dense0(x)
        x = self.dense1(x)
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


class SingleAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        return x0 + x1


class ConvWithBatchNorms(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=32)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
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
            self.acts.append(nn.ReLU())

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
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

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
            self.acts.append(nn.ReLU())

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


class SkipConnection(nn.Module):
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
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
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


class ConvReshapeAndDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False, padding=1, stride=2)
        self.dense0 = nn.Linear(in_features=768, out_features=32, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        return x


class ConvReshapeAndTwoDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False, padding=1, stride=2)
        self.dense0 = nn.Linear(in_features=768, out_features=64, bias=True)
        self.dense1 = nn.Linear(in_features=64, out_features=32, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        x = self.dense1(x)
        return x


# ── Poly-degree targeting modules ─────────────────────────────────────────────
#
# Level costs (ordinary style):
#   Conv (stride=1, ordinary): 1
#   Activation (ReLU → RangeNormPoly2d after prepare_for_fhe, order=4): ceil(log2(4)) + 1 = 3
#
# The no-BTP pipeline tries poly_n values in order [8192, 16384, 32768, 65536]
# with max_level [5, 9, 17, 33].  The input feature level equals the sum of
# level_cost along the critical path, so:
#
#   PolyDegreeN8192  : 1 Conv + 1 Act = 4 levels  → fits 8192  (max 5)
#   PolyDegreeN16384 : 3 Conv + 1 Act = 6 levels  → exceeds 8192, fits 16384 (max 9)
#   PolyDegreeN32768 : 4 Conv + 2 Act = 10 levels → exceeds 16384, fits 32768 (max 17)
#   PolyDegreeN65536 : 6 Conv + 4 Act = 18 levels → exceeds 32768, fits 65536 (max 33)
#   PolyDegreeNBtp   : 4 Conv + 10 Act = 34 levels → exceeds all non-BTP limits → BTP


class PolyDegreeN8192(nn.Module):
    """1 Conv + 1 Act = 4 levels total; fits poly_n=8192 (max_level=5)."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act0 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        return x


class PolyDegreeN16384(nn.Module):
    """3 Conv + 1 Act = 6 levels total; exceeds poly_n=8192 (max 5), fits poly_n=16384 (max 9)."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act0 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.act0(x)
        x = self.conv2(x)
        return x


class PolyDegreeN32768(nn.Module):
    """4 Conv + 2 Act = 10 levels total; exceeds poly_n=16384 (max 9), fits poly_n=32768 (max 17)."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act1 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        return x


class PolyDegreeN65536NoBtp(nn.Module):
    """6 Conv + 4 Act = 18 levels total; exceeds poly_n=32768 (max 17), fits poly_n=65536 non-BTP (max 33)."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act2 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
        self.act3 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act2(x)
        x = self.conv4(x)
        x = self.act3(x)
        x = self.conv5(x)
        return x


class PolyDegreeNBtp(nn.Module):
    """4 Conv + 10 Act = 34 levels total; exceeds all non-BTP limits (max 33) → forces BTP mode."""

    def __init__(self):
        super().__init__()
        self.n_acts = 10
        self.n_convs = 4
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(self.n_acts)])
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1)
                for _ in range(self.n_convs)
            ]
        )

    def forward(self, x):
        # Interleave: act, act, conv, act, act, conv, act, act, conv, act, act, conv, act, act
        for i in range(self.n_convs):
            x = self.acts[2 * i](x)
            x = self.acts[2 * i + 1](x)
            x = self.convs[i](x)
        x = self.acts[8](x)
        x = self.acts[9](x)
        return x


# ── End poly-degree targeting modules ─────────────────────────────────────────


class ConvAvgpoolReshapeAndDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, bias=False, padding=1, stride=4)
        self.pool0 = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)
        self.dense0 = nn.Linear(in_features=16, out_features=32, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        return x


# ── Models for E2E migration of test_fhe_layers_hetero ────────────────────────


class MultiChannelConv(nn.Module):
    """Conv2d with different input/output channels. Covers conv_mch_s1/s2."""

    def __init__(self, in_channels=3, out_channels=16, stride=1):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True, padding=1, stride=stride)

    def forward(self, x):
        x = self.conv0(x)
        return x


class DepthwiseConv(nn.Module):
    """Depthwise Conv2d (groups=in_channels). Covers dw_*ch_s*."""

    def __init__(self, channels=32, stride=1):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, bias=True, padding=1, stride=stride, groups=channels)

    def forward(self, x):
        x = self.conv0(x)
        return x


class ConvReshapeTwoFC(nn.Module):
    """Conv → Flatten → Linear → Linear. Covers fc_fc_0d."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False, padding=1, stride=2)
        self.dense0 = nn.Linear(in_features=768, out_features=128, bias=True)
        self.dense1 = nn.Linear(in_features=128, out_features=32, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        x = self.dense1(x)
        return x


class MuxConvLargeChannel(nn.Module):
    """Large-channel conv to trigger multiplexed packing. Covers mux_conv_varied_*."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=True, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return x


class SingleConv1dE2E(nn.Module):
    """Conv1d for E2E test. Covers conv1d."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, bias=True, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return x


class ConcatModel(nn.Module):
    """Two conv branches concatenated. Covers concat_layer."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        return torch.cat([a, b], dim=1)


class ConvUpsampleE2E(nn.Module):
    """Conv with stride=2 followed by nearest upsample. Covers upsample_layer / upsample_nearest_layer."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv0(x)
        x = self.resize(x)
        return x


class AvgpoolVariedStride(nn.Module):
    """Avgpool with configurable stride. Covers avgpool2d_layer varied strides."""

    def __init__(self, stride=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x
