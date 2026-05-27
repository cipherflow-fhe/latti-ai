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
    def __init__(self, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv0 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=kernel_size, bias=True, padding=padding, stride=stride
        )

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


class SingleAvgpool2d(nn.Module):
    """AvgPool2d with configurable kernel_size, stride, and padding."""

    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.pool(x)
        return x


class SingleAvgpool1d(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.pool0 = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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


class NestedForkJoinMultcoeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_left = nn.BatchNorm2d(32)
        self.bn_target = nn.BatchNorm2d(32)
        self.bn_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)

    def forward(self, x):
        root = self.bn_root(x)
        left = self.bn_left(root)
        target = self.bn_target(left)
        side = self.bn_side(left)
        right = self.bn_right(root)
        scaled = (target + right) * 0.5
        return scaled, side


class NestedForkJoinMultcoeffNonOptimal(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_left = nn.BatchNorm2d(32)
        self.bn_a = nn.BatchNorm2d(32)
        self.bn_b = nn.BatchNorm2d(32)
        self.bn_c = nn.BatchNorm2d(32)
        self.bn_left_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)
        self.bn_root_side = nn.BatchNorm2d(32)

    def forward(self, x):
        root = self.bn_root(x)
        left = self.bn_left(root)
        a = self.bn_a(left)
        b = self.bn_b(left)
        c = self.bn_c(left)
        left_side = self.bn_left_side(left)
        right = self.bn_right(root)
        root_side = self.bn_root_side(root)
        scaled = (a + b + c + right) * 0.5
        return scaled, left_side, root_side


class NestedForkJoinMultcoeffIntermediateNonOptimal(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_left = nn.BatchNorm2d(32)
        self.bn_a = nn.BatchNorm2d(32)
        self.bn_b = nn.BatchNorm2d(32)
        self.bn_c = nn.BatchNorm2d(32)
        self.bn_left_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)
        self.bn_root_sides = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(5)])

    def forward(self, x):
        root = self.bn_root(x)
        left = self.bn_left(root)
        a = self.bn_a(left)
        b = self.bn_b(left)
        c = self.bn_c(left)
        left_side = self.bn_left_side(left)
        right = self.bn_right(root)
        root_sides = [bn(root) for bn in self.bn_root_sides]
        scaled = (a + b + c + right) * 0.5
        return (scaled, left_side, *root_sides)


class NestedForkJoinMultcoeffDoublePre(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_left = nn.BatchNorm2d(32)
        self.bn_left_a = nn.BatchNorm2d(32)
        self.bn_left_b = nn.BatchNorm2d(32)
        self.bn_left_c = nn.BatchNorm2d(32)
        self.bn_left_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)
        self.bn_right_a = nn.BatchNorm2d(32)
        self.bn_right_b = nn.BatchNorm2d(32)
        self.bn_right_c = nn.BatchNorm2d(32)
        self.bn_right_side = nn.BatchNorm2d(32)
        # self.bn_root_sides = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(5)])
        self.bn_root_sides = nn.ModuleList([nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False) for _ in range(5)])
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        root = self.bn_root(x)
        left = self.bn_left(root)
        left_a = self.bn_left_a(left)
        left_a = self.conv(left_a)
        left_b = self.bn_left_b(left)
        left_c = self.bn_left_c(left)
        left_side = self.bn_left_side(left)
        right = self.bn_right(root)
        right_a = self.bn_right_a(right)
        right_b = self.bn_right_b(right)
        right_c = self.bn_right_c(right)
        right_side = self.bn_right_side(right)
        root_sides = [bn(root) for bn in self.bn_root_sides]
        scaled = torch.cat([left_a, left_b, left_c, right_a, right_b, right_c], dim=1) * 0.5
        return (scaled, left_side, right_side, *root_sides)


class AddOutputTargetSuboptimal(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_left = nn.BatchNorm2d(32)
        self.bn_a = nn.BatchNorm2d(32)
        self.bn_b = nn.BatchNorm2d(32)
        self.bn_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)
        self.bn_root_side = nn.BatchNorm2d(32)

    def forward(self, x):
        root = self.bn_root(x)
        left = self.bn_left(root)
        a = self.bn_a(left)
        b = self.bn_b(left)
        side = self.bn_side(left)
        merged = a + b
        right = self.bn_right(root)
        root_side = self.bn_root_side(root)
        scaled = torch.cat([merged, right], dim=1) * 0.5
        return scaled, side, root_side


class IntermediateGoUpTargetAware(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_root = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_path = nn.BatchNorm2d(32)
        self.bn_mid = nn.BatchNorm2d(32)
        self.conv_root_side = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_target0 = nn.BatchNorm2d(32)
        self.bn_target1 = nn.BatchNorm2d(32)
        self.bn_mid_side = nn.BatchNorm2d(32)

    def forward(self, x):
        root = self.conv_root(x)
        path = self.bn_path(root)
        mid = self.bn_mid(path)
        root_side = self.conv_root_side(root)
        target0 = self.bn_target0(mid)
        target1 = self.bn_target1(mid)
        mid_side = self.bn_mid_side(mid)
        scaled = torch.cat([target0, target1], dim=1) * 0.5
        return scaled, mid_side, root_side


class TargetAwareGlobalDpVsLocalSink(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_root = nn.BatchNorm2d(32)
        self.bn_path = nn.BatchNorm2d(32)
        self.bn_mid = nn.BatchNorm2d(32)
        self.conv_root_side = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_target0 = nn.BatchNorm2d(32)
        self.bn_target1 = nn.BatchNorm2d(32)
        self.bn_mid_side = nn.BatchNorm2d(32)
        self.bn_right = nn.BatchNorm2d(32)

    def forward(self, x):
        root = self.bn_root(x)
        path = self.bn_path(root)
        mid = self.bn_mid(path)
        root_side = self.conv_root_side(root)
        target0 = self.bn_target0(mid)
        target1 = self.bn_target1(mid)
        mid_side = self.bn_mid_side(mid)
        right = self.bn_right(root)
        scaled = torch.cat([target0, target1, right], dim=1) * 0.5
        return scaled, mid_side, root_side


class ConvResidualRelu(nn.Module):
    """relu(conv(x) + x): residual shortcut with relu on the sum."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ConvResidualMultcoeff(nn.Module):
    """(conv(x) + x) * 0.5: residual shortcut scaled by a mult_coeff."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.bn(x)
        return (self.conv(x) + x) * 0.5


class ConvResidualMultcoeffDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = 0.5 * x
        x = self.bn(x)
        return self.conv(x) + x


class DoubleResidualMultcoeff(nn.Module):
    """两层残差叠加后乘 0.5。
    结构：z = (conv2(y) + y) * 0.5，其中 y = conv1(x) + x
    测试多层残差场景，scale 需要穿透两个 add 节点。
    """

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.bn(x)
        y = (self.conv1(x)) + self.conv2(x)  # 第一个残差 add
        z = (self.conv2(y)) + y  # 第二个残差 add
        return z * 0.5


class DoubleResidualMultcoeffDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = 0.5 * x
        x = self.bn(x)
        y = (self.conv1(x)) + self.conv2(x)
        z = (self.conv2(y)) + y
        return z


class BranchInBranchMultcoeff(nn.Module):
    """分支里面还有分支，再乘 0.5。
    结构：((conv_a(x) + conv_b(x)) + x) * 0.5
    外层 add：内层 add 臂 + identity 臂；内层 add：两条 conv 臂。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.conv_a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(32)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x) + x
        inner = self.bn_a(self.conv_a(x)) + self.bn_b(x)  # 内层 add
        return (inner + x) * 0.5  # 外层 add + mult_coeff


class BranchInBranchMultcoeffDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.conv_a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(32)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(32)

    def forward(self, x):
        x = 0.5 * x
        x = self.conv(x) + x
        inner = self.bn_a(self.conv_a(x)) + self.bn_b(x)
        return inner + x


class MultiInputBranchInBranchMultcoeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.conv_a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(32)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(32)
        self.bn_c = nn.BatchNorm2d(32)

    def forward(self, x, x1):
        x = self.conv(x) + x
        # x_1_1 = self.conv(x1)
        # x_1_2 = self.bn(x1)
        # out1 = torch.cat([x_1_1, x_1_2, x], dim=1)
        # inner = self.bn_a(self.conv_a(x)) + self.bn_b(x)  # 内层 add
        # inner = torch.cat([self.bn_a(self.conv_a(x)), self.bn_b(x), self.bn_c(x)],dim=1)
        conv_out1 = self.conv_a(x)
        conv_out2 = self.conv_b(x)
        inner = self.conv_a(conv_out1) + self.bn_b(conv_out2) + self.bn_c(x)
        out2 = 0.5 * inner + x
        out2 = x + out2
        return out2  # 外层 add + mult_coeff


class ThreeBranchMultcoeff(nn.Module):
    """三条分支汇入同一个 add，再乘 0.5。
    结构：(conv_a(x) + conv_b(x) + x) * 0.5
    add 有三条臂：两条 conv 臂 + 一条 identity 臂，测试 input_index 多臂场景。
    """

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv_a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(32)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.bn(x)
        return (self.bn_a(self.conv_a(x)) + self.bn_b(self.bn(x)) + x) * 0.5


class ThreeBranchMultcoeffDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv_a = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(32)
        self.conv_b = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(32)

    def forward(self, x):
        x = 0.5 * x
        x = self.bn(x)
        return self.bn_a(self.conv_a(x)) + self.bn_b(self.bn(x)) + x


class DeepNestedMultcoeff(nn.Module):
    """三层线性堆叠残差，最外层乘 0.5。
    结构：x1=conv1(x)+x, x2=conv2(x1)+x1, x3=conv3(x2)+x2, out=x3*0.5
    测试 scale 能沿深链路穿透多层 add 到达共同祖先。
    """

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.bn1(x)
        x1 = self.bn1(self.conv1(x)) + x
        x2 = self.bn2(self.conv2(x1)) + x1
        x3 = self.bn3(self.conv3(x2)) + x2
        return x3 * 0.5


class DeepNestedMultcoeffDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = 0.5 * x
        x = self.bn1(x)
        x1 = self.bn1(self.conv1(x)) + x
        x2 = self.bn2(self.conv2(x1)) + x1
        x3 = self.bn3(self.conv3(x2)) + x2
        return x3


class MultCoeffThenResidual(nn.Module):
    """input → mc (×0.5) → mc_out → add(mc_out, conv(mc_out))
    add 后面是 graph output，无下游吸收点。
    测试 DOWN 方向到多输入 add、且 add 后无吸收点时的 ms 插入策略。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        mc_out = x * 0.5
        out1 = mc_out + self.bn(mc_out)
        out2 = self.conv(out1)
        return out2


class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_cat = 64
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)  # 内 cat 的 conv 臂
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.bn0 = nn.BatchNorm2d(32)
        # 内 cat 后 channel = 32 + 32 = 64，下游层按 64 设
        self.conv2 = nn.Conv2d(n_cat, 32, kernel_size=3, padding=1, bias=False)  # 外 cat 的 conv 臂
        self.bn1 = nn.BatchNorm2d(n_cat)  # 外 cat 的 bn 臂
        self.bn2 = nn.BatchNorm2d(n_cat)

    def forward(self, x):
        # mc_out = x * 0.5
        x = self.bn(x)
        out1 = torch.cat([self.bn(x), self.conv1(x)], dim=1)  # 32 + 32 = 64
        # 外层 cat：conv2(64→32) + bn1(64) + identity(64) = 32 + 64 + 64 = 160
        out2 = torch.cat([self.conv2(out1), self.bn1(out1), self.bn2(out1)], dim=1)
        out = out2 * 0.5
        return out


class NewModelDown(nn.Module):
    def __init__(self):
        super().__init__()
        n_cat = 64
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(n_cat, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_cat)
        self.bn2 = nn.BatchNorm2d(n_cat)

    def forward(self, x):
        x = 0.5 * x
        x = self.bn(x)
        out1 = torch.cat([self.bn(x), self.conv1(x)], dim=1)
        out2 = torch.cat([self.conv2(out1), self.bn1(out1), self.bn2(out1)], dim=1)
        return out2


class NewModel1(nn.Module):
    def __init__(self):
        super().__init__()
        n_cat = 96
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)  # 内 cat 的 conv 臂
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.bn0 = nn.BatchNorm2d(32)
        # 内 cat 后 channel = 32 + 32 = 64，下游层按 64 设
        self.conv2 = nn.Conv2d(n_cat, 32, kernel_size=3, padding=1, bias=False)  # 外 cat 的 conv 臂
        self.bn1 = nn.BatchNorm2d(n_cat)  # 外 cat 的 bn 臂
        self.bn2 = nn.BatchNorm2d(n_cat)

    def forward(self, x):
        # mc_out = x * 0.5
        x = self.bn(x)
        out1 = torch.cat([self.bn(x), self.bn0(x), self.conv1(x)], dim=1)  # 32 + 32 = 64
        # 外层 cat：conv2(64→32) + bn1(64) + identity(64) = 32 + 64 + 64 = 160
        out2 = torch.cat([self.conv2(out1), self.bn1(out1)], dim=1)
        out = out2 * 0.5
        return out


class NewModel1Down(nn.Module):
    def __init__(self):
        super().__init__()
        n_cat = 96
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)  # 内 cat 的 conv 臂
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.bn0 = nn.BatchNorm2d(32)
        # 内 cat 后 channel = 32 + 32 = 64，下游层按 64 设
        self.conv2 = nn.Conv2d(n_cat, 32, kernel_size=3, padding=1, bias=False)  # 外 cat 的 conv 臂
        self.bn1 = nn.BatchNorm2d(n_cat)  # 外 cat 的 bn 臂
        self.bn2 = nn.BatchNorm2d(n_cat)

    def forward(self, x):
        # mc_out = x * 0.5
        x = 0.5 * x
        x = self.bn(x)
        out1 = torch.cat([self.bn(x), self.bn0(x), self.conv1(x)], dim=1)  # 32 + 32 = 64
        # 外层 cat：conv2(64→32) + bn1(64) + identity(64) = 32 + 64 + 64 = 160
        out2 = torch.cat([self.conv2(out1), self.bn1(out1)], dim=1)
        out = out2
        return out


class ThreeConvConcatRelu(nn.Module):
    """cat([conv1, conv2, conv3], dim=1) → relu."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.relu(s)


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
            self.units.append(Unit(pairs=2))

    def forward(self, *xs):
        s = self.units[0](xs[0])
        for i in range(1, self.n_inputs):
            s = s + self.units[i](xs[i])
        x = self.units[self.n_inputs](s)
        return x


class MutipleOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_outputs = 3
        self.units = nn.ModuleList()
        for i in range(self.n_outputs + 1):
            self.units.append(Unit(pairs=2))

    def forward(self, x):
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

    def __init__(self, in_channels=3, out_channels=16, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=True, padding=padding, stride=stride
        )

    def forward(self, x):
        x = self.conv0(x)
        return x


class DepthwiseConv(nn.Module):
    """Depthwise Conv2d (groups=in_channels). Covers dw_*ch_s*."""

    def __init__(self, channels=32, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv0 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, bias=True, padding=padding, stride=stride, groups=channels
        )

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

    def __init__(self, in_channels=4, out_channels=4, stride=1):
        super().__init__()
        self.conv0 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=True, padding=1, stride=stride
        )

    def forward(self, x):
        x = self.conv0(x)
        return x


class DepthwiseConv1d(nn.Module):
    """Depthwise Conv1d (groups=in_channels). Covers dw_conv1d."""

    def __init__(self, channels=8, stride=1):
        super().__init__()
        self.conv0 = nn.Conv1d(channels, channels, kernel_size=3, bias=True, padding=1, stride=stride, groups=channels)

    def forward(self, x):
        x = self.conv0(x)
        return x


class Conv1dReshapeAndDense(nn.Module):
    """Conv1d → Flatten → Dense pipeline."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, bias=True, padding=1)
        self.dense0 = nn.Linear(in_features=256, out_features=32, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        return x


class Concat(nn.Module):
    """Two conv branches concatenated. Covers concat_layer."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        return torch.cat([a, b], dim=1)


class ConvConcatConv(nn.Module):
    """Shared-input concat structure with final add:
      concat1: [conv1_out(8ch), conv2_out(8ch)] → concat1_out (16ch)
      concat2: [conv2_out(8ch), conv3_out(4ch), conv4_out(4ch)] → concat2_out (16ch)
      add:     concat1_out + concat2_out → 16ch
    Note: conv2_out feeds both concat1 and concat2 (tests shared-FeatureNode edge ordering).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, bias=False, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        c1 = torch.cat([a, b], dim=1)  # concat1: [conv1_out, conv2_out], 16ch
        d = self.conv3(x)
        e = self.conv4(x)
        c2 = torch.cat([b, d, e], dim=1)  # concat2: [conv2_out, conv3_out, conv4_out], 16ch
        return c1 + c2  # add: concat1_out + concat2_out, 16ch


class UnevenConcatModel(nn.Module):
    """Two conv branches with uneven channels concatenated. Covers concat_layer uneven path."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=7, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=5, kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        a = self.conv0(x)
        b = self.conv1(x)
        c = self.conv2(x)
        d = torch.cat([a, b, c], dim=1)
        return self.conv3(d)


class ConvUpsample(nn.Module):
    """Conv with stride=2 followed by nearest upsample. Covers upsample_layer / upsample_nearest_layer."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, padding=1, stride=2)
        self.resize = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv0(x)
        x = self.resize(x)
        return x


class SingleAdaptiveAvgpool2d(nn.Module):
    """AdaptiveAvgPool2d for E2E test. Covers adaptive_avgpool2d_layer."""

    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x):
        return self.pool(x)


class SingleAdaptiveAvgpool1d(nn.Module):
    """AdaptiveAvgPool1d for E2E test. Covers adaptive_avgpool1d_layer."""

    def __init__(self, output_size=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=output_size)

    def forward(self, x):
        return self.pool(x)
