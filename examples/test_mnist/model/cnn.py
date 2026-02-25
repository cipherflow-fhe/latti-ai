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

"""
Simple CNN for MNIST.

Uses nn.ReLU modules (not F.relu) so that activations can be replaced
with polynomial approximations for encrypted inference.

Structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> AvgPool -> Reshape -> Dropout -> FC
"""

import torch.nn as nn
import torch.nn.init as init

__all__ = ['SimpleCNN', 'simple_cnn']


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv1: 1x16x16 -> 16x8x8
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        # Conv2: 16x8x8 -> 32x4x4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Reshape + Dropout + FC: 32 -> num_classes
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def simple_cnn(num_classes=10):
    return SimpleCNN(num_classes=num_classes)
