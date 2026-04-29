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
VGG for ImageNet.

Configuration-driven VGG implementation for 224x224 ImageNet input,
with explicit activation and pooling modules for later FHE-oriented replacement.

Classifier: AdaptiveAvgPool2d(1,1) + Linear(512, num_classes)
"""

from collections import OrderedDict

import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """Configuration-driven VGG adapted for ImageNet."""

    def __init__(self, features, num_classes=1000, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    conv_idx = 0
    pool_idx = 0

    for v in cfg:
        if v == "M":
            pool_idx += 1
            layers.append((f"pool{pool_idx}", nn.MaxPool2d(kernel_size=2, stride=2)))
            continue

        conv_idx += 1
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
        layers.append((f"conv{conv_idx}", conv2d))

        if batch_norm:
            layers.append((f"bn{conv_idx}", nn.BatchNorm2d(v)))

        layers.append((f"relu{conv_idx}", nn.ReLU(inplace=True)))
        in_channels = v

    return nn.Sequential(OrderedDict(layers))


def _vgg(model_name, in_channels=3, num_classes=1000, batch_norm=True, init_weights=True):
    if model_name not in cfgs:
        raise ValueError(f"Unsupported VGG model: {model_name}")

    features = make_layers(cfgs[model_name], in_channels=in_channels, batch_norm=batch_norm)
    return VGG(features=features, num_classes=num_classes, init_weights=init_weights)


def vgg11(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG11 model for ImageNet."""
    return _vgg("vgg11", in_channels=in_channels, num_classes=num_classes, batch_norm=False, init_weights=init_weights)


def vgg11_bn(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG11-BN model for ImageNet."""
    return _vgg("vgg11", in_channels=in_channels, num_classes=num_classes, batch_norm=True, init_weights=init_weights)


def vgg13(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG13 model for ImageNet."""
    return _vgg("vgg13", in_channels=in_channels, num_classes=num_classes, batch_norm=False, init_weights=init_weights)


def vgg13_bn(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG13-BN model for ImageNet."""
    return _vgg("vgg13", in_channels=in_channels, num_classes=num_classes, batch_norm=True, init_weights=init_weights)


def vgg16(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG16 model for ImageNet."""
    return _vgg("vgg16", in_channels=in_channels, num_classes=num_classes, batch_norm=False, init_weights=init_weights)


def vgg16_bn(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG16-BN model for ImageNet."""
    return _vgg("vgg16", in_channels=in_channels, num_classes=num_classes, batch_norm=True, init_weights=init_weights)


def vgg19(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG19 model for ImageNet."""
    return _vgg("vgg19", in_channels=in_channels, num_classes=num_classes, batch_norm=False, init_weights=init_weights)


def vgg19_bn(in_channels=3, num_classes=1000, init_weights=True):
    """Create VGG19-BN model for ImageNet."""
    return _vgg("vgg19", in_channels=in_channels, num_classes=num_classes, batch_norm=True, init_weights=init_weights)
