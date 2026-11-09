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
"""Custom modules for FHE-compatible model conversion."""

from typing import Union

import torch
import torch.nn as nn


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class _PolyMaxPool2dExport(torch.autograd.Function):
    """Export a MaxPool node while preserving the FHE approximation settings."""

    @staticmethod
    def forward(ctx, x, kernel, stride, padding, alpha, upper_bound, B):
        return torch.nn.functional.max_pool2d(x, kernel, stride, padding)

    @staticmethod
    def symbolic(g, x, kernel, stride, padding, alpha, upper_bound, B):
        return g.op(
            'nn_tools::PolyMaxPool2d', x,
            kernel_shape_i=kernel,
            strides_i=stride,
            pads_i=list(padding) + list(padding),
            alpha_i=alpha,
            upper_bound_f=upper_bound,
            B_f=B,
        ).setType(x.type())


class PolyMaxPool2d(nn.Module):
    """Drop-in MaxPool2d whose FHE settings survive ONNX export.

    Plain PyTorch execution remains exact max pooling. During compilation the
    exported node is lowered to the explicit polynomial MaxPool stages.
    """

    def __init__(self, kernel_size=3, stride=2, padding=1, *, alpha=7,
                 upper_bound=1.0, B=1.0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.alpha = alpha
        self.upper_bound = float(upper_bound)
        self.B = float(B)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _PolyMaxPool2dExport.apply(
                x, list(self.kernel_size), list(self.stride), list(self.padding),
                self.alpha, self.upper_bound, self.B)
        return torch.nn.functional.max_pool2d(
            x, self.kernel_size, self.stride, self.padding)


class DepthwiseAvgPool2d(nn.Module):
    """Drop-in replacement for ``nn.AvgPool2d`` using a depthwise convolution.

    Because ``nn.AvgPool2d`` does not store the channel count, the internal
    ``nn.Conv2d`` is created lazily on the first forward pass.  Weights are
    initialised to ``1 / (k0 * k1)`` and bias is zero; both are frozen by
    default so the module behaves identically to the original AvgPool.

    Args:
        kernel_size: Pooling kernel size (int or 2-tuple).
        stride:      Pooling stride (defaults to *kernel_size*).
        padding:     Padding applied before convolution.
        freeze:      If ``True`` (default), convolution parameters are not
                     updated during training.
    """

    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int], None] = None,
        padding: Union[int, tuple[int, int]] = 0,
        freeze: bool = True,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self.freeze = freeze
        self.conv: nn.Conv2d | None = None

    def _init_conv(self, channels: int) -> None:
        k0, k1 = self.kernel_size
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.conv.weight, 1.0 / (k0 * k1))
        if self.freeze:
            self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            self._init_conv(x.shape[1])
            self.conv = self.conv.to(x.device)
        return self.conv(x)

    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, freeze={self.freeze}'
