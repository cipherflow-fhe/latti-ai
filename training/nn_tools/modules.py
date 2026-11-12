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

import math
from typing import Union

import torch
import torch.nn as nn


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def _poly_maxpool_plaintext(x, kernel, stride, padding, alpha, upper_bound, B, running_max):
    """Differentiable deferred-half reduction matching the compiled MaxPool DAG."""
    from .fixed_maxpool_profiles import get_maxpool_profile

    if x.ndim not in (3, 4):
        raise ValueError('PolyMaxPool2d expects CHW or NCHW input')
    unbatched = x.ndim == 3
    if unbatched:
        x = x.unsqueeze(0)
    h, w = x.shape[-2:]
    oh = (h + 2 * padding[0] - kernel[0]) // stride[0] + 1
    ow = (w + 2 * padding[1] - kernel[1]) // stride[1] + 1
    normalized = x * (upper_bound / running_max)
    anchor = normalized[..., ::stride[0], ::stride[1]][..., :oh, :ow]
    padded = torch.nn.functional.pad(normalized, (padding[1], padding[1], padding[0], padding[0]))
    valid = torch.nn.functional.pad(torch.ones_like(normalized),
                                    (padding[1], padding[1], padding[0], padding[0]))
    taps = []
    for row in range(kernel[0]):
        for col in range(kernel[1]):
            part = padded[..., row:row + oh * stride[0]:stride[0], col:col + ow * stride[1]:stride[1]]
            mask = valid[..., row:row + oh * stride[0]:stride[0], col:col + ow * stride[1]:stride[1]]
            # Runtime masks duplicate the window anchor outside the image.
            taps.append(torch.where(mask.bool(), part, anchor))
    coefficients = get_maxpool_profile(alpha)['odd_coefficients']
    depth = 0
    while len(taps) > 1:
        bound = B * (1 if depth == 0 else 1.05) * 2 ** depth
        reduced = []
        for i in range(0, len(taps) - 1, 2):
            a, b = taps[i:i + 2]
            difference = a - b
            sign = difference / (2 * bound)
            for stage in coefficients:
                squared = sign * sign
                value = torch.zeros_like(sign) + float(stage[-1])
                for coefficient in reversed(stage[:-1]):
                    value = value * squared + float(coefficient)
                sign = sign * value
            reduced.append(a + b + difference * sign)
        if len(taps) % 2:
            reduced.append(2 * taps[-1])
        taps = reduced
        depth += 1
    result = taps[0] * (running_max / upper_bound / 2 ** depth)
    return result.squeeze(0) if unbatched else result


class _PolyMaxPool2dExport(torch.autograd.Function):
    """Export a MaxPool node while preserving the FHE approximation settings."""

    @staticmethod
    def forward(ctx, x, kernel, stride, padding, alpha, upper_bound, B, running_max):
        # Shape-only tracing placeholder; symbolic emits the polynomial operator.
        # Normal model execution always uses _poly_maxpool_plaintext.
        return torch.nn.functional.max_pool2d(x, kernel, stride, padding)

    @staticmethod
    def symbolic(g, x, kernel, stride, padding, alpha, upper_bound, B, running_max):
        sizes = x.type().varyingSizes()
        output_sizes = None if sizes is None else [
            *sizes[:-2],
            None if sizes[-2] is None else (sizes[-2] + 2 * padding[0] - kernel[0]) // stride[0] + 1,
            None if sizes[-1] is None else (sizes[-1] + 2 * padding[1] - kernel[1]) // stride[1] + 1,
        ]
        return g.op(
            'nn_tools::PolyMaxPool2d', x,
            kernel_shape_i=kernel,
            strides_i=stride,
            pads_i=list(padding) + list(padding),
            alpha_i=alpha,
            upper_bound_f=upper_bound,
            B_f=B,
            running_max_f=running_max,
        ).setType(x.type().with_sizes(output_sizes))


class PolyMaxPool2d(nn.Module):
    """Drop-in MaxPool2d whose FHE settings survive ONNX export.

    PyTorch evaluates the same polynomial reduction as the compiler. Training
    updates a scalar input bound; evaluation/export freezes it. Fine tuning
    remains optional. The compiler expands the custom node before global DP.
    """

    def __init__(self, kernel_size=3, stride=2, padding=1, *, alpha=7,
                 upper_bound=1.0, B=1.0):
        super().__init__()
        from .fixed_maxpool_profiles import get_maxpool_profile
        get_maxpool_profile(alpha)
        for name, value in (("upper_bound", upper_bound), ("B", B)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (_pair(kernel_size), _pair(stride), _pair(padding)) != ((3, 3), (2, 2), (1, 1)):
            raise ValueError("PolyMaxPool2d requires kernel=3, stride=2, padding=1")
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.alpha = alpha
        self.upper_bound = float(upper_bound)
        self.B = float(B)

    def forward(self, x):
        if self.training and not torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                observed = x.detach().abs().amax().clamp_min(torch.finfo(x.dtype).eps)
                if self.num_batches_tracked == 0:
                    self.running_max.copy_(observed)
                else:
                    self.running_max.copy_(torch.maximum(self.running_max, observed))
                self.num_batches_tracked.add_(1)
        if torch.onnx.is_in_onnx_export():
            return _PolyMaxPool2dExport.apply(
                x, list(self.kernel_size), list(self.stride), list(self.padding),
                self.alpha, self.upper_bound, self.B, float(self.running_max))
        return _poly_maxpool_plaintext(
            x, self.kernel_size, self.stride, self.padding,
            self.alpha, self.upper_bound, self.B, self.running_max.clone())


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
