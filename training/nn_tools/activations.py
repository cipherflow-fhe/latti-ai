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
"""Custom activation modules for encrypted inference.

Modules:
  - RangeNorm2d:      per-channel range normalization
  - Simple_Polyrelu:  Hermite polynomial approximation of ReLU
  - RangeNormPoly2d:  combined range-norm + polynomial activation
"""

import torch
import torch.nn as nn
import numpy as np


class RangeNorm2d(nn.Module):
    """Per-channel range normalization.

    Normalizes input to [-upper_bound, upper_bound] using a running
    estimate of the per-channel absolute maximum.

    Supports lazy initialization: pass ``num_features=0`` to defer buffer
    creation until the first forward call, where the channel count is
    inferred from ``x.shape[1]``.

    Args:
        num_features: Number of channels (0 for lazy initialization).
        upper_bound:  Normalization upper bound (default 3.0).
        eps:          Small constant to avoid division by zero.
        momentum:     Momentum for running-max update.
    """

    def __init__(self, num_features=0, upper_bound=3.0, eps=1e-3, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.upper_bound = upper_bound
        self.scale_factor = 1.0

        if num_features > 0:
            self.register_buffer('running_max', torch.ones(1, num_features, 1, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_max', None)
            self.register_buffer('num_batches_tracked', None)

    def _lazy_init(self, x: torch.Tensor):
        self.num_features = x.shape[1]
        self.running_max = torch.ones(1, self.num_features, 1, 1, device=x.device, dtype=x.dtype)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=x.device)

    def forward(self, x):
        if self.running_max is None:
            self._lazy_init(x)

        if self.training:
            abs_max = x.abs().amax(dim=(0, 2, 3), keepdim=True)

            if self.num_batches_tracked == 0:
                self.running_max.copy_(abs_max.detach())
            else:
                self.running_max.mul_(1 - self.momentum).add_(self.momentum * abs_max.detach())
            self.num_batches_tracked.add_(1)
        else:
            abs_max = self.running_max

        self.scale_factor = abs_max / self.upper_bound + self.eps
        return x / self.scale_factor

    def extra_repr(self):
        return (
            f'num_features={self.num_features}, upper_bound={self.upper_bound}, '
            f'eps={self.eps}, momentum={self.momentum}'
        )


class _SimplePolyreluExport(torch.autograd.Function):
    """ONNX export helper: emit Simple_Polyrelu as a single custom op."""

    @staticmethod
    def forward(ctx, x, scale_before, scale_after, degree, activation):
        if activation == 'relu':
            coeffs = Simple_Polyrelu._RELU_COEFF
        else:
            coeffs = Simple_Polyrelu._SILU_COEFF
        a0, a1, a2, a3, a4 = coeffs[degree]
        a0 *= scale_after
        a1 *= scale_after
        a2 *= scale_after
        a4 *= scale_after

        x = scale_before * x

        if degree == 2:
            return a0 + (a1 + a2 * x) * x - a2
        elif degree == 4:
            return a0 + a1 * x + a2 * (x**2 - 1) + a4 * (x**4 - 6 * x**2 + 3)
        else:
            raise ValueError(f'Unsupported degree: {degree}')

    @staticmethod
    def symbolic(g, x, scale_before, scale_after, degree, activation):
        return g.op(
            'nn_tools::Simple_Polyrelu',
            x,
            scale_before_f=scale_before,
            scale_after_f=scale_after,
            degree_i=degree,
            activation_s=activation,
        ).setType(x.type())


class Simple_Polyrelu(nn.Module):
    """Polynomial activation approximating ReLU or SiLU via Hermite expansion.

    Exported as a single ``nn_tools::Simple_Polyrelu`` custom op in ONNX.

    Args:
        scale_before: Input scaling factor.
        scale_after:  Output scaling factor.
        degree:       Polynomial degree (2 or 4).
        activation:   Target activation ('relu' or 'silu').
    """

    # Hermite coefficients for ReLU approximation
    _RELU_COEFF = {
        2: (0.39894228, 0.50000000, 0.28209479 / np.sqrt(2), 0.0, 0.0),
        4: (0.39894228, 0.50000000, 0.28209479 / np.sqrt(2), 0.0, -0.08143375 / np.sqrt(24)),
    }

    # Coefficients for SiLU approximation
    _SILU_COEFF = {
        2: (0.20662096, 0.50000000, 0.24808519 / np.sqrt(2), 0.0, 0.0),
        4: (0.20662096, 0.50000000, 0.24808519 / np.sqrt(2), 0.0, -0.03780501 / np.sqrt(24)),
    }

    def __init__(self, scale_before=1.0, scale_after=1.0, degree=4, activation='relu', **kwargs):
        super().__init__()
        self.scale_before = scale_before
        self.scale_after = scale_after
        self.degree = degree
        self.activation = activation

        if activation == 'relu':
            coeffs = self._RELU_COEFF
        elif activation == 'silu':
            coeffs = self._SILU_COEFF
        else:
            raise ValueError(
                f'Unsupported activation: {activation}. For other activations, use Chebyshev polynomial fitting.'
            )

        if degree not in coeffs:
            raise ValueError(f'Unsupported degree: {degree}. Use 2 or 4.')

        a0, a1, a2, a3, a4 = coeffs[degree]
        self.a0 = a0 * self.scale_after
        self.a1 = a1 * self.scale_after
        self.a2 = a2 * self.scale_after
        self.a4 = a4 * self.scale_after

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _SimplePolyreluExport.apply(
                x,
                self.scale_before,
                self.scale_after,
                self.degree,
                self.activation,
            )
        x = self.scale_before * x

        if self.degree == 2:
            return self.a0 + (self.a1 + self.a2 * x) * x - self.a2
        if self.degree == 4:
            return self.a0 + self.a1 * x + self.a2 * (x**2 - 1) + self.a4 * (x**4 - 6 * x**2 + 3)

    def extra_repr(self):
        return (
            f'degree={self.degree}, activation={self.activation}, '
            f'scale_before={self.scale_before}, scale_after={self.scale_after}'
        )


class _RangeNormPoly2dExport(torch.autograd.Function):
    """ONNX export helper: emit RangeNormPoly2d as a single custom op."""

    @staticmethod
    def forward(ctx, x, running_max, upper_bound, degree, eps, activation):
        scale_factor = running_max / upper_bound + eps
        x_norm = x / scale_factor

        if activation == 'relu':
            coeffs = Simple_Polyrelu._RELU_COEFF
        else:
            coeffs = Simple_Polyrelu._SILU_COEFF
        a0, a1, a2, a3, a4 = coeffs[degree]

        if degree == 2:
            poly_out = a0 + (a1 + a2 * x_norm) * x_norm - a2
        elif degree == 4:
            poly_out = a0 + a1 * x_norm + a2 * (x_norm**2 - 1) + a4 * (x_norm**4 - 6 * x_norm**2 + 3)
        else:
            raise ValueError(f'Unsupported degree: {degree}')

        return scale_factor * poly_out

    @staticmethod
    def symbolic(g, x, running_max, upper_bound, degree, eps, activation):
        return g.op(
            'nn_tools::RangeNormPoly2d',
            x,
            running_max,
            upper_bound_f=upper_bound,
            degree_i=degree,
            eps_f=eps,
            activation_s=activation,
        ).setType(x.type())


class RangeNormPoly2d(nn.Module):
    """Combined range normalization + polynomial activation.

    Applies per-channel range normalization, then a polynomial activation,
    and rescales back. Exported as a single ``nn_tools::RangeNormPoly2d``
    custom op in ONNX.

    Supports lazy initialization: pass ``num_features=0`` (default) to defer
    buffer creation until the first forward call.

    Args:
        num_features: Number of channels (0 for lazy initialization).
        upper_bound:  Normalization upper bound.
        degree:       Polynomial degree (2 or 4).
        activation:   Target activation ('relu' or 'silu').
    """

    def __init__(self, num_features=0, upper_bound=3.0, degree=4, activation='relu'):
        super().__init__()
        self.num_features = num_features
        self.upper_bound = upper_bound
        self.degree = degree
        self.activation = activation

        self.rangenorm = RangeNorm2d(num_features, upper_bound=upper_bound, eps=1e-3, momentum=0.1)
        self.poly = Simple_Polyrelu(degree=degree, activation=activation)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _RangeNormPoly2dExport.apply(
                x,
                self.rangenorm.running_max,
                self.upper_bound,
                self.degree,
                self.rangenorm.eps,
                self.activation,
            )
        x = self.rangenorm(x)
        x = self.rangenorm.scale_factor * self.poly(x)
        return x

    def extra_repr(self):
        return (
            f'num_features={self.num_features}, upper_bound={self.upper_bound}, '
            f'degree={self.degree}, activation={self.activation}'
        )
