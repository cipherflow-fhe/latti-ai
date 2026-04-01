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
  - Simple_Polyrelu:  Hermite polynomial approximation of activations
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


def _eval_hermite(x, hermite_coeffs, scale_after=1.0):
    """Evaluate Hermite polynomial expansion.

    Computes scale_after * sum_n a_n * He_n(x) where He_n is the probabilist's
    Hermite polynomial.
    """
    a0, a1, a2 = hermite_coeffs[0], hermite_coeffs[1], hermite_coeffs[2]
    degree = len(hermite_coeffs) - 1

    if degree == 2:
        return scale_after * (a0 + (a1 + a2 * x) * x - a2)

    if degree == 4:
        a4 = hermite_coeffs[4]
        return scale_after * (a0 + a1 * x + a2 * (x**2 - 1) + a4 * (x**4 - 6 * x**2 + 3))

    # General degree: recurrence He_n = x*He_{n-1} - (n-1)*He_{n-2}
    result = a0 * scale_after + a1 * scale_after * x
    he_prev2 = torch.ones_like(x)  # He_0
    he_prev1 = x  # He_1
    for n in range(2, degree + 1):
        he_n = x * he_prev1 - (n - 1) * he_prev2
        result = result + hermite_coeffs[n] * scale_after * he_n
        he_prev2, he_prev1 = he_prev1, he_n
    return result


class _SimplePolyreluExport(torch.autograd.Function):
    """ONNX export helper: emit Simple_Polyrelu as a single custom op."""

    @staticmethod
    def forward(ctx, x, scale_before, scale_after, hermite_coeffs):
        return _eval_hermite(scale_before * x, hermite_coeffs, scale_after)

    @staticmethod
    def symbolic(g, x, scale_before, scale_after, hermite_coeffs):
        return g.op(
            'nn_tools::Simple_Polyrelu',
            x,
            scale_before_f=scale_before,
            scale_after_f=scale_after,
            degree_i=len(hermite_coeffs) - 1,
        ).setType(x.type())


class Simple_Polyrelu(nn.Module):
    """Polynomial activation via Hermite expansion.

    Hermite coefficients are computed by ``get_hermite_coeffs_for_module()``
    and passed in via ``hermite_coeffs``.

    Reference coefficients (degree=4):
      ReLU: (0.39894228, 0.50000000, 0.28209479/sqrt(2), 0.0, -0.08143375/sqrt(24))
      SiLU: (0.20662096, 0.50000000, 0.24808519/sqrt(2), 0.0, -0.03780501/sqrt(24))

    Args:
        hermite_coeffs: Tuple of (degree+1) Hermite expansion coefficients.
        scale_before:   Input scaling factor.
        scale_after:    Output scaling factor.
    """

    def __init__(self, hermite_coeffs, scale_before=1.0, scale_after=1.0, **kwargs):
        super().__init__()
        self.hermite_coeffs = tuple(hermite_coeffs)
        self.degree = len(hermite_coeffs) - 1
        self.scale_before = scale_before
        self.scale_after = scale_after

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _SimplePolyreluExport.apply(
                x,
                self.scale_before,
                self.scale_after,
                self.hermite_coeffs,
            )
        return _eval_hermite(self.scale_before * x, self.hermite_coeffs, self.scale_after)

    def extra_repr(self):
        return f'degree={self.degree}, scale_before={self.scale_before}, scale_after={self.scale_after}'


class _RangeNormPoly1dExport(torch.autograd.Function):
    """ONNX export helper: emit RangeNormPoly1d as a single custom op."""

    @staticmethod
    def forward(ctx, x, running_max, upper_bound, eps, hermite_coeffs):
        scale_factor = running_max / upper_bound + eps
        poly_out = _eval_hermite(x / scale_factor, hermite_coeffs)
        return scale_factor * poly_out

    @staticmethod
    def symbolic(g, x, running_max, upper_bound, eps, hermite_coeffs):
        return g.op(
            'nn_tools::RangeNormPoly1d',
            x,
            running_max,
            upper_bound_f=upper_bound,
            degree_i=len(hermite_coeffs) - 1,
            eps_f=eps,
        ).setType(x.type())


class _RangeNormPoly2dExport(torch.autograd.Function):
    """ONNX export helper: emit RangeNormPoly2d as a single custom op."""

    @staticmethod
    def forward(ctx, x, running_max, upper_bound, eps, hermite_coeffs):
        scale_factor = running_max / upper_bound + eps
        poly_out = _eval_hermite(x / scale_factor, hermite_coeffs)
        return scale_factor * poly_out

    @staticmethod
    def symbolic(g, x, running_max, upper_bound, eps, hermite_coeffs):
        return g.op(
            'nn_tools::RangeNormPoly2d',
            x,
            running_max,
            upper_bound_f=upper_bound,
            degree_i=len(hermite_coeffs) - 1,
            eps_f=eps,
        ).setType(x.type())


class RangeNormPoly2d(nn.Module):
    """Combined range normalization + polynomial activation.

    Applies per-channel range normalization, then a Hermite polynomial
    activation, and rescales back.

    Args:
        hermite_coeffs: Tuple of Hermite expansion coefficients.
        num_features:   Number of channels (0 for lazy initialization).
        upper_bound:    Normalization upper bound.
    """

    def __init__(self, hermite_coeffs, num_features=0, upper_bound=3.0, **kwargs):
        super().__init__()
        self.hermite_coeffs = tuple(hermite_coeffs)
        self.num_features = num_features
        self.upper_bound = upper_bound

        self.rangenorm = RangeNorm2d(num_features, upper_bound=upper_bound, eps=1e-3, momentum=0.1)
        self.poly = Simple_Polyrelu(hermite_coeffs)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _RangeNormPoly2dExport.apply(
                x,
                self.rangenorm.running_max,
                self.upper_bound,
                self.rangenorm.eps,
                self.hermite_coeffs,
            )
        x = self.rangenorm(x)
        x = self.rangenorm.scale_factor * self.poly(x)
        return x

    def extra_repr(self):
        return (
            f'num_features={self.num_features}, upper_bound={self.upper_bound}, degree={len(self.hermite_coeffs) - 1}'
        )


class RangeNorm1d(nn.Module):
    """
    Range Normalization for 1D feature maps.

    Normalizes input to a specific range using running statistics.
    Expects input with shape (B, C, L).
    """

    def __init__(self, num_features=0, upper_bound=3.0, eps=1e-3, momentum=0.1):
        """
        Args:
            num_features: Number of channels (C), 0 for lazy initialization
            upper_bound: Upper bound for normalization
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super(RangeNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.upper_bound = upper_bound
        self.scale_factor = 1.0

        if num_features > 0:
            self.register_buffer('running_max', torch.ones(1, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_max', None)
            self.register_buffer('num_batches_tracked', None)

    def _lazy_init(self, x: torch.Tensor):
        self.num_features = x.shape[1]
        self.running_max = torch.ones(1, self.num_features, device=x.device, dtype=x.dtype)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=x.device)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, L) or (B, C)

        Returns:
            Normalized x
        """
        if self.running_max is None:
            self._lazy_init(x)

        if self.training:
            abs_max = x.abs().amax(dim=(0), keepdim=True)
            if self.num_batches_tracked == 0:
                self.running_max.copy_(abs_max.detach())
            else:
                self.running_max.mul_(1 - self.momentum).add_(self.momentum * abs_max.detach())
            self.num_batches_tracked.add_(1)
        else:
            abs_max = self.running_max

        self.scale_factor = abs_max / self.upper_bound + self.eps
        return x / self.scale_factor


class RangeNormPoly1d(nn.Module):
    """Combined range normalization + polynomial activation for 1D features.

    Args:
        hermite_coeffs: Tuple of Hermite expansion coefficients.
        num_features:   Number of channels (0 for lazy initialization).
        upper_bound:    Normalization upper bound.
    """

    def __init__(self, hermite_coeffs, num_features=0, upper_bound=3.0, **kwargs):
        super().__init__()
        self.hermite_coeffs = tuple(hermite_coeffs)
        self.num_features = num_features
        self.upper_bound = upper_bound

        self.rangenorm = RangeNorm1d(num_features, upper_bound=upper_bound, eps=1e-3, momentum=0.1)
        self.poly = Simple_Polyrelu(hermite_coeffs)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _RangeNormPoly1dExport.apply(
                x,
                self.rangenorm.running_max,
                self.upper_bound,
                self.rangenorm.eps,
                self.hermite_coeffs,
            )
        x = self.rangenorm(x)
        x = self.rangenorm.scale_factor * self.poly(x)
        return x

    def extra_repr(self):
        return (
            f'num_features={self.num_features}, upper_bound={self.upper_bound}, degree={len(self.hermite_coeffs) - 1}'
        )


class PolyBN(nn.Module):
    """
    Fused BatchNorm1D + degree-2 Hermite polynomial activation.

    BN normalizes x to ~N(0,1), then computes:
      a0 + a1 * x_norm + a2 * (x_norm^2 - 1) / sqrt(2)

    Since BN is fused internally, standalone BatchNorm1D layers should be
    removed when using this as act_layer (signaled by fuses_bn = True).
    """

    fuses_bn = True

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.a0 = 0.39894228
        self.a1 = 0.5
        self.a2 = 0.28209479 / np.sqrt(2)

    def forward(self, x):
        x_norm = self.bn(x)
        return self.a0 + self.a1 * x_norm + self.a2 * (x_norm**2 - 1)
