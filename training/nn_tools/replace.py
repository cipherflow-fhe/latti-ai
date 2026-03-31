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
"""Activation replacement utilities.

Replace standard activations (e.g. nn.ReLU) with custom polynomial
activations in an existing model, in-place.
"""

import logging
from typing import Type, Callable

import torch.nn as nn

from .activations import RangeNormPoly2d, Simple_Polyrelu
from .modules import DepthwiseAvgPool2d

log = logging.getLogger(__name__)


def replace_activation(
    module: nn.Module,
    old_cls: Type[nn.Module],
    new_module_factory: Callable,
    upper_bound: float,
    hermite_coeffs: tuple,
):
    """Replace all *old_cls* activations with *new_module_factory* in-place."""
    for name, child in list(module.named_children()):
        replace_activation(child, old_cls, new_module_factory, upper_bound, hermite_coeffs)

        if isinstance(child, old_cls):
            new_module = new_module_factory(hermite_coeffs=hermite_coeffs, upper_bound=upper_bound)
            setattr(module, name, new_module)
            log.debug('Replaced %s: %s -> %s', name, old_cls.__name__, new_module_factory.__name__)


def replace_activation_with_poly(
    model: nn.Module,
    old_cls: Type[nn.Module] = nn.ReLU,
    new_module_factory=RangeNormPoly2d,
    upper_bound: float = 3.0,
    degree: int = 4,
) -> nn.Module:
    """Replace all instances of *old_cls* activation with polynomial activation.

    Supports any ``nn.Module`` activation class. Hermite expansion coefficients
    are computed automatically via numerical integration by instantiating the
    activation module and evaluating it.

    Args:
        model:       PyTorch model (modified in-place).
        old_cls:     Activation class to replace (default ``nn.ReLU``).
        upper_bound: Normalization upper bound.
        degree:      Polynomial degree.

    Returns:
        The same model with activations replaced.

    Example::

        >>> model = resnet20()
        >>> replace_activation_with_poly(model, old_cls=nn.ReLU)
        >>> replace_activation_with_poly(model, old_cls=nn.GELU, degree=4)
    """
    from .eval_fn_hat_for_aespa import get_hermite_coeffs_for_module

    hermite_coeffs = get_hermite_coeffs_for_module(old_cls, degree=degree)
    log.info(
        'Hermite coefficients for %s (degree=%d): %s',
        old_cls.__name__,
        degree,
        ', '.join(f'{c:.8f}' for c in hermite_coeffs),
    )

    replace_activation(model, old_cls, new_module_factory, upper_bound, hermite_coeffs)
    return model


def replace_maxpool_with_avgpool(model: nn.Module) -> nn.Module:
    """Replace all ``nn.MaxPool2d`` with ``nn.AvgPool2d`` in-place.

    FHE does not support comparison operations, so MaxPool cannot be
    evaluated on ciphertexts.  AvgPool is a linear operation and can
    be computed directly.

    Args:
        model: PyTorch model (modified in-place).

    Returns:
        The same model with MaxPool layers replaced.

    Example::

        >>> model = resnet18()
        >>> replace_maxpool_with_avgpool(model)
    """
    for name, child in list(model.named_children()):
        replace_maxpool_with_avgpool(child)

        if isinstance(child, nn.MaxPool2d):
            avg = nn.AvgPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
            )
            setattr(model, name, avg)
            log.debug(
                'Replaced %s: MaxPool2d -> AvgPool2d(kernel=%s, stride=%s)', name, child.kernel_size, child.stride
            )
    return model


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _needs_depthwise_replacement(pool: nn.AvgPool2d) -> bool:
    """Return True if this AvgPool2d cannot be handled by the FHE multiplexed avgpool.

    The multiplexed avgpool requires stride == kernel_size and both are powers of 2.
    Any other configuration must be replaced with a depthwise convolution.
    """
    ks = pool.kernel_size if isinstance(pool.kernel_size, (tuple, list)) else (pool.kernel_size, pool.kernel_size)
    st = pool.stride if isinstance(pool.stride, (tuple, list)) else (pool.stride, pool.stride)

    if ks[0] != st[0] or ks[1] != st[1]:
        return True
    if not _is_power_of_2(ks[0]) or not _is_power_of_2(ks[1]):
        return True
    return False


def _replace_general_avgpool_recursive(model: nn.Module, freeze: bool) -> None:
    for name, child in list(model.named_children()):
        _replace_general_avgpool_recursive(child, freeze=freeze)

        if isinstance(child, nn.AvgPool2d) and _needs_depthwise_replacement(child):
            ks = (
                child.kernel_size
                if isinstance(child.kernel_size, (tuple, list))
                else (child.kernel_size, child.kernel_size)
            )
            st = child.stride if isinstance(child.stride, (tuple, list)) else (child.stride, child.stride)
            pad = child.padding if isinstance(child.padding, (tuple, list)) else (child.padding, child.padding)

            dw = DepthwiseAvgPool2d(
                kernel_size=ks,
                stride=st,
                padding=pad,
                freeze=freeze,
            )
            setattr(model, name, dw)
            log.debug(
                'Replaced %s: AvgPool2d -> DepthwiseAvgPool2d(kernel=%s, stride=%s, padding=%s)',
                name,
                ks,
                st,
                pad,
            )


def replace_general_avgpool_with_depthwise_conv(
    model: nn.Module,
    input_size: tuple,
    freeze: bool = True,
) -> nn.Module:
    """Replace general ``nn.AvgPool2d`` with ``DepthwiseAvgPool2d`` in-place.

    A "general" AvgPool is one where ``kernel_size != stride`` or
    ``kernel_size`` is not a power of 2.  These cannot be evaluated by the
    FHE multiplexed avgpool operator, so they are converted to an
    equivalent depthwise separable convolution with fixed weights
    ``1 / (k0 * k1)`` and zero bias.

    AvgPool layers where ``kernel_size == stride`` and both are powers of 2
    are left unchanged (handled by the existing multiplexed avgpool).

    This function should be called **after training** and **before ONNX
    export**, so that training accuracy is not affected.

    Args:
        model:      PyTorch model (modified in-place).
        input_size: Model input shape (e.g. ``(1, 3, 32, 32)``), used to
                    run a dummy forward pass that initialises the depthwise
                    conv layers.
        freeze:     If ``True`` (default), the depthwise conv weights are
                    frozen.

    Returns:
        The same model with general AvgPool layers replaced.

    Example::

        >>> model = resnet18()
        >>> replace_general_avgpool_with_depthwise_conv(model, input_size=(1, 3, 32, 32))
    """
    import torch

    _replace_general_avgpool_recursive(model, freeze=freeze)

    # Trigger lazy init of DepthwiseAvgPool2d via a dummy forward pass
    has_lazy = any(isinstance(m, DepthwiseAvgPool2d) and m.conv is None for m in model.modules())
    if has_lazy:
        model.eval()
        with torch.no_grad():
            model(torch.randn(*input_size))

    return model


def prepare_for_fhe(
    model: nn.Module,
    poly_module=RangeNormPoly2d,
    upper_bound: float = 3.0,
    degree: int = 4,
    input_size: tuple = None,
) -> nn.Module:
    """Convert a standard PyTorch model to be FHE-compatible.

    Performs three in-place replacements:

    1. ``nn.MaxPool2d`` → ``nn.AvgPool2d``
    2. General ``nn.AvgPool2d`` → ``DepthwiseAvgPool2d`` (depthwise conv)
    3. ``nn.ReLU`` → *poly_module* (default ``RangeNormPoly2d``)

    When *input_size* is provided, a dummy forward pass is run to trigger
    lazy initialization of ``RangeNormPoly2d`` and ``DepthwiseAvgPool2d``
    buffers (required before ONNX export).

    Args:
        model:       PyTorch model (modified in-place).
        poly_module: Polynomial activation constructor (default ``RangeNormPoly2d``).
        upper_bound: Normalization upper bound for the polynomial activation.
        degree:      Polynomial degree.
        input_size:  Input tensor shape (e.g. ``(1, 3, 32, 32)``).
                     If provided, runs a dummy forward pass after replacement.

    Returns:
        The same model with activations and pooling layers replaced.

    Example::

        >>> model = resnet20()
        >>> prepare_for_fhe(model, input_size=(1, 3, 32, 32))
    """
    import torch

    replace_maxpool_with_avgpool(model)
    _replace_general_avgpool_recursive(model, freeze=True)
    replace_activation_with_poly(model, new_module_factory=poly_module, upper_bound=upper_bound, degree=degree)

    if input_size is not None:
        has_lazy_poly = any(isinstance(m, RangeNormPoly2d) and m.rangenorm.running_max is None for m in model.modules())
        has_lazy_dw = any(isinstance(m, DepthwiseAvgPool2d) and m.conv is None for m in model.modules())
        if has_lazy_poly or has_lazy_dw:
            model.eval()
            with torch.no_grad():
                multi_input = isinstance(input_size[0], (list, tuple))
                if multi_input:
                    model(*[torch.randn(*s) for s in input_size])
                else:
                    model(torch.randn(*input_size))

    return model


def count_activations(module: nn.Module, activation_cls: Type[nn.Module] = nn.ReLU) -> int:
    """Count the number of *activation_cls* instances in *module*.

    Args:
        module:         PyTorch model.
        activation_cls: Activation class to count.

    Returns:
        Number of matching activations.
    """
    return sum(1 for m in module.modules() if isinstance(m, activation_cls))
