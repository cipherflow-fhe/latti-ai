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

import torch
import torch.nn as nn

from .activations import RangeNormPoly2d, RangeNormPoly1d, RangeNorm1d, RangeNorm2d, Simple_Polyrelu
from .modules import DepthwiseAvgPool2d

log = logging.getLogger(__name__)


def _is_1d_context(module: nn.Module) -> bool:
    """Return True if *module*'s direct children suggest a 1D spatial context."""
    for child in module.children():
        if isinstance(child, (nn.Conv1d, nn.BatchNorm1d, nn.AvgPool1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d)):
            return True
        if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            return False
    return False


def replace_activation(
    module: nn.Module,
    old_cls: Type[nn.Module],
    new_module_factory: Callable,
    upper_bound: float,
    hermite_coeffs: tuple,
    new_module_factory_1d: Callable | None = None,
):
    """Replace all *old_cls* activations with *new_module_factory* in-place.

    When *new_module_factory_1d* is provided, modules whose siblings are 1D
    operators (Conv1d, BatchNorm1d, etc.) will use the 1D factory instead.
    """
    for name, child in list(module.named_children()):
        replace_activation(child, old_cls, new_module_factory, upper_bound, hermite_coeffs, new_module_factory_1d)

        if isinstance(child, old_cls):
            factory = new_module_factory
            if new_module_factory_1d is not None and _is_1d_context(module):
                factory = new_module_factory_1d
            new_module = factory(hermite_coeffs=hermite_coeffs, upper_bound=upper_bound)
            setattr(module, name, new_module)
            log.debug('Replaced %s: %s -> %s', name, old_cls.__name__, factory.__name__)


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

    When *new_module_factory* is ``RangeNormPoly2d``, activations inside 1D
    contexts (siblings are Conv1d, BatchNorm1d, etc.) are automatically
    replaced with ``RangeNormPoly1d`` instead.

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

    # Auto-select 1D factory when 2D factory is RangeNormPoly2d
    factory_1d = RangeNormPoly1d if new_module_factory is RangeNormPoly2d else None

    replace_activation(model, old_cls, new_module_factory, upper_bound, hermite_coeffs, factory_1d)
    return model


def replace_maxpool_with_avgpool(model: nn.Module) -> nn.Module:
    """Replace all ``nn.MaxPool2d`` / ``nn.MaxPool1d`` with AvgPool in-place.

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
        elif isinstance(child, nn.MaxPool1d):
            avg = nn.AvgPool1d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
            )
            setattr(model, name, avg)
            log.debug(
                'Replaced %s: MaxPool1d -> AvgPool1d(kernel=%s, stride=%s)', name, child.kernel_size, child.stride
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

    Performs in-place replacements:

    1. ``nn.MaxPool2d`` / ``nn.MaxPool1d`` → ``nn.AvgPool2d`` / ``nn.AvgPool1d``
    2. General ``nn.AvgPool2d`` → ``DepthwiseAvgPool2d`` (depthwise conv)
    3. ``nn.ReLU`` → *poly_module* (``RangeNormPoly2d`` or ``RangeNormPoly1d``
       depending on context)

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
    replace_maxpool_with_avgpool(model)
    _replace_general_avgpool_recursive(model, freeze=True)
    replace_activation_with_poly(model, new_module_factory=poly_module, upper_bound=upper_bound, degree=degree)

    if input_size is not None:
        has_lazy_poly2d = any(
            isinstance(m, RangeNormPoly2d) and m.rangenorm.running_max is None for m in model.modules()
        )
        has_lazy_poly1d = any(
            isinstance(m, RangeNormPoly1d) and m.rangenorm.running_max is None for m in model.modules()
        )
        has_lazy_dw = any(isinstance(m, DepthwiseAvgPool2d) and m.conv is None for m in model.modules())
        if has_lazy_poly2d or has_lazy_poly1d or has_lazy_dw:
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


# ------------------------------------------------------------------
# 1D → 2D model conversion
# ------------------------------------------------------------------


def _to_scalar(x):
    """Extract scalar from a single-element tuple/list, or return as-is."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _conv1d_to_conv2d(conv: nn.Conv1d) -> nn.Conv2d:
    """Convert a Conv1d module to an equivalent Conv2d."""
    k = _to_scalar(conv.kernel_size)
    s = _to_scalar(conv.stride)
    p = _to_scalar(conv.padding)
    d = _to_scalar(conv.dilation)

    conv2d = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        dilation=(1, d),
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    # Weight: [out, in/g, k] -> [out, in/g, 1, k]
    with torch.no_grad():
        conv2d.weight.copy_(conv.weight.unsqueeze(2))
        if conv.bias is not None:
            conv2d.bias.copy_(conv.bias)
    return conv2d


def _batchnorm1d_to_batchnorm2d(bn: nn.BatchNorm1d) -> nn.BatchNorm2d:
    """Convert a BatchNorm1d module to an equivalent BatchNorm2d."""
    bn2d = nn.BatchNorm2d(
        num_features=bn.num_features,
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    with torch.no_grad():
        if bn.affine:
            bn2d.weight.copy_(bn.weight)
            bn2d.bias.copy_(bn.bias)
        if bn.track_running_stats and bn.running_mean is not None:
            bn2d.running_mean.copy_(bn.running_mean)
            bn2d.running_var.copy_(bn.running_var)
            bn2d.num_batches_tracked.copy_(bn.num_batches_tracked)
    return bn2d


def _rangenorm1d_to_rangenorm2d(rn: RangeNorm1d) -> RangeNorm2d:
    """Convert a RangeNorm1d module to an equivalent RangeNorm2d."""
    rn2d = RangeNorm2d(
        num_features=rn.num_features,
        upper_bound=rn.upper_bound,
        eps=rn.eps,
        momentum=rn.momentum,
    )
    if rn.running_max is not None:
        with torch.no_grad():
            # [1, C, 1] -> [1, C, 1, 1]
            rn2d.running_max = rn.running_max.unsqueeze(-1)
            rn2d.num_batches_tracked = rn.num_batches_tracked.clone()
    return rn2d


def _rangenormpoly1d_to_rangenormpoly2d(rnp: RangeNormPoly1d) -> RangeNormPoly2d:
    """Convert a RangeNormPoly1d module to an equivalent RangeNormPoly2d."""
    rnp2d = RangeNormPoly2d(
        hermite_coeffs=rnp.hermite_coeffs,
        num_features=rnp.num_features,
        upper_bound=rnp.upper_bound,
    )
    rnp2d.rangenorm = _rangenorm1d_to_rangenorm2d(rnp.rangenorm)
    return rnp2d


def _replace_1d_modules_recursive(module: nn.Module) -> None:
    """Recursively replace all 1D modules with 2D equivalents in-place."""
    for name, child in list(module.named_children()):
        # Handle compound modules first to avoid double-processing their internals
        if isinstance(child, RangeNormPoly1d):
            setattr(module, name, _rangenormpoly1d_to_rangenormpoly2d(child))
            log.debug('Replaced %s: RangeNormPoly1d -> RangeNormPoly2d', name)
            continue

        _replace_1d_modules_recursive(child)

        if isinstance(child, nn.Conv1d):
            setattr(module, name, _conv1d_to_conv2d(child))
            log.debug('Replaced %s: Conv1d -> Conv2d', name)

        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, _batchnorm1d_to_batchnorm2d(child))
            log.debug('Replaced %s: BatchNorm1d -> BatchNorm2d', name)

        elif isinstance(child, RangeNorm1d):
            setattr(module, name, _rangenorm1d_to_rangenorm2d(child))
            log.debug('Replaced %s: RangeNorm1d -> RangeNorm2d', name)

        elif isinstance(child, nn.AvgPool1d):
            k = _to_scalar(child.kernel_size)
            s = _to_scalar(child.stride)
            p = _to_scalar(child.padding)
            setattr(module, name, nn.AvgPool2d(kernel_size=(1, k), stride=(1, s), padding=(0, p)))
            log.debug('Replaced %s: AvgPool1d -> AvgPool2d', name)

        elif isinstance(child, nn.MaxPool1d):
            k = _to_scalar(child.kernel_size)
            s = _to_scalar(child.stride)
            p = _to_scalar(child.padding)
            setattr(module, name, nn.MaxPool2d(kernel_size=(1, k), stride=(1, s), padding=(0, p)))
            log.debug('Replaced %s: MaxPool1d -> MaxPool2d', name)

        elif isinstance(child, nn.AdaptiveAvgPool1d):
            out = _to_scalar(child.output_size)
            setattr(module, name, nn.AdaptiveAvgPool2d(output_size=(1, out)))
            log.debug('Replaced %s: AdaptiveAvgPool1d -> AdaptiveAvgPool2d', name)


class _Unsqueeze2dWrapper(nn.Module):
    """Wrapper that unsqueezes input from (B, C, L) to (B, C, 1, L)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, C, L) -> (B, C, 1, L)
        return self.model(x)


def _has_1d_modules(model: nn.Module) -> bool:
    """Return True if the model contains any 1D spatial modules."""
    _1d_types = (
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.AvgPool1d,
        nn.MaxPool1d,
        nn.AdaptiveAvgPool1d,
        RangeNormPoly1d,
        RangeNorm1d,
    )
    return any(isinstance(m, _1d_types) for m in model.modules())


def convert_1d_to_2d(model: nn.Module) -> nn.Module:
    """Convert a 1D CNN model to an equivalent 2D model for FHE inference.

    All 1D operators are replaced with their 2D equivalents:

    - ``Conv1d(k, s, p)`` → ``Conv2d((1,k), (1,s), (0,p))``
    - ``AvgPool1d(k, s, p)`` → ``AvgPool2d((1,k), (1,s), (0,p))``
    - ``MaxPool1d`` → ``MaxPool2d`` (same pattern)
    - ``AdaptiveAvgPool1d(L)`` → ``AdaptiveAvgPool2d((1, L))``
    - ``BatchNorm1d`` → ``BatchNorm2d``
    - ``RangeNormPoly1d`` → ``RangeNormPoly2d``
    - ``RangeNorm1d`` → ``RangeNorm2d``

    The returned model is wrapped so that a 1D input ``(B, C, L)`` is
    automatically unsqueezed to ``(B, C, 1, L)`` before the first layer.

    This function should be called **after training** and **before ONNX
    export** so that the exported model uses only 2D operators, which are
    fully supported by the FHE compilation and inference pipeline.

    Args:
        model: Trained PyTorch model with 1D operators (modified in-place).

    Returns:
        A wrapper module whose forward accepts ``(B, C, L)`` input and
        internally runs all computation in 2D.

    Example::

        >>> model = my_conv1d_model()
        >>> model_2d = convert_1d_to_2d(model)
        >>> export_to_onnx(model_2d, save_path='model.onnx', input_size=(1, 16, 128))
    """
    _replace_1d_modules_recursive(model)
    log.info('Converted all 1D modules to 2D equivalents')
    return _Unsqueeze2dWrapper(model)
