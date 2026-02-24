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

log = logging.getLogger(__name__)


def replace_activation(
    module: nn.Module,
    old_cls: Type[nn.Module] = nn.ReLU,
    new_module_factory: Callable = RangeNormPoly2d,
    upper_bound: float = 3.0,
    degree: int = 4,
    activation: str = 'relu',
):
    """Replace all *old_cls* activations with *new_module_factory*.

    Channel count is automatically inferred at the first forward pass
    via lazy initialization in ``RangeNormPoly2d``.

    Args:
        module:             Model to modify in-place.
        old_cls:            Activation class to replace (default ``nn.ReLU``).
        new_module_factory: Constructor ``(upper_bound, degree, activation) -> nn.Module``.
        upper_bound:        Normalization upper bound.
        degree:             Polynomial degree.
        activation:         Target activation name ('relu' or 'silu').

    Example::

        >>> model = resnet18()
        >>> replace_activation(model, nn.ReLU, RangeNormPoly2d, upper_bound=3.0, degree=4)
    """
    for name, child in list(module.named_children()):
        replace_activation(
            child,
            old_cls=old_cls,
            new_module_factory=new_module_factory,
            upper_bound=upper_bound,
            degree=degree,
            activation=activation,
        )

        if isinstance(child, old_cls):
            new_module = new_module_factory(upper_bound=upper_bound, degree=degree, activation=activation)
            setattr(module, name, new_module)
            log.debug(
                'Replaced %s: %s -> %s(activation=%s)', name, old_cls.__name__, new_module_factory.__name__, activation
            )


def replace_activation_with_poly(
    model: nn.Module,
    old_cls: Type[nn.Module] = nn.ReLU,
    new_module_factory=RangeNormPoly2d,
    upper_bound: float = 3.0,
    degree: int = 4,
) -> nn.Module:
    """Replace all instances of *old_cls* activation with ``RangeNormPoly2d``.

    Supported activation classes: ``nn.ReLU`` and ``nn.SiLU``.
    Channel count is automatically inferred at the first forward pass
    via lazy initialization.

    Args:
        model:       PyTorch model (modified in-place).
        old_cls:     Activation class to replace (default ``nn.ReLU``).
                     Supported: ``nn.ReLU``, ``nn.SiLU``.
        upper_bound: Normalization upper bound.
        degree:      Polynomial degree (2 or 4).

    Returns:
        The same model with activations replaced.

    Raises:
        ValueError: If *old_cls* is not ``nn.ReLU`` or ``nn.SiLU``.

    Example::

        >>> model = resnet20()
        >>> replace_activation_with_poly(model, old_cls=nn.ReLU)
        >>> # or replace SiLU activations
        >>> replace_activation_with_poly(model, old_cls=nn.SiLU, degree=4)
    """
    _supported = (nn.ReLU, nn.SiLU)
    if old_cls not in _supported:
        raise ValueError(
            f'Unsupported activation class: {old_cls.__name__}. '
            f'Supported: {", ".join(c.__name__ for c in _supported)}. '
            f'For other activations, use Chebyshev polynomial fitting.'
        )

    _activation_map = {nn.ReLU: 'relu', nn.SiLU: 'silu'}
    activation = _activation_map[old_cls]

    replace_activation(
        model,
        old_cls=old_cls,
        new_module_factory=new_module_factory,
        upper_bound=upper_bound,
        degree=degree,
        activation=activation,
    )
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


def count_activations(module: nn.Module, activation_cls: Type[nn.Module] = nn.ReLU) -> int:
    """Count the number of *activation_cls* instances in *module*.

    Args:
        module:         PyTorch model.
        activation_cls: Activation class to count.

    Returns:
        Number of matching activations.
    """
    return sum(1 for m in module.modules() if isinstance(m, activation_cls))
