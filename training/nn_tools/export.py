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
"""ONNX export and Conv+BN fusion utilities."""

import logging
import os
from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
import torch.onnx

log = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: Union[Tuple[int, ...], List[Tuple[int, ...]]] = (1, 3, 32, 32),
    opset_version: int = 13,
    dynamic_batch: bool = True,
    remove_identity: bool = True,
    save_h5: bool = True,
    verbose: bool = True,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    do_constant_folding=False,
) -> str:
    """Export a PyTorch model to ONNX.

    BatchNorm is kept as a full operator (not folded into Conv).

    Args:
        model:           PyTorch model.
        save_path:       Output ``.onnx`` file path.
        input_size:      Single input shape ``(N, C, H, W)`` or a list of shapes
                         for models with multiple inputs.
        opset_version:   ONNX opset version.
        dynamic_batch:   Enable dynamic batch-size axis.
        remove_identity: Remove Identity ops after export.
        save_h5:         Also save weights to an H5 file.
        verbose:         Log progress information.
        input_names:     Names for each input tensor. Auto-generated if None.
        output_names:    Names for each output tensor. Auto-generated if None.

    Returns:
        Path to the saved ONNX file.
    """
    export_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    export_model.eval()
    export_model.cpu()

    # Support both single input_size and list of input sizes
    multi_input = isinstance(input_size, (list, tuple)) and isinstance(input_size[0], (list, tuple))
    if multi_input:
        dummy_input = tuple(torch.randn(s) for s in input_size)
        n_inputs = len(input_size)
    else:
        dummy_input = torch.randn(input_size)
        n_inputs = 1

    if input_names is None:
        input_names = [f'input_{i}' for i in range(n_inputs)] if multi_input else ['input']
    if output_names is None:
        output_names = ['output']

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: 'batch_size'} for name in input_names + output_names}

    torch.onnx.export(
        export_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )

    if verbose:
        log.info('ONNX exported: %s', save_path)

    if remove_identity:
        removed_count = remove_identity_nodes(save_path)
        if verbose:
            log.info('Removed %d Identity operators', removed_count)

    try:
        import onnx

        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        if verbose:
            bn_count = sum(1 for n in onnx_model.graph.node if n.op_type == 'BatchNormalization')
            log.info('ONNX verified  |  BatchNorm ops: %d', bn_count)
    except ImportError:
        if verbose:
            log.warning('onnx package not installed, skipping verification')
    except Exception as e:
        if verbose:
            log.warning('ONNX verification failed: %s', e)

    if save_h5:
        save_onnx_weights_to_h5(save_path, verbose=verbose)

    return save_path


def remove_identity_nodes(onnx_path: str) -> int:
    """Remove all Identity operators from an ONNX model file.

    Args:
        onnx_path: Path to ``.onnx`` file (modified in-place).

    Returns:
        Number of removed Identity operators.
    """
    try:
        import onnx
    except ImportError:
        log.warning('onnx package not installed, cannot remove Identity nodes')
        return 0

    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    # Build mapping: Identity output -> Identity input
    identity_map = {}
    for node in graph.node:
        if node.op_type == 'Identity':
            identity_map[node.output[0]] = node.input[0]

    # Redirect all references
    for node in graph.node:
        for i, inp in enumerate(node.input):
            while inp in identity_map:
                inp = identity_map[inp]
                node.input[i] = inp

    # Update graph outputs
    for output in graph.output:
        if output.name in identity_map:
            output.name = identity_map[output.name]

    # Remove Identity nodes
    nodes_to_remove = [n for n in graph.node if n.op_type == 'Identity']
    for node in nodes_to_remove:
        graph.node.remove(node)

    onnx.save(onnx_model, onnx_path)
    return len(nodes_to_remove)


def save_onnx_weights_to_h5(
    onnx_path: str,
    h5_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Extract all weights from an ONNX model and save to H5.

    Args:
        onnx_path: Path to ``.onnx`` model.
        h5_path:   Output H5 path (default: ``<onnx_stem>_weights.h5``).
        verbose:   Log progress information.

    Returns:
        Path to the saved H5 file.
    """
    import onnx
    import h5py
    from onnx import numpy_helper

    if h5_path is None:
        h5_path = os.path.splitext(onnx_path)[0] + '_weights.h5'

    model = onnx.load(onnx_path)

    total_params = 0
    weight_count = 0

    with h5py.File(h5_path, 'w') as f:
        for initializer in model.graph.initializer:
            weight = numpy_helper.to_array(initializer).astype('float64')
            name = initializer.name.replace('/', '_').replace('.', '_')
            f.create_dataset(name, data=weight)
            total_params += weight.size
            weight_count += 1

    if verbose:
        log.info('Weights -> H5: %s  (%d tensors, %s params)', h5_path, weight_count, f'{total_params:,}')

    return h5_path


def load_h5_weights(h5_path: str) -> dict:
    """Load weights from an H5 file.

    Args:
        h5_path: Path to H5 file.

    Returns:
        ``{name: numpy_array}`` dictionary.
    """
    import h5py

    weights = {}
    with h5py.File(h5_path, 'r') as f:
        for name in f.keys():
            weights[name] = f[name][:]
    return weights


# ------------------------------------------------------------------
# Conv+BN fusion & export
# ------------------------------------------------------------------


def _fuse_conv_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d.

    Formulas::

        scale = bn_weight / sqrt(bn_var + eps)
        fused_weight[i] = conv_weight[i] * scale[i]
        fused_bias[i] = scale[i] * (conv_bias[i] - bn_mean[i]) + bn_bias[i]

    Args:
        conv_weight: ``[out_ch, in_ch, kH, kW]``
        conv_bias:   ``[out_ch]`` (pass zeros if no bias)
        bn_weight:   BN gamma ``[out_ch]``
        bn_bias:     BN beta  ``[out_ch]``
        bn_mean:     BN running_mean ``[out_ch]``
        bn_var:      BN running_var  ``[out_ch]``
        eps:         BN epsilon

    Returns:
        ``(fused_weight, fused_bias)``
    """
    import numpy as np

    out_ch = conv_weight.shape[0]
    fused_weight = conv_weight.copy()
    fused_bias = np.zeros(out_ch)

    for i in range(out_ch):
        scale = bn_weight[i] / np.sqrt(bn_var[i] + eps)
        fused_weight[i] = conv_weight[i] * scale
        fused_bias[i] = scale * (conv_bias[i] - bn_mean[i]) + bn_bias[i]

    return fused_weight, fused_bias


def _compute_poly_coeffs(running_max, upper_bound=3.0, eps=1e-3, degree=4):
    """Absorb RangeNormPoly2d scale into standard polynomial coefficients.

    Given::

        s = running_max / upper_bound + eps(per - channel)
        output = s * poly(x / s)

    this function returns per-channel coefficients ``c_k`` such that::

        output = c0 + c1 * x + c2 * x ^ 2 + ... + c_n * x ^ n

    Args:
        running_max: Per-channel running_max ``[C]`` (1-D numpy).
        upper_bound: Normalization upper bound.
        eps:         Epsilon.
        degree:      Polynomial degree (2, 4, or 8).

    Returns:
        ``numpy.ndarray`` of shape ``(degree+1, C)``
        (only even + const + linear terms are non-zero).
    """
    import numpy as np

    s = running_max / upper_bound + eps

    # Fixed Hermite-expansion coefficients
    _a0 = 0.39894228
    _a1 = 0.5
    _a2 = 0.28209479 / np.sqrt(2)
    _a4 = -0.08143375 / np.sqrt(24)
    _a6 = 0.04460310 / np.sqrt(720)
    _a8 = -0.02980170 / np.sqrt(40320)

    C = len(s)

    if degree == 2:
        c0 = s * (_a0 - _a2)
        c1 = np.full(C, _a1)
        c2 = _a2 / s
        return np.array([c0, c1, c2])  # [3, C]

    elif degree == 4:
        c0 = s * (_a0 - _a2 + 3 * _a4)
        c1 = np.full(C, _a1)
        c2 = (_a2 - 6 * _a4) / s
        c3 = np.zeros(C)
        c4 = _a4 / (s**3)
        return np.array([c0, c1, c2, c3, c4])  # [5, C]

    elif degree == 8:
        # Hermite -> standard monomial expansion
        # He2 = t^2 - 1
        # He4 = t^4 - 6t^2 + 3
        # He6 = t^6 - 15t^4 + 45t^2 - 15
        # He8 = t^8 - 28t^6 + 210t^4 - 420t^2 + 105
        #
        # output = s * poly(x/s) -> c_k = p_k / s^(k-1)
        p0 = _a0 - _a2 + 3 * _a4 - 15 * _a6 + 105 * _a8
        p1 = _a1
        p2 = _a2 - 6 * _a4 + 45 * _a6 - 420 * _a8
        p4 = _a4 - 15 * _a6 + 210 * _a8
        p6 = _a6 - 28 * _a8
        p8 = _a8

        c0 = s * p0
        c1 = np.full(C, p1)
        c2 = p2 / s
        c3 = np.zeros(C)
        c4 = p4 / (s**3)
        c5 = np.zeros(C)
        c6 = p6 / (s**5)
        c7 = np.zeros(C)
        c8 = p8 / (s**7)
        return np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8])  # [9, C]

    else:
        raise ValueError(f'Unsupported degree: {degree}')


def fuse_and_export_h5(model, h5_path, upper_bound=3.0, degree=4, eps=1e-3, verbose=True):
    """Fuse Conv+BN, absorb poly scale, and export all weights to H5.

    Automatically handles:

    1. ``Conv2d`` + ``BatchNorm2d`` pairs -> fused ``(weight, bias)``
    2. ``RangeNormPoly2d`` -> per-channel polynomial coefficients
    3. ``Linear`` -> ``(weight, bias)``
    4. Standalone ``Conv2d`` (no BN) -> ``(weight, bias)``

    Args:
        model:       Trained model (with ``RangeNormPoly2d`` activations).
        h5_path:     Output H5 file path.
        upper_bound: ``RangeNormPoly2d`` upper bound.
        degree:      Polynomial degree.
        eps:         ``RangeNormPoly2d`` epsilon.
        verbose:     Log progress information.

    Returns:
        *h5_path*
    """
    import numpy as np
    import h5py
    from .activations import RangeNormPoly2d as _RangeNormPoly2d
    from .activations import Simple_Polyrelu as _Simple_Polyrelu

    sd = model.state_dict()

    def get_np(key):
        return sd[key].detach().cpu().numpy()

    # Collect leaf modules in traversal order
    target_types = (nn.Conv2d, nn.BatchNorm2d, _RangeNormPoly2d, _Simple_Polyrelu, nn.Linear)
    modules_list = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, target_types)]

    fused = {}
    i = 0
    while i < len(modules_list):
        name, mod = modules_list[i]

        if isinstance(mod, nn.Conv2d):
            next_is_bn = i + 1 < len(modules_list) and isinstance(modules_list[i + 1][1], nn.BatchNorm2d)

            conv_w = get_np(f'{name}.weight')
            conv_b = get_np(f'{name}.bias') if mod.bias is not None else np.zeros(conv_w.shape[0])

            if next_is_bn:
                bn_name = modules_list[i + 1][0]
                bn_w = get_np(f'{bn_name}.weight')
                bn_b = get_np(f'{bn_name}.bias')
                bn_m = get_np(f'{bn_name}.running_mean')
                bn_v = get_np(f'{bn_name}.running_var')

                fw, fb = _fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_m, bn_v)
                fused[f'{name}.weight'] = fw
                fused[f'{name}.bias'] = fb
                if verbose:
                    log.info('Fuse Conv+BN: %s + %s', name, bn_name)
                i += 2
            else:
                fused[f'{name}.weight'] = conv_w
                fused[f'{name}.bias'] = conv_b
                if verbose:
                    log.info('Conv (no BN): %s', name)
                i += 1

        elif isinstance(mod, _RangeNormPoly2d):
            running_max = get_np(f'{name}.rangenorm.running_max').flatten()
            coeffs = _compute_poly_coeffs(running_max, upper_bound, eps, degree)
            fused[f'{name}.weight'] = coeffs.flatten()
            if verbose:
                log.info('Poly coeffs:  %s  (%d coeffs x %d ch)', name, coeffs.shape[0], coeffs.shape[1])
            i += 1

        elif isinstance(mod, _Simple_Polyrelu):
            # Determine channel count from preceding Conv/BN layer
            num_channels = 1
            for j in range(i - 1, -1, -1):
                prev_mod = modules_list[j][1]
                if isinstance(prev_mod, nn.BatchNorm2d):
                    num_channels = prev_mod.num_features
                    break
                elif isinstance(prev_mod, nn.Conv2d):
                    num_channels = prev_mod.out_channels
                    break
            # Simple_Polyrelu has no per-channel normalization, equivalent to s = 1
            s = np.ones(num_channels)
            coeffs = _compute_poly_coeffs(s, upper_bound=1.0, eps=0.0, degree=mod.degree)
            fused[f'{name}.weight'] = coeffs.flatten()
            if verbose:
                log.info('Poly coeffs:  %s  (%d coeffs x %d ch)', name, coeffs.shape[0], coeffs.shape[1])
            i += 1

        elif isinstance(mod, nn.Linear):
            fused[f'{name}.weight'] = get_np(f'{name}.weight')
            fused[f'{name}.bias'] = get_np(f'{name}.bias')
            if verbose:
                log.info('Linear:       %s', name)
            i += 1

        else:
            # Standalone BatchNorm2d (not consumed by a preceding Conv) -- skip
            i += 1

    # Save to H5
    h5_dir = os.path.dirname(h5_path)
    if h5_dir:
        os.makedirs(h5_dir, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        for name, data in fused.items():
            log.debug('  %s  shape=%s', name, data.shape)
            f.create_dataset(name, data=data.flatten())

    if verbose:
        total_params = sum(d.size for d in fused.values())
        log.info('Fused H5: %s  (%d tensors, %s params)', h5_path, len(fused), f'{total_params:,}')

    return h5_path
