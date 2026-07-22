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
"""Feature-mat H5 exporter.

This exporter treats the compiled CT JSON as the source of runtime layer
structure. Custom ONNX ops are only used as parameter sources for the final CT
layers produced by the compiler, e.g. CustomMultiHeadAttention is exported as
q/k/v CPMM weights, attention gamma/poly coefficients, and output projection.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import onnx
from onnx import helper, numpy_helper

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionSource:
    qkv_weight_path: str
    qkv_bias_path: str
    proj_weight_path: str
    proj_bias_path: str
    running_max_path: str
    coeff_paths: tuple[str, ...]
    gamma_path: str
    coeffs_path: str
    upper_bound: float
    eps: float


@dataclass(frozen=True)
class PolyActRNSource:
    running_max_path: str
    coeff_paths: tuple[str, ...]
    degree: int
    upper_bound: float
    eps: float


def export_feature_mat_h5_from_onnx(
    onnx_path: str,
    json_path: str,
    h5_path: str,
    verbose: bool = True,
    model_type: str = '',
) -> str:
    """Export H5 parameters for a compiled feature_mat CT graph.

    Args:
        onnx_path: Original ONNX model path.
        json_path: Final compiled ``nn_layers_ct_*.json`` path.
        h5_path: Output H5 path.
        verbose: Log exported tensor information.

    Returns:
        The output H5 path.
    """
    onnx_model = onnx.load(onnx_path)
    onnx_weights = {init.name: numpy_helper.to_array(init).astype('float64') for init in onnx_model.graph.initializer}
    attention_sources, polyact_sources = _index_onnx_sources(onnx_model)

    with open(json_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    if not model_type:
        task_config_path = os.path.join(os.path.dirname(json_path), 'task_config.json')
        if os.path.exists(task_config_path):
            with open(task_config_path, 'r', encoding='utf-8') as f:
                model_type = str(json.load(f).get('model_type', ''))

    features: dict[str, dict[str, Any]] = graph['feature']
    layers: dict[str, dict[str, Any]] = graph['layer']
    standalone_gamma_paths = _standalone_pcmgamma_paths(layers)
    out: dict[str, np.ndarray] = {}

    for layer_key, layer in layers.items():
        ltype = layer.get('type')
        if ltype == 'parcpmm':
            _export_parcpmm(layer_key, layer, features, onnx_weights, out)
        elif ltype == 'pdmpcmm':
            _export_pdmpcmm(layer_key, layer, features, onnx_weights, out, model_type=model_type)
        elif ltype == 'pcmgamma':
            _export_pcmgamma(layer_key, layer, features, onnx_weights, attention_sources, polyact_sources, out)
        elif ltype == 'pdmgamma':
            _export_pdmgamma(layer_key, layer, features, onnx_weights, attention_sources, polyact_sources, out)
        elif ltype == 'pdmupperpoly':
            _export_pdmupperpoly(layer_key, layer, features, out)
        elif ltype == 'pcmpoly':
            _export_pcmpoly(
                layer_key,
                layer,
                features,
                onnx_weights,
                attention_sources,
                polyact_sources,
                standalone_gamma_paths,
                out,
                model_type=model_type,
            )
        elif ltype == 'pdmpoly':
            _export_pdmpoly(
                layer_key,
                layer,
                features,
                onnx_weights,
                attention_sources,
                polyact_sources,
                standalone_gamma_paths,
                out,
                model_type=model_type,
            )
        elif ltype == 'pcmaffine':
            _export_pcmaffine(layer_key, layer, features, onnx_weights, out)
        elif ltype == 'pdmaffine':
            _export_pdmaffine(layer_key, layer, features, onnx_weights, out)
        elif ltype in ('add_pt', 'pcm_add_pt'):
            _export_add_pt(layer_key, layer, features, onnx_weights, out)
        elif ltype == 'pdm_add_pt':
            _export_pdm_add_pt(layer_key, layer, features, onnx_weights, out)

    h5_dir = os.path.dirname(h5_path)
    if h5_dir:
        os.makedirs(h5_dir, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        for name, data in out.items():
            f.create_dataset(name, data=np.asarray(data, dtype='float64').reshape(-1))

    if verbose:
        total_params = sum(data.size for data in out.values())
        log.info('FeatureMat ONNX→H5: %s  (%d tensors, %s params)', h5_path, len(out), f'{total_params:,}')

    return h5_path


def _index_onnx_sources(onnx_model: onnx.ModelProto) -> tuple[dict[str, AttentionSource], dict[str, PolyActRNSource]]:
    attention_sources: dict[str, AttentionSource] = {}
    polyact_sources: dict[str, PolyActRNSource] = {}

    for node in onnx_model.graph.node:
        attrs = _attrs(node)
        if node.op_type == 'CustomMultiHeadAttention':
            qkv_weight_path = _input_or_empty(node, 1)
            qkv_bias_path = _input_or_empty(node, 2)
            running_max_path = _input_or_empty(node, 5)
            coeff_paths = tuple(node.input[6:])
            gamma_path = _attention_gamma_path(running_max_path, node.name)
            coeffs_path = _attention_coeffs_path(coeff_paths, running_max_path, node.name)
            source = AttentionSource(
                qkv_weight_path=qkv_weight_path,
                qkv_bias_path=qkv_bias_path,
                proj_weight_path=_input_or_empty(node, 3),
                proj_bias_path=_input_or_empty(node, 4),
                running_max_path=running_max_path,
                coeff_paths=coeff_paths,
                gamma_path=gamma_path,
                coeffs_path=coeffs_path,
                upper_bound=float(attrs.get('upper_bound', 3.0)),
                eps=float(attrs.get('eps', 1e-3)),
            )
            attention_sources[gamma_path] = source
            attention_sources[coeffs_path] = source

        elif node.op_type == 'PolyActRN':
            running_max_path = _input_or_empty(node, 1)
            if not running_max_path:
                continue
            attrs = _attrs(node)
            polyact_sources[running_max_path] = PolyActRNSource(
                running_max_path=running_max_path,
                coeff_paths=tuple(node.input[2:]),
                degree=int(attrs.get('degree', 4)),
                upper_bound=float(attrs.get('upper_bound', 3.0)),
                eps=float(attrs.get('eps', 1e-3)),
            )

    return attention_sources, polyact_sources


def _standalone_pcmgamma_paths(layers: dict[str, dict[str, Any]]) -> set[str]:
    paths = set()
    for layer in layers.values():
        if layer.get('type') not in ('pcmgamma', 'pdmgamma'):
            continue
        path = layer.get('gamma_path') or layer.get('weight_path')
        if path:
            paths.add(path)
    return paths


def _export_parcpmm(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> None:
    weight_path = _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    fout = _feature(features, layer['feature_output'][0], layer_key)
    if layer.get('type') == 'pdmpcmm' and layer.get('weight_shape'):
        expected_shape = tuple(int(x) for x in layer['weight_shape'])
    elif layer.get('type') == 'pdmpcmm':
        expected_shape = (int(fout['shape'][0]), int(fin['shape'][0]))
    else:
        expected_shape = (int(fin['shape'][1]), int(fout['shape'][1]))

    weight = _resolve_parcpmm_weight(weight_path, layer, features, onnx_weights)
    weight = _reshape_checked(weight, expected_shape, layer_key, weight_path)
    _put(out, weight_path, weight, layer_key)

    bias_path = layer.get('bias_path', '')
    if bias_path:
        expected_bias_shape = (expected_shape[0] if layer.get('type') == 'pdmpcmm' else expected_shape[1],)
        bias = _resolve_parcpmm_bias(bias_path, layer, features, onnx_weights)
        bias = _reshape_checked(bias, expected_bias_shape, layer_key, bias_path)
        _put(out, bias_path, bias, layer_key)


def _export_pdmpcmm(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
    model_type: str = '',
) -> None:
    weight_path = _required_path(layer, 'weight_path', layer_key)
    fout = _feature(features, layer['feature_output'][0], layer_key)

    weight = _resolve_parcpmm_weight(weight_path, layer, features, onnx_weights)
    logical_shape = _pdmpcmm_logical_weight_shape(weight, layer, layer_key, weight_path)
    weight = _pdmpcmm_weight_matrix(weight, logical_shape, layer_key, weight_path, model_type=model_type)
    _put(out, weight_path, weight, layer_key)

    bias_path = layer.get('bias_path', '')
    if not bias_path:
        return

    bias = _resolve_parcpmm_bias(bias_path, layer, features, onnx_weights)
    bias = np.asarray(bias, dtype='float64')
    output_rows = logical_shape[0]
    output_shape = (output_rows, int(fout['shape'][1]))
    if bias.size == output_rows:
        _put(out, bias_path, bias.reshape(output_rows), layer_key)
    elif bias.size == output_shape[0] * output_shape[1]:
        zero_bias = np.zeros(output_rows, dtype=np.float64)
        _put(out, bias_path, zero_bias, layer_key)
    else:
        raise ValueError(
            f'{layer_key}: {bias_path} size {bias.size} does not match pdmpcmm vector bias '
            f'({output_rows},) or fused matrix bias {output_shape}'
        )


def _export_pdmgamma(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    attention_sources: dict[str, AttentionSource],
    polyact_sources: dict[str, PolyActRNSource],
    out: dict[str, np.ndarray],
) -> None:
    dst_path = layer.get('gamma_path') or _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][0]),)

    if 'btp_scale' in layer:
        gamma = np.full(expected_shape, float(layer['btp_scale']), dtype=np.float64)
    elif 'scalar_value' in layer:
        gamma = np.full(expected_shape, float(layer['scalar_value']), dtype=np.float64)
    elif dst_path in attention_sources:
        source = attention_sources[dst_path]
        gamma = 1.0 / _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
    else:
        source_path = layer.get('running_max_path') or dst_path
        if source_path in polyact_sources:
            source = polyact_sources[source_path]
            gamma = 1.0 / _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
        elif dst_path in onnx_weights:
            gamma = onnx_weights[dst_path].copy().reshape(-1)
        else:
            raise KeyError(f'{layer_key}: cannot resolve pdmgamma source for {dst_path}')

    gamma = _reshape_checked(gamma, expected_shape, layer_key, dst_path)
    _put(out, dst_path, gamma, layer_key)


def _export_pdmpoly(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    attention_sources: dict[str, AttentionSource],
    polyact_sources: dict[str, PolyActRNSource],
    standalone_gamma_paths: set[str],
    out: dict[str, np.ndarray],
    model_type: str = '',
) -> None:
    dst_path = layer.get('coeffs_path') or _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    degree = int(layer.get('degree', layer.get('order', 4)))
    expected_shape = (degree + 1, int(fin['shape'][0]))

    if dst_path in attention_sources:
        source = attention_sources[dst_path]
        scale = _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
        gamma_path = layer.get('gamma_path') or source.gamma_path
        coeffs = _coeff_matrix(
            source.coeff_paths,
            degree,
            scale,
            onnx_weights,
            absorb_input_gamma=gamma_path not in standalone_gamma_paths,
        )
        if model_type != 'bert':
            coeffs = coeffs / _sequence_length(fin, layer_key)
    else:
        source_path = layer.get('running_max_path') or dst_path
        if source_path in polyact_sources:
            source = polyact_sources[source_path]
            scale = _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
            gamma_path = layer.get('gamma_path') or _polyact_gamma_path(source.running_max_path)
            coeffs = _coeff_matrix(
                source.coeff_paths,
                degree,
                scale,
                onnx_weights,
                absorb_input_gamma=gamma_path not in standalone_gamma_paths,
            )
        elif dst_path in onnx_weights:
            coeffs = onnx_weights[dst_path].copy()
        else:
            raise KeyError(f'{layer_key}: cannot resolve pdmpoly source for {dst_path}')

    coeffs = _reshape_checked(coeffs, expected_shape, layer_key, dst_path)
    _put(out, dst_path, coeffs, layer_key)


def _export_pdmupperpoly(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    out: dict[str, np.ndarray],
) -> None:
    dst_path = layer.get('coeffs_path') or _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    order = int(layer.get('order', 15))
    expected_shape = (order + 1, int(fin['shape'][0]))
    if 'coefficients' not in layer:
        raise KeyError(f'{layer_key}: pdmupperpoly requires coefficients in layer JSON')
    coeffs = np.asarray(_parse_coefficients(layer['coefficients']), dtype='float64')
    if coeffs.size != order + 1:
        raise ValueError(f'{layer_key}: pdmupperpoly has {coeffs.size} coefficients, expected {order + 1}')
    coeffs = np.tile(coeffs.reshape(order + 1, 1), (1, expected_shape[1]))
    _put(out, dst_path, coeffs, layer_key)


def _export_pdmaffine(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> None:
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][0]),)
    for path_key in ('weight_path', 'bias_path', 'gamma_path', 'beta_path'):
        path = layer.get(path_key, '')
        if not path:
            continue
        if path not in onnx_weights:
            raise KeyError(f'{layer_key}: {path_key} not found in ONNX: {path}')
        data = _reshape_checked(onnx_weights[path], expected_shape, layer_key, path)
        _put(out, path, data, layer_key)


def _export_pdm_add_pt(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> None:
    path = layer.get('weight_path') or layer.get('bias_path')
    if not path:
        raise KeyError(f'{layer_key}: pdm_add_pt requires weight_path or bias_path')
    if path not in onnx_weights:
        raise KeyError(f'{layer_key}: pdm_add_pt source not found in ONNX: {path}')
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][0]), int(fin['shape'][1]))
    data = np.asarray(onnx_weights[path], dtype='float64')
    if data.ndim == 3 and data.shape[0] == 1:
        data = data.reshape(data.shape[1], data.shape[2])
    if layer.get('source_op_type') == 'CustomPositionEmbedding':
        if data.ndim == 2 and data.shape[1] == expected_shape[0] and data.shape[0] >= expected_shape[1]:
            data = data[: expected_shape[1], :].T.copy()
        elif data.ndim == 2 and tuple(data.T.shape) == expected_shape:
            data = data.T.copy()
        else:
            raise ValueError(
                f'{layer_key}: CustomPositionEmbedding {path} shape {tuple(data.shape)} cannot transpose to '
                f'pdm_add_pt matrix shape {expected_shape}'
            )
    elif data.ndim == 2 and tuple(data.T.shape) == expected_shape:
        data = data.T.copy()
    elif data.ndim == 2 and tuple(data.shape) == expected_shape:
        data = data.copy()
    elif data.size == expected_shape[0] * expected_shape[1]:
        data = data.reshape(expected_shape)
    else:
        raise ValueError(
            f'{layer_key}: {path} size {data.size} does not match pdm_add_pt matrix shape {expected_shape}'
        )
    _put(out, path, data, layer_key)


def _export_pcmgamma(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    attention_sources: dict[str, AttentionSource],
    polyact_sources: dict[str, PolyActRNSource],
    out: dict[str, np.ndarray],
) -> None:
    dst_path = layer.get('gamma_path') or _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][1]),)

    if 'btp_scale' in layer:
        gamma = np.full(expected_shape, float(layer['btp_scale']), dtype=np.float64)
    elif 'scalar_value' in layer:
        gamma = np.full(expected_shape, float(layer['scalar_value']), dtype=np.float64)
    elif dst_path in attention_sources:
        source = attention_sources[dst_path]
        gamma = 1.0 / _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
    else:
        source_path = layer.get('running_max_path') or dst_path
        if source_path in polyact_sources:
            source = polyact_sources[source_path]
            gamma = 1.0 / _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
        elif dst_path in onnx_weights:
            gamma = onnx_weights[dst_path].copy().reshape(-1)
        else:
            raise KeyError(f'{layer_key}: cannot resolve pcmgamma source for {dst_path}')

    gamma = _reshape_checked(gamma, expected_shape, layer_key, dst_path)
    _put(out, dst_path, gamma, layer_key)


def _export_pcmpoly(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    attention_sources: dict[str, AttentionSource],
    polyact_sources: dict[str, PolyActRNSource],
    standalone_gamma_paths: set[str],
    out: dict[str, np.ndarray],
    model_type: str = '',
) -> None:
    dst_path = layer.get('coeffs_path') or _required_path(layer, 'weight_path', layer_key)
    fin = _feature(features, layer['feature_input'][0], layer_key)
    degree = int(layer.get('degree', layer.get('order', 4)))
    expected_shape = (degree + 1, int(fin['shape'][1]))

    if dst_path in attention_sources:
        source = attention_sources[dst_path]
        scale = _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
        gamma_path = layer.get('gamma_path') or source.gamma_path
        coeffs = _coeff_matrix(
            source.coeff_paths,
            degree,
            scale,
            onnx_weights,
            absorb_input_gamma=gamma_path not in standalone_gamma_paths,
        )
        if model_type != 'bert':
            coeffs = coeffs / _sequence_length(fin, layer_key)
    else:
        source_path = layer.get('running_max_path') or dst_path
        if source_path in polyact_sources:
            source = polyact_sources[source_path]
            scale = _scale_factor(source.running_max_path, source.upper_bound, source.eps, onnx_weights)
            gamma_path = layer.get('gamma_path') or _polyact_gamma_path(source.running_max_path)
            coeffs = _coeff_matrix(
                source.coeff_paths,
                degree,
                scale,
                onnx_weights,
                absorb_input_gamma=gamma_path not in standalone_gamma_paths,
            )
        elif dst_path in onnx_weights:
            coeffs = onnx_weights[dst_path].copy()
        else:
            raise KeyError(f'{layer_key}: cannot resolve pcmpoly source for {dst_path}')

    coeffs = _reshape_checked(coeffs, expected_shape, layer_key, dst_path)
    _put(out, dst_path, coeffs, layer_key)


def _export_pcmaffine(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> None:
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][1]),)
    for path_key in ('weight_path', 'bias_path', 'gamma_path', 'beta_path'):
        path = layer.get(path_key, '')
        if not path:
            continue
        if path not in onnx_weights:
            raise KeyError(f'{layer_key}: {path_key} not found in ONNX: {path}')
        data = _reshape_checked(onnx_weights[path], expected_shape, layer_key, path)
        _put(out, path, data, layer_key)


def _export_add_pt(
    layer_key: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
    out: dict[str, np.ndarray],
) -> None:
    path = layer.get('weight_path') or layer.get('bias_path')
    if not path:
        raise KeyError(f'{layer_key}: add_pt/pcm_add_pt/pdm_add_pt requires weight_path or bias_path')
    if path not in onnx_weights:
        raise KeyError(f'{layer_key}: add_pt source not found in ONNX: {path}')
    fin = _feature(features, layer['feature_input'][0], layer_key)
    expected_shape = (int(fin['shape'][0]), int(fin['shape'][1]))
    data = np.asarray(onnx_weights[path], dtype='float64').reshape(-1)
    if data.size == expected_shape[0] * expected_shape[1]:
        data = data.reshape(expected_shape)
    else:
        data = _reshape_checked(data, expected_shape, layer_key, path)
    _put(out, path, data, layer_key)


def _pdmpcmm_logical_weight_shape(
    weight: np.ndarray,
    layer: dict[str, Any],
    layer_key: str,
    weight_path: str,
) -> tuple[int, int]:
    if layer.get('weight_shape'):
        shape = tuple(int(x) for x in layer['weight_shape'])
        if len(shape) != 2:
            raise ValueError(f'{layer_key}: {weight_path} expects 2D weight_shape, got {shape}')
        return shape

    weight = np.asarray(weight)
    if weight.ndim != 2:
        raise ValueError(f'{layer_key}: {weight_path} requires 2D weight or explicit weight_shape')
    return int(weight.shape[0]), int(weight.shape[1])


def _pdmpcmm_weight_matrix(
    weight: np.ndarray,
    logical_shape: tuple[int, int],
    layer_key: str,
    weight_path: str,
    model_type: str = '',
) -> np.ndarray:
    weight = np.asarray(weight, dtype='float64')
    if model_type == 'bert':
        if weight.ndim == 2 and tuple(weight.shape) == logical_shape:
            return weight.copy()
        return _reshape_checked(weight, logical_shape, layer_key, weight_path)
    if weight.ndim == 2 and tuple(weight.T.shape) == logical_shape:
        return weight.T.copy()
    if weight.ndim == 2 and tuple(weight.shape) == logical_shape:
        return weight.copy()
    return _reshape_checked(weight, logical_shape, layer_key, weight_path)


def _resolve_parcpmm_weight(
    weight_path: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
) -> np.ndarray:
    qkv_source, part = _qkv_source(weight_path, '.weight')
    if qkv_source in onnx_weights and part in ('q', 'k', 'v'):
        q, k, v = np.split(onnx_weights[qkv_source], 3, axis=1)
        weight = {'q': q, 'k': k, 'v': v}[part].copy()
        if part == 'q':
            weight *= 1.0 / math.sqrt(_layer_head_dim(layer, features))
        if layer.get('weight_multiplier') is not None:
            weight *= float(layer['weight_multiplier'])
        return weight

    if weight_path not in onnx_weights:
        raise KeyError(f'{layer.get("type", "parcpmm")}: weight not found in ONNX: {weight_path}')
    weight = onnx_weights[weight_path].copy()
    if layer.get('weight_multiplier') is not None:
        weight = weight * float(layer['weight_multiplier'])
    return weight


def _resolve_parcpmm_bias(
    bias_path: str,
    layer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    onnx_weights: dict[str, np.ndarray],
) -> np.ndarray:
    qkv_source, part = _qkv_source(bias_path, '.bias')
    if qkv_source in onnx_weights and part in ('q', 'k', 'v'):
        q, k, v = np.split(onnx_weights[qkv_source].reshape(-1), 3)
        bias = {'q': q, 'k': k, 'v': v}[part].copy()
        if part == 'q':
            bias *= 1.0 / math.sqrt(_layer_head_dim(layer, features))
        if layer.get('bias_multiplier') is not None:
            bias *= float(layer['bias_multiplier'])
        return bias

    if bias_path not in onnx_weights:
        raise KeyError(f'{layer.get("type", "parcpmm")}: bias not found in ONNX: {bias_path}')
    bias = onnx_weights[bias_path].copy()
    if layer.get('bias_multiplier') is not None:
        bias = bias * float(layer['bias_multiplier'])
    return bias


def _qkv_source(path: str, suffix: str) -> tuple[str, str]:
    for part in ('q', 'k', 'v'):
        part_suffix = f'.{part}{suffix}'
        if path.endswith(part_suffix):
            prefix = path[: -len(part_suffix)]
            return f'{prefix}.qkv{suffix}', part
    return '', ''


def _scale_factor(path: str, upper_bound: float, eps: float, onnx_weights: dict[str, np.ndarray]) -> np.ndarray:
    if path not in onnx_weights:
        raise KeyError(f'running_max not found in ONNX: {path}')
    return onnx_weights[path].reshape(-1).astype('float64') / upper_bound + eps


def _coeff_matrix(
    coeff_paths: tuple[str, ...],
    degree: int,
    scale: np.ndarray,
    onnx_weights: dict[str, np.ndarray],
    *,
    absorb_input_gamma: bool = False,
) -> np.ndarray:
    coeff = np.zeros(degree + 1, dtype='float64')
    for path in coeff_paths:
        idx = _coeff_index(path)
        if idx is None or idx > degree:
            continue
        if path not in onnx_weights:
            raise KeyError(f'polynomial coeff not found in ONNX: {path}')
        coeff[idx] = float(np.asarray(onnx_weights[path]).reshape(-1)[0])

    # Hermite basis → standard monomial basis:
    #   He_2(x) = x² - 1       →  a2 contributes -a2 to c0
    #   He_4(x) = x⁴ - 6x² + 3 →  a4 contributes +3a4 to c0, -6a4 to c2
    if degree >= 2:
        coeff[0] -= coeff[2]
    if degree >= 4:
        coeff[0] += 3 * coeff[4]
        coeff[2] -= 6 * coeff[4]

    if absorb_input_gamma:
        exponents = 1 - np.arange(degree + 1, dtype='float64').reshape(-1, 1)
        scale_factor = np.power(scale.reshape(1, -1), exponents)
    else:
        scale_factor = scale.reshape(1, -1)
    return coeff.reshape(-1, 1) * scale_factor


def _coeff_index(path: str) -> int | None:
    name = path.rsplit('.', 1)[-1]
    if len(name) >= 2 and name[0] == 'a' and name[1:].isdigit():
        return int(name[1:])
    return None


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    values = {}
    for attr in node.attribute:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        values[attr.name] = value
    return values


def _parse_coefficients(value: Any) -> list[float]:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(',') if item.strip()]
    return [float(item) for item in value]


def _input_or_empty(node: onnx.NodeProto, idx: int) -> str:
    return node.input[idx] if len(node.input) > idx else ''


def _attention_gamma_path(running_max_path: str, node_name: str) -> str:
    if running_max_path.endswith('.running_max_concat'):
        return running_max_path[: -len('.running_max_concat')] + '.gamma'
    return _format_id(node_name) + '.gamma'


def _polyact_gamma_path(running_max_path: str) -> str:
    suffix = '.rangenorm.running_max'
    if running_max_path.endswith(suffix):
        return running_max_path[: -len(suffix)] + '.gamma'
    return ''


def _attention_coeffs_path(coeff_paths: tuple[str, ...], running_max_path: str, node_name: str) -> str:
    if running_max_path.endswith('.running_max_concat'):
        attn_prefix = running_max_path[: -len('.running_max_concat')]
        if coeff_paths:
            coeff_prefix = coeff_paths[0].rsplit('.', 1)[0]
            marker = '.attn.'
            if marker in coeff_prefix and attn_prefix.endswith('.attn'):
                return f'{attn_prefix}.{coeff_prefix.split(marker, 1)[1]}.weight'
        return f'{attn_prefix}.poly.weight'
    if coeff_paths:
        first = coeff_paths[0]
        suffix = first.rsplit('.', 1)[-1]
        if len(suffix) >= 2 and suffix[0] == 'a' and suffix[1:].isdigit():
            return first.rsplit('.', 1)[0] + '.weight'
    return _format_id(node_name) + '.poly.weight'


def _format_id(onnx_id: str) -> str:
    return onnx_id.replace(':', '_').replace('/', '_').replace('.', '_')


def _feature(features: dict[str, dict[str, Any]], feature_id: str, layer_key: str) -> dict[str, Any]:
    if feature_id not in features:
        raise KeyError(f'{layer_key}: feature not found in CT JSON: {feature_id}')
    feature = features[feature_id]
    if feature.get('data_type') != 'feature_mat':
        raise ValueError(f'{layer_key}: expected feature_mat feature: {feature_id}')
    if 'shape' not in feature or len(feature['shape']) < 2:
        raise ValueError(f'{layer_key}: feature missing 2D shape: {feature_id}')
    return feature


def _layer_head_dim(layer: dict[str, Any], features: dict[str, dict[str, Any]]) -> int:
    feature_id = layer['feature_output'][0]
    feature = _feature(features, feature_id, layer.get('type', 'parcpmm'))
    head_shape = feature.get('head_shape')
    if not head_shape or len(head_shape) < 2:
        raise ValueError(f'missing head_shape for feature_mat output: {feature_id}')
    dims = [int(x) for x in head_shape[:2] if int(x) > 0]
    if not dims:
        raise ValueError(f'invalid head_shape for feature_mat output: {feature_id}')
    return min(dims)


def _sequence_length(feature: dict[str, Any], layer_key: str) -> int:
    head_shape = feature.get('head_shape')
    if head_shape and int(head_shape[0]) > 0:
        return int(head_shape[0])
    shape = feature.get('shape')
    if shape and int(shape[0]) > 0:
        return int(shape[0])
    raise ValueError(f'{layer_key}: cannot infer sequence length from feature shape/head_shape')


def _required_path(layer: dict[str, Any], key: str, layer_key: str) -> str:
    path = layer.get(key, '')
    if not path:
        raise KeyError(f'{layer_key}: missing {key}')
    return path


def _reshape_checked(data: np.ndarray, shape: tuple[int, ...], layer_key: str, path: str) -> np.ndarray:
    data = np.asarray(data, dtype='float64')
    expected_size = math.prod(shape)
    if data.size != expected_size:
        raise ValueError(f'{layer_key}: {path} size {data.size} does not match expected shape {shape}')
    return data.reshape(shape)


def _put(out: dict[str, np.ndarray], path: str, data: np.ndarray, layer_key: str) -> None:
    data = np.asarray(data, dtype='float64')
    if path in out:
        prev = out[path]
        if prev.shape != data.shape or not np.allclose(prev, data):
            raise ValueError(
                f'{layer_key}: H5 path conflict for {path}; compiled CT JSON maps different runtime tensors to the same path'
            )
    out[path] = data
