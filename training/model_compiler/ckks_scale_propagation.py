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

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ScaleMap = dict[str, float]


def annotate_ct_json_scales(
    ct_json: dict[str, Any],
    initial_scale: float | Mapping[str, float] | None = None,
    *,
    output_key: str = 'ckks_scale_inferred',
    strict: bool = True,
) -> dict[str, Any]:
    features = ct_json.get('feature')
    layers = ct_json.get('layer')
    params = ct_json.get('ckks_parameter')
    if not isinstance(features, dict) or not isinstance(layers, dict) or not isinstance(params, dict):
        raise ValueError('ct_json must contain dict fields: feature, layer, ckks_parameter')

    input_features = list(ct_json.get('input_feature') or _infer_input_features(features, layers))
    if not input_features:
        raise ValueError('ct_json has no input_feature and no inferred source features')

    layer_inputs: dict[str, list[str]] = {}
    layer_outputs: dict[str, list[str]] = {}
    feature_to_consumers: dict[str, list[str]] = defaultdict(list)
    feature_producer: dict[str, str] = {}

    for layer_id, layer in layers.items():
        inputs = list(layer.get('feature_input') or [])
        outputs = list(layer.get('feature_output') or [])
        if not inputs or not outputs:
            raise ValueError(f'layer {layer_id} must contain non-empty feature_input and feature_output')
        for feature_id in inputs + outputs:
            if feature_id not in features:
                raise ValueError(f'layer {layer_id} references unknown feature {feature_id}')
        for feature_id in inputs:
            feature_to_consumers[feature_id].append(layer_id)
        for feature_id in outputs:
            prev = feature_producer.get(feature_id)
            if prev is not None and strict:
                raise ValueError(f'feature {feature_id} is produced by multiple layers: {prev}, {layer_id}')
            feature_producer[feature_id] = layer_id
        layer_inputs[layer_id] = inputs
        layer_outputs[layer_id] = outputs

    scales = _initial_feature_scales(input_features, features, params, initial_scale, strict)
    processed: set[str] = set()
    queue: deque[str] = deque()
    for feature_id in input_features:
        queue.extend(feature_to_consumers.get(feature_id, []))

    while queue:
        layer_id = queue.popleft()
        if layer_id in processed:
            continue
        inputs = layer_inputs[layer_id]
        if any(feature_id not in scales for feature_id in inputs):
            continue

        outputs = layer_outputs[layer_id]
        layer = layers[layer_id]
        input_scales = [scales[feature_id] for feature_id in inputs]
        output_scale = _infer_layer_scale(layer_id, layer, input_scales, inputs, features, params, strict)

        for output_id in outputs:
            if output_id in scales and strict:
                raise ValueError(f'feature {output_id} already has inferred scale before processing {layer_id}')
            scales[output_id] = output_scale
            queue.extend(feature_to_consumers.get(output_id, []))
        processed.add(layer_id)

    if len(processed) != len(layers):
        unresolved = []
        for layer_id in layers:
            if layer_id in processed:
                continue
            missing = [feature_id for feature_id in layer_inputs[layer_id] if feature_id not in scales]
            unresolved.append({'layer': layer_id, 'type': layers[layer_id].get('type'), 'missing_inputs': missing})
        raise ValueError(f'could not propagate scales through all layers: {unresolved[:10]}')

    missing_features = [feature_id for feature_id in features if feature_id not in scales]
    if missing_features and strict:
        raise ValueError(f'features missing inferred scale: {missing_features[:10]}')

    annotated = copy.deepcopy(ct_json)
    for feature_id, scale in scales.items():
        annotated['feature'][feature_id][output_key] = float(scale)
    annotated['scale_propagation'] = {
        'output_key': output_key,
        'input_feature': {feature_id: float(scales[feature_id]) for feature_id in input_features},
    }
    return annotated


def annotate_ct_json_scales_from_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    initial_scale: float | Mapping[str, float] | None = None,
    *,
    output_key: str = 'ckks_scale_inferred',
    strict: bool = True,
) -> dict[str, Any]:
    with Path(input_path).open('r', encoding='utf8') as f:
        ct_json = json.load(f)

    annotated = annotate_ct_json_scales(ct_json, initial_scale, output_key=output_key, strict=strict)
    if output_path is not None:
        with Path(output_path).open('w', encoding='utf8') as f:
            json.dump(annotated, f, indent=4, ensure_ascii=False)
    return annotated


def _infer_input_features(features: dict[str, Any], layers: dict[str, Any]) -> list[str]:
    produced = set()
    consumed = set()
    for layer in layers.values():
        produced.update(layer.get('feature_output') or [])
        consumed.update(layer.get('feature_input') or [])
    return [feature_id for feature_id in features if feature_id in consumed and feature_id not in produced]


def _initial_feature_scales(
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
    initial_scale: float | Mapping[str, float] | None,
    strict: bool,
) -> ScaleMap:
    scales: ScaleMap = {}
    if isinstance(initial_scale, Mapping):
        for feature_id in input_features:
            if feature_id not in initial_scale:
                if strict:
                    raise ValueError(f'initial_scale is missing input feature {feature_id}')
                scales[feature_id] = _default_scale(_feature_param(features[feature_id], params))
            else:
                scales[feature_id] = float(initial_scale[feature_id])
        return scales

    for feature_id in input_features:
        if feature_id not in features:
            raise ValueError(f'input_feature references unknown feature {feature_id}')
        if initial_scale is None:
            scales[feature_id] = _default_scale(_feature_param(features[feature_id], params))
        else:
            scales[feature_id] = float(initial_scale)
    return scales


def _infer_layer_scale(
    layer_id: str,
    layer: dict[str, Any],
    input_scales: list[float],
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
    strict: bool,
) -> float:
    layer_type = layer.get('type')
    if strict:
        _check_same_ckks_param(layer_id, input_features, features)

    if layer_type in {'drop_level', 'dropout', 'identity', 'reshape'}:
        return _require_input(layer_id, input_scales, 1)[0]
    if layer_type == 'bootstrapping':
        return _require_input(layer_id, input_scales, 1)[0]
    if layer_type in {'add', 'add2d'}:
        return _max_scale(*input_scales)
    if layer_type in {'add_pt', 'pcm_add_pt'}:
        sx = _require_input(layer_id, input_scales, 1)[0]
        d = _default_scale(_param_for_input(input_features, features, params, 0))
        return _max_scale(sx, d)
    if layer_type in {'partranspose', 'pcmgamma'}:
        sx = _require_input(layer_id, input_scales, 1)[0]
        q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
        return _div(_mul(sx, q0), q0)
    if layer_type == 'parcpmm':
        sa = _require_input(layer_id, input_scales, 1)[0]
        q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
        q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
        s_block = _div(_mul(sa, q0), q0)
        return _div(_mul(s_block, q1), q1)
    if layer_type == 'parccmm':
        sa, sb = _require_input(layer_id, input_scales, 2)
        if strict:
            _check_same_level(layer_id, input_features, features)
        d = _default_scale(_param_for_input(input_features, features, params, 0))
        q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
        q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
        q2 = _q_for_input(input_features, features, params, layer_id, 0, 2)
        a_sigma = _div(_mul(sa, q0), q0)
        b_tau = _div(_mul(sb, q0), q0)
        p_psi = _mul(_div(q2, d), q1)
        b_psi = _div(_mul(b_tau, p_psi), q1)
        return _div(_mul(a_sigma, b_psi), q2)
    if layer_type == 'pcmpoly':
        sx = _require_input(layer_id, input_scales, 1)[0]
        order = int(layer.get('order', 4))
        return _pcmpoly_scale(layer_id, sx, order, input_features, features, params)
    if layer_type == 'pcmstats':
        sx = _require_input(layer_id, input_scales, 1)[0]
        return _pcmstats_scale(layer_id, sx, input_features, features, params)
    if layer_type == 'pcmcenter':
        sx = _require_input(layer_id, input_scales, 1)[0]
        return _pcmcenter_scale(layer_id, sx, input_features, features, params)
    if layer_type == 'pcminit':
        sa = _require_input(layer_id, input_scales, 1)[0]
        return _pcminit_scale(layer_id, sa, input_features, features, params)
    if layer_type == 'pcmgs':
        sy, sa = _require_input(layer_id, input_scales, 2)
        return _pcmgs_scale(layer_id, sy, sa, input_features, features, params)
    if layer_type == 'pcmaffine':
        sx, sy = _require_input(layer_id, input_scales, 2)
        return _pcmaffine_scale(layer_id, sx, sy, input_features, features, params)

    if strict:
        raise NotImplementedError(f'unsupported layer type for CKKS scale propagation: {layer_id} ({layer_type})')
    if len(input_scales) == 1:
        return input_scales[0]
    return _max_scale(*input_scales)


def _pcmpoly_scale(
    layer_id: str,
    sx: float,
    order: int,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    d = _default_scale(_param_for_input(input_features, features, params, 0))
    q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
    q2 = _q_for_input(input_features, features, params, layer_id, 0, 2)

    p_c1 = q0
    p_c2 = _mul(_div(q0, d), q1)
    s_x2 = _div(_mul(sx, sx), q0)
    s_c1x_drop = _div(_mul(sx, p_c1), q0)
    s_c2x2 = _div(_mul(s_x2, p_c2), q1)
    out_degree2 = _max_scale(_max_scale(s_c1x_drop, s_c2x2), d)

    if order == 2:
        return out_degree2
    if order != 4:
        raise NotImplementedError(f'unsupported pcmpoly order at {layer_id}: {order}')

    p_c3 = _mul(_div(_mul(_div(q0, d), q0), d), q2)
    p_c4 = _mul(_div(_mul(_div(_mul(_div(q0, d), q0), d), q1), d), q2)
    s_c3x_drop = _div(_mul(sx, p_c3), q0)
    s_c4x2 = _div(_mul(s_x2, p_c4), q1)
    s_high = _max_scale(s_c3x_drop, s_c4x2)
    s_x2_high = _div(_mul(s_x2, s_high), q2)
    return _max_scale(out_degree2, s_x2_high)


def _pcmstats_scale(
    layer_id: str,
    sx: float,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    d = _default_scale(_param_for_input(input_features, features, params, 0))
    q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
    q2 = _q_for_input(input_features, features, params, layer_id, 0, 2)
    q3 = _q_for_input(input_features, features, params, layer_id, 0, 3)

    p_h0 = q0
    p_inv_n = q1
    p_iv = _mul(_div(q2, d), q3)
    s_x2 = _div(_mul(sx, sx), q0)
    s_sum_x = _div(_mul(sx, p_h0), q0)
    s_sum_x_sq_row = _div(_mul(s_x2, p_h0), q1)
    s_mean = _div(_mul(s_sum_x, p_inv_n), q1)
    s_mean_sq = _div(_mul(s_mean, s_mean), q2)
    s_e_x_sq = _div(_mul(s_sum_x_sq_row, p_inv_n), q2)
    s_var = _max_scale(s_e_x_sq, s_mean_sq)
    s_a = _div(_mul(s_var, p_iv), q3)
    return _max_scale(s_a, d)


def _pcmcenter_scale(
    layer_id: str,
    sx: float,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
    s_sum_x = _div(_mul(sx, q0), q0)
    s_mean = _div(_mul(s_sum_x, q1), q1)
    return _max_scale(sx, s_mean)


def _pcminit_scale(
    layer_id: str,
    sa: float,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    d = _default_scale(_param_for_input(input_features, features, params, 0))
    q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
    p_c1 = q0
    p_c2 = _mul(_div(q0, d), q1)
    s_a2 = _div(_mul(sa, sa), q0)
    s_c2a2 = _div(_mul(s_a2, p_c2), q1)
    s_c1a_drop = _div(_mul(sa, p_c1), q0)
    return _max_scale(_max_scale(s_c1a_drop, s_c2a2), d)


def _pcmgs_scale(
    layer_id: str,
    sy: float,
    sa: float,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    d = _default_scale(_param_for_input(input_features, features, params, 0))
    q0 = _q_for_input(input_features, features, params, layer_id, 0, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 0, 1)
    q2 = _q_for_input(input_features, features, params, layer_id, 0, 2)
    p_three = _mul(_div(_mul(_div(d, q0), d), q1), d)
    p_half = _mul(_div(_mul(_div(_mul(_div(q0, d), q0), d), q1), d), q2)
    s_ya = _div(_mul(sy, sa), q0)
    s_yy = _div(_mul(sy, sy), q0)
    s_ya_yy = _div(_mul(s_ya, s_yy), q1)
    s_three_y_drop = _div(_mul(sy, p_three), q0)
    s_diff = _max_scale(s_three_y_drop, s_ya_yy)
    return _div(_mul(s_diff, p_half), q2)


def _pcmaffine_scale(
    layer_id: str,
    sx: float,
    sy: float,
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
) -> float:
    if len(input_features) < 2:
        raise ValueError(f'layer {layer_id} requires x_centered and y inputs')
    d = _default_scale(_param_for_input(input_features, features, params, 1))
    q0 = _q_for_input(input_features, features, params, layer_id, 1, 0)
    q1 = _q_for_input(input_features, features, params, layer_id, 1, 1)
    p_gamma = _mul(_div(q0, d), q1)
    s_yw = _div(_mul(sy, p_gamma), q0)
    s_out_no_beta = _div(_mul(sx, s_yw), q1)
    return _max_scale(s_out_no_beta, d)


def _require_input(layer_id: str, input_scales: list[float], n: int) -> list[float]:
    if len(input_scales) < n:
        raise ValueError(f'layer {layer_id} requires at least {n} input(s), got {len(input_scales)}')
    return input_scales[:n]


def _feature_param(feature: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    param_id = feature.get('ckks_parameter_id', 'param0')
    if param_id not in params:
        raise ValueError(f'unknown ckks_parameter_id: {param_id}')
    return params[param_id]


def _param_for_input(
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
    input_index: int,
) -> dict[str, Any]:
    return _feature_param(features[input_features[input_index]], params)


def _default_scale(param: dict[str, Any]) -> float:
    return float(2 ** int(param['log_default_scale']))


def _q_for_input(
    input_features: list[str],
    features: dict[str, Any],
    params: dict[str, Any],
    layer_id: str,
    input_index: int,
    offset: int,
) -> float:
    feature = features[input_features[input_index]]
    level = int(feature['level'])
    index = level - offset
    param = _feature_param(feature, params)
    q = param.get('q')
    if not isinstance(q, list):
        raise ValueError(f'ckks parameter for layer {layer_id} must contain q list')
    if index < 0 or index >= len(q):
        raise ValueError(f'layer {layer_id} requires q[{index}] from input level {level}')
    return float(q[index])


def _check_same_ckks_param(layer_id: str, input_features: list[str], features: dict[str, Any]) -> None:
    param_ids = {features[feature_id].get('ckks_parameter_id', 'param0') for feature_id in input_features}
    if len(param_ids) > 1:
        raise ValueError(f'layer {layer_id} has inputs with different ckks_parameter_id values: {sorted(param_ids)}')


def _check_same_level(layer_id: str, input_features: list[str], features: dict[str, Any]) -> None:
    levels = {int(features[feature_id]['level']) for feature_id in input_features}
    if len(levels) > 1:
        raise ValueError(f'layer {layer_id} has inputs with different levels: {sorted(levels)}')


def _mul(a: float, b: float) -> float:
    return a * b


def _div(a: float, b: float) -> float:
    return a / b


def _max_scale(*values: float) -> float:
    if not values:
        raise ValueError('max_scale requires at least one value')
    return max(values)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Annotate a compiled ct_json with inferred CKKS scale values for every feature node.'
    )
    parser.add_argument('input_path', type=Path, help='Path to compiled ct_json, e.g. nn_layers_ct_0.json')
    parser.add_argument(
        '-o',
        '--output',
        dest='output_path',
        type=Path,
        default=None,
        help='Optional path to write annotated ct_json. If omitted, only a summary is printed.',
    )
    parser.add_argument(
        '--initial-scale',
        type=float,
        default=None,
        help='Initial input feature scale. Defaults to 2 ** log_default_scale from ckks_parameter.',
    )
    parser.add_argument(
        '--output-key',
        default='ckks_scale_inferred',
        help='Feature JSON key used for inferred scale values.',
    )
    parser.add_argument(
        '--non-strict',
        action='store_true',
        help='Pass unknown single-input ops through instead of raising an error.',
    )
    return parser.parse_args(argv)


def _print_summary(annotated: dict[str, Any], input_path: Path, output_path: Path | None, output_key: str) -> None:
    features = annotated['feature']
    annotated_count = sum(output_key in feature for feature in features.values())
    print(f'input = {input_path}')
    if output_path is not None:
        print(f'output = {output_path}')
    print(f'features annotated = {annotated_count} / {len(features)}')
    print(f'layers = {len(annotated.get("layer", {}))}')
    for feature_id in annotated.get('output_feature', []):
        print(f'output_feature {feature_id} {output_key} = {features[feature_id][output_key]}')


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    annotated = annotate_ct_json_scales_from_file(
        args.input_path,
        args.output_path,
        args.initial_scale,
        output_key=args.output_key,
        strict=not args.non_strict,
    )
    _print_summary(annotated, args.input_path, args.output_path, args.output_key)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
