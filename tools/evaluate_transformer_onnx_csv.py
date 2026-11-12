#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Plain NumPy reference for the custom operators in exported BERT and ViT ONNX.

Reads the original graph, attributes and initializers, without compiler artifacts.
Matrix arithmetic uses float64 to isolate FHE error from reference rounding.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def polynomial(x, coefficients):
    if isinstance(coefficients, bytes):
        coefficients = coefficients.decode()
    if isinstance(coefficients, str):
        coefficients = [float(v) for v in coefficients.split(',')]
    return np.polynomial.polynomial.polyval(x, coefficients)


def inverse(x, epsilon, alpha, attrs):
    # Match PolySoftmax._normalize / _approx_inverse in the ONNX exporter.
    # epsilon and alpha control convergence; neither is added to the input.
    a = np.ones_like(x)
    b = x
    lower = epsilon
    for _ in range(attrs.get('max_inverse_iterations', 15)):
        if not attrs.get('use_asor', 1):
            a = a * (2 - b * a)
        else:
            if lower >= 1 - alpha:
                break
            k = 2 / (1 + lower)
            factor = k * (2 - k * b)
            a = a * factor
            b = b * factor
            lower = k * lower * (2 - k * lower)
    return a


def layernorm(x, weight, bias, attrs):
    eps = attrs.get('eps', attrs.get('epsilon', 1e-5))
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True) + eps
    if 'max_var' in attrs:
        iv = 1 / ((attrs['max_var'] * attrs.get('w_buffer', 1.05) + eps) * attrs.get('input_scale', 1.0) ** 2)
        iterations = attrs.get('max_inverse_sqrt_iterations', 7)
    else:
        iv = attrs.get('inv_var_scale', 1.0)
        iterations = attrs.get('num_iters', 2)
    a = variance * iv
    c0 = attrs.get('minimax_c0', attrs.get('c0', 6.19067182))
    c1 = attrs.get('minimax_c1', attrs.get('c1', -16.15885111))
    c2 = attrs.get('minimax_c2', attrs.get('c2', 11.52830778))
    y = c0 + a * (c1 + a * c2)
    for _ in range(iterations):
        y = 0.5 * y * (3 - a * y * y)
    return centered * y * np.sqrt(iv) * weight + bias


def range_polynomial(x, running_max, coefficients, attrs):
    scale = running_max / attrs.get('upper_bound', 3.0) + attrs.get('eps', 1e-3)
    z = x / scale
    a0, a1, a2, a4 = coefficients
    z2 = z * z
    return scale * (a0 + a1 * z + a2 * (z2 - 1) + a4 * (z2 * z2 - 6 * z2 + 3))


def attention(values, attrs, bert):
    x = values[0]
    heads, dim = attrs['num_heads'], attrs['head_dim']
    if bert:
        q, k, v = [x @ values[i].T + values[i + 1] for i in (1, 3, 5)]
        projection, bias = values[7:9]
    else:
        q, k, v = np.split(x @ values[1] + values[2], 3, axis=-1)
        projection, bias = values[3:5]

    def split(t):
        return t.reshape(t.shape[0], t.shape[1], heads, dim).transpose(0, 2, 1, 3)

    q, k, v = map(split, (q, k, v))
    scores = (q @ k.swapaxes(-1, -2)) * attrs.get('scaling', dim**-0.5)
    if bert:
        mid = (attrs['range_min'] + attrs['range_max']) / 2
        a = polynomial((scores - mid) / attrs['exp_divisor'], attrs['exp_coefficients'])
        a = a ** attrs.get('delta_1', 2)
        epsilon = attrs['inverse_epsilon']
        a = a * inverse(a.sum(axis=-1, keepdims=True), epsilon, attrs['inverse_alpha'], attrs)
        delta = attrs.get('delta_2', 2)
        while delta > 1:
            a = a * a
            delta >>= 1
            key = 'final_inverse_alpha' if delta == 1 else 'inverse_alpha'
            a = a * inverse(a.sum(axis=-1, keepdims=True), epsilon / x.shape[1], attrs[key], attrs)
    else:
        n = x.shape[1]
        scale = values[5].reshape(heads, n)[None, :, None, :]
        a = range_polynomial(scores, scale, values[6:10], attrs) / n
    out = (a @ v).transpose(0, 2, 1, 3).reshape(x.shape)
    return out @ (projection.T if bert else projection) + bias


def _load_inputs(model, csv_path, input_shapes=None):
    model_inputs = list(model.graph.input)
    if isinstance(csv_path, dict):
        paths = csv_path
    elif len(model_inputs) == 1:
        paths = {model_inputs[0].name: csv_path}
    else:
        raise ValueError('Multiple model inputs require named CSV paths')
    expected_names = {value.name for value in model_inputs}
    if set(paths) != expected_names:
        raise ValueError(f'CSV input names do not match model inputs: {sorted(paths)} vs {sorted(expected_names)}')

    feeds = {}
    for index, value in enumerate(model_inputs):
        array = np.loadtxt(paths[value.name], delimiter=',', dtype=np.float64)
        shape = (input_shapes or {}).get(value.name)
        if shape is not None:
            shape = tuple(int(dim) for dim in shape)
            if array.size != int(np.prod(shape)):
                raise ValueError(f'CSV size for {value.name} is {array.size}, expected {shape}')
            array = array.reshape(shape)
        elif index == 0 and array.ndim == 2 and len(value.type.tensor_type.shape.dim) == 3:
            # Transformer CSVs contain token rows without the batch dimension.
            array = array[None, :, :]
        else:
            static_shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            if static_shape and all(dim > 0 for dim in static_shape) and array.size == int(np.prod(static_shape)):
                array = array.reshape(static_shape)
        feeds[value.name] = array
    return feeds


def execute(model, feeds, requested_outputs=None, model_kind=None, intermediates=None):
    """Preserve the original float64 semantics for named prefix/suffix feeds."""
    bert = model_kind == 'bert' or (
        model_kind is None
        and any(
            n.op_type in ('CustomGELU', 'CustomTanh')
            or (n.op_type == 'CustomMultiHeadAttention' and any(a.name == 'exp_coefficients' for a in n.attribute))
            for n in model.graph.node
        )
    )
    if set(feeds) != {value.name for value in model.graph.input}:
        raise ValueError('Named feeds do not match model inputs')
    values = {w.name: numpy_helper.to_array(w).astype(np.float64) for w in model.graph.initializer}
    values.update({name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()})
    # BERT exports a padded length of 128, with L_prepad and an attention mask
    # selecting the actual CSV tokens. Masked keys cannot influence valid rows.
    attention_node = next((n for n in model.graph.node if n.op_type == 'CustomMultiHeadAttention'), None)
    if bert and attention_node is not None:
        x = values[attention_node.input[0]] if attention_node.input[0] in feeds else None
        attrs = {a.name: helper.get_attribute_value(a) for a in attention_node.attribute}
        mask = numpy_helper.to_array(attrs['attention_mask']).reshape(-1)
        token_count = int(mask.sum())
        if (x is not None and x.shape[1] != token_count) or not np.all(mask[:token_count] == 1):
            raise ValueError('CSV does not match the ONNX attention mask')
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        args = [values[i] for i in node.input]
        if node.op_type == 'Constant':
            result = numpy_helper.to_array(attrs['value']).astype(np.float64)
        elif node.op_type == 'Identity':
            result = args[0]
        elif node.op_type == 'Linear':
            source, w, b = args
            if attrs.get('select_first_token'):
                source = source[:, 0, :]
            result = source @ (w.T if bert else w) + b
        elif node.op_type == 'Add':
            result = args[0] + args[1]
        elif node.op_type == 'CustomLayerNorm':
            result = layernorm(*args, attrs)
        elif node.op_type == 'CustomMultiHeadAttention':
            result = attention(args, attrs, bert)
        elif node.op_type == 'PolyActRN':
            result = range_polynomial(args[0], args[1], args[2:], attrs)
        elif node.op_type == 'CustomGELU':
            z = polynomial(args[0] / attrs['scale'], attrs['p1_coefficients'])
            z = polynomial(z / attrs['f2_input_scale'], attrs['p2_coefficients'])
            z = polynomial(z / attrs['f3_input_scale'], attrs['p3_coefficients'])
            result = args[0] * (z + 0.5)
        elif node.op_type == 'CustomTanh':
            z = polynomial(args[0] / (attrs['scale'] * attrs['f1_input_scale']), attrs['p1_coefficients'])
            result = polynomial(z / attrs['f2_input_scale'], attrs['p2_coefficients'])
        else:
            raise ValueError(f'Unsupported original ONNX operator: {node.op_type}')
        if not np.isfinite(result).all():
            raise ValueError(f'Nonfinite reference output at {node.name}')
        values[node.output[0]] = result
    if intermediates is not None:
        np.savez(intermediates, **{n.output[0]: values[n.output[0]] for n in model.graph.node})
    output_names = list(requested_outputs) if requested_outputs is not None else [o.name for o in model.graph.output]
    return {name: values[name] for name in output_names}


def evaluate(onnx_path, csv_path, intermediates=None, requested_outputs=None, input_shapes=None, model_kind=None):
    started = time.perf_counter()
    model = onnx.load(onnx_path)
    feeds = _load_inputs(model, csv_path, input_shapes=input_shapes)
    outputs = execute(model, feeds, requested_outputs, model_kind, intermediates)
    return outputs, {
        'wall_s': time.perf_counter() - started,
        'outputs': {
            k: {'shape': list(v.shape), 'min': float(v.min()), 'max': float(v.max())} for k, v in outputs.items()
        },
        'reference': 'original ONNX custom operators evaluated with NumPy float64',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outputs, report = evaluate(args.onnx, args.input_csv)
    np.savez(args.output, **outputs)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
