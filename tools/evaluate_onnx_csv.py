#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an original ONNX and CSV, expanding its polynomial ops in memory."""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def expand_polynomials(model):
    from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_module
    from training.nn_tools.export import _resolve_activation_cls

    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    prefix = '__plaintext_poly_'
    while any(name.startswith(prefix) for name in names):
        prefix += '_'
    nodes, counter = [], 0
    dtype = helper.tensor_dtype_to_np_dtype(model.graph.input[0].type.tensor_type.elem_type)

    def unique():
        nonlocal counter
        counter += 1
        return f'{prefix}{counter}'

    def constant(value):
        name = unique()
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, *inputs):
        output = unique()
        nodes.append(helper.make_node(kind, inputs, [output]))
        return output

    supported = {'RangeNormPoly1d', 'RangeNormPoly2d', 'PolyAct', 'PolyActRNPoly'}
    for node in model.graph.node:
        if node.op_type == 'Simple_Polyrelu':
            terms = []
            for name in node.input[1:]:
                power = int(name.rsplit('.a', 1)[1])
                terms.append(name if power == 0 else op('Mul', name, op('Pow', node.input[0], constant(power))))
            y = terms[0]
            for term in terms[1:]:
                y = op('Add', y, term)
            nodes.append(helper.make_node('Identity', [y], list(node.output)))
            continue
        if node.op_type not in supported:
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        activation = attrs.get('activation', b'relu' if node.op_type != 'PolyActRNPoly' else b'gelu')
        if isinstance(activation, bytes):
            activation = activation.decode()
        coeff = get_hermite_coeffs_for_module(_resolve_activation_cls(activation), int(attrs['degree']))
        x = node.input[0]
        if node.op_type.startswith('RangeNormPoly'):
            scale = op('Add', op('Div', node.input[1], constant(attrs.get('upper_bound', 3.0))),
                       constant(attrs.get('eps', 1e-3)))
            x = op('Div', x, scale)
        else:
            scale = constant(attrs.get('scale_after', 1.0))
            x = op('Mul', x, constant(attrs.get('scale_before', 1.0)))

        # Match the training forward's special evaluation order for degree 2/4.
        if len(coeff) == 3:
            y = op('Sub', op('Add', constant(coeff[0]),
                            op('Mul', op('Add', constant(coeff[1]), op('Mul', constant(coeff[2]), x)), x)),
                   constant(coeff[2]))
        elif len(coeff) == 5:
            x2 = op('Mul', x, x)
            x4 = op('Mul', x2, x2)
            y = op('Add', op('Add', op('Add', constant(coeff[0]), op('Mul', constant(coeff[1]), x)),
                                   op('Mul', constant(coeff[2]), op('Sub', x2, constant(1)))),
                   op('Mul', constant(coeff[4]),
                      op('Add', op('Sub', x4, op('Mul', constant(6), x2)), constant(3))))
        else:
            previous2, previous1 = constant(1), x
            y = op('Add', constant(coeff[0]), op('Mul', constant(coeff[1]), x))
            for degree in range(2, len(coeff)):
                current = op('Sub', op('Mul', x, previous1), op('Mul', constant(degree - 1), previous2))
                y = op('Add', y, op('Mul', constant(coeff[degree]), current))
                previous2, previous1 = previous1, current
        nodes.append(helper.make_node('Mul', [scale, y], list(node.output)))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def evaluate(onnx_path, csv_path, warmup=1, repeats=3):
    import onnxruntime as ort

    started = time.perf_counter()
    model = expand_polynomials(onnx.load(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if isinstance(csv_path, dict):
        paths = csv_path
    elif len(inputs) == 1:
        paths = {inputs[0].name: csv_path}
    else:
        raise ValueError('Multiple model inputs require named CSV paths')
    if set(paths) != {meta.name for meta in inputs}:
        raise ValueError('CSV input names do not match model inputs')
    feeds = {}
    for meta in inputs:
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        tensor = next(x for x in model.graph.input if x.name == meta.name)
        dtype = helper.tensor_dtype_to_np_dtype(tensor.type.tensor_type.elem_type)
        feeds[meta.name] = np.loadtxt(paths[meta.name], delimiter=',', dtype=dtype).reshape(shape)
    preparation_ms = (time.perf_counter() - started) * 1000
    samples, outputs = [], None
    for i in range(warmup + repeats):
        started = time.perf_counter()
        outputs = session.run(None, feeds)
        samples.append({'warmup': i < warmup, 'wall_ms': (time.perf_counter() - started) * 1000})
    results = {meta.name: values.reshape(-1) for meta, values in zip(session.get_outputs(), outputs)}
    if not all(np.isfinite(v).all() for v in results.values()):
        raise ValueError('Plaintext logits contain nonfinite values')
    return results, {
        'onnx': str(onnx_path),
        'input_csv': {name: str(path) for name, path in paths.items()} if isinstance(csv_path, dict) else str(csv_path),
        'provider': 'CPUExecutionProvider',
        'preparation_wall_ms': preparation_ms,
        'plaintext_wall_ms': statistics.median(s['wall_ms'] for s in samples if not s['warmup']),
        'samples': samples, 'output_counts': {name: len(v) for name, v in results.items()},
        'top1': {name: int(np.argmax(v)) for name, v in results.items()},
    }


def write_logits(path, outputs):
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['output', 'index', 'value'])
        for name, values in outputs.items():
            writer.writerows((name, i, format(float(value), '.17g')) for i, value in enumerate(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-csv', type=Path)
    inputs.add_argument('--input', action='append', metavar='NAME=CSV', help='Named input; repeat for multiple inputs')
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative and repeats must be positive')
    paths = args.input_csv
    if args.input:
        paths = {}
        for binding in args.input:
            name, sep, path = binding.partition('=')
            if not sep or not name or not path or name in paths:
                parser.error('Each input must be a unique NAME=CSV binding')
            paths[name] = Path(path)
    outputs, report = evaluate(args.onnx, paths, args.warmup, args.repeats)
    write_logits(args.output_csv, outputs)
    text = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
