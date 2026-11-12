#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an ONNX plaintext prefix and FHE suffix with logical CSV inputs."""

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import tempfile
import time

import networkx as nx
import numpy as np
import onnx
from onnx import helper, numpy_helper

import evaluate_onnx_csv as standard_reference
import evaluate_transformer_onnx_csv as transformer_reference


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_files(path):
    path = Path(path).resolve()
    model = onnx.load(path, load_external_data=False)
    files = [path]
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = next(item.value for item in tensor.external_data if item.key == 'location')
            external = (path.parent / location).resolve()
            if external not in files:
                files.append(external)
    return [{'path': str(file), 'sha256': sha256(file)} for file in files]


def split_graph(model, ratio_text):
    try:
        ratio = Decimal(ratio_text)
    except InvalidOperation as exc:
        raise ValueError('Invalid decimal split ratio') from exc
    if not ratio.is_finite() or not 0 < ratio < 1:
        raise ValueError('Split ratio must be finite and strictly between 0 and 1')
    if model.graph.sparse_initializer or model.graph.quantization_annotation:
        raise ValueError('ONNX extraction does not support sparse initializers or quantization annotations')
    for node in model.graph.node:
        if any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute):
            raise ValueError(f'Control-flow/subgraph extraction is unsupported: {node.op_type}')
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == 'BatchNormalization' and attrs.get('training_mode', 0):
            raise ValueError('Training BatchNormalization is unsupported')
        if node.op_type == 'Dropout':
            raise ValueError('Dropout must be removed from the inference model before splitting')
    producer = {
        name: (i, port) for i, node in enumerate(model.graph.node) for port, name in enumerate(node.output) if name
    }
    dag = nx.MultiDiGraph()
    dag.add_nodes_from(range(len(model.graph.node)))
    for consumer, node in enumerate(model.graph.node):
        for port, tensor in enumerate(node.input):
            if tensor in producer:
                source, source_port = producer[tensor]
                dag.add_edge(source, consumer, key=port, tensor=tensor, producer_port=source_port)
    try:
        topo = list(nx.lexicographical_topological_sort(dag, key=lambda i: i))
    except nx.NetworkXUnfeasible as exc:
        raise ValueError('ONNX node dependencies contain a cycle') from exc
    compute = [i for i in topo if model.graph.node[i].op_type != 'Constant']
    count = int(ratio * len(compute))
    if not 1 <= count < len(compute):
        raise ValueError('Split ratio produces an empty prefix or suffix')
    prefix, suffix = set(compute[:count]), set(compute[count:])
    positions = {i: position for position, i in enumerate(topo)}
    crossing = {data['tensor'] for _, _, data in nx.edge_boundary(dag, prefix, suffix, data=True)}
    crossing = sorted(crossing, key=lambda name: (positions[producer[name][0]], producer[name][1]))
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value.name for value in model.graph.input if value.name not in initializers]

    def consumed(group):
        return {name for i in group for name in model.graph.node[i].input if name}

    prefix_inputs = [name for name in inputs if name in consumed(prefix)]
    forwarded = [name for name in inputs if name in consumed(suffix)]
    routes = []
    for value in model.graph.output:
        name = value.name
        if name in inputs:
            route = 'input'
        elif name in initializers or (name in producer and model.graph.node[producer[name][0]].op_type == 'Constant'):
            route = 'static'
        elif name in producer:
            route = 'prefix' if producer[name][0] in prefix else 'suffix'
        else:
            raise ValueError(f'Cannot route original output {name}')
        routes.append({'name': name, 'route': route})
    prefix_outputs = list(dict.fromkeys(crossing + [r['name'] for r in routes if r['route'] == 'prefix']))
    suffix_outputs = [r['name'] for r in routes if r['route'] == 'suffix']
    if not suffix_outputs:
        raise ValueError('The suffix has no original outputs to execute')
    return dict(
        dag=dag,
        producer=producer,
        topo=topo,
        prefix=prefix,
        suffix=suffix,
        crossing=crossing,
        prefix_inputs=prefix_inputs,
        forwarded=forwarded,
        prefix_outputs=prefix_outputs,
        suffix_inputs=crossing + forwarded,
        suffix_outputs=suffix_outputs,
        routes=routes,
        node_count=len(compute),
        prefix_count=count,
    )


def load_inputs(model, entries, base_dir, model_kind='standard'):
    if len(entries) != len({entry['name'] for entry in entries}):
        raise ValueError('Duplicate input binding')
    entries = {entry['name']: dict(entry) for entry in entries}
    initializers = {value.name for value in model.graph.initializer}
    model_inputs = [value for value in model.graph.input if value.name not in initializers]
    if set(entries) != {value.name for value in model_inputs}:
        raise ValueError('Input manifest must bind every original graph input exactly once')
    feeds = {}
    for value in model_inputs:
        entry = entries[value.name]
        declared = [
            d.dim_value if d.HasField('dim_value') and d.dim_value > 0 else None
            for d in value.type.tensor_type.shape.dim
        ]
        if model_kind == 'bert':
            attention = next(
                (
                    node
                    for node in model.graph.node
                    if node.op_type == 'CustomMultiHeadAttention' and node.input[0] == value.name
                ),
                None,
            )
            if attention is not None:
                attrs = {a.name: helper.get_attribute_value(a) for a in attention.attribute}
                mask = numpy_helper.to_array(attrs['attention_mask']).reshape(-1)
                tokens = int(mask.sum())
                if not 0 < tokens <= declared[1] or not np.all(mask[:tokens] == 1) or np.any(mask[tokens:] != 0):
                    raise ValueError('BERT input requires the existing contiguous attention mask convention')
                declared[1] = tokens
        shape = entry.get('shape', declared)
        if not shape or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError(f'Explicit positive shape required for {value.name}: {declared}')
        if len(shape) != len(declared) or any(d is not None and d != actual for d, actual in zip(declared, shape)):
            raise ValueError(f'Input shape disagrees with ONNX: {value.name}: {shape} vs {declared}')
        dtype = np.dtype(entry.get('dtype', helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)))
        if dtype != np.dtype(helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)):
            raise ValueError(f'Input dtype disagrees with ONNX: {value.name}')
        if dtype.kind not in 'fiub':
            raise ValueError(f'Unsupported CSV dtype: {dtype}')
        path = (base_dir / entry['csv']).resolve()
        array = np.loadtxt(path, delimiter=',', dtype=dtype, ndmin=1)
        if array.size != int(np.prod(shape)) or not np.isfinite(array).all():
            raise ValueError(f'CSV shape/count or finite-value check failed: {value.name}')
        feeds[value.name] = array.reshape(shape)
        entry.update(csv=str(path), sha256=sha256(path), shape=shape, dtype=str(dtype), order='C')
    return feeds, entries


def execute(model, feeds, model_kind, requested=None):
    if model_kind in ('bert', 'vit'):
        return transformer_reference.execute(model, feeds, requested, model_kind=model_kind)
    return standard_reference.execute(model, feeds, requested)


def observe(model, feeds, names, model_kind):
    if model_kind in ('bert', 'vit'):
        return execute(model, feeds, model_kind, names)
    probe = standard_reference.expand_polynomials(copy.deepcopy(model))
    probe = onnx.shape_inference.infer_shapes(probe)
    values = {v.name: v for v in [*probe.graph.input, *probe.graph.value_info, *probe.graph.output]}
    existing = {v.name for v in probe.graph.output}
    for name in names:
        if name not in existing:
            if name not in values:
                raise ValueError(f'No inferred ValueInfo for observed tensor {name}')
            probe.graph.output.append(values[name])
    return standard_reference.execute(probe, feeds, names)


def compare_arrays(expected, actual, context):
    if set(expected) != set(actual):
        raise ValueError(f'{context}: output names differ')
    errors = {}
    for name, value in expected.items():
        other = actual[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'{context}: shape/dtype differs at {name}')
        if not np.isfinite(other).all():
            raise ValueError(f'{context}: nonfinite result at {name}')
        np.testing.assert_allclose(other, value, rtol=1e-6, atol=1e-7, err_msg=f'{context}: {name}')
        errors[name] = float(np.max(np.abs(value.astype(np.float64) - other.astype(np.float64))))
    return errors


def tensor_semantics(model, plan, entries, hints, name, array, model_kind):
    """Require proven axes; never select a physical packing from CSV rank."""
    hints = dict(hints)
    for input_name, entry in entries.items():
        if entry.get('axes'):
            hints.setdefault(input_name, {'axes': entry['axes']})
    axes = {key: value['axes'] for key, value in hints.items() if 'axes' in value}
    passthrough = {
        'Add',
        'Sub',
        'Mul',
        'Div',
        'Identity',
        'BatchNormalization',
        'Relu',
        'LeakyRelu',
        'Cast',
        'Resize',
        'Concat',
        'Simple_Polyrelu',
        'RangeNormPoly1d',
        'RangeNormPoly2d',
        'PolyAct',
        'PolyActRNPoly',
    }
    for i in plan['topo']:
        node = model.graph.node[i]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        output_axes = None
        if node.op_type in ('Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool') and node.input[0] in axes:
            output_axes = axes[node.input[0]]
        elif node.op_type in passthrough and node.input and node.input[0] in axes:
            candidates = {tuple(axes[t]) for t in node.input if t in axes}
            if len(candidates) == 1:
                output_axes = axes[node.input[0]]
        elif node.op_type == 'Transpose' and node.input[0] in axes:
            previous = axes[node.input[0]]
            perm = attrs.get('perm', list(reversed(range(len(previous)))))
            output_axes = [previous[i] for i in perm]
        elif node.op_type == 'Flatten' and attrs.get('axis', 1) == 1:
            output_axes = ['N', 'F']
        elif node.op_type == 'Gemm':
            output_axes = ['N', 'F']
        elif model_kind in ('bert', 'vit') and node.domain == 'custom_domain':
            output_axes = ['N', 'F'] if attrs.get('select_first_token') else ['N', 'T', 'D']
        if output_axes:
            for output in node.output:
                axes.setdefault(output, list(output_axes))
    labels = axes.get(name)
    representations = {
        ('N', 'C', 'H', 'W'): 'Feature2D',
        ('N', 'C', 'L'): 'Feature1D',
        ('N', 'F'): 'Feature0D',
        ('N', 'T', 'D'): 'FeatureMat',
        ('M', 'N'): 'FeatureMat',
    }
    representation = representations.get(tuple(labels or []))
    if not representation or len(labels) != array.ndim:
        raise ValueError(f'Boundary hint with unambiguous axes required: {name}, shape={array.shape}')
    return {'axes': labels, 'representation': representation}


def write_tensor(root, tensor_id, array):
    path = root / f'{tensor_id}.csv'
    fmt = '%.17g' if array.dtype.kind == 'f' else '%d'
    np.savetxt(path, array.reshape(1, -1), delimiter=',', fmt=fmt)
    restored = np.loadtxt(path, delimiter=',', dtype=array.dtype, ndmin=1).reshape(array.shape)
    np.testing.assert_array_equal(array, restored)
    return {
        'csv': path.name,
        'sha256': sha256(path),
        'dtype': str(array.dtype),
        'shape': list(array.shape),
        'element_count': array.size,
        'order': 'C',
    }


def prepare(
    onnx_path, output_dir, entries, ratio_text='0.9', model_kind='standard', boundary_hints=None, base_dir=None
):
    started = time.perf_counter()
    onnx_path, output_dir = Path(onnx_path).resolve(), Path(output_dir).resolve()
    onnx.checker.check_model(str(onnx_path))
    model = onnx.load(onnx_path)
    plan = split_graph(model, ratio_text)
    feeds, entries = load_inputs(model, entries, Path(base_dir or '.').resolve(), model_kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'split_manifest.json'
    if manifest_path.exists():
        raise ValueError(f'Refusing to overwrite published manifest: {manifest_path}')
    if any(output_dir.iterdir()):
        raise ValueError(f'Split output directory must be empty: {output_dir}')
    # Specialize only the execution/extraction copy; original node identities stay unchanged.
    prepared = copy.deepcopy(model)
    dynamic_inputs = [value for value in prepared.graph.input if value.name in feeds]
    del prepared.graph.input[:]
    prepared.graph.input.extend(dynamic_inputs)
    for value in prepared.graph.input:
        value.CopyFrom(
            helper.make_tensor_value_info(
                value.name, helper.np_dtype_to_tensor_dtype(feeds[value.name].dtype), list(feeds[value.name].shape)
            )
        )
    original_outputs = [value.name for value in model.graph.output]
    names = list(dict.fromkeys(original_outputs + plan['prefix_outputs']))
    reference = observe(prepared, feeds, names, model_kind)
    original = {name: reference[name] for name in original_outputs}
    if model_kind in ('bert', 'vit'):
        # The established transformer reference reads CSV decimals as float64,
        # independently of the source ONNX input dtype used for split feeds.
        original, _ = transformer_reference.evaluate(
            onnx_path,
            {name: entry['csv'] for name, entry in entries.items()},
            input_shapes={name: entry['shape'] for name, entry in entries.items()},
            model_kind=model_kind,
        )
    np.savez(output_dir / 'original_full_onnx_outputs.npz', **original)
    observed = {name: reference[name] for name in plan['prefix_outputs']}
    # Custom ops often lack shape inference: supply measured interfaces, not guessed rank.
    known = {value.name: value for value in prepared.graph.value_info}
    for name, array in reference.items():
        value = helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape))
        if name in known:
            known[name].CopyFrom(value)
        else:
            prepared.graph.value_info.append(value)
        for output in prepared.graph.output:
            if output.name == name:
                output.CopyFrom(value)
    metadata = {value.key: value.value for value in model.metadata_props}
    metadata['latti.model_kind'] = model_kind
    helper.set_model_props(prepared, metadata)
    prefix_path, suffix_path = output_dir / 'prefix.onnx', output_dir / 'suffix.onnx'
    with tempfile.TemporaryDirectory(dir=output_dir) as temp:
        prepared_path = Path(temp) / 'prepared.onnx'
        onnx.save_model(
            prepared,
            prepared_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location='prepared.data',
            size_threshold=1024,
        )
        for path, inputs, outputs in [
            (prefix_path, plan['prefix_inputs'], plan['prefix_outputs']),
            (suffix_path, plan['suffix_inputs'], plan['suffix_outputs']),
        ]:
            if not inputs or not outputs:
                raise ValueError('A split side has no execution interface')
            onnx.utils.extract_model(
                str(prepared_path), str(path), inputs, outputs, infer_shapes=model_kind == 'standard'
            )
            extracted = onnx.load(path)
            helper.set_model_props(extracted, metadata)
            onnx.save_model(
                extracted,
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=path.name + '.data',
                size_threshold=1024,
            )
            onnx.checker.check_model(str(path))
    prefix_model, suffix_model = onnx.load(prefix_path), onnx.load(suffix_path)
    actual_counts = {}
    for label, extracted in [('prefix', prefix_model), ('suffix', suffix_model)]:
        indices = []
        for node in extracted.graph.node:
            if node.op_type == 'Constant':
                continue
            index = plan['producer'][next(name for name in node.output if name)][0]
            if index not in plan[label]:
                raise ValueError(f'Extracted {label} includes a node from the other partition')
            indices.append(index)
        actual_counts[label] = len(indices)
    prefix_result = execute(prefix_model, {name: feeds[name] for name in plan['prefix_inputs']}, model_kind)
    prefix_errors = compare_arrays(observed, prefix_result, 'prefix equivalence')
    suffix_feeds = {}
    inputs = []
    for index, name in enumerate(plan['suffix_inputs']):
        array = prefix_result[name] if name in prefix_result else feeds[name]
        consumers = [
            {'node_index': i, 'input_port': port}
            for i in plan['topo']
            if i in plan['suffix']
            for port, tensor in enumerate(model.graph.node[i].input)
            if tensor == name
        ]
        entry = {
            'name': name,
            'id': f'boundary_{index:03d}',
            'consumers': consumers,
            'source': 'prefix' if name in plan['crossing'] else 'original_input',
            **tensor_semantics(model, plan, entries, boundary_hints or {}, name, array, model_kind),
        }
        if name in plan['producer']:
            i, port = plan['producer'][name]
            entry['producer'] = {'node_index': i, 'output_port': port}
        entry.update(write_tensor(output_dir, entry['id'], array))
        suffix_feeds[name] = np.loadtxt(
            output_dir / entry['csv'], delimiter=',', dtype=entry['dtype'], ndmin=1
        ).reshape(entry['shape'])
        inputs.append(entry)
    suffix_result = execute(suffix_model, suffix_feeds, model_kind)
    merged = {
        route['name']: (
            suffix_result[route['name']]
            if route['route'] == 'suffix'
            else prefix_result[route['name']]
            if route['route'] == 'prefix'
            else original[route['name']]
        )
        for route in plan['routes']
    }
    suffix_errors = compare_arrays(original, merged, 'full split equivalence')
    np.savez(output_dir / 'prefix_outputs.npz', **prefix_result)
    for route in plan['routes']:
        value = original[route['name']]
        route.update(shape=list(value.shape), dtype=str(value.dtype))
    manifest = {
        'schema_version': 'onnx_split_v1',
        'source': {'path': str(onnx_path), 'sha256': sha256(onnx_path), 'files': model_files(onnx_path)},
        'split': {
            'ratio_text': ratio_text,
            'count_policy': 'top_level_non_constant_v1',
            'node_count': plan['node_count'],
            'prefix_count': plan['prefix_count'],
            'topology': 'networkx_lexicographical_original_index',
            'nodes': [
                {
                    'original_index': i,
                    'topological_position': position,
                    'partition': 'static'
                    if model.graph.node[i].op_type == 'Constant'
                    else 'prefix'
                    if i in plan['prefix']
                    else 'suffix',
                }
                for position, i in enumerate(plan['topo'])
            ],
        },
        'models': {'prefix': prefix_path.name, 'suffix': suffix_path.name},
        'model_files': {'prefix': model_files(prefix_path), 'suffix': model_files(suffix_path)},
        'model_kind': model_kind,
        'original_inputs': list(entries.values()),
        'inputs': inputs,
        'output_routes': plan['routes'],
        'reference': 'original_full_onnx_outputs.npz',
        'prefix_outputs': 'prefix_outputs.npz',
        'artifacts': [
            {'path': name, 'sha256': sha256(output_dir / name)}
            for name in ('original_full_onnx_outputs.npz', 'prefix_outputs.npz')
        ],
        'validation': {
            'prefix_max_abs': prefix_errors,
            'split_max_abs': suffix_errors,
            'actual_node_counts': actual_counts,
            'prepare_wall_s': time.perf_counter() - started,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--ratio', default='0.9')
    parser.add_argument('--output-dir', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--inputs-manifest', type=Path, help='JSON: inputs [{name,csv,shape,dtype,axes}], model_kind, boundary_hints'
    )
    source.add_argument('--input-csv', type=Path, help='Single static input; axes must be supplied')
    parser.add_argument('--axes', help='Comma-separated logical input axes, e.g. N,C,H,W')
    args = parser.parse_args()
    if args.inputs_manifest:
        config = json.loads(args.inputs_manifest.read_text())
        base_dir = args.inputs_manifest.resolve().parent
    else:
        model = onnx.load(args.onnx, load_external_data=False)
        if len(model.graph.input) != 1 or not args.axes:
            parser.error('--input-csv requires one input and explicit --axes')
        config = {
            'inputs': [
                {'name': model.graph.input[0].name, 'csv': str(args.input_csv.resolve()), 'axes': args.axes.split(',')}
            ]
        }
        base_dir = Path.cwd()
    manifest = prepare(
        args.onnx,
        args.output_dir,
        config['inputs'],
        args.ratio,
        config.get('model_kind', 'standard'),
        config.get('boundary_hints'),
        base_dir,
    )
    print(
        json.dumps(
            {
                'manifest': str(args.output_dir / 'split_manifest.json'),
                'split': {k: v for k, v in manifest['split'].items() if k != 'nodes'},
                'validation': manifest['validation'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
