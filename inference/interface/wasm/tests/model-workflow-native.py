#!/usr/bin/env python3
"""Isolated native oracle and evidence checks; never supplies main-chain results."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import h5py
import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]
from prepare_split_onnx import prepare, compare_arrays, load_inputs, observe
from onnx_split_contract import file_hash, validate_contract


def normalize(value, aliases=None):
    if aliases is None:
        aliases = {}
        if isinstance(value, dict) and 'layer' in value and 'feature' in value:
            for name, layer in value['layer'].items():
                if layer.get('type') != 'drop_level':
                    continue
                source, target = layer['feature_input'][0], layer['feature_output'][0]
                suffix = '_drop_level_output'
                if (
                    target.startswith(source)
                    and target.endswith(suffix)
                    and target[len(source) : -len(suffix)].isdigit()
                ):
                    aliases[target] = name + '__output_0'
    if isinstance(value, str):
        for original, canonical in aliases.items():
            value = value.replace(original, canonical)
        return re.sub(r'_refreshed\d+', '_refreshed<ID>', re.sub(r'_ts\d+', '_ts<ID>', value))
    if isinstance(value, list):
        return [normalize(item, aliases) for item in value]
    if isinstance(value, dict):
        result = {normalize(key, aliases): normalize(item, aliases) for key, item in value.items()}
        if len(result) != len(value):
            raise ValueError('Semantic IDs collide after compiler normalization')
        return result
    return value


def artifact(directory):
    manifest = json.loads((directory / 'artifact-manifest.json').read_text())
    for name, entry in manifest['files'].items():
        path = directory / name
        if (
            not path.is_relative_to(directory)
            or path.stat().st_size != entry['bytes']
            or file_hash(path) != entry['sha256']
        ):
            raise ValueError(f'Browser artifact changed: {path}')
    return manifest


def graph_fingerprint(graph):
    """Preserve all port-labelled dependencies while removing temporary node IDs."""
    dag = nx.DiGraph()
    for name, data in graph['data'].items():
        attributes = normalize(data)
        if name in graph['inputs'] and not re.fullmatch('[a-z]{12}', name):
            attributes['interface'] = normalize(name)
        dag.add_node(('data', name), attributes=attributes)
    for name, computation in graph['compute'].items():
        attributes = normalize({key: value for key, value in computation.items() if key not in ('inputs', 'outputs')})
        node = ('compute', name)
        dag.add_node(node, attributes=attributes)
        for port, source in enumerate(computation['inputs']):
            data = ('data', source)
            if not dag.has_node(data):
                raise ValueError('MegaAG references missing data')
            # Preserve repeated input ports in the edge's label.
            if dag.has_edge(data, node):
                dag[data][node]['ports'].append(('in', port))
            else:
                dag.add_edge(data, node, ports=[('in', port)])
        for port, target in enumerate(computation['outputs']):
            dag.add_edge(node, ('data', target), ports=[('out', port)])
    fingerprints = {}
    for node in nx.topological_sort(dag):
        attributes = dag.nodes[node].get('attributes')
        if attributes is None:
            raise ValueError('MegaAG references missing output data')
        predecessors = sorted((fingerprints[parent], dag[parent][node]['ports']) for parent in dag.predecessors(node))
        text = json.dumps([node[0], attributes, predecessors], sort_keys=True, separators=(',', ':'))
        fingerprints[node] = hashlib.sha256(text.encode()).hexdigest()
    return {
        'nodes': sorted(Counter(fingerprints.values()).items()),
        'outputs': [fingerprints['data', name] for name in graph['outputs']],
    }


def oracle(args):
    from training.run_compile import compile_model
    from inference.interface.gen_mega_ag import generate_low_level

    browser = args.browser.resolve()
    split = json.loads((browser / 'split/split_manifest.json').read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = artifact(browser)
    model = browser / split['source']['path'].removeprefix(metadata['virtualRoot'] + '/')
    entries = [
        {**entry, 'csv': str(browser / entry['csv'].removeprefix(metadata['virtualRoot'] + '/'))}
        for entry in split['original_inputs']
    ]
    model_kind = split.get('model_kind', 'standard')
    prepared = prepare(model, root / 'split', entries, split['split']['ratio_text'], model_kind=model_kind)
    checks = {}
    for key in ('split', 'security'):
        if split[key] != prepared[key]:
            raise ValueError(f'Browser/native split differs: {key}')
    for old, new in zip(split['inputs'], prepared['inputs'], strict=True):
        fields = ('name', 'shape', 'dtype', 'axes', 'representation', 'producer', 'consumers')
        if any(old.get(key) != new.get(key) for key in fields):
            raise ValueError('Browser/native crossing interfaces differ')
    # The normative prefix/suffix check is inside each runtime's prepare().
    # CPU MLAS and WASM SIMD can round differently even at the same ORT version;
    # preserve the exact cross-runtime differences and audit the first Conv.
    for name in ('original_full_onnx_outputs.npz', 'prefix_outputs.npz'):
        with np.load(browser / 'split' / name) as main, np.load(root / 'split' / name) as reference:
            if set(main.files) != set(reference.files):
                raise ValueError('Browser/native observation names differ')
            checks[name] = {}
            for key in main.files:
                a, b = main[key], reference[key]
                if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Browser/native observation shape/dtype/finite check failed')
                checks[name][key] = {
                    'max_abs': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
                    'matches_compare_arrays': bool(np.allclose(a, b, atol=1e-7, rtol=1e-6)),
                    'atol': 1e-7,
                    'rtol': 1e-6,
                }
    for mode in args.modes.split(','):
        output = root / mode
        config = {'bert': 'BERT_pcm_config.json', 'vit': 'ViT_pcm_config.json'}.get(model_kind)
        if config is None and mode == 'server_provisioned_runner':
            config = 'storage_aware_config.json'
        task = compile_model(
            root / 'split/suffix.onnx',
            output,
            deployment_mode=mode,
            split_manifest=root / 'split/split_manifest.json',
            temperature=0.0,
            num_workers=1,
            random_seed=0,
            config_path=str(REPO / 'training/config' / config) if config else '',
        )
        generate_low_level(
            task,
            deployment_mode=mode,
            random_seed=0,
            parameter_loading_mode='runtime_lazy' if mode == 'server_provisioned_runner' else None,
            streaming=mode == 'server_provisioned_runner' and split['split']['prefix_count'] == 0,
        )
    (root / 'oracle.json').write_text(
        json.dumps({'provenance': 'isolated native oracle', 'checks': checks}, indent=2) + '\n'
    )


def kernel_audit(args):
    import onnx
    import onnxruntime

    root = args.browser.resolve()
    metadata = artifact(root)
    nodes = json.loads((root / 'diagnostic/nodes.json').read_text())
    model = onnx.load(root / metadata['sourceModel'])
    source = root / 'source'
    source_manifest = json.loads((source / 'inputs.json').read_text())
    inputs = source_manifest['inputs']
    model_kind = source_manifest.get('model_kind', 'standard')
    feeds, _ = load_inputs(model, inputs, source, model_kind)
    native = observe(model, feeds, [node['name'] for node in nodes], model_kind)
    checks = []
    with np.load(root / 'diagnostic/observations.npz') as browser:
        for node in nodes:
            name = node['name']
            a, b = browser[name], native[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Kernel audit shape/dtype differs')
            checks.append(
                {
                    **node,
                    'dtype': str(a.dtype),
                    'shape': a.shape,
                    'max_abs': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
                    'matches_compare_arrays': bool(np.allclose(a, b, atol=1e-7, rtol=1e-6)),
                }
            )
    first = next((entry for entry in checks if entry['max_abs']), None)
    args.output.write_text(
        json.dumps(
            {
                'native_ort': onnxruntime.__version__,
                'native_provider': 'CPUExecutionProvider / MLAS',
                'browser_provider': 'WASM SIMD, one thread',
                'expansion': (
                    'shared per-node standard float64 ONNX lowering'
                    if model_kind in ('bert', 'vit')
                    else 'shared expand_polynomials with float32 constants'
                ),
                'first_difference': first,
                'checkpoints': checks,
                'normative_equivalence': 'Browser original/prefix/suffix compare_arrays retains atol=1e-7, rtol=1e-6.',
            },
            indent=2,
        )
        + '\n'
    )


def graph_records(path):
    """Read compiler JSON records without retaining the entire graph."""
    decoder = json.JSONDecoder()
    with path.open() as file:
        buffer, offset = '', 0

        def fill():
            nonlocal buffer, offset
            chunk = file.read(1024 * 1024)
            buffer, offset = buffer[offset:] + chunk, 0
            return bool(chunk)

        def token(expected):
            nonlocal offset
            while True:
                while offset < len(buffer) and buffer[offset].isspace():
                    offset += 1
                if offset < len(buffer):
                    break
                if not fill():
                    raise ValueError('Truncated graph JSON')
            if buffer[offset] != expected:
                raise ValueError('Invalid graph JSON delimiter')
            offset += 1

        def value():
            nonlocal offset
            while True:
                while offset < len(buffer) and buffer[offset].isspace():
                    offset += 1
                try:
                    result, end = decoder.raw_decode(buffer, offset)
                    offset = end
                    return result
                except json.JSONDecodeError:
                    if not fill():
                        raise

        token('{')
        for section in ('name', 'algorithm', 'inputs', 'outputs', 'data', 'compute'):
            if value() != section:
                raise ValueError('Unexpected streamed graph section')
            token(':')
            if section not in ('data', 'compute'):
                yield section, value()
            else:
                token('{')
                first = True
                while True:
                    while offset >= len(buffer) or buffer[offset].isspace():
                        if offset < len(buffer):
                            offset += 1
                        elif not fill():
                            raise ValueError('Truncated graph section')
                    if buffer[offset] == '}':
                        offset += 1
                        break
                    if not first:
                        token(',')
                    key = value()
                    token(':')
                    yield section, key, value()
                    first = False
            token('}' if section == 'compute' else ',')
        if buffer[offset:].strip() or file.read().strip():
            raise ValueError('Trailing graph JSON')


def compare_streamed_graphs(main, reference):
    digest = hashlib.sha256()
    counts = Counter()
    for index, (a, b) in enumerate(zip(graph_records(main), graph_records(reference), strict=True)):
        a, b = normalize(list(a)), normalize(list(b))
        if a != b:
            raise ValueError(f'Browser/native graph record differs at {index}: {a[:2]} vs {b[:2]}')
        digest.update(json.dumps(a, sort_keys=True, separators=(',', ':')).encode())
        counts[a[0]] += 1
    return dict(sha256=digest.hexdigest(), records=dict(counts))


def verify_task(args):
    root = args.browser.resolve()
    metadata = artifact(root)
    task = root / metadata['entry']
    validate_contract(task, path_map={metadata['virtualRoot']: root})
    role = 'runner' if metadata['deploymentMode'] == 'server_provisioned_runner' else 'server'
    native = args.oracle / metadata['deploymentMode'] / 'task'
    checks = {}
    for filename in ('ckks_parameter.json', 'nn_layers_ct_0.json', 'task_signature.json', 'key_requirements.json'):
        main = normalize(json.loads((task / role / filename).read_text()))
        reference = normalize(json.loads((native / role / filename).read_text()))
        if main != reference:
            raise ValueError('Browser/native compiler artifact differs: ' + filename)
        checks[filename] = True
    if (task / role / 'graph_key_data.json').exists():
        checks['mega_ag.json'] = compare_streamed_graphs(task / role / 'mega_ag.json', native / role / 'mega_ag.json')
        checks['compiled_mega_ag.json'] = compare_streamed_graphs(
            task / role / 'compiled_mega_ag.json', native / role / 'compiled_mega_ag.json'
        )
    else:
        main_graph = json.loads((task / role / 'mega_ag.json').read_text())
        reference_graph = json.loads((native / role / 'mega_ag.json').read_text())
        if graph_fingerprint(main_graph) != graph_fingerprint(reference_graph):
            raise ValueError('Browser/native MegaAG interface/topology differs')
        checks['mega_ag.json'] = True
    datasets = {}
    with (
        h5py.File(task / 'server/model_parameters.h5') as main,
        h5py.File(native / 'server/model_parameters.h5') as reference,
    ):
        names, reference_names = {}, {}
        main.visititems(
            lambda name, data: names.__setitem__(normalize(name), name) if isinstance(data, h5py.Dataset) else None
        )
        reference.visititems(
            lambda name, data: reference_names.__setitem__(normalize(name), name)
            if isinstance(data, h5py.Dataset)
            else None
        )
        if names.keys() != reference_names.keys():
            raise ValueError('Browser/native H5 dataset names differ')
        for key in names:
            a, b = main[names[key]], reference[reference_names[key]]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError('Browser/native H5 shape/dtype differs: ' + key)
            np.testing.assert_allclose(a[...], b[...], atol=1e-12, rtol=1e-12)
            datasets[key] = {'shape': a.shape, 'dtype': str(a.dtype), 'max_abs': float(np.max(np.abs(a[...] - b[...])))}
    args.output.write_text(
        json.dumps(
            {'provenance': 'browser compared with isolated native oracle', 'checks': checks, 'h5': datasets}, indent=2
        )
        + '\n'
    )


def compare(args):
    root = args.browser.resolve()
    metadata = artifact(root)
    contract = json.loads((root / metadata['entry'] / 'input_contract.json').read_text())
    split = json.loads((root / 'split/split_manifest.json').read_text())
    result = json.loads((args.results / 'manifest.json').read_text())
    if [entry['name'] for entry in result['outputs']] != [route['name'] for route in split['output_routes']]:
        raise ValueError('Decrypted output order/names differ')
    reports = {}
    with np.load(root / 'split/original_full_onnx_outputs.npz') as reference:
        for entry, route in zip(result['outputs'], split['output_routes'], strict=True):
            expected = reference[entry['name']]
            if entry['shape'] != list(expected.shape) or entry['dtype'] != str(expected.dtype):
                raise ValueError('Original output shape/dtype differs')
            values = np.loadtxt(args.results / entry['file'], delimiter=',', dtype=np.float64, ndmin=1).reshape(-1)
            if values.size != expected.size or not np.isfinite(values).all():
                raise ValueError('Decrypted output count/finite check failed')
            errors = np.abs(values - expected.reshape(-1).astype(np.float64))
            reports[entry['name']] = {
                'shape': entry['shape'],
                'dtype': entry['dtype'],
                'numeric_dtype': 'float64',
                'count': values.size,
                'max_abs_error': float(errors.max()),
                'max_error_index': int(errors.argmax()),
                'route': route['route'],
            }
    maximum = max(entry['max_abs_error'] for entry in reports.values())
    report = {
        'reference': 'original_full_onnx_csv_plaintext',
        'outputs': reports,
        'max_abs_error': maximum,
        'deployment_mode': contract['deployment_mode'],
        'tolerance': 0.1,
        'pass': maximum < 0.1,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['pass']:
        raise ValueError('Maximum error must be strictly below 0.1')


def verify_bundle(args):
    from inference.model_generator.parameter_materialization import ParameterMaterializer

    metadata = artifact(args.browser)
    materializer = ParameterMaterializer(args.browser / metadata['entry'])
    manifest = json.loads((args.bundle / 'encrypted_model_parameters/manifest.json').read_text())
    plan = {item['id']: item for item in materializer.plan()}
    if manifest['format'] != 'latti-ai.encrypted-parameter-store.v2' or plan.keys() != manifest['arguments'].keys():
        raise ValueError('Encrypted parameter argument set differs')
    total, maximum, checked = 0, 0, {}
    for name, expected in plan.items():
        entry = manifest['arguments'][name]
        for key in (
            'size',
            'level',
            'scale',
            'layout',
            'generator',
            'param_kind',
            'layer_id',
            'count',
            'slots',
            'use_site_level',
        ):
            if entry.get(key) != expected.get(key):
                raise ValueError(f'Parameter signature/layout differs: {name}/{key}')
        with (args.bundle / 'encrypted_model_parameters' / entry['file']).open('rb') as file:
            (count,) = struct.unpack('<Q', file.read(8))
            if count != expected['count'] or len(entry['elements']) != count:
                raise ValueError('Parameter flat count differs')
            for span in entry['elements']:
                (length,) = struct.unpack('<Q', file.read(8))
                if length != span['length'] or file.tell() != span['offset'] or not length:
                    raise ValueError('Parameter offset/length index differs')
                file.seek(length, 1)
                maximum = max(maximum, length)
            if file.tell() != file.seek(0, 2):
                raise ValueError('Parameter container has trailing data')
        if entry.get('verified_count') != count or not entry.get('encoding_max_abs_error', math.inf) < 1e-5:
            raise ValueError('Missing browser parameter decryption verification')
        total += count
        checked[name] = {
            key: entry.get(key)
            for key in (
                'size',
                'level',
                'scale',
                'layout',
                'generator',
                'param_kind',
                'layer_id',
                'count',
                'slots',
                'use_site_level',
                'encoding_max_abs_error',
                'verified_count',
            )
        }
    args.output.write_text(
        json.dumps({'arguments': checked, 'ciphertexts': total, 'max_ciphertext_bytes': maximum}, indent=2) + '\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('oracle', 'verify-task', 'compare', 'verify-bundle', 'kernel-audit'))
    parser.add_argument('--browser', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--oracle', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--modes', default='client_encrypted_input,server_provisioned_runner')
    args = parser.parse_args()
    {
        'oracle': oracle,
        'verify-task': verify_task,
        'compare': compare,
        'verify-bundle': verify_bundle,
        'kernel-audit': kernel_audit,
    }[args.operation](args)


if __name__ == '__main__':
    main()
