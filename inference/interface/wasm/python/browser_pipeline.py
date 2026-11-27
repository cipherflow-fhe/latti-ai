# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Pyodide entry points. Inference is supplied by the model Worker's ORT Web."""

import copy
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

from prepare_split_onnx import prepare_steps
from prepare_split_onnx import load_inputs
from prepare_split_onnx import observation_model
from evaluate_onnx_csv import expand_polynomials
from onnx_external_weights import externalize, model_metadata, tensor_array


def relative_path(root, name):
    """Resolve a portable source reference without traversal or absolute paths."""
    if not isinstance(name, str) or not name or any(c in name for c in ('\\', '\0', ':')):
        raise ValueError('Invalid relative artifact path')
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('Artifact path must be a safe relative path')
    root = Path(root).resolve()
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Artifact path escapes its directory')
    return result


def validate_sources(source_directory, model_path, inputs_manifest):
    """Check references before ONNX checker/load can follow external data."""
    model_path = relative_path(source_directory, model_path)
    model = model_metadata(model_path)
    if len(model.graph.node) > 256:
        raise ValueError('Model exceeds the 256-node browser profile')
    initializer_sizes = [
        math.prod(tensor.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        for tensor in model.graph.initializer
    ]
    if sum(initializer_sizes) > 384 * 1024**2 or max(initializer_sizes, default=0) > 32 * 1024**2:
        raise ValueError('Model initializer storage exceeds the validated browser profile')
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = onnx.external_data_helper.ExternalDataInfo(tensor).location
            relative_path(model_path.parent, location)
    manifest = validate_inputs(source_directory, inputs_manifest)
    return model_path, manifest


def validate_inputs(source_directory, inputs_manifest):
    """Apply the same allocation and CSV gates to initial and subsequent samples."""
    from onnx_split_contract import validate_csv

    manifest = inputs_manifest if isinstance(inputs_manifest, dict) else json.loads(Path(inputs_manifest).read_text())
    if manifest.get('model_kind', 'standard') not in ('standard', 'bert', 'vit'):
        raise ValueError('Unknown ONNX execution semantics')
    for entry in manifest['inputs']:
        relative_path(source_directory, entry['csv'])
        shape, axes = entry.get('shape'), entry.get('axes')
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Browser inputs require a complete positive shape')
        if math.prod(shape) > 2**22:
            raise ValueError('Input tensor exceeds the validated browser profile')
        if not isinstance(axes, list) or len(axes) != len(shape) or len(set(axes)) != len(axes):
            raise ValueError('Browser inputs require complete, unambiguous axes')
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError('Browser inputs require named axes')
        validate_csv(relative_path(source_directory, entry['csv']), entry.get('dtype', 'float64'), math.prod(shape))
    return manifest


def write_ort_request(request, directory):
    """Persist the FFI boundary; tensors retain dtype, shape and C order."""
    directory = Path(directory)
    directory.mkdir()
    model = request['model'] if request.get('standardized') else expand_polynomials(copy.deepcopy(request['model']))
    capacity = copy.deepcopy(model)
    del capacity.graph.value_info[:]
    capacity = onnx.shape_inference.infer_shapes(capacity)
    total_bytes, maximum_bytes = 0, 0
    for value in [*capacity.graph.input, *capacity.graph.value_info, *capacity.graph.output]:
        tensor = value.type.tensor_type
        if not tensor.HasField('shape') or any(not dim.HasField('dim_value') for dim in tensor.shape.dim):
            raise ValueError('Ambiguous tensor shape before ORT allocation: ' + value.name)
        size = (
            math.prod(dim.dim_value for dim in tensor.shape.dim)
            * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        )
        total_bytes += size
        maximum_bytes = max(maximum_bytes, size)
    if maximum_bytes > 32 * 1024**2 or total_bytes > 2 * 1024**3:
        raise ValueError('ORT tensor geometry exceeds the validated browser memory profile')
    del capacity
    model_path = directory / 'model.onnx'
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location='model.data',
        size_threshold=1024,
    )
    external = (
        [{'path': 'model.data', 'file': str(directory / 'model.data')}] if (directory / 'model.data').exists() else []
    )
    feeds = []
    for index, (name, value) in enumerate(request['feeds'].items()):
        path = directory / f'input-{index}.bin'
        value = np.ascontiguousarray(value)
        value.tofile(path)
        feeds.append({'name': name, 'file': str(path), 'dtype': str(value.dtype), 'shape': list(value.shape)})
    return {
        'stage': request['stage'],
        'model': str(model_path),
        'externalData': external,
        'feeds': feeds,
        'outputs': request['outputs'],
        'directory': str(directory),
        'maximumTensorBytes': maximum_bytes,
        'totalTensorBytes': total_bytes,
    }


def read_ort_result(directory, result):
    result = json.loads(result)
    arrays = {}
    for output in result['outputs']:
        name = output['name']
        if name in arrays:
            raise ValueError('Duplicate ORT output name')
        shape = output['shape']
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError('Invalid ORT output shape')
        dtype = np.dtype(output['dtype'])
        if dtype.kind not in 'fiub':
            raise ValueError('Unsupported ORT output dtype')
        path = relative_path(directory, output['file'])
        count = int(np.prod(shape, dtype=object))
        if path.stat().st_size != count * dtype.itemsize:
            raise ValueError('ORT output length differs from shape/dtype')
        value = np.fromfile(path, dtype=dtype).reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError('ORT output contains nonfinite values')
        arrays[name] = value
    return arrays


async def prepare_split(source_directory, model_path, inputs_manifest, output_directory, ratio, ort_run):
    """Await the browser executor at each shared split inference boundary."""
    if not isinstance(ratio, str):
        raise ValueError('Split ratio must be a decimal string')
    model_path, manifest = validate_sources(source_directory, model_path, inputs_manifest)
    output_directory = Path(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    steps = prepare_steps(
        model_path,
        output_directory,
        manifest['inputs'],
        ratio,
        model_kind=manifest.get('model_kind', 'standard'),
        boundary_hints=manifest.get('boundary_hints'),
        base_dir=source_directory,
    )
    return await _drive_split(steps, output_directory.parent, ort_run)


def export_prefix(split_manifest, output_directory):
    """Publish a self-contained prefix; no suffix, sample CSV or private paths."""
    from onnx_split_contract import file_hash, load_manifest
    from training.model_export.operations import format_id

    split_manifest = Path(split_manifest)
    split = load_manifest(split_manifest)
    if not split['models']['prefix']:
        raise ValueError('This split has no prefix to export')
    output_directory = Path(output_directory)
    model_path = externalize(
        relative_path(split_manifest.parent, split['models']['prefix']), output_directory / 'model.onnx'
    )
    model = onnx.load(model_path, load_external_data=False)
    model_kind = split.get('model_kind', 'standard')
    if model_kind == 'standard':
        model = expand_polynomials(model)
    # A full-model output may be an original input or a static constant. Include
    # just that public output, without copying the original model or its weights.
    for route in split['output_routes']:
        if route['route'] not in ('input', 'static'):
            continue
        value = onnx.helper.make_tensor_value_info(
            route['name'], onnx.helper.np_dtype_to_tensor_dtype(np.dtype(route['dtype'])), route['shape']
        )
        if route['route'] == 'input' and route['name'] not in {v.name for v in model.graph.input}:
            model.graph.input.append(value)
        defined = {v.name for v in model.graph.initializer} | {
            name for node in model.graph.node for name in node.output
        }
        if route['route'] == 'static' and route['name'] not in defined:
            reference = relative_path(split_manifest.parent, route['plaintext']['csv'])
            array = np.loadtxt(reference, delimiter=',', dtype=route['dtype'], ndmin=1).reshape(route['shape'])
            model.graph.initializer.append(onnx.numpy_helper.from_array(array, route['name']))
        if route['name'] not in {v.name for v in model.graph.output}:
            model.graph.output.append(value)
    del model.metadata_props[:]
    onnx.save_model(model, str(model_path))
    # Repack newly introduced constants as well, keeping all weight I/O bounded.
    compact = externalize(model_path, output_directory / 'compact.onnx')
    model_path.unlink()
    (output_directory / 'model.onnx.data').unlink()
    compact.rename(model_path)
    # externalize names the data file after the graph; preserve that reference.
    onnx.checker.check_model(str(model_path))
    names = {v.name for v in model.graph.input}
    inputs = [
        {k: v for k, v in entry.items() if k in ('name', 'shape', 'dtype', 'axes', 'order')}
        for entry in split['original_inputs']
        if entry['name'] in names
    ]
    outputs = [
        {k: v for k, v in entry.items() if k in ('name', 'shape', 'dtype', 'axes', 'representation')}
        for entry in split['inputs']
    ]
    for entry in outputs:
        entry['feature_id'] = format_id(entry['name'])
    specification = dict(
        schema='latti-ai.onnx-prefix.v1',
        model='model.onnx',
        externalData=['compact.onnx.data'],
        model_kind=model_kind,
        inputs=inputs,
        outputs=outputs,
        output_routes=[
            {k: v for k, v in entry.items() if k in ('name', 'route', 'shape', 'dtype')}
            for entry in split['output_routes']
        ],
        sourceModelSHA256=split['source']['sha256'],
        splitManifestSHA256=file_hash(split_manifest),
    )
    (output_directory / 'prefix.json').write_text(json.dumps(specification, indent=2) + '\n')


async def run_exported_prefix(prefix_manifest, inputs_directory, inputs_manifest, output_directory, ort_run):
    """Run an independent prefix for one or more samples, serially.

    The legacy ``{"inputs": [...]}`` shape remains accepted for one sample.
    Batch callers use ``{"samples": [{"case_id": ..., "inputs": [...]}]}``;
    every sample is executed and persisted before the next one starts.
    """
    from onnx_split_contract import file_hash
    from prepare_split_onnx import write_tensor
    import shutil
    import time

    prefix_manifest, output_directory = Path(prefix_manifest), Path(output_directory)
    spec = json.loads(prefix_manifest.read_text())
    if spec.get('schema') != 'latti-ai.onnx-prefix.v1' or spec.get('model_kind') not in ('standard', 'bert', 'vit'):
        raise ValueError('Unknown prefix artifact semantics')
    model_path = relative_path(prefix_manifest.parent, spec['model'])
    model = model_metadata(model_path)
    if len(model.graph.node) > 4096:
        raise ValueError('Expanded prefix exceeds the browser node budget')
    storage = 0
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        size = math.prod(tensor.dims) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        if size > 32 * 1024**2:
            raise ValueError('Prefix weight exceeds the browser tensor budget')
        storage += size
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            relative_path(model_path.parent, onnx.external_data_helper.ExternalDataInfo(tensor).location)
    if storage > 384 * 1024**2:
        raise ValueError('Prefix weights exceed the browser storage budget')
    expected = {entry['name']: entry for entry in spec['inputs']}
    raw = json.loads(Path(inputs_manifest).read_text())
    batch = 'samples' in raw
    samples = raw.get('samples')
    if samples is None:
        samples = [{'case_id': 'sample-0', 'inputs': raw.get('inputs', [])}]
    if not isinstance(samples, list) or not samples or len(samples) > 100000:
        raise ValueError('Prefix batch requires one to one hundred thousand samples')
    output_directory.mkdir(parents=True, exist_ok=True)
    cases_directory = output_directory / 'cases'
    if batch:
        cases_directory.mkdir()
    metrics = dict(reads=0, bytes=0, largestTensorBytes=0)
    records = []
    seen = set()
    for sample_index, sample_input in enumerate(samples):
        started = time.perf_counter()
        case_id = sample_input.get('case_id', f'sample-{sample_index}')
        if not isinstance(case_id, str) or '/' in case_id:
            raise ValueError('Invalid prefix sample ID')
        relative_path(output_directory, case_id)
        import unicodedata

        case_key = unicodedata.normalize('NFC', case_id).casefold()
        if case_key in seen:
            raise ValueError('Invalid or duplicate prefix sample ID')
        seen.add(case_key)
        entries = sample_input.get('inputs')
        if not isinstance(entries, list):
            raise ValueError('Prefix sample inputs must be a list')
        if len(entries) != len(expected) or {e.get('name') for e in entries} != expected.keys():
            raise ValueError('Prefix input names differ')
        for entry in entries:
            declared = expected[entry['name']]
            if entry.get('shape') != declared['shape'] or entry.get('axes') != declared['axes']:
                raise ValueError('Prefix input shape/axes differ')
            relative_path(inputs_directory, entry['csv'])
        # Validate this sample's CSVs with the same allocation gates as the
        # single-sample path, without loading any other sample.
        validate_inputs(inputs_directory, {'inputs': entries, 'model_kind': spec['model_kind']})
        feeds, _ = load_inputs(model, entries, Path(inputs_directory), spec['model_kind'])
        request = dict(
            model=model,
            feeds=feeds,
            outputs=[v.name for v in model.graph.output],
            model_kind=spec['model_kind'],
            base_dir=str(model_path.parent),
            stage='prefix',
            weight_metrics=metrics,
        )
        temporary = output_directory.parent / f'ort-prefix-{sample_index}'
        values = await (
            run_transformer_request(request, temporary, ort_run)
            if spec['model_kind'] in ('bert', 'vit')
            else run_standard_request(request, temporary, ort_run)
        )
        shutil.rmtree(temporary, ignore_errors=True)
        case_directory = cases_directory / case_id if batch else output_directory
        case_directory.mkdir(exist_ok=True)
        crossings = []
        for index, entry in enumerate(spec['outputs']):
            array = values[entry['name']]
            if list(array.shape) != entry['shape'] or str(array.dtype) != entry['dtype']:
                raise ValueError('Prefix crossing shape/dtype differs')
            crossings.append({**entry, **write_tensor(case_directory, f'input_{index:03d}', array)})
        routes = copy.deepcopy(spec['output_routes'])
        for index, route in enumerate(routes):
            if route['route'] != 'suffix':
                array = values[route['name']]
                if list(array.shape) != route['shape'] or str(array.dtype) != route['dtype']:
                    raise ValueError('Prefix output route shape/dtype differs')
                route['plaintext'] = write_tensor(case_directory, f'output_{index:03d}', array)
        sample = dict(case_id=case_id, inputs=crossings, output_routes=routes)
        sample_path = case_directory / 'sample_manifest.json'
        sample_path.write_text(json.dumps(sample, indent=2) + '\n')
        records.append(
            {
                'case_id': case_id,
                'manifest': f'cases/{case_id}/sample_manifest.json' if batch else 'sample_manifest.json',
                'manifest_sha256': file_hash(sample_path),
                'duration_ms': (time.perf_counter() - started) * 1000,
            }
        )
        del feeds, values, request
    binding = dict(
        schema='latti-ai.prefix-run.v2',
        split_manifest_sha256=spec['splitManifestSHA256'],
        prefix_manifest_sha256=file_hash(prefix_manifest),
        samples=records,
        count=len(records),
        weightLoading='on_demand',
        weightReads=metrics,
    )
    (output_directory / 'batch_manifest.json').write_text(json.dumps(binding, indent=2) + '\n')
    if not batch:
        binding.update(sample_manifest=records[0]['manifest'], sample_manifest_sha256=records[0]['manifest_sha256'])
    (output_directory / 'run_manifest.json').write_text(json.dumps(binding, indent=2) + '\n')


def _read_weight(request, tensor):
    array = tensor_array(tensor, request['base_dir'])
    metrics = request.get('weight_metrics')
    if metrics is not None:
        metrics['reads'] += 1
        metrics['bytes'] += array.nbytes
        metrics['largestTensorBytes'] = max(metrics['largestTensorBytes'], array.nbytes)
    return array


async def _drive_split(steps, temporary_parent, ort_run):
    result = None
    try:
        with tempfile.TemporaryDirectory(dir=temporary_parent) as bridge:
            index = 0
            while True:
                try:
                    request = steps.send(result)
                except StopIteration as finished:
                    return json.dumps(finished.value)
                directory = Path(bridge) / str(index)
                if request.get('model_kind') in ('bert', 'vit'):
                    result = await run_transformer_request(request, directory, ort_run)
                else:
                    metadata = write_ort_request(request, directory)
                    response = await ort_run(json.dumps(metadata))
                    result = read_ort_result(directory, response)
                # No subsequent inference needs the bridge's persisted copies.
                import shutil

                shutil.rmtree(directory)
                index += 1
    finally:
        steps.close()


async def run_transformer_request(request, directory, ort_run):
    """Run each original transformer node through the shared standard ONNX lowering."""
    from collections import Counter
    import shutil
    from expand_transformer_onnx import lower_node

    directory = Path(directory)
    directory.mkdir(parents=True)
    model = request['model']
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    requested = request['outputs'] or [value.name for value in model.graph.output]
    values = {name: np.asarray(value, dtype=np.float64) for name, value in request['feeds'].items()}
    remaining = Counter(name for node in model.graph.node for name in node.input)
    for index, node in enumerate(model.graph.node):
        feeds = {name: values[name] for name in node.input if name not in initializers}
        loader = (lambda tensor: _read_weight(request, tensor)) if request.get('base_dir') else None
        lowered, feeds = lower_node(node, feeds, initializers, request['model_kind'], weight_loader=loader)
        current = directory / str(index)
        metadata = write_ort_request(
            {
                'model': lowered,
                'standardized': True,
                'feeds': feeds,
                'outputs': list(node.output),
                'stage': request['stage'] + '-node-' + str(index),
            },
            current,
        )
        values.update(read_ort_result(current, await ort_run(json.dumps(metadata))))
        shutil.rmtree(current)
        del lowered, feeds, metadata
        for name in node.input:
            remaining[name] -= 1
            if remaining[name] == 0 and name not in requested:
                values.pop(name, None)
    return {name: values[name] for name in requested}


async def diagnose_ort(source_directory, model_path, inputs_manifest, output_directory, ort_run):
    """Independent per-node observations for the acceptance oracle's kernel audit."""
    source, manifest = validate_sources(source_directory, model_path, inputs_manifest)
    working = externalize(source, Path(output_directory).parent / 'working/model.onnx')
    model = onnx.load(working, load_external_data=False)
    model_kind = manifest.get('model_kind', 'standard')
    feeds, _ = load_inputs(model, manifest['inputs'], Path(source_directory), model_kind)
    for value in model.graph.input:
        if value.name in feeds:
            array = feeds[value.name]
            value.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    onnx.helper.np_dtype_to_tensor_dtype(array.dtype),
                    list(array.shape),
                )
            )
    names = list(
        dict.fromkeys(
            [name for node in model.graph.node[:8] for name in node.output if name]
            + [value.name for value in model.graph.output]
        )
    )
    request = {
        'model': observation_model(model, names, model_kind),
        'feeds': feeds,
        'outputs': names,
        'stage': 'diagnostic',
        'base_dir': str(working.parent),
        'model_kind': model_kind,
    }
    output = Path(output_directory)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        if model_kind in ('bert', 'vit'):
            result = await run_transformer_request(request, Path(temporary) / 'ort', ort_run)
        else:
            result = await run_standard_request(request, Path(temporary) / 'ort', ort_run)
    np.savez(output / 'observations.npz', **result)
    (output / 'nodes.json').write_text(
        json.dumps(
            [
                {'name': name, 'node_index': index, 'op_type': node.op_type}
                for index, node in enumerate(model.graph.node)
                for name in node.output
                if name in names
            ],
            indent=2,
        )
        + '\n'
    )


def compile_high_level(
    model_path,
    output_directory,
    deployment_mode,
    config_path='',
    split_manifest=None,
    on_stage=None,
    profile_memory=False,
):
    from training.run_compile import compile_model
    import time
    import tracemalloc

    stages = []
    current, started = None, None
    tracing = profile_memory and not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()

    def stage(name):
        nonlocal current, started
        now = time.perf_counter()
        if current is not None:
            record = dict(phase=current, durationMs=(now - started) * 1000)
            if profile_memory:
                record['pythonCurrentBytes'], record['pythonPeakBytes'] = tracemalloc.get_traced_memory()
            stages.append(record)
            if on_stage is not None:
                on_stage(json.dumps(record))
        current, started = name, now
        if profile_memory:
            tracemalloc.reset_peak()

    try:
        task = compile_model(
            model_path,
            output_directory,
            deployment_mode=deployment_mode,
            config_path=config_path,
            split_manifest=split_manifest,
            temperature=0.0,
            num_workers=1,
            random_seed=0,
            on_stage=stage,
            stream_weights=True,
        )
    finally:
        if tracing:
            tracemalloc.stop()
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('Browser ONNX compilation imported torch')
    import platform
    import scipy
    import h5py

    (task / 'browser_compile.json').write_text(
        json.dumps(
            {
                'execution': 'pyodide' if sys.platform == 'emscripten' else sys.platform,
                'python': platform.python_version(),
                'onnx': onnx.__version__,
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'h5py': h5py.__version__,
                'torch_modules': [],
                'temperature': 0.0,
                'random_seed': 0,
                'num_workers': 1,
                'deployment_mode': deployment_mode,
                'compile_config': json.loads(Path(config_path).read_text()) if config_path else {},
                'stages': stages,
            },
            indent=2,
        )
        + '\n'
    )
    return str(task)


def prepare_compile_source(source_directory, model_path, inputs_manifest, destination):
    source, manifest = validate_sources(source_directory, model_path, inputs_manifest)
    onnx.checker.check_model(str(source))
    model = onnx.load(source)
    feeds, _ = load_inputs(model, manifest['inputs'], Path(source_directory))
    for value in model.graph.input:
        if value.name in feeds:
            array = feeds[value.name]
            value.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    onnx.helper.np_dtype_to_tensor_dtype(array.dtype),
                    list(array.shape),
                )
            )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        destination,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destination.name + '.data',
        size_threshold=1024,
    )
    onnx.checker.check_model(str(destination))
    return str(destination)


def generate_low_level(task_directory, deployment_mode):
    from inference.interface.gen_mega_ag import generate_low_level as generate

    streaming = False
    contract_path = Path(task_directory, 'input_contract.json')
    if deployment_mode == 'server_provisioned_runner' and contract_path.exists():
        contract = json.loads(contract_path.read_text())
        split = json.loads(Path(contract['manifest']).read_text())
        streaming = split['split']['prefix_count'] == 0
    return str(
        generate(
            task_directory,
            deployment_mode=deployment_mode,
            parameter_loading_mode='runtime_lazy' if deployment_mode == 'server_provisioned_runner' else None,
            streaming=streaming,
            random_seed=0,
        )
    )


def source_references(source_directory, model_path, inputs_manifest):
    """Inspect references without retaining source weights in Python globals."""
    source = relative_path(source_directory, model_path)
    model = model_metadata(source)
    references = []
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = onnx.external_data_helper.ExternalDataInfo(tensor).location
            external = relative_path(source.parent, location)
            references.append(str(external.relative_to(source_directory)))
    manifest = json.loads(Path(inputs_manifest).read_text())
    for entry in manifest['inputs']:
        relative_path(source_directory, entry['csv'])
        references.append(entry['csv'])
    return json.dumps(list(dict.fromkeys(references)))


def begin_split(
    state_directory,
    source_directory,
    model_path,
    inputs_manifest,
    output_directory,
    ratio,
):
    """Persist private job state before the owner replaces this Worker."""
    from onnx_split_contract import file_hash

    source, manifest = validate_sources(source_directory, model_path, inputs_manifest)
    state_directory = Path(state_directory)
    if not isinstance(ratio, str):
        raise ValueError('Split ratio must be a decimal string')
    working_model = Path(output_directory) / 'weights.onnx'
    if not working_model.exists():
        externalize(source, working_model)
    model = model_metadata(source)
    paths = [source]
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            name = onnx.external_data_helper.ExternalDataInfo(tensor).location
            path = relative_path(source.parent, name)
            if path not in paths:
                paths.append(path)
    state = dict(
        source_directory=str(source_directory),
        model_path=str(source),
        inputs_manifest=str(inputs_manifest),
        output_directory=str(output_directory),
        ratio=ratio,
        working_model=str(working_model),
        source_files=[dict(path=str(path), sha256=file_hash(path)) for path in paths],
        model_kind=manifest.get('model_kind', 'standard'),
    )
    (state_directory / 'step-state.json').write_text(json.dumps(state))


async def split_step(state_directory, ort_run):
    """Replay metadata and persisted observations; execute exactly one missing phase."""
    import shutil

    state_directory = Path(state_directory)
    state = json.loads((state_directory / 'step-state.json').read_text())
    manifest = validate_inputs(state['source_directory'], state['inputs_manifest'])
    checkpoint = state_directory / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    steps = prepare_steps(
        state['model_path'],
        state['output_directory'],
        manifest['inputs'],
        state['ratio'],
        model_kind=state['model_kind'],
        boundary_hints=manifest.get('boundary_hints'),
        base_dir=state['source_directory'],
        working_model=state['working_model'],
        checkpoints=checkpoint,
        source_files=state['source_files'],
    )
    result = None
    try:
        while True:
            try:
                request = steps.send(result)
            except StopIteration as completed:
                shutil.rmtree(checkpoint)
                (state_directory / 'step-state.json').unlink()
                return json.dumps({'complete': True})
            stage = request['stage']
            marker = checkpoint / (stage + '.json')
            if marker.exists():
                if request.get('checkpoint'):
                    result = None
                else:
                    with np.load(checkpoint / (stage + '.npz'), allow_pickle=False) as archive:
                        result = {name: archive[name] for name in archive.files}
                continue
            if not request.get('checkpoint'):
                directory = checkpoint / 'ort'
                result = await (
                    run_transformer_request(request, directory, ort_run)
                    if request['model_kind'] in ('bert', 'vit')
                    else run_standard_request(request, directory, ort_run)
                )
                np.savez(checkpoint / (stage + '.npz'), **result)
                shutil.rmtree(directory)
            marker.write_text(json.dumps({'phase': stage}))
            return json.dumps({'complete': False, 'phase': stage})
    finally:
        steps.close()


async def run_standard_request(request, directory, ort_run):
    """Execute standard nodes with only their live inputs and required weights."""
    from collections import Counter
    import shutil
    from onnx import helper, numpy_helper

    directory = Path(directory)
    directory.mkdir(parents=True)
    model = expand_polynomials(copy.deepcopy(request['model']))
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    requested = request['outputs'] or [value.name for value in model.graph.output]
    values = dict(request['feeds'])
    remaining = Counter(name for node in model.graph.node for name in node.input)
    for index, node in enumerate(model.graph.node):
        if node.op_type == 'Constant':
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            if 'value' in attrs:
                values[node.output[0]] = _read_weight(request, attrs['value'])
                continue
        feeds, embedded, array = {}, [], None
        controls = {
            'Reshape': (1,),
            'Resize': (1, 2, 3),
            'Expand': (1,),
            'Tile': (1,),
            'Slice': (1, 2, 3, 4),
            'Pad': (1, 2, 3),
            'Split': (1,),
            'Squeeze': (1,),
            'Unsqueeze': (1,),
            'ReduceSum': (1,),
            'ReduceMean': (1,),
        }
        control_names = {node.input[i] for i in controls.get(node.op_type, ()) if i < len(node.input)}
        for name in dict.fromkeys(node.input):
            if not name:
                continue
            if name in initializers:
                array = _read_weight(request, initializers[name])
                if array.nbytes <= 1024:
                    embedded.append(numpy_helper.from_array(array, name))
                else:
                    feeds[name] = array
            else:
                array = values[name]
                if array.nbytes <= 1024 and (array.dtype.kind in 'iub' or name in control_names):
                    embedded.append(numpy_helper.from_array(array, name))
                else:
                    feeds[name] = array
        inputs = [
            helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
            for name, value in feeds.items()
        ]
        graph = helper.make_graph(
            [node],
            'on-demand-node',
            inputs,
            [helper.make_empty_tensor_value_info(name) for name in node.output],
            embedded,
        )
        lowered = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
        lowered = onnx.shape_inference.infer_shapes(lowered)
        current = directory / str(index)
        metadata = write_ort_request(
            dict(
                model=lowered,
                feeds=feeds,
                outputs=list(node.output),
                standardized=True,
                stage=request['stage'] + '-node-' + str(index),
            ),
            current,
        )
        values.update(read_ort_result(current, await ort_run(json.dumps(metadata))))
        shutil.rmtree(current)
        del feeds, embedded, inputs, graph, lowered, metadata, array
        for name in node.input:
            remaining[name] -= 1
            if remaining[name] == 0 and name not in requested:
                values.pop(name, None)
    for name in requested:
        if name not in values and name in initializers:
            values[name] = _read_weight(request, initializers[name])
    return {name: values[name] for name in requested}
