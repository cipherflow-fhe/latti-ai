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

    manifest = json.loads(Path(inputs_manifest).read_text())
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


async def run_prefix(split_manifest, inputs_directory, inputs_manifest, output_directory, ort_run):
    from onnx_split_contract import load_manifest

    original = load_manifest(split_manifest)
    inputs = validate_inputs(inputs_directory, inputs_manifest)
    steps = prepare_steps(
        original['source']['path'],
        output_directory,
        inputs['inputs'],
        original['split']['ratio_text'],
        model_kind=original.get('model_kind', 'standard'),
        boundary_hints=inputs.get('boundary_hints'),
        base_dir=inputs_directory,
    )
    result = json.loads(await _drive_split(steps, Path(output_directory).parent, ort_run))
    _bind_run(split_manifest, output_directory, result)
    return (Path(output_directory) / 'run_manifest.json').read_text()


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
        loader = (lambda tensor: tensor_array(tensor, request['base_dir'])) if request.get('base_dir') else None
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
        original_split=None,
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
                if state['original_split']:
                    _bind_run(state['original_split'], state['output_directory'], completed.value)
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


def _bind_run(split_manifest, output_directory, result):
    from onnx_split_contract import file_hash, load_manifest

    original = load_manifest(split_manifest)
    if result['split'] != original['split']:
        raise ValueError('New sample changed the compiled node partition')
    fields = ('name', 'route', 'shape', 'dtype')
    if [[r[k] for k in fields] for r in result['output_routes']] != [
        [r[k] for k in fields] for r in original['output_routes']
    ]:
        raise ValueError('New sample changed the compiled output routes')
    fields = ('name', 'shape', 'dtype', 'axes', 'representation', 'producer', 'consumers')
    for old, new in zip(original['inputs'], result['inputs'], strict=True):
        if any(old.get(key) != new.get(key) for key in fields):
            raise ValueError('New sample changed a crossing interface')
    binding = dict(
        split_manifest=str(split_manifest),
        split_manifest_sha256=file_hash(split_manifest),
        sample_manifest='split_manifest.json',
        sample_manifest_sha256=file_hash(Path(output_directory) / 'split_manifest.json'),
    )
    (Path(output_directory) / 'run_manifest.json').write_text(json.dumps(binding, indent=2) + '\n')


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
                values[node.output[0]] = tensor_array(attrs['value'], request['base_dir'])
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
                array = tensor_array(initializers[name], request['base_dir'])
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
            values[name] = tensor_array(initializers[name], request['base_dir'])
    return {name: values[name] for name in requested}


def begin_run(split_manifest, inputs_directory, inputs_manifest, output_directory, state_directory):
    from onnx_split_contract import load_manifest

    original = load_manifest(split_manifest)
    validate_inputs(inputs_directory, inputs_manifest)
    source = Path(original['source']['path'])
    state_directory = Path(state_directory)
    state_directory.mkdir(parents=True, exist_ok=True)
    working = externalize(source, Path(output_directory) / 'weights.onnx')
    state = dict(
        source_directory=str(inputs_directory),
        model_path=str(source),
        inputs_manifest=str(inputs_manifest),
        output_directory=str(output_directory),
        ratio=original['split']['ratio_text'],
        working_model=str(working),
        source_files=original['source']['files'],
        original_split=str(split_manifest),
        model_kind=original.get('model_kind', 'standard'),
    )
    (state_directory / 'step-state.json').write_text(json.dumps(state))
