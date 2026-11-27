"""Independent-prefix boundaries, weight isolation and reference numerics."""

import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

repo = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(repo), str(repo / 'tools')]
from prepare_split_onnx import prepare
from inference.interface.wasm.python import browser_pipeline as pipeline

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)


def fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)

    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 512])

    prefix_weight = np.ones((1, 512), np.float32)
    suffix_weight = np.full((1, 512), 117.125, np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node('Add', ['input', 'prefix_weight'], ['crossing']),
                helper.make_node('Mul', ['crossing', 'suffix_weight'], ['output']),
            ],
            'independent-prefix',
            [value('input')],
            [value(n) for n in ('crossing', 'output', 'input', 'constant')],
            [
                numpy_helper.from_array(prefix_weight, 'prefix_weight'),
                numpy_helper.from_array(suffix_weight, 'suffix_weight'),
                numpy_helper.from_array(np.full((1, 512), 3.5, np.float32), 'constant'),
            ],
        ),
        opset_imports=[helper.make_opsetid('', 17)],
        ir_version=8,
    )
    onnx.save_model(
        model,
        directory / 'model.onnx',
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location='weights.data',
        size_threshold=0,
    )
    np.savetxt(directory / 'input.csv', np.arange(512, dtype=np.float32), delimiter=',')
    entry = dict(name='input', csv='input.csv', shape=[1, 512], axes=['N', 'F'])
    (directory / 'inputs.json').write_text(json.dumps(dict(inputs=[entry])))
    return entry


async def ort_run(text):
    request = json.loads(text)
    session = ort.InferenceSession(request['model'], providers=['CPUExecutionProvider'])
    feeds = {e['name']: np.fromfile(e['file'], dtype=e['dtype']).reshape(e['shape']) for e in request['feeds']}
    names = request['outputs'] or [v.name for v in session.get_outputs()]
    outputs = []
    for index, (name, array) in enumerate(zip(names, session.run(names, feeds))):
        file = f'output-{index}.bin'
        array.tofile(Path(request['directory']) / file)
        outputs.append(dict(name=name, file=file, shape=list(array.shape), dtype=str(array.dtype)))
    return json.dumps(dict(outputs=outputs))


async def check():
    source = root / 'source'
    entry = fixtures(source)
    split = root / 'split'
    prepare(source / 'model.onnx', split, [entry], '0.5', base_dir=source)
    zero = root / 'zero'
    prepare(source / 'model.onnx', zero, [entry], '0', base_dir=source)
    try:
        pipeline.export_prefix(zero / 'split_manifest.json', root / 'no-prefix')
    except ValueError as error:
        assert 'no prefix' in str(error)
    else:
        raise AssertionError('A ratio=0 split exported a prefix')
    public = root / 'prefix'
    pipeline.export_prefix(split / 'split_manifest.json', public)
    assert {p.name for p in public.iterdir()} == {'model.onnx', 'compact.onnx.data', 'prefix.json'}
    spec = json.loads((public / 'prefix.json').read_text())
    model = onnx.load(public / 'model.onnx')
    assert 'suffix_weight' not in {v.name for v in model.graph.initializer}
    assert np.float32(117.125).tobytes() * 32 not in (public / 'compact.onnx.data').read_bytes()
    assert str(root) not in (public / 'prefix.json').read_text()
    assert all('csv' not in entry for entry in spec['inputs'])
    # Make every original and suffix path unavailable before independent inference.
    source.rename(root / 'private-source')
    split.rename(root / 'private-split')
    sample = root / 'sample'
    sample.mkdir()
    np.savetxt(sample / 'input.csv', np.arange(512, dtype=np.float32) + 0.25, delimiter=',')
    (sample / 'inputs.json').write_text(json.dumps(dict(inputs=[entry])))
    result = root / 'result'
    await pipeline.run_exported_prefix(public / 'prefix.json', sample, sample / 'inputs.json', result, ort_run)
    manifest = json.loads((result / 'sample_manifest.json').read_text())
    actual = np.loadtxt(result / manifest['inputs'][0]['csv'], delimiter=',')
    np.testing.assert_array_equal(actual, np.arange(512, dtype=np.float32) + 1.25)
    binding = json.loads((result / 'run_manifest.json').read_text())
    assert binding['weightLoading'] == 'on_demand'
    assert binding['weightReads']['largestTensorBytes'] == 2048
    assert all('plaintext' not in r for r in manifest['output_routes'] if r['route'] == 'suffix')
    for route in manifest['output_routes']:
        if route['route'] == 'suffix':
            continue
        actual = np.loadtxt(result / route['plaintext']['csv'], delimiter=',')
        expected = (
            np.full(512, 3.5)
            if route['name'] == 'constant'
            else np.arange(512) + (1.25 if route['name'] == 'crossing' else 0.25)
        )
        np.testing.assert_array_equal(actual, expected)
    batch = dict(samples=[dict(case_id=name, inputs=[entry]) for name in ('case-a', 'case-b')])
    (sample / 'batch.json').write_text(json.dumps(batch))
    batch_result = root / 'batch-result'
    await pipeline.run_exported_prefix(public / 'prefix.json', sample, sample / 'batch.json', batch_result, ort_run)
    batch_binding = json.loads((batch_result / 'run_manifest.json').read_text())
    assert batch_binding['count'] == 2
    for record in batch_binding['samples']:
        sample_file = batch_result / record['manifest']
        record_manifest = json.loads(sample_file.read_text())
        actual = np.loadtxt(sample_file.parent / record_manifest['inputs'][0]['csv'], delimiter=',')
        np.testing.assert_array_equal(actual, np.arange(512, dtype=np.float32) + 1.25)
    batch['samples'][1]['case_id'] = '../escape'
    (sample / 'unsafe.json').write_text(json.dumps(batch))
    try:
        await pipeline.run_exported_prefix(
            public / 'prefix.json', sample, sample / 'unsafe.json', root / 'unsafe', ort_run
        )
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe case ID accepted')
    bad = dict(inputs=[{**entry, 'axes': ['C', 'N']}])
    (sample / 'bad.json').write_text(json.dumps(bad))
    try:
        await pipeline.run_exported_prefix(
            public / 'prefix.json', sample, sample / 'bad.json', root / 'invalid', ort_run
        )
    except ValueError as error:
        assert 'shape/axes' in str(error)
    else:
        raise AssertionError('Mismatched independent input axes accepted')
    # A static full-model output may already be a Constant node used by the
    # prefix. Export must keep that definition, without adding a duplicate.
    constants = root / 'constant-source'
    constants.mkdir()
    reused = onnx.load(root / 'private-source/model.onnx')
    kept = [tensor for tensor in reused.graph.initializer if tensor.name != 'constant']
    del reused.graph.initializer[:]
    reused.graph.initializer.extend(kept)
    reused.graph.node[0].input[1] = 'constant'
    reused.graph.node.insert(
        0,
        helper.make_node(
            'Constant', [], ['constant'], value=numpy_helper.from_array(np.full((1, 512), 3.5, np.float32))
        ),
    )
    onnx.save_model(reused, constants / 'model.onnx')
    np.savetxt(constants / 'input.csv', np.arange(512, dtype=np.float32), delimiter=',')
    prepare(constants / 'model.onnx', root / 'constant-split', [entry], '0.5', base_dir=constants)
    pipeline.export_prefix(root / 'constant-split/split_manifest.json', root / 'constant-prefix')
    onnx.checker.check_model(str(root / 'constant-prefix/model.onnx'))
    print(json.dumps(dict(status='passed', weightReads=binding['weightReads'], routes=len(manifest['output_routes']))))


asyncio.run(check())
