# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""External weights and persisted browser stages against full native ONNX."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

from inference.interface.wasm.python import browser_pipeline as pipeline
from tools.onnx_external_weights import externalize, model_metadata, tensor_array


class ExternalWeightTest(unittest.TestCase):
    def fixture(self, root):
        weights = np.arange(512, dtype=np.float32).reshape(1, 512) / 512
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node('Add', ['input', 'weight'], ['crossing']),
                    helper.make_node('Mul', ['crossing', 'crossing'], ['output']),
                ],
                'fixture',
                [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 512])],
                [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 512])],
                [numpy_helper.from_array(weights, 'weight')],
            ),
            opset_imports=[helper.make_opsetid('', 17)],
            ir_version=8,
        )
        onnx.save(model, root / 'source.onnx')
        np.savetxt(root / 'input.csv', weights, delimiter=',')
        (root / 'inputs.json').write_text(
            json.dumps({'inputs': [dict(name='input', csv='input.csv', shape=[1, 512], axes=['N', 'F'])]})
        )
        return weights

    def test_large_inline_weights_are_read_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = np.arange(1024 * 1024, dtype=np.float32)
            tensor = numpy_helper.from_array(expected, 'weight')
            model = helper.make_model(
                helper.make_graph(
                    [],
                    'large',
                    [],
                    [helper.make_tensor_value_info('weight', TensorProto.FLOAT, [expected.size])],
                    [tensor],
                )
            )
            source = root / 'source.onnx'
            onnx.save(model, source)
            original_open = Path.open
            reads = []

            class Guard:
                def __init__(self, stream):
                    self.stream = stream

                def __getattr__(self, name):
                    return getattr(self.stream, name)

                def read(self, count=-1):
                    if not 0 <= count <= 1024 * 1024:
                        raise AssertionError('Unbounded weight read')
                    reads.append(count)
                    return self.stream.read(count)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.stream.close()

            def guarded(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                return Guard(stream) if path == source and args == ('rb',) else stream

            with patch.object(Path, 'open', guarded):
                externalize(source, root / 'external/model.onnx')
            loaded = onnx.load(root / 'external/model.onnx', load_external_data=False)
            np.testing.assert_array_equal(tensor_array(loaded.graph.initializer[0], root / 'external'), expected)
            self.assertEqual(max(reads), 1024 * 1024)

    def test_inline_and_existing_external_weights_preserve_values_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.fixture(root)
            original = (root / 'source.onnx').read_bytes()
            metadata = model_metadata(root / 'source.onnx')
            self.assertFalse(metadata.graph.initializer[0].HasField('raw_data'))
            for source, destination in [
                (root / 'source.onnx', root / 'external/model.onnx'),
                (root / 'external/model.onnx', root / 'again/model.onnx'),
            ]:
                externalize(source, destination)
                model = onnx.load(destination, load_external_data=False)
                tensor = model.graph.initializer[0]
                np.testing.assert_array_equal(tensor_array(tensor, destination.parent), expected)
                self.assertFalse(tensor.HasField('raw_data'))
                onnx.checker.check_model(str(destination))
                self.assertLess(destination.stat().st_size, 512)
            self.assertEqual(original, (root / 'source.onnx').read_bytes())

    def test_derived_shape_controls_are_available_to_node_shape_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = numpy_helper.from_array(np.asarray([1, 4], dtype=np.int64), 'shape')
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node('Identity', ['shape'], ['derived_shape']),
                        helper.make_node('Reshape', ['input', 'derived_shape'], ['output']),
                    ],
                    'reshape',
                    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2])],
                    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])],
                    [shape],
                ),
                opset_imports=[helper.make_opsetid('', 17)],
                ir_version=8,
            )

            async def execute(text):
                request = json.loads(text)
                session = ort.InferenceSession(request['model'])
                feeds = {
                    x['name']: np.fromfile(x['file'], dtype=x['dtype']).reshape(x['shape']) for x in request['feeds']
                }
                outputs = []
                for index, (name, value) in enumerate(zip(request['outputs'], session.run(request['outputs'], feeds))):
                    file = f'out-{index}.bin'
                    value.tofile(Path(request['directory']) / file)
                    outputs.append(dict(name=name, file=file, shape=value.shape, dtype=str(value.dtype)))
                return json.dumps(dict(outputs=outputs))

            result = asyncio.run(
                pipeline.run_standard_request(
                    dict(
                        model=model,
                        feeds={'input': np.arange(4, dtype=np.float32).reshape(2, 2)},
                        outputs=['output'],
                        base_dir=str(root),
                        stage='fixture',
                    ),
                    root / 'ort',
                    execute,
                )
            )
            np.testing.assert_array_equal(result['output'], [[0, 1, 2, 3]])

    def test_stages_recover_from_disk_and_match_full_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = self.fixture(root)
            expected = ort.InferenceSession(str(root / 'source.onnx')).run(None, {'input': weights})[0]
            pipeline.begin_split(root, root, 'source.onnx', root / 'inputs.json', root / 'split', '0.5')
            phases = []

            async def execute(text):
                request = json.loads(text)
                session = ort.InferenceSession(request['model'])
                feeds = {
                    x['name']: np.fromfile(x['file'], dtype=x['dtype']).reshape(x['shape']) for x in request['feeds']
                }
                outputs = []
                for index, (name, value) in enumerate(zip(request['outputs'], session.run(request['outputs'], feeds))):
                    file = f'out-{index}.bin'
                    value.tofile(Path(request['directory']) / file)
                    outputs.append(dict(name=name, file=file, shape=value.shape, dtype=str(value.dtype)))
                return json.dumps(dict(outputs=outputs))

            for _ in range(6):
                # No generator or runtime state crosses these calls.
                result = json.loads(asyncio.run(pipeline.split_step(root, execute)))
                if result['complete']:
                    break
                phases.append(result['phase'])
            self.assertEqual(phases, ['observe', 'extract', 'prefix', 'suffix'])
            with np.load(root / 'split/original_full_onnx_outputs.npz') as actual:
                np.testing.assert_array_equal(actual['output'], expected)
            self.assertFalse((root / 'step-state.json').exists())
            self.assertFalse((root / 'checkpoints').exists())
            for name in ['prefix', 'suffix']:
                model = onnx.load(root / 'split' / (name + '.onnx'), load_external_data=False)
                self.assertTrue(all(not tensor.HasField('raw_data') for tensor in model.graph.initializer))

    def test_truncated_and_escaping_external_ranges_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for content in [b'\x3a\x80', b'\x3a\x04\x01']:
                (root / 'invalid.onnx').write_bytes(content)
                with self.assertRaises(ValueError):
                    externalize(root / 'invalid.onnx', root / 'output.onnx')
            model = onnx.load(root / 'source.onnx')
            tensor = model.graph.initializer[0]
            onnx.external_data_helper.set_external_data(tensor, location='../secret', offset=0, length=2048)
            tensor.ClearField('raw_data')
            (root / 'invalid.onnx').write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(ValueError, 'safe relative'):
                externalize(root / 'invalid.onnx', root / 'output.onnx')

    def test_new_sample_keeps_original_transformer_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node('Linear', ['input', 'weight', 'bias'], ['crossing'], domain='custom_domain'),
                        helper.make_node('Add', ['crossing', 'bias'], ['output']),
                    ],
                    'transformer',
                    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 4])],
                    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 4])],
                    [
                        numpy_helper.from_array(np.eye(4, dtype=np.float32), 'weight'),
                        numpy_helper.from_array(np.ones(4, dtype=np.float32), 'bias'),
                    ],
                ),
                opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('custom_domain', 1)],
                ir_version=8,
            )
            onnx.save(model, root / 'source.onnx')
            (root / 'input.csv').write_text(','.join(['1'] * 8))
            inputs = dict(inputs=[dict(name='input', csv='input.csv', shape=[1, 2, 4], axes=['N', 'T', 'D'])])
            (root / 'inputs.json').write_text(json.dumps(dict(inputs, model_kind='vit')))

            async def execute(text):
                request = json.loads(text)
                session = ort.InferenceSession(request['model'])
                feeds = {
                    x['name']: np.fromfile(x['file'], dtype=x['dtype']).reshape(x['shape']) for x in request['feeds']
                }
                outputs = []
                for index, (name, value) in enumerate(zip(request['outputs'], session.run(request['outputs'], feeds))):
                    file = f'out-{index}.bin'
                    value.tofile(Path(request['directory']) / file)
                    outputs.append(dict(name=name, file=file, shape=value.shape, dtype=str(value.dtype)))
                return json.dumps(dict(outputs=outputs))

            pipeline.begin_split(root, root, 'source.onnx', root / 'inputs.json', root / 'split', '0.5')
            for _ in range(7):
                if json.loads(asyncio.run(pipeline.split_step(root, execute)))['complete']:
                    break
            samples = root / 'samples'
            samples.mkdir()
            (samples / 'input.csv').write_text(','.join(['2'] * 8))
            (samples / 'inputs.json').write_text(json.dumps(inputs))
            pipeline.begin_run(
                root / 'split/split_manifest.json',
                samples,
                samples / 'inputs.json',
                root / 'new-run',
                root / 'run-state',
            )
            for _ in range(7):
                if json.loads(asyncio.run(pipeline.split_step(root / 'run-state', execute)))['complete']:
                    break
            with np.load(root / 'new-run/original_full_onnx_outputs.npz') as values:
                np.testing.assert_array_equal(values['output'], np.full((1, 2, 4), 4.0))
            self.assertEqual(json.loads((root / 'new-run/split_manifest.json').read_text())['model_kind'], 'vit')
            self.assertTrue((root / 'new-run/run_manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
