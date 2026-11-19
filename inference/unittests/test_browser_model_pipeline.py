# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared browser preparation/compilation regressions; no browser acceptance claims."""

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from inference.interface.wasm.python.browser_pipeline import prepare_split, relative_path, validate_sources, run_prefix
from training.nn_tools.eval_fn_hat_for_aespa import activation_function, get_hermite_coeffs_for_activation

ROOT = Path(__file__).resolve().parents[2]


class BrowserModelPipelineTest(unittest.TestCase):
    def test_async_executor_preserves_split_checks_and_named_external_data(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node('Add', ['input', 'weight'], ['crossing']),
                        helper.make_node('Mul', ['crossing', 'crossing'], ['output']),
                    ],
                    'fixture',
                    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])],
                    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])],
                    [numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), 'weight')],
                ),
                opset_imports=[helper.make_opsetid('', 17)],
                ir_version=8,
            )
            onnx.save_model(
                model, root / 'source.onnx', save_as_external_data=True, location='weights.data', size_threshold=0
            )
            (root / 'input.csv').write_text('1,2,3,4\n')
            (root / 'inputs.json').write_text(
                json.dumps(
                    {
                        'inputs': [
                            {'name': 'input', 'csv': 'input.csv', 'shape': [1, 4], 'axes': ['N', 'F']},
                        ]
                    }
                )
            )
            stages = []

            async def executor(text):
                # File boundary mirrors ORT Web; this regression uses a native oracle.
                await asyncio.sleep(0)
                request = json.loads(text)
                stages.append(request['stage'])
                session = ort.InferenceSession(request['model'], providers=['CPUExecutionProvider'])
                feeds = {
                    entry['name']: np.fromfile(entry['file'], dtype=entry['dtype']).reshape(entry['shape'])
                    for entry in request['feeds']
                }
                names = request['outputs'] or [entry.name for entry in session.get_outputs()]
                outputs = []
                for index, (name, array) in enumerate(zip(names, session.run(names, feeds))):
                    file = f'output-{index}.bin'
                    array.tofile(Path(request['directory']) / file)
                    outputs.append({'name': name, 'file': file, 'shape': list(array.shape), 'dtype': str(array.dtype)})
                return json.dumps({'outputs': outputs})

            manifest = json.loads(
                asyncio.run(
                    prepare_split(
                        root,
                        'source.onnx',
                        root / 'inputs.json',
                        root / 'split',
                        '0.5',
                        executor,
                    )
                )
            )
            self.assertEqual(stages, ['observe', 'prefix', 'suffix'])
            self.assertEqual(manifest['validation']['actual_node_counts'], {'prefix': 1, 'suffix': 1})
            self.assertEqual(manifest['inputs'][0]['name'], 'crossing')
            with np.load(root / 'split/original_full_onnx_outputs.npz') as reference:
                np.testing.assert_array_equal(reference['output'], [[4, 9, 16, 25]])
            self.assertEqual(len(manifest['source']['files']), 2)
            committed = (root / 'split/split_manifest.json').read_bytes()
            samples = root / 'samples'
            samples.mkdir()
            (samples / 'input.csv').write_text('2,3,4,5\n')
            binding = json.loads(
                asyncio.run(
                    run_prefix(
                        root / 'split/split_manifest.json',
                        samples,
                        root / 'inputs.json',
                        root / 'new-run',
                        executor,
                    )
                )
            )
            self.assertEqual((root / 'split/split_manifest.json').read_bytes(), committed)
            self.assertEqual(binding['sample_manifest'], 'split_manifest.json')
            with np.load(root / 'new-run/original_full_onnx_outputs.npz') as reference:
                np.testing.assert_array_equal(reference['output'], [[9, 16, 25, 36]])

    def test_source_paths_rejected_before_external_data_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('../secret', '/absolute', 'a/../b', './a', 'a//b', 'C:a', 'a\\b', 'a\0b'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    relative_path(root, name)
            tensor = numpy_helper.from_array(np.ones((4,), dtype=np.float32), 'weight')
            onnx.external_data_helper.set_external_data(tensor, location='../private.data')
            tensor.ClearField('raw_data')
            tensor.data_location = TensorProto.EXTERNAL
            model = helper.make_model(helper.make_graph([], 'invalid', [], [], [tensor]))
            (root / 'model.onnx').write_bytes(model.SerializeToString())
            (root / 'inputs.json').write_text('{"inputs": []}')
            with self.assertRaisesRegex(ValueError, 'safe relative path'):
                validate_sources(root, 'model.onnx', root / 'inputs.json')

    def test_import_and_onnx_export_do_not_require_torch(self):
        code = """
import sys
class BlockTraining:
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch', 'matplotlib'):
            raise ImportError('forbidden: ' + name)
sys.meta_path.insert(0, BlockTraining())
from training.nn_tools import export_h5_from_onnx
from training.run_compile import compile_model
from inference.interface.gen_mega_ag import generate_low_level
from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_activation
for name in ('relu', 'relu6', 'silu', 'gelu'):
    assert len(get_hermite_coeffs_for_activation(name, 4)) == 5
assert not any(n.split('.')[0] == 'torch' for n in sys.modules)
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True, timeout=60)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'training oracle requires torch')
    def test_float64_activation_and_hermite_match_training(self):
        import torch
        from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_module

        x = np.linspace(-20, 20, 4001, dtype=np.float64)
        for name, module in [
            ('relu', torch.nn.ReLU),
            ('relu6', torch.nn.ReLU6),
            ('silu', torch.nn.SiLU),
            ('gelu', torch.nn.GELU),
        ]:
            with self.subTest(activation=name):
                np.testing.assert_allclose(
                    activation_function(name)(x), module()(torch.from_numpy(x)).numpy(), atol=1e-12, rtol=1e-12
                )
                for degree in (0, 2, 4, 6):
                    np.testing.assert_allclose(
                        get_hermite_coeffs_for_activation(name, degree),
                        get_hermite_coeffs_for_module(module, degree),
                        atol=1e-12,
                        rtol=1e-12,
                    )
        np.testing.assert_allclose(
            activation_function('gelu', approximate='tanh')(x),
            torch.nn.GELU(approximate='tanh')(torch.from_numpy(x)).numpy(),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            get_hermite_coeffs_for_activation('gelu', 4, approximate='tanh'),
            get_hermite_coeffs_for_module(lambda: torch.nn.GELU(approximate='tanh'), 4),
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == '__main__':
    unittest.main()
