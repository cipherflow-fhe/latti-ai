# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exercise split dependency closure, logical CSVs, and full-output acceptance."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from prepare_split_onnx import load_inputs, prepare, split_graph
from run_split_inference import compare
from onnx_split_contract import load_manifest, validate_signature_size


def value(name, shape=(1, 4)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def model(nodes, inputs, outputs, initializers=()):
    return helper.make_model(
        helper.make_graph(nodes, 'split_test', inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid('', 17)],
        ir_version=8,
    )


class OnnxSplitTest(unittest.TestCase):
    def test_signature_count_uses_matrix_heads_and_vector_packing(self):
        contract = {
            'mat_pack_style': 'par_block_col_major',
            'matmul_block_size': 64,
            'n_heads': 12,
            'block_shape': [64, 64],
            'ckks': {'param0': {'n_slots': 32768}},
        }
        matrix = {'data_type': 'feature_mat', 'head_shape': [44, 64], 'ckks_parameter_id': 'param0'}
        vector = {'dim': 0, 'channel': 512, 'pack_num': 256}
        for packing in (matrix, vector):
            validate_signature_size({'id': 'input', 'size': [2]}, packing, contract)
            with self.assertRaisesRegex(ValueError, 'count differs'):
                validate_signature_size({'id': 'input', 'size': [1]}, packing, contract)

    def test_residual_multiple_outputs_forwarded_input_static_and_external_data(self):
        constant = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), 'constant')
        source = model(
            [
                helper.make_node('Constant', [], ['c'], value=constant),
                helper.make_node('Add', ['x', 'c'], ['a']),
                helper.make_node('Split', ['a'], ['left', 'right'], axis=1),
                helper.make_node('Add', ['left', 'z'], ['b']),
                helper.make_node('Add', ['b', 'right'], ['out']),
            ],
            [value('x'), value('z', (1, 2))],
            [value('out', (1, 2)), value('a'), value('x'), value('w')],
            [numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), 'w')],
        )
        plan = split_graph(source, '0.5')
        self.assertEqual(plan['node_count'], 4)
        self.assertEqual(plan['crossing'], ['left', 'right'])
        self.assertEqual(plan['forwarded'], ['z'])
        self.assertEqual([r['route'] for r in plan['routes']], ['suffix', 'prefix', 'input', 'static'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx.save_model(
                source, root / 'source.onnx', save_as_external_data=True, location='weights.data', size_threshold=0
            )
            np.savetxt(root / 'x.csv', [[1, 2, 3, 4]], delimiter=',')
            np.savetxt(root / 'z.csv', [[10, 20]], delimiter=',')
            entries = [{'name': n, 'csv': n + '.csv', 'axes': ['N', 'F']} for n in ('x', 'z')]
            hints = {n: {'axes': ['N', 'F']} for n in ('left', 'right')}
            manifest = prepare(
                root / 'source.onnx', root / 'split', entries, '0.5', boundary_hints=hints, base_dir=root
            )
            self.assertEqual(len(manifest['source']['files']), 2)
            self.assertEqual(manifest['validation']['actual_node_counts'], {'prefix': 2, 'suffix': 2})
            self.assertEqual([e['name'] for e in manifest['inputs']], ['left', 'right', 'z'])
            np.testing.assert_array_equal(np.loadtxt(root / 'split/boundary_000.csv', delimiter=','), [2, 3])
            for name in ('prefix', 'suffix'):
                onnx.checker.check_model(str(root / f'split/{name}.onnx'))
            with np.load(root / 'split/original_full_onnx_outputs.npz') as reference:
                expected = reference['out'].reshape(-1)
            actual = root / 'decrypted.csv'

            def write_actual(offset):
                actual.write_text(
                    'output,index,value\n' + ''.join(f'out,{i},{v + offset}\n' for i, v in enumerate(expected))
                )

            contract = {
                'manifest': str(root / 'split/split_manifest.json'),
                'manifest_sha256': 'test',
                'suffix_sha256': 'test',
                'output_map': {'out': 'out'},
                'deployment_mode': 'client_encrypted_input',
                'ckks': {'param0': {'param_name': 'PN15Default'}},
            }
            write_actual(0)
            report = compare(contract, actual, root / 'comparison.json')
            self.assertEqual(list(report['outputs']), ['out', 'a', 'x', 'w'])
            self.assertTrue(report['pass'])
            write_actual(0.1)
            with self.assertRaisesRegex(ValueError, 'strictly below'):
                compare(contract, actual, root / 'comparison.json')
            actual.write_text('output,index,value\n')
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                compare(contract, actual, root / 'comparison.json')
            load_manifest(root / 'split/split_manifest.json')
            with (root / 'split/original_full_onnx_outputs.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                load_manifest(root / 'split/split_manifest.json')

    def test_transformer_baseline_reads_original_csv_as_float64(self):
        source = model(
            [helper.make_node('Identity', ['x'], ['a']), helper.make_node('Identity', ['a'], ['out'])],
            [value('x')],
            [value('out')],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx.save(source, root / 'source.onnx')
            decimals = np.array([[0.123456789012345, 0.333333333333333, 0.2, 0.7]])
            np.savetxt(root / 'x.csv', decimals, delimiter=',', fmt='%.17g')
            manifest = prepare(
                root / 'source.onnx',
                root / 'split',
                [{'name': 'x', 'csv': 'x.csv', 'axes': ['N', 'F']}],
                '0.5',
                model_kind='vit',
                base_dir=root,
            )
            with np.load(root / 'split/original_full_onnx_outputs.npz') as reference:
                np.testing.assert_array_equal(reference['out'], decimals)
            self.assertGreater(manifest['validation']['split_max_abs']['out'], 0)

    def test_stable_topology_parallel_ports_and_decimal_floor(self):
        source = model(
            [
                helper.make_node('Identity', ['x'], ['a']),
                helper.make_node('Identity', ['x'], ['b']),
                helper.make_node('Add', ['a', 'a'], ['aa']),
                helper.make_node('Add', ['aa', 'b'], ['out']),
            ],
            [value('x')],
            [value('out')],
        )
        plan = split_graph(source, '0.5')
        self.assertEqual(plan['topo'], [0, 1, 2, 3])
        self.assertEqual(plan['dag'].number_of_edges(0, 2), 2)
        self.assertEqual(plan['crossing'], ['a', 'b'])
        ten = model(
            [helper.make_node('Identity', ['x' if i == 0 else f'v{i - 1}'], [f'v{i}']) for i in range(10)],
            [value('x')],
            [value('v9')],
        )
        self.assertEqual(split_graph(ten, '0.9')['prefix_count'], 9)
        for ratio in ('0', '1', '-1', 'nan', 'Infinity', '0.01'):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                split_graph(ten, ratio)

    def test_input_shape_dtype_and_initializer_binding(self):
        source = model(
            [helper.make_node('Add', ['x', 'w'], ['out'])],
            [value('x'), value('w')],
            [value('out')],
            [numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), 'w')],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savetxt(root / 'x.csv', [[1, 2, 3, 4]], delimiter=',')
            entry = {'name': 'x', 'csv': 'x.csv', 'shape': [1, 4]}
            self.assertEqual(set(load_inputs(source, [entry], root)[0]), {'x'})
            for change in ({'shape': [2, 2]}, {'dtype': 'float64'}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'disagrees with ONNX'):
                    load_inputs(source, [{**entry, **change}], root)

    def test_ambiguous_axes_and_subgraphs_fail_before_encryption(self):
        source = model(
            [helper.make_node('Identity', ['x'], ['a']), helper.make_node('Identity', ['a'], ['out'])],
            [value('x', (1, 2, 3))],
            [value('out', (1, 2, 3))],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx.save(source, root / 'source.onnx')
            np.savetxt(root / 'x.csv', np.arange(6).reshape(1, -1), delimiter=',')
            with self.assertRaisesRegex(ValueError, 'unambiguous axes'):
                prepare(root / 'source.onnx', root / 'split', [{'name': 'x', 'csv': 'x.csv'}], '0.5', base_dir=root)
            self.assertFalse((root / 'split/split_manifest.json').exists())
        subgraph = helper.make_graph([], 'branch', [], [value('x')])
        source.graph.node[0].attribute.append(helper.make_attribute('body', subgraph))
        with self.assertRaisesRegex(ValueError, 'subgraph'):
            split_graph(source, '0.5')


if __name__ == '__main__':
    unittest.main()
