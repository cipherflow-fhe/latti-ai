# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from benchmark_server_provisioned_runner import check_structure, metrics, recovery_operations
from evaluate_onnx_csv import evaluate
from summarize_parameter_bundle import element_lengths


class ParameterAcceptanceToolsTest(unittest.TestCase):
    def test_explicit_polynomial_coefficients_survive_import_and_export(self):
        import h5py
        from training.model_export.onnx_to_json import onnx_to_json
        from training.nn_tools.export import export_h5_from_onnx

        x = np.linspace(-1.5, 1.5, 12, dtype=np.float32).reshape(1, 2, 2, 3)
        weights = np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
        bias = np.array([0.25, -0.5], dtype=np.float32)
        coefficients = {power: np.array(values, dtype=np.float32).reshape(1, 2, 1, 1)
                        for power, values in {0: [0.7, -0.2], 1: [1.1, 0.3],
                                              2: [0.4, 0.6], 4: [-0.02, 0.03]}.items()}
        parameters = {'conv.weight': weights, 'conv.bias': bias,
                      **{f'act.a{power}': value for power, value in coefficients.items()}}
        nodes = [helper.make_node('Conv', ['input', 'conv.weight', 'conv.bias'], ['conv_output'],
                                  name='conv', kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
                                  group=1, dilations=[1, 1]),
                 helper.make_node('Simple_Polyrelu', ['conv_output', *[f'act.a{p}' for p in coefficients]],
                                  ['output'], name='activation', domain='custom', degree=4)]
        graph = helper.make_graph(nodes, 'explicit_polynomial',
                                  [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, x.shape)],
                                  [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, x.shape)],
                                  [numpy_helper.from_array(value, name) for name, value in parameters.items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17),
                                                       helper.make_opsetid('custom', 1)], ir_version=8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx.save(model, root / 'model.onnx')
            np.savetxt(root / 'input.csv', x.reshape(-1), delimiter=',')
            outputs, _ = evaluate(root / 'model.onnx', root / 'input.csv', warmup=0, repeats=1)
            conv_output = x + bias.reshape(1, 2, 1, 1)
            expected = sum(value * conv_output ** power for power, value in coefficients.items())
            np.testing.assert_allclose(outputs['output'], expected.reshape(-1), atol=2e-6)
            onnx_to_json(str(root / 'model.onnx'), str(root / 'layers.json'), 'multiplexed')
            layers = json.loads((root / 'layers.json').read_text())
            self.assertEqual(layers['layer']['activation']['weight_path'], 'act.weight')
            self.assertEqual(layers['layer']['activation']['order'], 4)
            # The compiler may attach an absorption edge; explicit coefficients
            # must leave both convolution weights and biases unchanged.
            layers['layer']['conv'].update(absorb_path=['activation'], absorb_type=['polyact'])
            (root / 'layers.json').write_text(json.dumps(layers))
            export_h5_from_onnx(str(root / 'model.onnx'), str(root / 'layers.json'), str(root / 'params.h5'),
                                verbose=False)
            with h5py.File(root / 'params.h5') as params:
                np.testing.assert_array_equal(params['conv.weight'][:], weights.reshape(-1))
                np.testing.assert_array_equal(params['conv.bias'][:], bias)
                actual = params['act.weight'][:].reshape(5, 2)
                for power in range(5):
                    np.testing.assert_array_equal(actual[power], coefficients[power].reshape(-1)
                                                  if power in coefficients else np.zeros(2))

    def test_named_csv_inputs_follow_model_names_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, count])
                      for name, count in [('short', 2), ('long', 3)]]
            graph = helper.make_graph([helper.make_node('Concat', ['short', 'long'], ['output'], axis=1)],
                                      'multi_input', inputs,
                                      [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 5])])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8),
                      root / 'model.onnx')
            np.savetxt(root / 'short.csv', [2, 3], delimiter=',')
            np.savetxt(root / 'long.csv', [5, 7, 11], delimiter=',')
            paths = {'long': root / 'long.csv', 'short': root / 'short.csv'}
            outputs, report = evaluate(root / 'model.onnx', paths, warmup=0, repeats=1)
            np.testing.assert_array_equal(outputs['output'], [2, 3, 5, 7, 11])
            self.assertEqual(report['input_csv'], {k: str(v) for k, v in paths.items()})
            with self.assertRaisesRegex(ValueError, 'names'):
                evaluate(root / 'model.onnx', {'short': paths['short']}, warmup=0, repeats=1)

    def test_complete_logits_and_finite_values_are_required(self):
        reference = {('output', i): float(i + 1) for i in range(1000)}
        result = metrics(reference, dict(reference))
        self.assertEqual(result['max_abs_error'], 0)
        self.assertTrue(result['top1_match'])
        truncated = dict(reference)
        del truncated[('output', 999)]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            metrics(reference, truncated)
        invalid = dict(reference)
        invalid[('output', 999)] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            metrics(reference, invalid)
        changed = dict(reference)
        changed[('output', 999)] += 0.25
        self.assertEqual(metrics(reference, changed)['max_abs_error'], 0.25)

    def test_container_measurement_rejects_truncation_and_incorrect_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.bin'
            payload = struct.pack('<QQ', 2, 3) + b'abc' + struct.pack('<Q', 5) + b'12345'
            path.write_bytes(payload)
            argument = {'size': [1, 2]}
            self.assertEqual(element_lengths(path, argument), [3, 5])
            argument['elements'] = [{'offset': 16, 'length': 3}, {'offset': 27, 'length': 5}]
            self.assertEqual(element_lengths(path, argument), [3, 5])
            argument['elements'][1]['offset'] += 1
            with self.assertRaisesRegex(ValueError, 'index'):
                element_lengths(path, argument)
            del argument['elements']
            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                element_lengths(path, argument)

    def test_manifest_contract_and_parameter_only_recovery_count(self):
        layout = {'group_size': 4, 'weight': True, 'tile_grid': [2, 2]}
        argument = dict(id='densew_fc', size=[1, 1], level=3, type='ct', scale=1024.0,
                        layer_id='fc', param_kind='weight', layout=layout)
        graph = {'data': {'0': {'type': 'ct'}, '1': {'type': 'pt_ringt'}, '2': {'type': 'ct'},
                          '3': {'type': 'ct'}, '4': {'type': 'ct'}, '5': {'type': 'ct'}},
                 'compute': {
                     '0': dict(type='load_encrypted_param_ct', inputs=[], outputs=[0],
                               attributes=dict(arg_id='densew_fc', flat_index=0, expected_level=3)),
                     '1': dict(type='encode_pt', inputs=[], outputs=[1], attributes={'type': 'tile_mask_pt'}),
                     '2': dict(type='mult', inputs=[0, 1], outputs=[2]),
                     '3': dict(type='rescale', inputs=[2], outputs=[3]),
                     '4': dict(type='mult', inputs=[3, 4], outputs=[5]),
                 }}
        self.assertEqual(recovery_operations(graph, {'densew_fc': argument}),
                         {'encode_pt': 1, 'mult': 1, 'rescale': 1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / 'runner'
            params = runner / 'encrypted_model_parameters'
            params.mkdir(parents=True)
            fixtures = {
                'task_config.json': dict(deployment_mode='server_provisioned_runner', input_mode='plaintext',
                                         parameter_mode='encrypted_offline', parameter_loading_mode='runtime_lazy'),
                'task_signature.json': {'encrypted_parameters': [argument]},
                'nn_layers_ct_0.json': {'layer': {'fc': {'dense_seed_tiles': layout}}},
                'mega_ag.json': graph,
            }
            for name, value in fixtures.items():
                (runner / name).write_text(json.dumps(value))
            manifest = {'arguments': {'densew_fc': argument.copy()}}
            path = params / 'manifest.json'
            path.write_text(json.dumps(manifest))
            self.assertTrue(check_structure(root)['shape_level_scale_layout_match'])
            for field, wrong in [('scale', 2048.0), ('level', 2), ('size', [1, 2]),
                                 ('layout', dict(layout, group_size=8)), ('param_kind', 'mask')]:
                with self.subTest(field=field):
                    manifest['arguments']['densew_fc'] = dict(argument, **{field: wrong})
                    path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        check_structure(root)

    def test_plaintext_polynomial_expansion_matches_training_forward(self):
        from training.nn_tools.activations import RangeNormPoly2d
        from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_module

        x = np.linspace(-2, 2, 24, dtype=np.float32).reshape(1, 2, 3, 4)
        running_max = np.array([1.25, 2.75], dtype=np.float32).reshape(1, 2, 1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savetxt(root / 'input.csv', x.reshape(-1), delimiter=',')
            for degree in (2, 4, 6):
                layer = RangeNormPoly2d(get_hermite_coeffs_for_module(torch.nn.ReLU, degree), num_features=2)
                layer.eval()
                layer.rangenorm.running_max.copy_(torch.from_numpy(running_max))
                expected = layer(torch.from_numpy(x)).detach().numpy().reshape(-1)
                node = helper.make_node('RangeNormPoly2d', ['input', 'max'], ['output'],
                                        domain='nn_tools', degree=degree, activation='relu', upper_bound=3.0, eps=1e-3)
                graph = helper.make_graph([node], 'polynomial',
                    [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, x.shape)],
                    [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, x.shape)],
                    [numpy_helper.from_array(running_max, 'max')])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17),
                                                               helper.make_opsetid('nn_tools', 1)], ir_version=8)
                onnx.save(model, root / 'model.onnx')
                outputs, _ = evaluate(root / 'model.onnx', root / 'input.csv', warmup=0, repeats=1)
                np.testing.assert_allclose(outputs['output'], expected, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
