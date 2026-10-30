# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference.model_generator.deploy_cmds import gen_custom_task
from inference.model_generator.layers import inverse_multiplexed_conv2d_layer as inverse
from inference.model_generator.layers import inverse_multiplexed_depthwise_conv2d_layer as inverse_dw
from inference.unittests.test_server_provisioned_runner_generator import _data_with_missing_source


class MobileNetEncryptedParamsTest(unittest.TestCase):
    def test_relu6_onnx_exports_polynomial_coefficients(self):
        import h5py
        import onnx
        import torch
        from onnx import helper, numpy_helper
        from training.nn_tools.activations import _eval_hermite
        from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_module
        from training.nn_tools.export import export_h5_from_onnx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            running_max = np.array([3.0, 6.0], dtype=np.float32)
            node = helper.make_node('RangeNormPoly2d', ['input', 'act.rangenorm.running_max'], ['output'],
                                    activation='relu6', degree=4, upper_bound=3.0, eps=0.001)
            graph = helper.make_graph([node], 'relu6', [], [],
                                      [numpy_helper.from_array(running_max, 'act.rangenorm.running_max')])
            onnx.save(helper.make_model(graph), root / 'model.onnx')
            (root / 'layers.json').write_text(json.dumps({'layer': {
                'act': {'type': 'polyact', 'weight_path': 'act.weight'}
            }}))
            export_h5_from_onnx(str(root / 'model.onnx'), str(root / 'layers.json'),
                                str(root / 'parameters.h5'), verbose=False)
            with h5py.File(root / 'parameters.h5') as h5:
                coefficients = h5['act.weight'][:].reshape(5, 2)
            x = torch.tensor([-2.0, 1.5], dtype=torch.float64)
            expected = (running_max / 3.0 + 0.001) * _eval_hermite(
                x, get_hermite_coeffs_for_module(torch.nn.ReLU6, 4)).numpy()
            actual = sum(coefficients[i] * x.numpy() ** i for i in range(5))
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_inverse_plaintext_rotation_matches_ciphertext_convolution(self):
        rng = np.random.default_rng(17)
        for module in (inverse, inverse_dw):
            for shape, stride, kernel in (([8, 8], [2, 2], [3, 3]),
                                          ([8, 8], [4, 4], [1, 1]),
                                          ([8, 8], [4, 4], [3, 3])):
                with self.subTest(layer=module.op_class, stride=stride, kernel=kernel), ExitStack() as stack:
                    if module is inverse:
                        layer = module.InverseMultiplexedConv2DLayer(3, 2, shape, [-1, -1], kernel, stride, [4, 4])
                    else:
                        layer = module.InverseMultiplexedDepthwiseConv2DLayer(3, shape, [-1, -1], kernel, stride, [4, 4])
                    x = [rng.normal(size=32) for _ in range(layer.n_in_channel * 4)]
                    inner = np.prod(kernel) * np.prod(layer.output_step)
                    weights = ([[rng.normal(size=(inner, 32)) for _ in range(2)] for _ in range(3)]
                               if module is inverse else [rng.normal(size=(inner, 32)) for _ in range(3)])
                    bias = [rng.normal(size=32) for _ in range(3)]
                    mask = rng.integers(0, 2, size=32)
                    for name, fn in {
                        'rotate_cols': lambda x, steps: [np.roll(x, -s) for s in steps],
                        'mult': lambda x, y: x * y,
                        'add': lambda x, y: x + y,
                        'add_encrypted_param_bias': lambda x, y: x + y,
                        'sub': lambda x, y: x - y,
                        'rescale': lambda x: x,
                        'drop_level': lambda x, _: x,
                        'materialize_encrypted_param': lambda w, _: w,
                        'multiply_with_encrypted_param': lambda x, w: x * w,
                        'accumulate_encrypted_param_terms': lambda xs, ws: sum(x * w for x, w in zip(xs, ws)),
                    }.items():
                        stack.enter_context(patch.object(module, name, fn))
                    expected = layer.call_param_ct(x, weights, bias, 64, mask)
                    actual = layer.call_param_ct(x, weights, bias, 64, mask, input_is_plaintext=True)
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_eager_and_lazy_graphs_cover_inverse_depthwise_and_partial_seed_group(self):
        cases = (
            {'shape': [16, 16], 'stride': [2, 2], 'kernel_shape': [3, 3], 'ci': 1, 'co': 2,
             'groups': 1, 'pack': 1, 'is_big_size': True, 'seed_weight_subgrid_stride': [4, 4]},
            {'shape': [16, 16], 'stride': [2, 2], 'kernel_shape': [3, 3], 'ci': 3, 'co': 3,
             'groups': 3, 'pack': 1, 'is_big_size': True, 'seed_weight_subgrid_stride': [4, 4]},
            {'shape': [4, 4], 'stride': [2, 2], 'kernel_shape': [3, 3], 'ci': 3, 'co': 3,
             'groups': 3, 'pack': 2, 'is_big_size': False, 'seed_weight_subgrid_stride': [4, 4]},
            {'shape': [4, 4], 'stride': [1, 1], 'kernel_shape': [1, 1], 'ci': 6, 'co': 2,
             'groups': 1, 'pack': 4, 'is_big_size': False, 'seed_weight_subgrid_stride': [4, 4]},
        )
        for case in cases:
            for mode in ('eager', 'runtime_lazy'):
                with self.subTest(case=case, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    feat = {'dim': 2, 'channel': case['ci'], 'shape': case['shape'], 'skip': [1, 1],
                            'pack_num': case['pack'], 'level': 4}
                    out = dict(feat, channel=case['co'], level=2,
                               shape=[v // s for v, s in zip(case['shape'], case['stride'])])
                    layer = dict(case, type='conv2d', channel_input=case['ci'], channel_output=case['co'],
                                 feature_input=['input'], feature_output=['output'])
                    graph = {'feature': {'input': feat, 'output': out}, 'layer': {'conv': layer},
                             'input_feature': ['input'], 'output_feature': ['output']}
                    (root / 'nn_layers_ct_0.json').write_text(json.dumps(graph))
                    (root / 'task_config.json').write_text(json.dumps({'block_shape': [8, 8]}))
                    gen_custom_task(str(root), param_name='PN13QP218', style='multiplexed',
                                    parameter_mode='encrypted_offline', input_mode='plaintext',
                                    deployment_mode='server_provisioned_runner', parameter_loading_mode=mode)
                    mega = json.loads((root / 'mega_ag.json').read_text())
                    sig = json.loads((root / 'task_signature.json').read_text())
                    self.assertEqual(_data_with_missing_source(mega), [])
                    params = sig.get('encrypted_parameters', sig['offline'])
                    self.assertTrue(all(p['type'] == 'ct' for p in params))
                    if case['groups'] > 1 and not case['is_big_size']:
                        self.assertIn('convm_conv', {p['id'] for p in params})
                    loads = [c for c in mega['compute'].values() if c['type'] == 'load_encrypted_param_ct']
                    self.assertEqual(bool(loads), mode == 'runtime_lazy')
                    for load in loads:
                        self.assertEqual(len(load['wait_inputs']), 1)


if __name__ == '__main__':
    unittest.main()
