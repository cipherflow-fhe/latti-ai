# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Standard ONNX lowering against the independent float64 transformer reference."""

from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from evaluate_transformer_onnx_csv import execute
from expand_transformer_onnx import lower_node


class TransformerLoweringTest(unittest.TestCase):
    def compare(self, node, feeds, weights, kind):
        tensors = [numpy_helper.from_array(value, name) for name, value in weights.items()]
        model = helper.make_model(
            helper.make_graph(
                [node],
                'generic',
                [
                    helper.make_tensor_value_info(name, TensorProto.DOUBLE, list(value.shape))
                    for name, value in feeds.items()
                ],
                [helper.make_tensor_value_info('output', TensorProto.DOUBLE, None)],
                tensors,
            ),
            ir_version=8,
        )
        expected = execute(model, feeds, model_kind=kind)['output']
        lowered, actual_feeds = lower_node(node, feeds, {tensor.name: tensor for tensor in tensors}, kind)
        onnx.checker.check_model(lowered)
        session = ort.InferenceSession(lowered.SerializeToString(), providers=['CPUExecutionProvider'])
        actual = session.run(['output'], actual_feeds)[0]
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_linear_normalization_and_custom_activation(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(1, 3, 4)) * 0.2
        for kind, shape in [('bert', (2, 4)), ('vit', (4, 2))]:
            self.compare(
                helper.make_node('Linear', ['input', 'weight', 'bias'], ['output']),
                {'input': x},
                {'weight': rng.normal(size=shape), 'bias': rng.normal(size=(2,))},
                kind,
            )
        self.compare(
            helper.make_node(
                'CustomLayerNorm', ['input', 'weight', 'bias'], ['output'], eps=1e-5, inv_var_scale=0.0625, num_iters=2
            ),
            {'input': x},
            {'weight': np.arange(4.0) / 4, 'bias': np.arange(4.0) / 8},
            'vit',
        )
        attrs = dict(
            scale=2.0,
            f2_input_scale=1.2,
            f3_input_scale=0.8,
            p1_coefficients=[0.0, 0.5, 0.0, 0.01],
            p2_coefficients=[0.0, 1.0, 0.0, 1e-5],
            p3_coefficients=[0.0, 0.8, 0.0, 1e-5],
        )
        self.compare(helper.make_node('CustomGELU', ['input'], ['output'], **attrs), {'input': x}, {}, 'bert')

    def test_attention_preserves_masks_polynomial_and_projection_order(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(1, 3, 4)) * 0.1
        attrs = dict(
            num_heads=2,
            head_dim=2,
            scaling=2**-0.5,
            range_min=-4.0,
            range_max=4.0,
            exp_divisor=8.0,
            exp_coefficients=[0.5, 0.1, 0.01],
            delta_1=2,
            delta_2=2,
            inverse_epsilon=0.1,
            inverse_alpha=0.001,
            final_inverse_alpha=0.000001,
            max_inverse_iterations=5,
            use_asor=1,
        )
        attrs['attention_mask'] = numpy_helper.from_array(np.ones(3, dtype=np.int64))
        weights = {
            name: rng.normal(size=shape) * 0.1
            for name, shape in [
                ('qw', (4, 4)),
                ('qb', (4,)),
                ('kw', (4, 4)),
                ('kb', (4,)),
                ('vw', (4, 4)),
                ('vb', (4,)),
                ('ow', (4, 4)),
                ('ob', (4,)),
            ]
        }
        self.compare(
            helper.make_node('CustomMultiHeadAttention', ['input', *weights], ['output'], **attrs),
            {'input': x},
            weights,
            'bert',
        )
        weights = {
            'qkv': rng.normal(size=(4, 12)) * 0.1,
            'qkvb': np.zeros(12),
            'projection': rng.normal(size=(4, 4)) * 0.1,
            'bias': np.zeros(4),
            'running_max': np.ones((2, 3)) * 2,
        }
        weights.update(
            {name: np.asarray(value) for name, value in zip(['c0', 'c1', 'c2', 'c4'], [0.5, 0.2, 0.03, -0.002])}
        )
        self.compare(
            helper.make_node(
                'CustomMultiHeadAttention',
                ['input', *weights],
                ['output'],
                num_heads=2,
                head_dim=2,
                upper_bound=3.0,
                eps=1e-3,
            ),
            {'input': x},
            weights,
            'vit',
        )

    def test_unknown_operator_fails_before_runtime_allocation(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported transformer'):
            lower_node(helper.make_node('Unsupported', ['input'], ['output']), {'input': np.ones((1, 2))}, {}, 'bert')


if __name__ == '__main__':
    unittest.main()
