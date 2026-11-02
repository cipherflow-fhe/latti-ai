# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from inference.lattisense.frontend import custom_task as task
from inference.model_generator.layers.poly_relu2d import PolyRelu2D


class PolySeedTest(unittest.TestCase):
    def tearDown(self):
        task.g_dag.clear()
        task.g_swk_node_dict.clear()

    def test_degrees_keep_their_levels_and_independent_canonical_seeds(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        layer = PolyRelu2D([16, 16], 4, [2, 2], 32)
        layout = layer.choose_seed_layout([2, 2], 16384, 5, 5, lambda lv: (lv + 1) * 100)
        self.assertEqual(layout['use_site_levels'], [2, 3, 3, 3, 4])
        weights = layer.make_param_ct_nodes('poly', 5, 5)
        self.assertEqual([len(row) for row in weights], [2] * 5)
        self.assertEqual([row[0].level for row in weights], [3, 4, 4, 4, 5])
        self.assertEqual(layer.get_encrypted_param_count(5, 5),
                         {3: {'coefficient': 2}, 4: {'coefficient': 6}, 5: {'coefficient': 2}})
        store = task.CustomDataNode(type='encrypted_parameter_store', id='store')
        for degree, row in enumerate(weights):
            for i, node in enumerate(row):
                node.encrypted_parameter_store_node = store
                node.encrypted_parameter_arg_id = f'poly_reluw_poly_{degree}'
                node.encrypted_parameter_flat_index = i
        outputs = layer.call_bsgs_param_ct([task.CkksCiphertextNode(f'x{i}', level=5) for i in range(5)], weights)
        self.assertEqual([o.level for o in outputs], [2] * 5)
        loads = [n for n in task.g_dag.nodes if getattr(n, 'type', None) == 'load_encrypted_param_ct']
        self.assertEqual(len(loads), 10)
        masks = [n for n in task.g_dag.nodes if getattr(n, 'type', None) == 'encode_pt']
        self.assertEqual(len(masks), 25)
        self.assertEqual(Counter(n.level for row in weights for n in row), {3: 2, 4: 6, 5: 2})
        layout['use_site_levels'][0] += 1
        with self.assertRaisesRegex(ValueError, 'metadata'):
            layer.make_param_ct_nodes('poly', 5, 5)

    def test_incompatible_geometry_and_nonpositive_benefit_use_legacy(self):
        for shape, zero, expansion, stride, groups in [
            ([16, 16], [1, 1], [1, 1], [1, 1], 5),
            ([16, 16], [2, 1], [1, 1], [2, 2], 5),
            ([1, 16], [1, 1], [1, 1], [2, 2], 5),
            ([16, 16], [1, 1], [1, 1], [2, 2], 1),
        ]:
            layer = PolyRelu2D(shape, 4, [1, 1], 32, zero, expansion)
            self.assertIsNone(layer.choose_seed_layout(stride, 16384, groups, 5, lambda lv: (lv + 1) * 100))
        layer = PolyRelu2D([16, 16], 4, [1, 1], 32)
        self.assertIsNone(layer.choose_seed_layout([2, 2], 16384, 5, 5, lambda lv: 100 ** lv))

    def test_big_rectangular_blocks_use_physical_seed_geometry(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        layer = PolyRelu2D([128, 128], 8, [1, 1], 1, block_expansion=[2, 1])
        layout = layer.choose_seed_layout([16, 16], 16384, 6, 8, lambda lv: (lv + 1) * 100)
        self.assertEqual(layout['input_shape'], [64, 128])
        self.assertEqual(layout['blocks'], 1)
        self.assertEqual(layout['channel_groups'], 6)
        weights = layer.make_param_ct_nodes('big', 6, 8)
        self.assertEqual([len(row) for row in weights], [1] * 9)
        result = layer.call_bsgs_param_ct([task.CkksCiphertextNode(f'x{i}', level=8) for i in range(6)], weights)
        self.assertEqual([node.level for node in result], [4] * 6)
        masks = [n for n in task.g_dag.nodes if getattr(n, 'type', None) == 'encode_pt']
        self.assertEqual(len(masks), 54)
        for mask in masks:
            self.assertEqual(mask.attributes['seed_shape'], [64, 128])

    def test_degree_scales_keep_modulus_drift_at_equal_levels(self):
        layer = PolyRelu2D([16, 16], 4, [1, 1], 32)
        param = SimpleNamespace(scale=1024.0, q=[1093, 1151, 1223, 1301, 1423, 1543])
        s, q = param.scale, param.q
        # Independent degree-4 expansion: x^2=S^2/q5 and x^3=S^3/(q5*q4).
        expected = [s, q[3], q[3] * q[5] / s,
                    q[3] * q[5] * q[4] / s ** 2,
                    q[3] * q[5] * q[4] ** 2 / s ** 3]
        np.testing.assert_allclose(layer.bsgs_coefficient_scales(5, param), expected, rtol=1e-14)
        self.assertEqual(layer._bsgs_encrypted_coeff_levels(5)[1:4], [3, 3, 3])
        self.assertNotEqual(expected[1], expected[2])


if __name__ == '__main__':
    unittest.main()
