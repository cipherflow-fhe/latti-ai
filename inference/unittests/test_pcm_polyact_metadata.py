# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest

from frontend import custom_task as task
from inference.model_generator.layers.par_block_col_major_polyactrn import ParBlockColMajorPolyActRNPoly


class PCMPolyActMetadataTest(unittest.TestCase):
    def tearDown(self):
        task.g_dag.clear()
        task.g_evk_node_dict.clear()

    def test_degree_two_and_four_coefficients_match_native_encoder_scales(self):
        for degree in (2, 4):
            with self.subTest(degree=degree):
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                param = task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant)
                task.set_fhe_param(param)
                level, scale = 6, param.default_scale
                q = param.q
                expected = [scale, q[level], q[level] / scale * q[level - 1],
                            q[level] / scale * q[level] / scale * q[level - 2],
                            q[level] / scale * q[level] / scale * q[level - 1] / scale * q[level - 2]]
                layer = ParBlockColMajorPolyActRNPoly((4, 4), 8, 1, (1 << param.log_max_slots()), degree)
                source = task.CustomDataNode(type='poly_data_source', id='source')
                outputs = layer.call_custom_compute([task.CkksCiphertextNode(level=level, id='x')], source)
                self.assertEqual([node.metadata.level for node in outputs], [level - (2 if degree == 2 else 3)])
                encoders = [node for node in task.g_dag.nodes if getattr(node, 'type', None) == 'encode_pt']
                self.assertEqual({node.attributes['coeff_idx'] for node in encoders}, set(range(degree + 1)))
                for encoder in encoders:
                    output, = task.g_dag.successors(encoder)
                    self.assertEqual(output.metadata.level, 0)
                    self.assertTrue(output.metadata.is_ringt)
                    self.assertAlmostEqual(output.metadata.scale / expected[encoder.attributes['coeff_idx']], 1.0)

    def test_coefficient_ids_are_distinct_across_layers(self):
        param = task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant)
        task.set_fhe_param(param)
        layer = ParBlockColMajorPolyActRNPoly((4, 4), 8, 1, 1 << param.log_max_slots(), 2)
        x = task.CkksCiphertextNode(level=6, id='x')
        for name in ('first', 'second'):
            layer.call_custom_compute([x], task.CustomDataNode(type='poly_data_source', id=name))
        ids = [node.id for node in task.g_dag.nodes if isinstance(node, task.DataNode)]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
