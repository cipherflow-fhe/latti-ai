# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference.lattisense.frontend import custom_task as task
from inference.model_generator.layers import dense_packed_layer as dense


class DenseSeedTileTest(unittest.TestCase):
    def tearDown(self):
        task.g_dag.clear()
        task.g_swk_node_dict.clear()

    def test_restore_distinct_spatial_values_blocks_bias_and_tail(self):
        for shape, skip, fill in [([2, 2], [4, 4], [2, 2]), ([1, 1], [16, 16], [4, 4]),
                                   ([2, 1], [8, 4], [4, 2])]:
            slots = 2 * math.prod(a * b for a, b in zip(shape, skip))
            layer = dense.DensePackedLayer(5, 3 * math.prod(shape), shape, skip, 2, 1, 1, invalid_fill=fill)
            layout = layer.choose_seed_tile_layout(slots * 2, 3, lambda level: (level + 1) * 100)
            self.assertIsNotNone(layout)
            block = slots // 2
            width = shape[1] * skip[1]
            index = np.arange(slots)
            h, w = (index % block) // width, index % width
            for kind in ('weight', 'bias'):
                support = ((h == 0) & (w == 0) if kind == 'bias' else
                           ((h % skip[0] < skip[0] // fill[0]) & (w % skip[1] < skip[1] // fill[1])))
                capacity = layout['group_size']
                # Include a partial seed, with different values in both physical blocks.
                originals = [np.where(support, np.arange(slots) + 1000 * (u + 1), 0.0)
                             for u in range(capacity + 1)]
                seeds = [SimpleNamespace(id=f'seed_{q}', values=np.zeros(slots))
                         for q in range(math.ceil(len(originals) / capacity))]
                masks = []
                for sub in range(capacity):
                    r, c = divmod(sub, fill[1])
                    delta = r * (skip[0] // fill[0]) * width + c * (skip[1] // fill[1])
                    masks.append(np.roll(support.astype(float), delta))
                    for u in range(sub, len(originals), capacity):
                        seeds[u // capacity].values += np.roll(originals[u], delta)

                def encode(inputs, output, type, attributes):
                    self.assertEqual(type, 'encode_pt')
                    self.assertEqual(inputs, [])
                    self.assertEqual(attributes['level'], 3 + (kind == 'weight'))
                    output.values = masks[attributes['subgrid']]

                with ExitStack() as stack:
                    for name, fn in {
                        'CkksPlaintextRingtNode': lambda name: SimpleNamespace(id=name),
                        'custom_compute': encode, 'materialize_encrypted_param': lambda x, _: x,
                        'mult': lambda x, y: x.values * y.values, 'rescale': lambda x: x,
                        'rotate_cols': lambda x, steps: [np.roll(x, -s) for s in steps],
                    }.items():
                        stack.enter_context(patch.object(dense, name, fn))
                    for u, expected in enumerate(originals):
                        actual = layer.tile_parameter(seeds if kind == 'bias' else [seeds], u, kind, None)
                        np.testing.assert_array_equal(actual, expected)
                # Every unused tile in the final seed is zero.
                for mask in masks[1:]:
                    np.testing.assert_array_equal(seeds[-1].values * mask, np.zeros(slots))

    def test_graph_levels_and_one_lazy_load_per_canonical_seed(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        layer = dense.DensePackedLayer(5, 1025, [1, 1], [64, 64], 2, 1, 1, invalid_fill=[4, 4])
        layout = layer.choose_seed_tile_layout(16384, 3, lambda lv: (lv + 1) * 100)
        weights, biases = layer.make_param_ct_nodes_multiplexed('tiles', 16384, 3)
        params = weights[0] + biases
        self.assertEqual(len(weights[0]), 2)
        self.assertEqual(len(biases), 1)
        self.assertEqual(Counter(p.level for p in params), {4: 2, 3: 1})
        self.assertEqual(layer.get_encrypted_param_count_multiplexed(16384, 3),
                         {4: {'weight': 2}, 3: {'bias': 1}})
        store = task.CustomDataNode(type='encrypted_parameter_store', id='store')
        for i, param in enumerate(params):
            param.encrypted_parameter_store_node = store
            param.encrypted_parameter_arg_id = 'parameters'
            param.encrypted_parameter_flat_index = i
        outputs = layer.call_param_ct_multiplexed(
            [task.CkksCiphertextNode(f'x_{i}', level=3) for i in range(3)], weights, biases, 16384)
        self.assertEqual([o.level for o in outputs], [2, 2, 2])
        loads = [node for node in task.g_dag.nodes if getattr(node, 'type', None) == 'load_encrypted_param_ct']
        self.assertEqual(len(loads), len(params))
        layout['slots'] //= 2
        with self.assertRaisesRegex(ValueError, 'metadata'):
            layer.make_param_ct_nodes_multiplexed('tiles', 16384, 3)

    def test_geometry_and_negative_byte_benefit_disable_tiles(self):
        for shape, skip, fill in [([1, 1], [4, 4], [1, 1]), ([1, 1], [4, 4], [3, 2]),
                                   ([128, 128], [4, 4], [2, 2])]:
            layer = dense.DensePackedLayer(1, 1, shape, skip, 1, 1, 1, invalid_fill=fill)
            self.assertIsNone(layer.choose_seed_tile_layout(16384, 3, lambda lv: (lv + 1) * 100))

    def test_concat_0d_uses_tiles_without_folding_invalid_slots(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        slots, inputs, outputs, level = 8192, 3, 5, 1
        layer = dense.DensePackedLayer(outputs, inputs, [1, 1], [1, slots], 1, inputs, outputs,
                                       invalid_fill=[1, slots])
        layout = layer.choose_seed_tile_layout(2 * slots, level, lambda lv: (lv + 1) * 100)
        self.assertEqual(layout['weight_shape'], [outputs, inputs])
        weights, biases = layer.make_param_ct_nodes_skip_0d('concat', level)
        self.assertEqual([len(weights), len(weights[0]), len(biases)], [1, 1, 1])
        self.assertEqual(layer.get_encrypted_param_count_skip_0d(level), {2: {'weight': 1}, 1: {'bias': 1}})
        for plaintext in (False, True):
            task.g_dag.clear()
            node_type = task.CkksPlaintextNode if plaintext else task.CkksCiphertextNode
            x = [node_type(f'x{i}', level=level) for i in range(inputs)]
            with patch.object(dense, 'rotate_cols', wraps=dense.rotate_cols) as rotate:
                result = layer.call_param_ct_skip_0d(x, weights, biases, slots, input_is_plaintext=plaintext)
            self.assertEqual([node.level for node in result], [0] * outputs)
            # Only restoration rotates; no spatial fold can broadcast a 0D result into invalid slots.
            self.assertEqual([call.args[1][0] for call in rotate.call_args_list],
                             [step for out in range(outputs)
                              for step in [*(out * inputs + i for i in range(inputs)), out] if step])
        layer = dense.DensePackedLayer(1, 1, [1, 1], [64, 64], 2, 1, 1, invalid_fill=[4, 4])
        self.assertIsNone(layer.choose_seed_tile_layout(16384, 3, lambda lv: 100 ** lv))


if __name__ == '__main__':
    unittest.main()
