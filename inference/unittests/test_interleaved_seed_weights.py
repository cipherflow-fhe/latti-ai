# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from inference.lattisense.frontend import custom_task as task
from inference.model_generator.layers import multiplexed_conv2d_pack_layer as conv


class InterleavedSeedWeightsTest(unittest.TestCase):
    def tearDown(self):
        task.g_dag.clear()
        task.g_swk_node_dict.clear()

    def test_expansion_preserves_channel_blocks_and_skip_positions(self):
        # Different values in every block/skip position expose rotations that
        # accidentally copy a coefficient across a row or channel boundary.
        for skip in ([1, 1], [2, 2], [1, 2]):
            for subgrid_stride in ([2, 2], [4, 2], [2, 4], [1, 4], [4, 1]):
                with self.subTest(skip=skip, subgrid_stride=subgrid_stride):
                    layer = conv.MultiplexedConv2DPackedLayer(
                        8, 8, [8, 8], [1, 1], [1, 1], skip, 8, 1, 1,
                        seed_weight_subgrid_stride=subgrid_stride,
                    )
                    blocks = layer.n_block_per_ct
                    values = np.arange(1, blocks * np.prod(skip) + 1).reshape(blocks, 1, skip[0], 1, skip[1])
                    expected = np.broadcast_to(values, (blocks, 8, skip[0], 8, skip[1])).reshape(-1)
                    a, b = subgrid_stride
                    masks = []
                    seed = np.zeros_like(expected)
                    for subgrid in range(a * b):
                        mask = np.zeros((blocks, 8, skip[0], 8, skip[1]), dtype=int)
                        r, c = divmod(subgrid, b)
                        mask[:, r::a, :, c::b, :] = 1
                        mask = mask.reshape(-1)
                        masks.append([[mask, mask]])
                        seed += expected * (subgrid + 1) * mask
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(conv, '_multiply_mask_param', lambda x, y: x * y))
                        stack.enter_context(patch.object(conv, 'rescale', lambda x: x))
                        stack.enter_context(patch.object(conv, 'add', lambda x, y: x + y))
                        stack.enter_context(patch.object(
                            conv, 'rotate_cols', lambda x, steps: [np.roll(x, -step) for step in steps]
                        ))
                        for subgrid in range(a * b):
                            actual = layer._expand_seed_weight(seed, subgrid, 0, masks)
                            np.testing.assert_array_equal(actual, expected * (subgrid + 1))

    def test_encrypted_seed_is_loaded_once_for_all_subgrids(self):
        for kernel in ([1, 1], [3, 3], [1, 5]):
            for plaintext_input in (False, True):
                for level in (1, 4):
                    with self.subTest(kernel=kernel, plaintext_input=plaintext_input, level=level):
                        task.g_dag.clear()
                        task.g_swk_node_dict.clear()
                        task.set_fhe_param(task.CkksParam.create_default_param(16384))
                        layer = conv.MultiplexedConv2DPackedLayer(
                            4, 3, [4, 4], kernel, [1, 1], [1, 1], 4, 1, 1,
                            seed_weight_subgrid_stride=[2, 2],
                        )
                        weights, biases, masks = layer.make_param_ct_nodes('seed', level)
                        store = task.CustomDataNode(type='encrypted_parameter_store', id='store')
                        params = [w for group in weights for row in group for w in row] + biases
                        for i, param in enumerate(params):
                            param.encrypted_parameter_store_node = store
                            param.encrypted_parameter_arg_id = 'params'
                            param.encrypted_parameter_flat_index = i
                        x = (task.CkksPlaintextRingtNode('input') if plaintext_input
                             else task.CkksCiphertextNode('input', level=level))
                        seed_masks = layer.make_seed_mask_pt_nodes('seed', level)
                        outputs = layer.call_param_ct(
                            [x], weights, biases, masks, input_is_plaintext=plaintext_input,
                            seed_mask_pt=seed_masks,
                        )
                        self.assertEqual([out.level for out in outputs], [level - 1])
                        for param in params:
                            producers = list(task.g_dag.predecessors(param))
                            self.assertEqual(len(producers), 1)
                            producer = producers[0]
                            self.assertEqual(producer.type, 'load_encrypted_param_ct')
                            self.assertEqual(producer.attributes['expected_level'], param.level)
                            self.assertEqual(len(producer.wait_inputs), 1)
                        self.assertEqual(weights[0][0][0].level, level + layer.seed_weight_level_cost)

    def test_seed_expansion_restores_weight_level_zero(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        for kernel in ([1, 1], [3, 3], [1, 5]):
            with self.subTest(kernel=kernel):
                layer = conv.MultiplexedConv2DPackedLayer(
                    4, 3, [4, 4], kernel, [1, 1], [1, 1], 4, 1, 1,
                    seed_weight_subgrid_stride=[2, 2],
                )
                weights, _, _ = layer.make_param_ct_nodes('seed', 0)
                masks = layer.make_seed_mask_pt_nodes('seed', 0)
                expanded = layer._expand_seed_weight(weights[0][0][0], 0, 0, masks[0][0])
                self.assertEqual(expanded.level, 0)

    def test_seed_recovery_operations_use_public_masks_and_post_rescale_levels(self):
        for kernel in ([1, 1], [3, 3], [1, 5]):
            with self.subTest(kernel=kernel):
                args = (4, 3, [4, 4], kernel, [1, 1], [1, 1], 4, 1, 1)
                legacy = conv.MultiplexedConv2DPackedLayer(*args)
                seed = conv.MultiplexedConv2DPackedLayer(*args, seed_weight_subgrid_stride=[2, 2])
                baseline, actual = legacy.get_fhe_op_count(4), seed.get_fhe_op_count(4)
                recovery = {level: {op: value - baseline.get(level, {}).get(op, 0)
                                    for op, value in counts.items()}
                            for level, counts in actual.items()}
                terms = seed.weight_input_count * np.prod(kernel)
                seed_level = 4 + seed.seed_weight_level_cost
                self.assertEqual(recovery[seed_level]['mult_plain'], terms)
                self.assertEqual(recovery[seed_level]['rescale'], terms)
                self.assertEqual(recovery[seed_level - 1]['rotate'], 2 * terms)
                self.assertEqual(recovery[seed_level - 1]['add'], 2 * terms)
                self.assertTrue(all(counts['mult'] == 0 for counts in recovery.values()))
                if kernel != [1, 1]:
                    self.assertEqual(recovery[seed_level - 1]['mult_plain'], terms)
                    self.assertEqual(recovery[seed_level - 1]['rescale'], terms)


if __name__ == '__main__':
    unittest.main()
