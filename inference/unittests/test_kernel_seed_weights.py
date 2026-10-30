# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from inference.lattisense.frontend import custom_task as task
from inference.model_generator.layers import kernel_seed_packing as seed
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import InverseMultiplexedConv2DLayer
from inference.model_generator.layers.inverse_multiplexed_depthwise_conv2d_layer import InverseMultiplexedDepthwiseConv2DLayer
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import MultiplexedConv2DPackedLayerDepthwise


def flatten(items):
    if isinstance(items, list):
        return [node for item in items for node in flatten(item)]
    return [] if items is None else [items]


class KernelSeedWeightsTest(unittest.TestCase):
    def tearDown(self):
        task.g_dag.clear()
        task.g_swk_node_dict.clear()

    @staticmethod
    def layers(kernel, subgrid):
        for skip in ([1, 1], [1, 2], [2, 1], [2, 2]):
            for stride in ([1, 1], [2, 2]):
                yield MultiplexedConv2DPackedLayerDepthwise(
                    8, 8, [8, 8], kernel, stride, skip, 8, 1, 1,
                    seed_weight_subgrid_stride=subgrid)
        for cls, channels in ((InverseMultiplexedConv2DLayer, (3, 2)),
                              (InverseMultiplexedDepthwiseConv2DLayer, (3,))):
            for shape, stride in (([16, 16], [2, 2]), ([8, 8], [4, 4])):
                yield cls(*channels, shape, [-1, -1], kernel, stride, [4, 4], subgrid)

    def test_kernel_expansion_matches_legacy_boundary_and_preserves_blocks(self):
        for kernel in ([1, 1], [3, 3], [1, 5]):
            for subgrid in ([2, 2], [4, 2], [1, 4]):
                for layer in self.layers(kernel, subgrid):
                    with self.subTest(layer=type(layer).__name__, kernel=kernel,
                                      stride=subgrid, skip=layer.seed_skip):
                        h, w = layer.seed_shape
                        sh, sw = layer.seed_skip
                        blocks = layer.seed_blocks
                        a, b = subgrid
                        slots = 2048
                        # A distinct coefficient in every channel and skip position.
                        values = np.arange(1, blocks * sh * sw + 1).reshape(blocks, 1, sh, 1, sw)
                        periodic = np.zeros(slots)
                        periodic[:blocks * h * sh * w * sw] = np.broadcast_to(
                            values, (blocks, h, sh, w, sw)).reshape(-1)
                        masks, row = [], [np.zeros(slots) for _ in range(layer.seed_weight_kernel_count)]
                        for k, step in enumerate(layer.seed_kernel_steps):
                            r, c = divmod(k % (a * b), b)
                            select = np.zeros((blocks, h, sh, w, sw))
                            select[:, r::a, :, c::b, :] = 1
                            boundary = np.zeros((blocks, h, sh, w, sw))
                            for i in range(h):
                                for j in range(w):
                                    if 0 <= i + step[0] < h and 0 <= j + step[1] < w:
                                        boundary[:, i, :, j, :] = 1
                            select = np.pad(select.reshape(-1), (0, slots - select.size))
                            boundary = np.pad(boundary.reshape(-1), (0, slots - boundary.size))
                            masks.append([select, boundary])
                            row[k // (a * b)] += periodic * (k + 1) * select
                        with ExitStack() as stack:
                            for name, fn in {'mult': lambda x, y: x * y, 'rescale': lambda x: x,
                                             'add': lambda x, y: x + y,
                                             'rotate_cols': lambda x, steps: [np.roll(x, -s) for s in steps]}.items():
                                stack.enter_context(patch.object(seed, name, fn))
                            for k in range(layer.kernel_weight_count):
                                actual = layer.kernel_weight(row, k, masks)
                                np.testing.assert_array_equal(actual, periodic * (k + 1) * masks[k][1])

    def test_levels_counts_and_single_lazy_load_per_seed(self):
        for kernel in ([1, 1], [3, 3], [1, 5]):
            for layer in self.layers(kernel, [2, 2]):
                for level in (2, 4):
                    task.g_dag.clear()
                    task.g_swk_node_dict.clear()
                    task.set_fhe_param(task.CkksParam.create_default_param(16384))
                    weights, biases, masks = layer.make_param_ct_nodes('seed', level)
                    params = flatten(weights) + flatten(biases)
                    if isinstance(layer, MultiplexedConv2DPackedLayerDepthwise):
                        params += flatten(masks)
                        inputs = [task.CkksCiphertextNode('x', level=level)]
                    else:
                        inputs = [task.CkksCiphertextNode(f'x{i}', level=level)
                                  for i in range(layer.n_in_channel * math.prod(layer.input_shape) // math.prod(layer.block_shape))]
                    store = task.CustomDataNode(type='encrypted_parameter_store', id='store')
                    for i, node in enumerate(params):
                        node.encrypted_parameter_store_node = store
                        node.encrypted_parameter_arg_id = 'params'
                        node.encrypted_parameter_flat_index = i
                    seed_masks = layer.make_seed_mask_pt_nodes('seed', level)
                    with self.subTest(layer=type(layer).__name__, kernel=kernel, level=level):
                        if isinstance(layer, MultiplexedConv2DPackedLayerDepthwise):
                            output = layer.call_param_ct(inputs, weights, biases, masks, seed_mask_pt=seed_masks)
                        else:
                            output = layer.call_param_ct(inputs, weights, biases, 2 * math.prod(layer.block_shape),
                                                         repack_mask_pt=masks, seed_mask_pt=seed_masks)
                        repack = (layer.stride != [1, 1] if isinstance(layer, MultiplexedConv2DPackedLayerDepthwise)
                                  else layer.need_repack)
                        expected_level = level - 1 - int(repack)
                        self.assertTrue(output)
                        self.assertTrue(all(isinstance(x, task.CkksCiphertextNode) for x in output))
                        self.assertTrue(all(x.level == expected_level for x in output))
                        actual = {}
                        for node in params:
                            actual[node.level] = actual.get(node.level, 0) + 1
                            producers = list(task.g_dag.predecessors(node))
                            self.assertEqual(len(producers), 1)
                            self.assertEqual(producers[0].type, 'load_encrypted_param_ct')
                            self.assertEqual(producers[0].attributes['expected_level'], node.level)
                            self.assertEqual(len(producers[0].wait_inputs), 1)
                        expected = {lv: sum(counts.values())
                                    for lv, counts in layer.get_encrypted_param_count(level).items()}
                        self.assertEqual(actual, expected)
                        self.assertEqual(flatten(weights)[0].level, level + layer.seed_weight_level_cost)

    def test_level_zero_recovery_and_invalid_layout_fallback(self):
        task.set_fhe_param(task.CkksParam.create_default_param(16384))
        for kernel in ([1, 1], [3, 3]):
            for layer in self.layers(kernel, [2, 2]):
                weights, _, _ = layer.make_param_ct_nodes('seed', 0)
                masks = layer.make_seed_mask_pt_nodes('seed', 0)
                row = weights[0][0] if isinstance(layer, InverseMultiplexedConv2DLayer) else weights[0]
                self.assertEqual(layer.kernel_weight(row, 0, masks).level, 0)
        for stride in ([1, 1], [3, 2], [16, 2], [0, 2], [-2, 2]):
            for layer in self.layers([3, 3], stride):
                self.assertFalse(layer.use_seed_weight_pack)
                self.assertEqual(layer.seed_weight_level_cost, 0)
                self.assertEqual(layer.seed_weight_kernel_count, layer.kernel_weight_count)
                self.assertEqual(layer.make_seed_mask_pt_nodes('legacy', 4), [])


if __name__ == '__main__':
    unittest.main()
