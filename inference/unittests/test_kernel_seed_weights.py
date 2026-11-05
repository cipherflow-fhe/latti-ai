# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from frontend import custom_task as task
from inference.model_generator.layers import kernel_seed_packing as seed
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import InverseMultiplexedConv2DLayer
from inference.model_generator.layers.inverse_multiplexed_depthwise_conv2d_layer import InverseMultiplexedDepthwiseConv2DLayer
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import MultiplexedConv2DPackedLayerDepthwise


def flatten(items):
    if isinstance(items, list):
        return [node for item in items for node in flatten(item)]
    return [] if items is None else [items]


class KernelSeedWeightsTest(unittest.TestCase):
    def setUp(self):
        task.set_fhe_param(task.CkksParam.create_default_param(14))

    def test_encrypted_inverse_repack_uses_full_real_slot_capacity(self):
        param = task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant)
        task.set_fhe_param(param)
        slots = 1 << param.log_max_slots()
        for cls, channels in ((InverseMultiplexedConv2DLayer, (2, 1)),
                              (InverseMultiplexedDepthwiseConv2DLayer, (2,))):
            with self.subTest(layer=cls.__name__):
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                layer = cls(*channels, [128, 256], [-1, -1], [3, 3], [2, 2], [128, 256])
                self.assertTrue(layer.need_repack)
                weights, biases, mask = layer.make_param_ct_nodes('repack', 3)
                x = [task.CkksCiphertextNode(level=3, id=f'x{i}') for i in range(layer.n_in_channel)]
                output = layer.call_param_ct(x, weights, biases, slots, repack_mask_pt=mask)
                self.assertEqual(len(output), 1)
                self.assertEqual(output[0].metadata.level, 1)

    def test_inverse_lazy_bias_ids_are_unique_for_all_output_blocks(self):
        for cls, channels in ((InverseMultiplexedConv2DLayer, (2, 1)),
                              (InverseMultiplexedDepthwiseConv2DLayer, (2,))):
            with self.subTest(layer=cls.__name__):
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                layer = cls(*channels, [16, 16], [-1, -1], [3, 3], [2, 2], [4, 4])
                count = layer.n_in_channel * math.prod(layer.stride) * math.prod(layer.output_step)
                x = [task.CkksCiphertextNode(level=3, id=f'x{i}') for i in range(count)]
                layer.call_custom_compute(x, task.CustomDataNode(type='conv_data_source', id='source'), 8192)
                ids = [node.id for node in task.g_dag.nodes if isinstance(node, task.DataNode)]
                self.assertEqual(len(ids), len(set(ids)))

    def test_selection_uses_bytes_at_seed_level_and_chain_availability(self):
        def layer():
            return MultiplexedConv2DPackedLayerDepthwise(
                5, 5, [8, 8], [3, 3], [1, 1], [1, 2], 2, 3, 3,
                seed_weight_subgrid_stride=[2, 2], seed_weight_layout='joint')

        packed = layer()
        choice = seed.choose_convolution_seed_layout(packed, 3, lambda lv: (lv + 1) * 100)
        self.assertTrue(packed.use_seed_weight_pack)
        self.assertEqual(choice['level_cost'], 2)
        self.assertLess(choice['candidate_bytes'], choice['legacy_bytes'])
        expensive = layer()
        choice = seed.choose_convolution_seed_layout(expensive, 3, lambda lv: 100 ** lv)
        self.assertFalse(expensive.use_seed_weight_pack)
        self.assertEqual(choice['reason'], 'no_byte_benefit')
        self.assertEqual(choice['stride'], [1, 1])

        def limited_chain(lv):
            if lv > 4:
                raise ValueError('modulus chain exhausted')
            return (lv + 1) * 100

        unavailable = layer()
        choice = seed.choose_convolution_seed_layout(unavailable, 3, limited_chain)
        self.assertFalse(unavailable.use_seed_weight_pack)
        self.assertEqual(choice['reason'], 'seed_level_unavailable')

    def tearDown(self):
        task.g_dag.clear()
        task.g_evk_node_dict.clear()

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
                    task.g_evk_node_dict.clear()
                    task.set_fhe_param(task.CkksParam.create_default_param(14))
                    weights, biases, masks = layer.make_param_ct_nodes('seed', level)
                    params = flatten(weights) + flatten(biases)
                    if isinstance(layer, MultiplexedConv2DPackedLayerDepthwise):
                        masks = layer.make_runner_mask_pt_nodes('seed', level)
                        inputs = [task.CkksCiphertextNode(id='x', level=level)]
                    else:
                        inputs = [task.CkksCiphertextNode(id=f'x{i}', level=level)
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
                        self.assertTrue(all(x.metadata.level == expected_level for x in output))
                        actual = {}
                        for node in params:
                            actual[node.metadata.level] = actual.get(node.metadata.level, 0) + 1
                            producers = list(task.g_dag.predecessors(node))
                            self.assertEqual(len(producers), 1)
                            self.assertEqual(producers[0].type, 'load_encrypted_param_ct')
                            self.assertEqual(producers[0].attributes['expected_level'], node.metadata.level)
                            self.assertEqual(len(list(task.g_dag.predecessors(producers[0]))), 2)
                        expected = {lv: sum(counts.values())
                                    for lv, counts in layer.get_encrypted_param_count(level).items()}
                        self.assertEqual(actual, expected)
                        self.assertEqual(flatten(weights)[0].metadata.level, level + layer.seed_weight_level_cost)

    def test_level_zero_recovery_and_invalid_layout_fallback(self):
        task.set_fhe_param(task.CkksParam.create_default_param(14))
        for kernel in ([1, 1], [3, 3]):
            for layer in self.layers(kernel, [2, 2]):
                weights, _, _ = layer.make_param_ct_nodes('seed', 0)
                masks = layer.make_seed_mask_pt_nodes('seed', 0)
                row = weights[0][0] if isinstance(layer, InverseMultiplexedConv2DLayer) else weights[0]
                self.assertEqual(layer.kernel_weight(row, 0, masks).metadata.level, 0)
        for stride in ([1, 1], [3, 2], [16, 2], [0, 2], [-2, 2]):
            for layer in self.layers([3, 3], stride):
                self.assertFalse(layer.use_seed_weight_pack)
                self.assertEqual(layer.seed_weight_level_cost, 0)
                self.assertEqual(layer.seed_weight_kernel_count, layer.kernel_weight_count)
                self.assertEqual(layer.make_seed_mask_pt_nodes('legacy', 4), [])

    def test_inverse_canonical_kernels_keep_execution_boundaries(self):
        for cls, channels in ((InverseMultiplexedConv2DLayer, (3, 2)),
                              (InverseMultiplexedDepthwiseConv2DLayer, (3,))):
            layer = cls(*channels, [16, 16], [-1, -1], [3, 3], [2, 2], [4, 4],
                        [2, 2], seed_weight_layout='canonical')
            self.assertEqual(layer.kernel_weight_count, 36)
            self.assertEqual(layer.seed_weight_kernel_count, 3)
            positions = list(seed.inverse_kernel_positions(layer.kernel_shape, layer.stride,
                                                           layer.output_step, layer.padding))
            slots, h, w = 32, 4, 4
            packed = [np.zeros(slots) for _ in range(3)]
            for canonical in range(9):
                select = np.zeros((h, w))
                r, c = divmod(canonical % 4, 2)
                select[r::2, c::2] = 1
                packed[canonical // 4][:h*w] += (canonical + 1) * select.reshape(-1)
            masks, expected = [], []
            for k, (kernel, step) in enumerate(positions):
                canonical = kernel[0] * 3 + kernel[1]
                self.assertEqual(layer.seed_request_index(k), canonical)
                select = np.zeros((h, w))
                r, c = divmod(canonical % 4, 2)
                select[r::2, c::2] = 1
                boundary = np.array([[0 <= i + step[0] < h and 0 <= j + step[1] < w
                                      for j in range(w)] for i in range(h)])
                masks.append([np.pad(select.reshape(-1), (0, slots-h*w)),
                              np.pad(boundary.reshape(-1), (0, slots-h*w))])
                expected.append((canonical + 1) * masks[-1][1])
            with ExitStack() as stack:
                for name, fn in {'mult': lambda x, y: x * y, 'rescale': lambda x: x,
                                 'add': lambda x, y: x + y,
                                 'rotate_cols': lambda x, steps: [np.roll(x, -s) for s in steps]}.items():
                    stack.enter_context(patch.object(seed, name, fn))
                for k in range(36):
                    np.testing.assert_array_equal(layer.kernel_weight(packed, k, masks), expected[k])
            task.set_fhe_param(task.CkksParam.create_default_param(14))
            graph_masks = layer.make_seed_mask_pt_nodes('inverse', 2)
            for k, pair in enumerate(graph_masks):
                select, = task.g_dag.predecessors(pair[0])
                boundary, = task.g_dag.predecessors(pair[1])
                self.assertEqual(select.attributes['subgrid'], layer.seed_request_index(k) % 4)
                self.assertEqual(boundary.attributes['seed_step'], positions[k][1])
            task.g_dag.clear()
            task.g_evk_node_dict.clear()

    def test_depthwise_public_mask_graph_and_legacy_signature(self):
        for plaintext_input in (False, True):
            for public_mask in (False, True):
                with self.subTest(plaintext_input=plaintext_input, public_mask=public_mask):
                    task.g_dag.clear()
                    task.g_evk_node_dict.clear()
                    task.set_fhe_param(task.CkksParam.create_default_param(14))
                    layer = MultiplexedConv2DPackedLayerDepthwise(
                        3, 3, [8, 8], [3, 3], [2, 2], [1, 2], 2, 2, 1)
                    weights, bias, legacy_masks = layer.make_param_ct_nodes('dw', 4)
                    masks = layer.make_runner_mask_pt_nodes('dw', 4) if public_mask else legacy_masks
                    self.assertTrue(all(isinstance(m, task.CkksCiphertextNode) for m in legacy_masks))
                    node_type = task.CkksPlaintextNode if plaintext_input else task.CkksCiphertextNode
                    inputs = [node_type(id=f'x{i}', **({} if plaintext_input else {'level': 4})) for i in range(2)]
                    outputs = layer.call_param_ct(inputs, weights, bias, masks, input_is_plaintext=plaintext_input)
                    self.assertEqual([o.metadata.level for o in outputs], [2])
                    for channel, mask in enumerate(masks):
                        consumers = list(task.g_dag.successors(mask))
                        self.assertEqual(len(consumers), 1)
                        self.assertEqual(consumers[0].type, task.OperationType.Mult)
                        product, = task.g_dag.successors(consumers[0])
                        self.assertEqual(product.metadata.degree, 1 if public_mask else 2)
                        if public_mask:
                            producer, = task.g_dag.predecessors(mask)
                            self.assertEqual(producer.type, 'encode_pt')
                            self.assertEqual(producer.attributes['level'], 3)
                            self.assertEqual((producer.attributes['ct_idx'], producer.attributes['local_idx']),
                                             divmod(channel, 2))

    def test_depthwise_joint_requests_cross_channel_groups_and_keep_tail_zero(self):
        for kernel in ([1, 1], [3, 3]):
            for groups in (2, 3, 5):
                for subgrid in ([2, 2], [4, 2]):
                    layer = MultiplexedConv2DPackedLayerDepthwise(
                        groups * 2 - 1, groups * 2 - 1, [8, 8], kernel, [1, 1], [1, 2],
                        2, groups, groups, seed_weight_subgrid_stride=subgrid, seed_weight_layout='joint')
                    count = groups * math.prod(kernel)
                    self.assertEqual(layer.weight_storage_shape, [1, math.ceil(count / math.prod(subgrid))])
                    slots, a, b = 128, *subgrid
                    weights = [[np.zeros(slots) for _ in range(layer.weight_storage_shape[1])]]
                    masks, expected = [], []
                    for request in range(count):
                        group, k = divmod(request, math.prod(kernel))
                        value = np.zeros((8, 8, 2))
                        value[:, :, 0] = (group + 1) * 100 + k
                        if group != groups - 1:
                            value[:, :, 1] = (group + 1) * 100 + k + 0.5
                        select = np.zeros_like(value)
                        row, col = divmod(request % (a * b), b)
                        select[row::a, col::b, :] = 1
                        step = layer.seed_kernel_steps[k]
                        boundary = np.zeros_like(value)
                        for h in range(8):
                            for w in range(8):
                                if 0 <= h + step[0] < 8 and 0 <= w + step[1] < 8:
                                    boundary[h, w, :] = 1
                        weights[0][request // (a * b)] += (value * select).reshape(-1)
                        masks.append([select.reshape(-1), boundary.reshape(-1)])
                        expected.append((value * boundary).reshape(-1))
                    with ExitStack() as stack:
                        for name, fn in {'mult': lambda x, y: x * y, 'rescale': lambda x: x,
                                         'add': lambda x, y: x + y,
                                         'rotate_cols': lambda x, steps: [np.roll(x, -s) for s in steps]}.items():
                            stack.enter_context(patch.object(seed, name, fn))
                        for request in range(count):
                            group, k = divmod(request, math.prod(kernel))
                            np.testing.assert_array_equal(layer.weight_for_request(weights, group, k, masks), expected[request])

    def test_depthwise_joint_graph_has_one_lazy_load_per_canonical_seed(self):
        for plaintext in (False, True):
            for kernel in ([1, 1], [3, 3]):
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                task.set_fhe_param(task.CkksParam.create_default_param(14))
                layer = MultiplexedConv2DPackedLayerDepthwise(
                    5, 5, [8, 8], kernel, [1, 1], [1, 2], 2, 3, 3,
                    seed_weight_subgrid_stride=[2, 2], seed_weight_layout='joint')
                weights, bias, _ = layer.make_param_ct_nodes('joint', 3)
                store = task.CustomDataNode(type='encrypted_parameter_store', id='store')
                for i, node in enumerate(flatten(weights) + bias):
                    node.encrypted_parameter_store_node = store
                    node.encrypted_parameter_arg_id = 'params'
                    node.encrypted_parameter_flat_index = i
                inputs = [(task.CkksPlaintextNode(f'x{i}') if plaintext else task.CkksCiphertextNode(id=f'x{i}', level=3))
                          for i in range(3)]
                masks = layer.make_seed_mask_pt_nodes('joint', 3)
                outputs = layer.call_param_ct(inputs, weights, bias, [], input_is_plaintext=plaintext, seed_mask_pt=masks)
                self.assertEqual([x.metadata.level for x in outputs], [2, 2, 2])
                self.assertEqual(len(flatten(weights)), math.ceil(3 * math.prod(kernel) / 4))
                for node in flatten(weights):
                    producer, = task.g_dag.predecessors(node)
                    self.assertEqual(producer.type, 'load_encrypted_param_ct')
                for request, pair in enumerate(masks):
                    producer, = task.g_dag.predecessors(pair[0])
                    self.assertEqual(producer.attributes['subgrid'], request % 4)
                    self.assertEqual(producer.attributes['kernel_index'], request % math.prod(kernel))

        single = MultiplexedConv2DPackedLayerDepthwise(
            1, 1, [8, 8], [1, 1], [1, 1], [1, 1], 1, 1, 1,
            seed_weight_subgrid_stride=[4, 4], seed_weight_layout='joint')
        self.assertFalse(single.use_seed_weight_pack)
        self.assertEqual(single.weight_storage_shape, [1, 1])


if __name__ == '__main__':
    unittest.main()
