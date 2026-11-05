# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ActivationComputeNode, ComputeNode, FeatureNode, LayerAbstractGraph, PoolComputeNode, config
from transforms import set_graph_scale, set_level_costs
from processor import update_level_cost_for_btp


class PoolBootstrapScaleTest(unittest.TestCase):
    def test_pool_level_cost_includes_normalization_for_each_layout(self):
        before = config.__dict__.copy()
        try:
            config.style = 'multiplexed'
            config.block_shape = [8, 8]
            for dim in (1, 2):
                for size, kernel, adaptive in ((8, 2, False), (8, 8, True), (64, 2, False), (64, 16, False)):
                    with self.subTest(dim=dim, size=size, kernel=kernel, adaptive=adaptive):
                        graph = LayerAbstractGraph()
                        x = FeatureNode('input', dim, 4, shape=[size] * dim)
                        y = FeatureNode('pooled', dim, 4, shape=[size // kernel] * dim)
                        pool = PoolComputeNode('pool', f'avgpool{dim}d', 4, 4, dim=dim,
                            kernel_shape=[kernel] * dim, stride=[kernel] * dim)
                        graph.dag.add_edges_from([(x, pool), (pool, y)])
                        if adaptive:
                            flatten = ComputeNode('flatten', 'reshape', 4, 4)
                            graph.dag.add_edges_from([(y, flatten), (flatten, FeatureNode('output', 0, 4))])
                        set_level_costs(graph)
                        self.assertEqual(graph.dag.nodes[pool]['level_cost'], 1)
                        self.assertEqual(pool.is_adaptive_avgpool, adaptive)
                        update_level_cost_for_btp(graph)
                        self.assertEqual(graph.dag.nodes[pool]['level_cost'], 1)
        finally:
            config.__dict__.clear()
            config.__dict__.update(before)

    def test_pool_normalization_is_not_absorbed_into_adjacent_layers(self):
        for dim in (1, 2):
            for with_bootstrap in (False, True):
                with self.subTest(dim=dim, bootstrap=with_bootstrap):
                    graph = LayerAbstractGraph()
                    poly = ActivationComputeNode('activation', 'polyact', 4, 4)
                    pool = PoolComputeNode('pool', f'avgpool{dim}d', 4, 4, dim=dim,
                                           kernel_shape=[16] * dim, is_adaptive_avgpool=True)
                    dense = ComputeNode('dense', 'fc0', 4, 2)
                    layers = [poly, pool, ComputeNode('flatten', 'reshape', 4, 4)]
                    if with_bootstrap:
                        layers.append(ComputeNode('refresh', 'bootstrapping', 4, 4))
                    layers.append(dense)
                    previous = FeatureNode('input', dim, 4)
                    for index, layer in enumerate(layers):
                        output = FeatureNode(f'feature_{index}', dim, layer.channel_output)
                        graph.dag.add_edges_from([(previous, layer), (layer, output)])
                        previous = output
                    set_graph_scale(graph)
                    self.assertEqual(poly.scale_up, 1)
                    self.assertEqual(dense.scale_down, 1)
                    self.assertEqual(pool.up_scale_str, [])
                    self.assertEqual(pool.down_scale_str, [])


if __name__ == '__main__':
    unittest.main()
