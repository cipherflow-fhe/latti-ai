# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ComputeNode, FeatureNode, LayerAbstractGraph, PN15Default, config
from graph_partition_dp import generate_param_dict_for_graph
from score import FheScoreParam
from inference.model_generator.parameter_sources import pcm_parameter_sources


class PCMStorageScoreTest(unittest.TestCase):
    def test_affine_counts_follow_input_index_with_reversed_predecessors(self):
        previous = config.__dict__.copy()
        self.addCleanup(lambda: (config.__dict__.clear(), config.__dict__.update(previous)))
        config.fhe_param = PN15Default
        config.style = 'multiplexed'
        config.matmul_block_size = 64
        config.n_heads = 3
        config.pcm_encrypted_parameters = True
        graph = LayerAbstractGraph()
        x = FeatureNode('x', 2, 1, shape=[197, 192])
        y = FeatureNode('y', 2, 1, shape=[197, 1])
        output = FeatureNode('output', 2, 1, shape=[197, 192])
        affine = ComputeNode('affine', 'pcmaffine', 1, 1)
        affine.K = 1
        affine.parameter_sources = pcm_parameter_sources(
            dict(type='pcmaffine', gamma_path='gamma', bias_path='beta'))
        for feature, level in [(x, 4), (y, 4), (output, 2)]:
            graph.dag.add_node(feature, level=level, skip=[1, 1], pack_num=1)
        graph.dag.add_node(affine, level_cost=2)
        # NetworkX insertion order differs from the operator's input order.
        graph.dag.add_edge(y, affine, input_index=1)
        graph.dag.add_edge(x, affine, input_index=0)
        graph.dag.add_edge(affine, output)
        score = FheScoreParam(graph.dag, affine, generate_param_dict_for_graph(), 4)
        self.assertEqual(score.get_encrypted_param_count(),
                         {4: {'pt:pt_idx=0': 4}, 2: {'pt:pt_idx=1': 4}})
        self.assertGreater(score.get_score(), 0)


if __name__ == '__main__':
    unittest.main()
