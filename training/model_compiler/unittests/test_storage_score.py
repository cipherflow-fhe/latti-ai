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

"""Verify storage scoring at candidate levels and after inserted level drops."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ConvComputeNode, FeatureNode, LayerAbstractGraph, N16QP1546H192H32, config
from graph_partition_dp import GraphPartitioner, generate_param_dict_for_graph, get_compute_score
from score import FheScoreParam, encrypted_parameter_bytes, get_encrypted_parameter_storage_report
from transforms import insert_drop_level_layers


class StorageScoreTest(unittest.TestCase):
    def setUp(self):
        self.config_before = config.__dict__.copy()
        config.fhe_param = N16QP1546H192H32
        config.style = 'multiplexed'
        config.use_gpu = True
        config.encrypted_parameter_storage_weight = 1.0
        self.params = generate_param_dict_for_graph()
        self.graph = LayerAbstractGraph()
        self.input = FeatureNode('input', 2, 16, shape=[8, 8])
        self.output = FeatureNode('output', 2, 32, shape=[8, 8])
        self.conv = ConvComputeNode('conv', 'conv2d', 16, 32, groups=1,
                                    stride=[1, 1], kernel_shape=[3, 3])
        self.conv.depth = 0
        self.graph.dag.add_node(self.input, level=9, skip=[1, 1], pack_num=32)
        self.graph.dag.add_node(self.output, level=1, skip=[1, 1], pack_num=32)
        self.graph.dag.add_node(self.conv, level_cost=2)
        self.graph.dag.add_edges_from([(self.input, self.conv), (self.conv, self.output)])

    def tearDown(self):
        config.__dict__.clear()
        config.__dict__.update(self.config_before)

    def score_param(self):
        return FheScoreParam(self.graph.dag, self.conv, self.params, 9)

    def test_zero_weight_skips_storage_and_positive_weight_adds_exact_bytes(self):
        runtime = self.score_param().get_score()
        config.encrypted_parameter_storage_weight = 0.0
        with patch.object(FheScoreParam, 'get_encrypted_parameter_storage', side_effect=AssertionError):
            self.assertEqual(get_compute_score(self.graph.dag, self.conv, self.params), runtime)
        config.encrypted_parameter_storage_weight = 2.0
        storage = self.score_param().get_encrypted_parameter_storage()
        self.assertAlmostEqual(get_compute_score(self.graph.dag, self.conv, self.params),
                               runtime + 2 * storage['bytes'] / (1 << 30))

    def test_actual_use_site_matches_inserted_drop_level(self):
        score_param = self.score_param()
        self.assertEqual(score_param.get_parameter_use_site_level(), 3)
        expected = score_param.get_encrypted_param_count()
        insert_drop_level_layers(self.graph)
        actual = self.score_param()
        self.assertEqual(actual.input_mult_level, 3)
        self.assertEqual(actual.get_encrypted_param_count(), expected)
        report = get_encrypted_parameter_storage_report(self.graph.dag, self.params)
        self.assertEqual(report['unaccounted'], [])
        self.assertEqual(report['layers'][0]['counts'], expected)

    def test_cache_distinguishes_successor_levels(self):
        partitioner = GraphPartitioner(self.graph.dag, temperature=0)
        first = partitioner.get_compute_score_cached(self.graph.dag, self.conv)
        self.graph.dag.nodes[self.output]['level'] = 2
        second = partitioner.get_compute_score_cached(self.graph.dag, self.conv)
        self.assertNotEqual(first, second)
        self.assertEqual(second, get_compute_score(self.graph.dag, self.conv, self.params))
        self.assertEqual(len(partitioner.compute_score_cache), 2)

    def test_final_level_rewrite_rejected(self):
        import pipeline

        previous = get_encrypted_parameter_storage_report(self.graph.dag, self.params)
        self.graph.encrypted_parameter_dp_report = previous
        self.graph.dag.nodes[self.output]['level'] = 2
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(pipeline, 'insert_btp_scale_gamma_layers'), \
                    patch.object(pipeline, 'absorb_pcmgamma_layers'), \
                    patch.object(pipeline, 'recompute_final_level'):
                with self.assertRaisesRegex(ValueError, 'Final level recompute changed'):
                    pipeline.dump_graph(self.graph, Path(output_dir), 0.0, True)

    def test_unsupported_calibration_rejected(self):
        parameter = copy.copy(self.params['param0'])
        parameter.log_default_scale += 1
        with self.assertRaisesRegex(ValueError, 'No encrypted parameter size calibration'):
            encrypted_parameter_bytes(parameter, 3)
        with self.assertRaisesRegex(ValueError, 'Uncalibrated encrypted parameter level'):
            encrypted_parameter_bytes(self.params['param0'], 10)


if __name__ == '__main__':
    unittest.main()
