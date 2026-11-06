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
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ConvComputeNode, FeatureNode, LayerAbstractGraph, PN15Default, config
from graph_partition_dp import GraphPartitioner, generate_param_dict_for_graph, get_compute_score
from score import FheScoreParam, encrypted_parameter_bytes, get_encrypted_parameter_storage_report
from transforms import insert_drop_level_layers


class StorageScoreTest(unittest.TestCase):
    def setUp(self):
        self.config_before = config.__dict__.copy()
        config.fhe_param = PN15Default
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

    def test_concat2d_is_public_in_final_storage_report(self):
        self.conv.layer_type = 'concat2d'
        report = get_encrypted_parameter_storage_report(self.graph.dag, self.params)
        self.assertEqual(report['unaccounted'], [])
        self.assertEqual(report['layers'], [])
        self.assertEqual(report['storage_bytes'], 0)

    def test_concat_0d_dense_scores_the_exported_skip_and_sparse_tiles(self):
        config.seed_weight_subgrid_stride = [16, 16]
        slots = self.params['param0'].n_slots
        self.input.dim = self.output.dim = 0
        self.conv.layer_type = 'fc0'
        self.conv.channel_input, self.conv.channel_output = 96, 64
        self.graph.dag.nodes[self.input].update(skip=[slots], pack_num=1)
        self.graph.dag.nodes[self.output].update(skip=[slots], pack_num=1, level=0)
        self.graph.dag.nodes[self.conv]['level_cost'] = 1
        score = self.score_param()
        self.assertEqual(score.get_encrypted_param_count(), {2: {'weight': 1}, 1: {'bias': 1}})
        self.assertEqual(score.dense_seed_tiles['weight_shape'], [64, 96])
        self.assertEqual(score.dense_seed_tiles['skip'], [1, slots])
        config.seed_weight_subgrid_stride = [1, 1]
        self.assertEqual(self.score_param().get_encrypted_param_count(),
                         {1: {'weight': 6144}, 0: {'bias': 64}})

    def test_big_poly_counts_spatial_blocks_in_each_degree(self):
        config.seed_weight_subgrid_stride = [16, 16]
        config.block_shape = [128, 256]
        self.input.shape = self.output.shape = [256, 256]
        self.conv.layer_type = 'polyact'
        self.conv.order = 8
        self.conv.channel_input = self.conv.channel_output = 64
        self.graph.dag.nodes[self.input].update(skip=[1, 1], pack_num=1)
        self.graph.dag.nodes[self.output].update(skip=[1, 1], pack_num=1, level=4)
        self.graph.dag.nodes[self.conv]['level_cost'] = 4
        score = self.score_param()
        self.assertEqual(score.get_encrypted_param_count(), {5: {'coefficient': 1},
                                                            6: {'coefficient': 4}, 7: {'coefficient': 4}})
        self.assertEqual(score.poly_seed_layout['channel_groups'], 128)
        self.assertEqual(score.poly_seed_layout['input_shape'], [128, 256])

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
            encrypted_parameter_bytes(self.params['param0'], self.params['param0'].max_level + 1)

    def test_seed_weights_use_the_configured_layout_in_both_scores(self):
        from pipeline import set_seed_weight_subgrid_stride

        config.seed_weight_subgrid_stride = [2, 4]
        baseline = self.score_param().get_encrypted_parameter_storage()
        set_seed_weight_subgrid_stride(self.graph)
        score_param = self.score_param()
        self.assertEqual(self.conv.seed_weight_subgrid_stride, [2, 4])
        self.assertEqual(score_param.get_encrypted_param_count(), {5: {'weight': 18}, 2: {'bias': 1}})
        self.assertLess(score_param.get_encrypted_parameter_storage()['bytes'], baseline['bytes'])
        self.assertGreater(score_param.get_score(), 0)
        expected = score_param.get_encrypted_param_count()
        runtime = score_param.get_score()
        insert_drop_level_layers(self.graph)
        self.assertEqual(self.score_param().get_encrypted_param_count(), expected)
        self.assertEqual(self.score_param().get_score(), runtime)

    def test_seed_storage_above_feature_level_limit(self):
        self.conv.seed_weight_subgrid_stride = [2, 2]
        self.graph.dag.nodes[self.output]['level'] = 7
        self.assertEqual(self.score_param().get_encrypted_param_count(), {11: {'weight': 36}, 8: {'bias': 1}})
        self.assertEqual(self.score_param().get_encrypted_parameter_storage()['bytes'],
                         36 * encrypted_parameter_bytes(self.params['param0'], 11) + encrypted_parameter_bytes(self.params['param0'], 8))

    def test_seed_config_reading_and_pipeline_defaults(self):
        import pipeline
        from training.run_compile import read_compile_config

        with tempfile.TemporaryDirectory() as output_dir:
            config_path = Path(output_dir) / 'config.json'
            for values, expected in [({}, [1, 1]), ({'seed_weight_subgrid_stride': [2, 4]}, [2, 4])]:
                config_path.write_text(json.dumps(values))
                self.assertEqual(read_compile_config(str(config_path))['seed_weight_subgrid_stride'], expected)
            for stride in ([2, 4], [1, 1], [0, 2], [-2, 2], [3, 2], [2], [2.0, 2], [True, 2], None):
                with self.subTest(stride=stride), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(pipeline.LayerAbstractGraph, 'from_json', return_value=self.graph), \
                        patch.object(pipeline, 'try_no_btp', return_value=(True, self.graph, 0.0)), \
                        patch.object(pipeline, 'dump_graph'):
                    pipeline.run_pipeline(
                        1, Path('unused.json'), Path(output_dir), deployment_mode='server_provisioned_runner',
                        seed_weight_subgrid_stride=stride,
                    )
                    self.assertEqual(config.seed_weight_subgrid_stride, [2, 4] if stride == [2, 4] else [1, 1])

    def test_seed_config_is_exported_and_unsupported_layers_keep_legacy_layout(self):
        from pipeline import set_seed_weight_subgrid_stride

        config.seed_weight_subgrid_stride = [2, 4]
        self.conv.parameter_paths = {'weight': 'weight', 'bias': 'bias'}
        set_seed_weight_subgrid_stride(self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'graph.json'
            self.graph.to_json({}, str(path))
            self.assertEqual(json.loads(path.read_text())['layer']['conv']['seed_weight_subgrid_stride'], [2, 4])
        for attrs in ({'groups': 16}, {'is_conv_transpose': True}, {'dim': 1}):
            graph = copy.deepcopy(self.graph)
            node = next(n for n in graph.dag if isinstance(n, ConvComputeNode))
            del node.seed_weight_subgrid_stride
            for key, value in attrs.items():
                setattr(node, key, value)
            set_seed_weight_subgrid_stride(graph)
            self.assertFalse(hasattr(node, 'seed_weight_subgrid_stride'))
        for style, stride in [('ordinary', [2, 4]), ('multiplexed', [1, 1]), ('multiplexed', [16, 2])]:
            del self.conv.seed_weight_subgrid_stride
            config.style, config.seed_weight_subgrid_stride = style, stride
            set_seed_weight_subgrid_stride(self.graph)
            self.assertFalse(hasattr(self.conv, 'seed_weight_subgrid_stride'))
            # Restore only the test fixture for the next case.
            self.conv.seed_weight_subgrid_stride = [2, 4]

    def test_depthwise_and_inverse_share_seed_config_and_storage_scoring(self):
        from pipeline import set_seed_weight_subgrid_stride

        config.seed_weight_subgrid_stride = [2, 4]
        config.block_shape = [4, 4]
        baseline = copy.deepcopy(self.graph)
        for big in (False, True):
            for depthwise in (False, True):
                with self.subTest(big=big, depthwise=depthwise):
                    self.graph = copy.deepcopy(baseline)
                    self.conv = next(n for n in self.graph.dag if isinstance(n, ConvComputeNode))
                    self.output = next(self.graph.dag.successors(self.conv))
                    self.conv.is_big_size = big
                    self.conv.groups = 16 if depthwise else 1
                    self.conv.channel_output = 16 if depthwise else 32
                    self.output.channel = self.conv.channel_output
                    set_seed_weight_subgrid_stride(self.graph)
                    self.assertEqual(self.conv.seed_weight_subgrid_stride, [2, 4])
                    param = self.score_param()
                    counts = param.get_encrypted_param_count()
                    self.assertGreater(sum(c.get('weight', 0) for c in counts.values()), 0)
                    self.assertGreater(param.get_encrypted_parameter_storage()['bytes'], 0)
                    self.assertGreater(param.get_score(), 0)
                    insert_drop_level_layers(self.graph)
                    self.assertEqual(self.score_param().get_encrypted_param_count(), counts)


if __name__ == '__main__':
    unittest.main()
