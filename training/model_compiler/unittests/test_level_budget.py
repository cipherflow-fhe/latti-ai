# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Residual-Q bounds are hard constraints, independent of storage scoring."""

import copy
import sys
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import ComputeNode, FeatureNode, LayerAbstractGraph, N16QP1546H192H32, PN15Default, config
from graph_partition_dp import get_min_feature_level
import pipeline
import transforms


class LevelBudgetTest(unittest.TestCase):
    def setUp(self):
        previous = copy.copy(config.__dict__)
        self.addCleanup(lambda: (config.__dict__.clear(), config.__dict__.update(previous)))
        config.mpc_refresh = False
        config.graph_type = 'btp'
        config.model_type = ''
        config.mat_pack_style = ''
        config.encrypted_parameter_storage_weight = 0.0

    def configure(self, profile, encrypted, scale):
        config.deployment_mode = 'server_provisioned_runner' if encrypted else 'client_encrypted_input'
        config.set_btp_scale = scale
        pipeline.set_fhe_param(profile)

    def test_profile_bounds_and_serialized_physical_chain(self):
        for profile in (N16QP1546H192H32, PN15Default):
            for encrypted in (False, True):
                for scale in (None, 1.0):
                    with self.subTest(profile=profile.name, encrypted=encrypted, scale=scale):
                        self.configure(profile, encrypted, scale)
                        reserve = 2 * encrypted + int(scale is not None)
                        self.assertEqual(config.fhe_param.max_level, profile.max_level - reserve)
                        self.assertEqual(get_min_feature_level(), int(scale is not None))
                        self.assertEqual(config.fhe_param.to_dict()['n_mult_level'], len(profile.q) - 1)
                        self.assertEqual(config.fhe_param.q, profile.q)
                        self.assertEqual(config.fhe_param.p, profile.p)
                        self.assertEqual(profile.max_level, len(profile.q) - 1)

    def make_graph(self):
        graph = LayerAbstractGraph()
        c = config.fhe_param.max_level
        input_feature = FeatureNode('input', 2, 1, shape=[8, 8])
        refreshed = FeatureNode('refreshed', 2, 1, shape=[8, 8])
        boot = ComputeNode('boot', 'bootstrapping', 1, 1)
        boot.depth = 0
        graph.dag.add_node(input_feature, level=get_min_feature_level(), skip=[1, 1], pack_num=1)
        graph.dag.add_node(refreshed, level=c, skip=[1, 1], pack_num=1)
        graph.dag.add_node(boot, level_cost=get_min_feature_level() - c)
        graph.dag.add_edges_from([(input_feature, boot), (boot, refreshed)])
        previous = refreshed
        for level in range(c - 1, get_min_feature_level() - 1, -1):
            compute = ComputeNode(f'square_{level}', 'square', 1, 1)
            compute.depth = 0
            feature = FeatureNode(f'feature_{level}', 2, 1, shape=[8, 8])
            graph.dag.add_node(compute, level_cost=1)
            graph.dag.add_node(feature, level=level, skip=[1, 1], pack_num=1)
            graph.dag.add_edges_from([(previous, compute), (compute, feature)])
            previous = feature
        return graph, boot

    def test_bootstrap_outputs_and_explicit_drops_for_all_modes(self):
        for profile in (N16QP1546H192H32, PN15Default):
            for encrypted in (False, True):
                for scale in (None, 1.0):
                    with self.subTest(profile=profile.name, encrypted=encrypted, scale=scale):
                        self.configure(profile, encrypted, scale)
                        graph, boot = self.make_graph()
                        pipeline.insert_btp_scale_gamma_layers(graph)
                        pipeline.absorb_pcmgamma_layers(graph)
                        pipeline.recompute_final_level(graph)
                        pipeline.restore_bootstrap_output_levels(graph)
                        transforms.insert_drop_level_layers(graph)
                        pipeline.validate_final_feature_levels(graph)
                        output = next(graph.dag.successors(boot))
                        self.assertEqual(graph.dag.nodes[output]['level'], len(profile.q) - 1)
                        pred = next(graph.dag.predecessors(boot))
                        self.assertEqual(
                            graph.dag.nodes[boot]['level_cost'], graph.dag.nodes[pred]['level'] - (len(profile.q) - 1)
                        )
                        if encrypted:
                            drop = next(graph.dag.successors(output))
                            self.assertEqual(drop.layer_type, 'drop_level')
                            target = next(graph.dag.successors(drop))
                            self.assertEqual(
                                graph.dag.nodes[target]['level'], config.fhe_param.max_level + int(scale is not None)
                            )

    def test_unrelated_features_cannot_use_bootstrap_scale_exception(self):
        for scale in (None, 1.0):
            self.configure(PN15Default, True, scale)
            graph, _ = self.make_graph()
            root = next(node for node in graph.dag if isinstance(node, FeatureNode) and node.node_id == 'input')
            graph.dag.nodes[root]['level'] = config.fhe_param.max_level + 1
            with self.assertRaisesRegex(ValueError, 'out of range'):
                pipeline.recompute_final_level(graph)

    def test_final_export_rejects_an_ordinary_feature_above_budget(self):
        self.configure(N16QP1546H192H32, True, None)
        graph, _ = self.make_graph()
        root = next(node for node in graph.dag if isinstance(node, FeatureNode) and node.node_id == 'input')
        graph.dag.nodes[root]['level'] = config.fhe_param.max_level + 1
        with self.assertRaisesRegex(ValueError, 'exceeds allowed level'):
            pipeline.validate_final_feature_levels(graph)

    def test_parameter_validation_runs_with_zero_storage_weight_for_cnn(self):
        self.configure(N16QP1546H192H32, True, None)
        config.pcm_encrypted_parameters = False
        graph, _ = self.make_graph()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                pipeline.FheScoreParam, 'get_encrypted_param_count', return_value={10: {'weight': 1}}
            ) as get_count,
        ):
            with self.assertRaisesRegex(ValueError, 'level 10 exceeds residual Q maximum 9'):
                pipeline.dump_graph(graph, Path(directory), 0.0, True)
            get_count.assert_called_once()

    def test_fused_post_bootstrap_gamma_does_not_raise_parameter_use_site(self):
        self.configure(N16QP1546H192H32, True, 1.0)
        graph, boot = self.make_graph()
        refreshed = next(graph.dag.successors(boot))
        target = next(graph.dag.successors(refreshed))
        target.layer_type = 'parcpmm'
        pipeline.insert_btp_scale_gamma_layers(graph)
        pipeline.absorb_pcmgamma_layers(graph)
        pipeline.recompute_final_level(graph)
        pipeline.restore_bootstrap_output_levels(graph)
        transforms.insert_drop_level_layers(graph)
        pipeline.validate_final_feature_levels(graph)
        use_site = next(graph.dag.predecessors(target))
        self.assertEqual(graph.dag.nodes[use_site]['level'], config.fhe_param.max_level)
        self.assertTrue(target.fuse_gama_info)


if __name__ == '__main__':
    unittest.main()
