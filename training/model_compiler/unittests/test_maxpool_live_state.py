"""Compiler live-set metadata and Bootstrap cost regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import components
import transforms
from score import BtpScoreParam


class MaxPoolLiveStateTest(unittest.TestCase):
    def setUp(self):
        self.saved = copy.deepcopy(components.config.__dict__)
        components.config.fhe_param = copy.deepcopy(components.PN15Default)
        components.config.block_shape = [128, 128]
        components.config.style = 'multiplexed'
        components.config.mpc_refresh = False

    def tearDown(self):
        components.config.__dict__.clear()
        components.config.__dict__.update(self.saved)

    def test_shape_json_and_full_live_set_bootstrap(self):
        graph = components.LayerAbstractGraph()
        source = components.FeatureNode('x', 2, 64, 1, 'param0', 1, [256, 256])
        output = components.FeatureNode('y', 2, 64, 1, 'param0', 1, [128, 128])
        pool = components.PoolComputeNode('pool', 'maxpool2d', 64, 64,
            stride=[2, 2], kernel_shape=[3, 3], padding=[1, 1])
        pool.running_max = 2.5
        pool.alpha = 7
        pool.upper_bound = .5
        pool.B = 1.3
        graph.dag.add_edges_from([(source, pool), (pool, output)])
        for feature in (source, output):
            graph.dag.nodes[feature].update(skip=[1, 1], level=1, pack_num=1)
        transforms.expand_maxpool_for_btp(graph)
        transforms.infer_shapes_skips_and_pack_num(graph)
        transforms.set_level_costs(graph)
        features = [n for n in graph.dag if hasattr(n, 'maxpool_state')]
        self.assertEqual([f.maxpool_state['groups'] for f in features], [9, 5, 3, 2, 1])
        self.assertEqual(output.shape, [128, 128])
        params = {'param0': components.config.fhe_param}
        for feature in features:
            self.assertEqual(feature.shape, [128, 128])
            btp = transforms.add_btp_layer(graph.dag, feature, params, 8)
            refreshed = next(graph.dag.successors(btp))
            self.assertEqual(refreshed.maxpool_state, feature.maxpool_state)
            score = BtpScoreParam(graph.dag, btp, params)
            self.assertEqual(score.ct_num, feature.maxpool_state['groups'] * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'stages.json')
            graph.to_json(params, path)
            raw = json.loads(Path(path).read_text())
            self.assertEqual(raw['feature']['y']['shape'], [128, 128])
            restored = components.LayerAbstractGraph.from_json(path)
            restored_features = {n.node_id: n for n in restored.dag if isinstance(n, components.FeatureNode)}
            for feature in features:
                self.assertEqual(restored_features[feature.node_id].maxpool_state, feature.maxpool_state)
            stages = [n for n in restored.dag if getattr(n, 'layer_type', '').startswith('maxpool2d_')]
            self.assertEqual([n.level_cost for n in stages], [1, 7, 7, 7, 7, 1, 1])
            for node in stages:
                self.assertEqual((node.alpha, node.upper_bound, node.B), (7, .5, 1.3))


if __name__ == '__main__':
    unittest.main()
