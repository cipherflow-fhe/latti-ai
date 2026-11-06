"""Real DP planning and operation-count checks for lowered MaxPool."""
import copy
import math
import unittest
from collections import defaultdict
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

import networkx as nx
import components
import transforms
from graph_partition_dp import GraphPartitioner, get_compute_score
from score import BtpScoreParam
from inference.model_generator.layers.poly_maxpool_cost import get_maxpool_stage_op_count, get_maxpool_compare_cost
from inference.model_generator.layers import poly_maxpool2d_layer as reduction
from inference.model_generator.layers import poly2d
from inference.model_generator.layers import poly_maxpool_stages as stages
import frontend.custom_task as custom_task
from training.model_compiler.maxpool_plan import build_maxpool_plan


class MaxPoolDPTest(unittest.TestCase):
    def setUp(self):
        self.saved = copy.deepcopy(components.config.__dict__)
        components.config.fhe_param = copy.deepcopy(components.PN15Default)
        components.config.block_shape = [128, 128]
        components.config.style = 'multiplexed'
        components.config.mpc_refresh = False
        components.config.use_gpu = True

    def tearDown(self):
        components.config.__dict__.clear()
        components.config.__dict__.update(self.saved)

    def graph(self, alpha=7, channels=64):
        graph = components.LayerAbstractGraph()
        x = components.FeatureNode('x', 2, channels, 1, 'param0', 1, [256, 256])
        y = components.FeatureNode('y', 2, channels, 1, 'param0', 1, [128, 128])
        pool = components.PoolComputeNode('pool', 'maxpool2d', channels, channels,
            stride=[2, 2], kernel_shape=[3, 3], padding=[1, 1])
        pool.running_max = 2.5
        pool.alpha = alpha
        graph.dag.add_edges_from([(x, pool), (pool, y)])
        for feature in (x, y):
            graph.dag.nodes[feature].update(skip=[1, 1], level=9, pack_num=1)
        transforms.expand_maxpool_for_btp(graph)
        transforms.infer_shapes_skips_and_pack_num(graph)
        transforms.set_level_costs(graph)
        return graph

    def test_real_dp_levels_live_sets_and_score(self):
        graph = self.graph()
        score, result = GraphPartitioner(graph.dag, temperature=0).run()
        self.assertTrue(nx.is_directed_acyclic_graph(result))
        params = {'param0': components.config.fhe_param}
        actual_score = 0.
        refresh_groups = []
        for node in result:
            if not isinstance(node, components.ComputeNode):
                self.assertTrue(0 <= result.nodes[node]['level'] <= components.config.fhe_param.max_level)
                continue
            source = next(result.predecessors(node))
            output = next(result.successors(node))
            before, after = result.nodes[source]['level'], result.nodes[output]['level']
            if node.layer_type == 'bootstrapping':
                self.assertEqual(after, components.config.fhe_param.max_level)
                self.assertEqual(source.maxpool_state, output.maxpool_state)
                cost = BtpScoreParam(result, node, params, use_gpu=True)
                self.assertEqual(cost.ct_num, source.maxpool_state['groups'] * 64)
                refresh_groups.append(source.maxpool_state['groups'])
                actual_score += cost.get_score()
            else:
                self.assertGreaterEqual(before - after, result.nodes[node]['level_cost'])
                value = get_compute_score(result, node, params)
                self.assertGreater(value, 0)
                actual_score += value
        self.assertEqual(len(refresh_groups), 3)  # discovered by DP, not supplied to it
        self.assertAlmostEqual(score, actual_score)

    def test_round_counts_match_executed_stockmeyer_including_level_zero(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [256, 256])
        class Ct:
            def __init__(self, level):
                self.level = level
                self.metadata = SimpleNamespace(level=level)
        class Pt: pass
        param = SimpleNamespace(n=65536, default_scale=2.**40, q=[2.**40] * 33)
        for alpha, level in ((a, get_maxpool_compare_cost(a) + extra)
                             for a in range(7, 15) for extra in (0, 2)):
            for r in range(4):
                observed = defaultdict(lambda: dict.fromkeys(('rotate', 'mult_plain', 'mult', 'add', 'rescale'), 0))
                def binary(kind):
                    def call(a, b):
                        lv = a.level if isinstance(a, Ct) else b.level
                        key = ('mult_plain' if isinstance(b, Pt) else 'mult') if kind == 'mult' else 'add'
                        observed[lv][key] += 1
                        return Ct(lv)
                    return call
                def rescale(a):
                    observed[a.level]['rescale'] += 1
                    return Ct(a.level - 1)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(custom_task, 'g_param', param))
                    for module in (reduction, poly2d):
                        for name in ('add', 'sub', 'mult'):
                            stack.enter_context(patch.object(module, name, binary(name)))
                        stack.enter_context(patch.object(module, 'rescale', rescale))
                        stack.enter_context(patch.object(module, 'relin', lambda a: a))
                        stack.enter_context(patch.object(module, 'drop_level', lambda a, n: Ct(a.level - n)))
                    pts = stages.bind_maxpool_stage_plaintexts(f'reduce_round_{r + 1}', plan, 3,
                        32768, level, 2., lambda name, count, desc: [Pt() for _ in range(count)], alpha=alpha)
                    groups = 9 if r == 0 else len(plan['rounds'][r - 1]['pairs']) + len(plan['rounds'][r - 1]['carry'])
                    outputs = stages.call_maxpool_stage(f'reduce_round_{r + 1}',
                        [Ct(level) for _ in range(groups * 3)], plan, 3, 32768, pts)
                self.assertEqual({ct.level for ct in outputs}, {level - get_maxpool_compare_cost(alpha)})
                expected = get_maxpool_stage_op_count(f'reduce_round_{r + 1}', [128, 128], 3, 32768, level, alpha=alpha)
                self.assertEqual(dict(observed), expected, (alpha, level, r))

    def test_other_profiles_dp_and_budget_rejection(self):
        for alpha in range(8, 15):
            graph = self.graph(alpha)
            cost = get_maxpool_compare_cost(alpha)
            rounds = [n for n in graph.dag if getattr(n, 'maxpool_stage', '').startswith('reduce_round_')]
            self.assertEqual(sum(graph.dag.nodes[n]['level_cost'] for n in rounds), cost * 4)
            score, result = GraphPartitioner(graph.dag, temperature=0).run()
            self.assertTrue(math.isfinite(score))
            params = {'param0': components.config.fhe_param}
            actual_score = 0.
            internal_refreshes = 0
            for node in result:
                if isinstance(node, components.ComputeNode) and node.layer_type == 'bootstrapping':
                    source = next(result.predecessors(node))
                    output = next(result.successors(node))
                    state = getattr(source, 'maxpool_state', None)
                    self.assertEqual(state, getattr(output, 'maxpool_state', None))
                    btp = BtpScoreParam(result, node, params, use_gpu=True)
                    if state is not None:
                        self.assertEqual(btp.ct_num, state['groups'] * 64)
                    self.assertEqual(result.nodes[output]['level'], components.config.fhe_param.max_level)
                    internal_refreshes += state is not None and 'layout' in state
                    actual_score += btp.get_score()
                elif isinstance(node, components.ComputeNode):
                    source = next(result.predecessors(node))
                    output = next(result.successors(node))
                    self.assertGreaterEqual(result.nodes[source]['level'] - result.nodes[output]['level'],
                                            result.nodes[node]['level_cost'])
                    self.assertGreater(get_compute_score(result, node, params), 0)
                    actual_score += get_compute_score(result, node, params)
            self.assertAlmostEqual(score, actual_score)
            if cost > components.config.fhe_param.max_level:
                self.assertGreater(internal_refreshes, 0)

    def test_higher_profiles_lower_with_sufficient_real_budget(self):
        # Validate lowering independently of timing tables/DP: no claim that
        # this artificial test budget represents a supported CKKS parameter.
        components.config.fhe_param.max_level = 16
        for alpha in range(10, 15):
            graph = self.graph(alpha)
            rounds = [n for n in graph.dag if getattr(n, 'maxpool_stage', '').startswith('reduce_round_')]
            self.assertEqual([n.level_cost for n in rounds], [get_maxpool_compare_cost(alpha)] * 4)
        rounds[0].level_cost = 0
        with self.assertRaisesRegex(ValueError, 'expected'):
            transforms.set_level_costs(graph)

    def test_cost_scales_with_channels_and_rejects_unknown_stage(self):
        for stage in ('normalize_taps', 'reduce_round_1', 'repack', 'restore'):
            counts = get_maxpool_stage_op_count(stage, [128, 128], 64, 32768, 9)
            self.assertGreater(sum(sum(v.values()) for v in counts.values()), 0)
        a = get_maxpool_stage_op_count('reduce_round_1', [128, 128], 1, 32768, 9)
        b = get_maxpool_stage_op_count('reduce_round_1', [128, 128], 64, 32768, 9)
        for level, ops in a.items():
            self.assertEqual(b[level], {k: 64 * v for k, v in ops.items()})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            get_maxpool_stage_op_count('unknown', [128, 128], 64, 32768, 9)


if __name__ == '__main__':
    unittest.main()
