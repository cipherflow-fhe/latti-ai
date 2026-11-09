"""Execute the compiler's expanded graph against the fixed-profile reference."""
import copy
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import components
import transforms
import frontend.custom_task as custom_task
from inference.model_generator.layers import poly2d, poly_maxpool2d_layer as reduction
from inference.model_generator.layers import poly_maxpool_primitives as primitives
from inference.unittests.test_interleaved_maxpool_reduction import Cipher
from training.model_compiler.maxpool_plan import build_maxpool_plan
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


class MaxPoolPrimitiveTest(unittest.TestCase):
    def test_expanded_graph_matches_reference_all_profiles(self):
        saved = copy.deepcopy(components.config.__dict__)
        self.addCleanup(lambda: (components.config.__dict__.clear(), components.config.__dict__.update(saved)))
        components.config.fhe_param = copy.deepcopy(components.PN15Default)
        components.config.fhe_param.poly_modulus_degree = 64
        components.config.block_shape = [4, 4]
        components.config.style = 'multiplexed'
        channels, slots = 3, 32
        image = np.random.default_rng(11).uniform(-.7, .7, (channels, 8, 8))
        plan = build_maxpool_plan({'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8])
        param = SimpleNamespace(n=64, default_scale=2.**40, q=[2.**40] * 33)

        def binary(a, b, fn):
            if isinstance(b, Cipher):
                self.assertEqual(a.level, b.level)
                b = b.values
            return Cipher(fn(a.values, b), a.level)

        with ExitStack() as stack:
            stack.enter_context(patch.object(custom_task, 'g_param', param))
            for module in (primitives, poly2d, reduction):
                for name, fn in {
                    'add': lambda a, b: binary(a, b, np.add),
                    'sub': lambda a, b: binary(a, b, np.subtract),
                    'mult': lambda a, b: binary(a, b, np.multiply),
                    'relin': lambda a: a,
                    'rescale': lambda a: Cipher(a.values, a.level - 1),
                    'drop_level': lambda a, n: Cipher(a.values, a.level - n),
                    'rotate_cols': lambda a, steps: [Cipher(np.roll(a.values, -s), a.level) for s in steps],
                }.items():
                    stack.enter_context(patch.object(module, name, fn, create=True))
            for alpha in range(7, 15):
                with self.subTest(alpha=alpha):
                    graph = components.LayerAbstractGraph()
                    source = components.FeatureNode('x', 2, channels, 1, 'param0', 1, [8, 8])
                    output = components.FeatureNode('y', 2, channels, 1, 'param0', 1, [4, 4])
                    pool = components.PoolComputeNode('pool', 'maxpool2d', channels, channels,
                        kernel_shape=[3, 3], stride=[2, 2], padding=[1, 1])
                    pool.alpha, pool.running_max, pool.upper_bound, pool.B = alpha, 2., .5, 1.3
                    graph.dag.add_edges_from([(source, pool), (pool, output)])
                    for feature in (source, output):
                        graph.dag.nodes[feature].update(skip=[1, 1], pack_num=1, level=12)
                    transforms.expand_maxpool_for_btp(graph)
                    transforms.infer_shapes_skips_and_pack_num(graph)
                    transforms.set_level_costs(graph)
                    self.assertNotIn(pool, graph.dag)
                    nodes = [n for n in graph.dag if isinstance(n, components.ComputeNode)]
                    self.assertEqual(sum(n.maxpool_primitive['op'] == 'poly' for n in nodes),
                                     8 * len(get_maxpool_profile(alpha)['degrees']))
                    self.assertEqual(sum(n.maxpool_primitive['op'] == 'mult' for n in nodes), 8)
                    values = {source: []}
                    for channel in image:
                        for r in range(2):
                            for c in range(2):
                                vector = np.zeros(slots)
                                vector[:16] = channel[r::2, c::2].ravel()
                                values[source].append(Cipher(vector))

                    def make_pt(name, count, descriptor):
                        vector = np.zeros(slots)
                        if descriptor['kind'] == 'constant':
                            vector[:] = descriptor['value']
                        elif descriptor['kind'] == 'prefix':
                            vector[:descriptor['length']] = descriptor['value']
                        else:
                            vector[descriptor['indices']] = descriptor['value']
                        return [vector] * count

                    for node in nx.topological_sort(graph.dag):
                        if not isinstance(node, components.ComputeNode):
                            continue
                        preds = sorted(graph.dag.predecessors(node),
                                       key=lambda f: graph.dag.edges[f, node]['input_index'])
                        # Numerical refresh simulates independent live features;
                        # real CKKS Bootstrap selection is tested by DP integration.
                        inputs = [[Cipher(ct.values, 32) for ct in values[f]] for f in preds]
                        pts = primitives.bind_maxpool_primitive(node.__dict__, plan, channels, slots, 32, make_pt)
                        values[next(graph.dag.successors(node))] = primitives.call_maxpool_primitive(
                            node.maxpool_primitive, inputs, plan, channels, slots, pts)
                    reference = []
                    for kr in range(3):
                        for kc in range(3):
                            tap = np.empty((channels, 4, 4))
                            for r in range(4):
                                for c in range(4):
                                    rr, cc = 2 * r + kr - 1, 2 * c + kc - 1
                                    if not (0 <= rr < 8 and 0 <= cc < 8):
                                        rr, cc = 2 * r, 2 * c
                                    tap[:, r, c] = image[:, rr, cc] * .25
                            reference.append(tap)
                    for r, round_plan in enumerate(plan['rounds']):
                        next_values = []
                        for i, j in round_plan['pairs']:
                            a, b = reference[i], reference[j]
                            bound = 1.3 * (1 if r == 0 else 1.05) * 2**r
                            sign = (a - b) / (2 * bound)
                            for odd in get_maxpool_profile(alpha)['odd_coefficients']:
                                coeffs = [v for coefficient in odd for v in (0., float(coefficient))]
                                sign = np.polynomial.polynomial.polyval(sign, coeffs)
                            next_values.append(a + b + (a - b) * sign)
                        next_values.extend(2 * reference[i] for i in round_plan['carry'])
                        reference = next_values
                    expected = reference[0] / 4  # restore running_max / upper_bound / 16
                    self.assertEqual(len(values[output]), 2)
                    for c in range(channels):
                        actual = values[output][c // 2].values[(c % 2)*16:(c % 2 + 1)*16]
                        np.testing.assert_allclose(actual.reshape(4, 4), expected[c], atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
