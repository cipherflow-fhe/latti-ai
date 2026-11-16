"""Real Stockmeyer execution across complete live-set refresh boundaries."""
from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

import frontend.custom_task as custom_task
from inference.model_generator.layers import poly2d, poly_maxpool_stages as stages
from inference.model_generator.layers.poly_maxpool_cost import get_maxpool_stage_op_count
from inference.unittests.test_interleaved_maxpool_reduction import Cipher
from training.model_compiler.maxpool_plan import build_maxpool_plan
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


class Plain:
    def __init__(self, value):
        self.value = value


class SplitStageTest(unittest.TestCase):
    def test_all_profiles_full_tree_refresh_semantics_and_counts(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8])
        param = SimpleNamespace(n=65536, default_scale=2.**40, q=[2.**40] * 33)
        observed = {}

        def binary(a, b, op, name):
            observed[a.level][name] += 1
            if isinstance(b, Cipher):
                self.assertEqual(a.level, b.level)
                value = b.values
            else:
                value = b.value
            return Cipher(op(a.values, value), a.level)

        def multiply(a, b):
            return binary(a, b, np.multiply, 'mult' if isinstance(b, Cipher) else 'mult_plain')

        def rescale(ct):
            observed[ct.level]['rescale'] += 1
            return Cipher(ct.values, ct.level - 1)

        def drop(ct, count):
            self.assertGreaterEqual(count, 0)
            return Cipher(ct.values, ct.level - count)

        with ExitStack() as stack:
            stack.enter_context(patch.object(custom_task, 'g_param', param))
            for module in (poly2d, stages):
                for name, function in {
                    'add': lambda a, b: binary(a, b, np.add, 'add'),
                    'sub': lambda a, b: binary(a, b, np.subtract, 'add'),
                    'mult': multiply, 'rescale': rescale, 'relin': lambda ct: ct,
                    'drop_level': drop,
                }.items():
                    stack.enter_context(patch.object(module, name, function, create=True))
            for alpha in range(7, 15):
                profile = get_maxpool_profile(alpha)
                channels, B = 2, 1.3
                reference = list(np.random.default_rng(71).uniform(-.7, .7, (9, channels, 19)))
                nodes = [Cipher(v, 9) for group in reference for v in group]
                for r, round_plan in enumerate(plan['rounds']):
                    bound = B * (1 if r == 0 else 1.05) * 2**r
                    expected = []
                    for i, j in round_plan['pairs']:
                        a, b = reference[i], reference[j]
                        sign = (a - b) / (2 * bound)
                        for coefficients in profile['odd_coefficients']:
                            sign = np.polynomial.polynomial.polyval(
                                sign, [v for c in coefficients for v in (0., float(c))])
                        expected.append(a + b + (a - b) * sign)
                    expected.extend(2 * reference[i] for i in round_plan['carry'])
                    for phase in list(range(len(profile['degrees']))) + ['finish']:
                        suffix = 'finish' if phase == 'finish' else f'poly_{phase}'
                        name = f'reduce_round_{r + 1}_{suffix}'
                        # Refresh every live ciphertext. Actual placement is
                        # verified separately with the real DP implementation.
                        nodes = [Cipher(ct.values.copy(), 9) for ct in nodes]
                        observed = defaultdict(lambda: dict.fromkeys(
                            ('rotate', 'mult_plain', 'mult', 'add', 'rescale'), 0))
                        pts = stages.bind_maxpool_stage_plaintexts(
                            name, plan, channels, 32768, 9, 1.,
                            lambda key, count, desc: [Plain(desc['value']) for _ in range(count)],
                            alpha=alpha, comparison_B=B)
                        nodes = stages.call_maxpool_stage(name, nodes, plan, channels, 32768, pts)
                        self.assertEqual(dict(observed), get_maxpool_stage_op_count(
                            name, [4, 4], channels, 32768, 9, alpha=alpha), (alpha, name))
                        self.assertEqual({ct.level for ct in nodes}, {9 - pts['level_cost']})
                        pairs, carry = len(round_plan['pairs']), len(round_plan['carry'])
                        self.assertEqual(len(nodes), channels * ((pairs if phase == 'finish' else 3 * pairs) + carry))
                    np.testing.assert_allclose(np.array([ct.values for ct in nodes]).reshape(
                        len(expected), channels, -1), expected, atol=1e-8, rtol=1e-8)
                    reference = expected


if __name__ == '__main__':
    unittest.main()
