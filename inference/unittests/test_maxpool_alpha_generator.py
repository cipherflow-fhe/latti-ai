"""Execute the real Stockmeyer generator with numerical arithmetic nodes.

This checks polynomial semantics and level accounting, not CKKS noise or
encoding accuracy. No polynomial evaluator is replaced by a reference stub.
"""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import frontend.custom_task as custom_task
from inference.model_generator.layers import poly2d, poly_maxpool2d_layer as reduction
from inference.model_generator.layers import poly_maxpool_stages as stages
from inference.unittests.test_interleaved_maxpool_reduction import Cipher
from training.model_compiler.maxpool_plan import build_maxpool_plan
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile


class AlphaGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8])
        self.param = SimpleNamespace(n=65536, default_scale=2.**40,
                                     q=[2.**40 + 4096 * i for i in range(33)])

    def test_profiles_bind_all_coefficients_and_reject_insufficient_levels(self):
        for alpha in range(7, 15):
            with self.subTest(alpha=alpha), patch.object(custom_task, 'g_param', self.param):
                declarations = {}

                def make_pt(name, count, descriptor):
                    self.assertNotIn(name, declarations)
                    declarations[name] = descriptor
                    return [descriptor['value']] * count

                profile = get_maxpool_profile(alpha)
                cost = sum(poly2d.Poly2D.compute_stockmeyer_level_cost(d)
                           for d in profile['degrees']) + 1
                with self.assertRaisesRegex(ValueError, 'requires'):
                    stages.bind_maxpool_stage_plaintexts(
                        'reduce_round_1', self.plan, 2, 32768, cost - 1, 1., make_pt,
                        alpha=alpha)
                self.assertEqual(declarations, {})
                bound = stages.bind_maxpool_stage_plaintexts(
                    'reduce_round_1', self.plan, 2, 32768, cost, 1., make_pt, alpha=alpha)
                self.assertEqual(bound['level_cost'], cost)
                self.assertEqual([s['degree'] for s in bound['sign_stages']], profile['degrees'])
                start = cost
                for index, degree in enumerate(profile['degrees']):
                    poly = poly2d.Poly2D([4, 4], degree, [1, 1], 1)
                    poly.init_stockmeyer(start)
                    self.assertEqual(len(bound[f'p{index}']), degree + 1)
                    for power in range(degree + 1):
                        descriptor = declarations[f'p{index}_degree_{power}']
                        self.assertEqual(descriptor['level'], poly.cached_stockmeyer_level_order[power])
                        self.assertEqual(descriptor['scale'], poly.cached_stockmeyer_coeff_scale[power])
                        if power % 2 == 0:
                            self.assertEqual(descriptor['value'], 0.)
                    start -= poly.compute_stockmeyer_level_cost(degree)
                self.assertEqual(start, 1)

    def test_real_stockmeyer_chain_matches_fixed_polynomial(self):
        def binary(a, b, op):
            if isinstance(b, Cipher):
                self.assertEqual(a.level, b.level)
                return Cipher(op(a.values, b.values), a.level)
            return Cipher(op(a.values, b), a.level)

        def drop(ct, count):
            self.assertGreaterEqual(count, 0)
            return Cipher(ct.values, ct.level - count)

        with ExitStack() as stack:
            stack.enter_context(patch.object(custom_task, 'g_param', self.param))
            for module in (poly2d, reduction):
                for name, function in {
                    'add': lambda a, b: binary(a, b, np.add),
                    'sub': lambda a, b: binary(a, b, np.subtract),
                    'mult': lambda a, b: binary(a, b, np.multiply),
                    'rescale': lambda ct: Cipher(ct.values, ct.level - 1),
                    'relin': lambda ct: ct,
                    'drop_level': drop,
                }.items():
                    stack.enter_context(patch.object(module, name, function, create=True))
            for alpha in range(7, 15):
                for round_index in range(4):
                    with self.subTest(alpha=alpha, round=round_index):
                        B = 1.3
                        bound = B * (1 if round_index == 0 else 1.05) * 2**round_index
                        weights = stages.bind_maxpool_stage_plaintexts(
                            f'reduce_round_{round_index + 1}', self.plan, 1, 32768, 32, 1.,
                            lambda name, count, desc: [desc['value']] * count,
                            alpha=alpha, comparison_B=B)
                        a = np.linspace(-.7, .8, 257) * 2**round_index
                        b = np.linspace(.6, -.9, 257) * 2**round_index
                        expected_sign = (a - b) / (2 * bound)
                        for coefficients in get_maxpool_profile(alpha)['odd_coefficients']:
                            full = [value for c in coefficients for value in (0., float(c))]
                            expected_sign = np.polynomial.polynomial.polyval(expected_sign, full)
                        expected = a + b + (a - b) * expected_sign
                        actual = reduction.binary_max_stockmeyer(
                            [Cipher(a)], [Cipher(b)], weights['p0'], weights['p1'], None,
                            [4, 4], [1, 1], 1, sign_stages=weights['sign_stages'])[0]
                        np.testing.assert_allclose(actual.values, expected, rtol=1e-8, atol=1e-8)
                        self.assertEqual(actual.level, 32 - weights['level_cost'])
                        if alpha == 7:
                            legacy = reduction.binary_max_stockmeyer(
                                [Cipher(a)], [Cipher(b)], weights['p0'], weights['p1'], None,
                                [4, 4], [1, 1], 1)[0]
                            np.testing.assert_array_equal(legacy.values, actual.values)
                            self.assertEqual(legacy.level, actual.level)

    def test_invalid_chain_is_rejected_before_polynomial_emission(self):
        with patch.object(reduction, '_evaluate_sign_stage') as emit:
            with self.assertRaisesRegex(ValueError, 'weights must match'):
                reduction._evaluate_sign_chain(
                    [Cipher([.2])],
                    [{'degree': 7, 'weights': [[0.]] * 8},
                     {'degree': 15, 'weights': [[0.]] * 8}], [4, 4], [1, 1], 1)
            emit.assert_not_called()

    def test_normalization_and_restore_are_inverse_independent_of_B(self):
        for B in (.6, 1.7):
            descriptors = {}

            def bind(name, count, descriptor):
                descriptors[name] = descriptor
                return [descriptor['value']] * count

            stages.bind_maxpool_stage_plaintexts(
                'normalize_taps', self.plan, 1, 32768, 9, 4., bind,
                alpha=7, upper_bound=.5, comparison_B=B)
            self.assertTrue(descriptors)
            self.assertTrue(all(d['value'] == .125 for d in descriptors.values()))
            restore = stages.bind_maxpool_stage_plaintexts(
                'restore', self.plan, 1, 32768, 9, 4., bind,
                alpha=7, upper_bound=.5, comparison_B=B)
            self.assertEqual(.125 * restore['restore'][0] * 16, 1.)
        for value in (0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'upper_bound'):
                stages.bind_maxpool_stage_plaintexts(
                    'restore', self.plan, 1, 32768, 9, 4., bind, upper_bound=value)


if __name__ == '__main__':
    unittest.main()
