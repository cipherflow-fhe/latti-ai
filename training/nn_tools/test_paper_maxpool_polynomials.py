import unittest
from decimal import Decimal

import numpy as np

from training.nn_tools.eval_fn_hat_for_aespa import get_hermite_coeffs_for_module
from training.nn_tools.paper_maxpool_polynomials import (
    PAPER_CONFIG,
    PAPER_ODD_COEFFS,
    evaluate_composite_sign,
    initial_gap,
    validate_metadata,
)
from training.nn_tools.polynomial_validation import (
    build_maxpool_execution_config, deferred_half_binary_max,
    configure_poly_maxpool, deferred_half_maxpool2d_reference, paper_maxpool2d_reference,
    real_roots_in_interval, validate_paper_coefficients,
)
from training.nn_tools.fixed_maxpool_profiles import available_maxpool_alphas, get_maxpool_profile


class TestPaperMaxPoolPolynomials(unittest.TestCase):
    def test_public_config_uses_only_user_knobs(self):
        config = configure_poly_maxpool(7, 1.0, 1.2, running_max=8.0)
        self.assertEqual(config['alpha'], 7)
        self.assertEqual(config['degrees'], [7, 7])
        self.assertEqual(config['comparison_B'], 1.2)
        self.assertEqual(config['running_max'], 8.0)
        self.assertEqual(config['round_template'][0]['effective_B'], 1.2)
        with self.assertRaises(ValueError):
            configure_poly_maxpool(4, 1.0, 1.2, running_max=8.0)

    def test_deferred_config_is_alpha_driven(self):
        for alpha in range(7, 15):
            config = build_maxpool_execution_config(alpha, 1.0, 8.0, comparison_B=1.2)
            self.assertEqual(config['degrees'], PAPER_CONFIG[alpha]['degrees'])
            self.assertEqual(len(config['stages']), len(config['degrees']))
            self.assertEqual(config['input_bound'], 8.)
            self.assertEqual(config['comparison_B'], 1.2)
        with self.assertRaises(ValueError):
            build_maxpool_execution_config(4, 1., 8.)

    def test_deferred_binary_max_matches_two_x_max_on_safe_values(self):
        config = build_maxpool_execution_config(7, 1., 8., comparison_B=1.2)
        for a, b in ((-.5, .25), (0., 0.), (.8, -.7)):
            result = deferred_half_binary_max(a, b, config)
            self.assertLess(abs(result - 2*max(a, b)), .04)

    def test_deferred_reference_exposes_round_parameters(self):
        rng = np.random.default_rng(13)
        x = rng.uniform(-2, 2, (1, 2, 7, 9))
        result, config = deferred_half_maxpool2d_reference(x, 7, 2., upper_bound=1.)
        exact = __import__('torch').nn.functional.max_pool2d(__import__('torch').from_numpy(x), 3, 2, 1).numpy()
        self.assertEqual(result.shape, exact.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(np.max(np.abs(result-exact)), 3.0)
        self.assertAlmostEqual(config['input_normalization_multiplier'], 1/(2.+1e-3))

    def test_fixed_profiles_match_validated_coefficients(self):
        self.assertEqual(available_maxpool_alphas(), tuple(range(7, 15)))
        for alpha in available_maxpool_alphas():
            profile = get_maxpool_profile(alpha)
            self.assertEqual(profile['degrees'], PAPER_CONFIG[alpha]['degrees'])
            self.assertEqual(profile['paper_depth'], PAPER_CONFIG[alpha]['depth'])
            self.assertEqual([[float(c) for c in stage] for stage in profile['odd_coefficients']],
                             PAPER_ODD_COEFFS[alpha])
            actual = validate_paper_coefficients(alpha, samples=2)
            self.assertEqual(profile['stage_domains'], actual['stage_domains'])
            self.assertEqual(profile['measured_max_error'], actual['measured_max_error'])
            self.assertLessEqual(Decimal(profile['measured_max_error']), Decimal(2)**(-(alpha-1)))
            self.assertIsNone(profile['actual_ckks_level_cost'])

    def test_fixed_lookup_does_not_mutate_or_fallback(self):
        profile = get_maxpool_profile(7)
        profile['odd_coefficients'][0][0] = '0'
        self.assertEqual(get_maxpool_profile(7)['odd_coefficients'][0][0], '7.30445164958251')
        for alpha in (4, 5, 6, 15):
            with self.assertRaises(ValueError):
                get_maxpool_profile(alpha)
        for alpha in (True, 7.0, '7'):
            with self.assertRaises(TypeError):
                get_maxpool_profile(alpha)

    def test_high_precision_alpha7_stage_images(self):
        result = validate_paper_coefficients(7, samples=65)
        self.assertTrue(result['numerical_sign_bound_pass'])
        first = result['stage_domains'][0]
        self.assertEqual(len(first['critical_points']), 3)
        self.assertGreater(float(first['positive_output'][1]), 1.39)
        self.assertLess(float(first['positive_output'][0]), .61)
        self.assertLess(float(result['float64_reference_max_difference']), 1e-12)

    def test_root_isolation_repeated_root(self):
        import mpmath as mp
        with mp.workdps(60):
            roots = real_roots_in_interval([mp.mpf(1), mp.mpf(-2), mp.mpf(1)], mp.mpf(0), mp.mpf(2))
            self.assertEqual(len(roots), 1)
            self.assertAlmostEqual(float(roots[0]), 1.)

    def test_paper_pool_padding_and_range(self):
        import torch
        rng = np.random.default_rng(7)
        for x in (rng.uniform(-2, 2, (2, 3, 9, 11)),
                  rng.uniform(-2, -.5, (1, 2, 8, 8))):
            result = paper_maxpool2d_reference(x, 7, bound=2.)
            exact = torch.nn.functional.max_pool2d(torch.from_numpy(x), 3, 2, 1).numpy()
            self.assertEqual(result.shape, exact.shape)
            # Four comparisons on a critical path, normalized Max target.
            scale = 2./(.5-3*2.**(-7))
            self.assertLessEqual(np.max(np.abs(result-exact)), scale*4*2.**(-7))
            if np.all(x < 0):
                self.assertTrue(np.all(result < 0))
        with self.assertRaises(ValueError):
            paper_maxpool2d_reference(np.ones((1, 1, 3, 3))*3, 7, bound=2.)

    def test_metadata_and_odd_degree_layout(self):
        for alpha, config in PAPER_CONFIG.items():
            if alpha not in PAPER_ODD_COEFFS:
                continue
            validate_metadata(alpha)
            self.assertEqual(len(config["degrees"]), len(PAPER_ODD_COEFFS[alpha]))
            for coeffs, degree in zip(PAPER_ODD_COEFFS[alpha], config["degrees"]):
                self.assertEqual(len(coeffs), (degree + 1) // 2)

    def test_composite_sign_meets_paper_bound_on_dense_grid(self):
        for alpha in range(7, 15):
            x = np.linspace(initial_gap(alpha), 1.0, 20001)
            error = np.max(np.abs(evaluate_composite_sign(x, alpha) - 1.0))
            self.assertLessEqual(error, 2.0 ** (-(alpha - 1)) + 2e-6)

    def test_maxpool_coefficients_are_not_relu_hermite_coefficients(self):
        relu_degree7 = get_hermite_coeffs_for_module(__import__("torch").nn.ReLU, degree=7)
        p0 = PAPER_ODD_COEFFS[7][0]
        # ReLU Hermite coefficients include an even constant component and
        # are not the odd-power Appendix-A P0 coefficient vector.
        self.assertNotEqual(tuple(round(v, 12) for v in relu_degree7), tuple(round(v, 12) for v in p0))


if __name__ == "__main__":
    unittest.main()
