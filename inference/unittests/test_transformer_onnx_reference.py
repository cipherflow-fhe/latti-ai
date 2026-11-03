# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Keep ONNX reference softmax independent of compiler denominator guards."""
import unittest

import numpy as np

from tools.evaluate_transformer_onnx_csv import attention, inverse


class TransformerOnnxReferenceTest(unittest.TestCase):
    def test_non_asor_inverse_preserves_exported_iteration_count(self):
        denominators = np.array([1e-7, 1e-5, .1, 1.0])
        attrs = {'use_asor': 0, 'max_inverse_iterations': 15}
        # Newton's iteration starting at one has this closed form. Small
        # denominators deliberately do not converge to an exact reciprocal.
        expected = -np.expm1(2**15 * np.log1p(-denominators[:-1])) / denominators[:-1]
        expected = np.append(expected, 1.0)
        np.testing.assert_allclose(inverse(denominators, 2**-18, .01, attrs), expected, rtol=1e-12)
        np.testing.assert_array_equal(inverse(denominators, .2, .3, attrs),
                                      inverse(denominators, 2**-18, .01, attrs))

    def test_wide_softmax_does_not_add_epsilon_or_alpha(self):
        x = np.array([[[.1, .2], [.3, -.1], [-.2, .4]]])
        identity, zero = np.eye(2), np.zeros(2)
        values = [x, identity, zero, identity, zero, identity, zero, identity, zero]
        attrs = dict(num_heads=1, head_dim=2, scaling=.5, range_min=-1., range_max=1.,
                     exp_divisor=1., exp_coefficients=b'.05,.01', delta_1=2, delta_2=4,
                     inverse_epsilon=2**-18, inverse_alpha=.01, final_inverse_alpha=.001,
                     use_asor=0, max_inverse_iterations=15, softmax_profile=b'thor-wide')
        original = attention(values, attrs, bert=True)
        changed_controls = dict(attrs, inverse_epsilon=.1, inverse_alpha=.2, final_inverse_alpha=.3)
        np.testing.assert_array_equal(original, attention(values, changed_controls, bert=True))
        probabilities = (.05 + .01 * (.5 * (x @ x.swapaxes(-1, -2)))) ** 2
        for power in (1, 2, 2):
            probabilities = probabilities ** power
            denominator = probabilities.sum(axis=-1, keepdims=True)
            reciprocal = np.ones_like(denominator)
            for _ in range(15):
                reciprocal *= 2 - denominator * reciprocal
            probabilities *= reciprocal
        np.testing.assert_allclose(original, probabilities @ x, atol=1e-15, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
