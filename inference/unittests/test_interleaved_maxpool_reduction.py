"""Numerical simulation of tap routing plus the existing reduction tree."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from training.model_compiler.maxpool_plan import build_maxpool_plan
from inference.model_generator.layers import poly_maxpool_taps as taps
from inference.model_generator.layers import poly_maxpool2d_layer as reduction
from inference.model_generator.layers.poly_maxpool_generic import call_interleaved_deferred_half_maxpool_dag
from inference.model_generator.layers.poly_maxpool_generic import call_maxpool_round_stage, flatten_maxpool_state


class Cipher:
    def __init__(self, values, level=32):
        self.values = np.asarray(values)
        self.level = level
        from types import SimpleNamespace
        self.metadata = SimpleNamespace(level=level)


class InterleavedReductionTest(unittest.TestCase):
    def test_round_state_refresh_preserves_every_group(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8],
        )
        # Stub only binary arithmetic: exercise real round grouping, carry
        # doubling and drop-level handling across externally refreshed states.
        def compare(a, b, **kwargs):
            self.assertTrue(all(x.level == y.level for x, y in zip(a, b)))
            return [Cipher(2 * np.maximum(x.values, y.values), x.level - 7) for x, y in zip(a, b)]

        def drop(ct, count):
            return Cipher(ct.values.copy(), ct.level - count)

        with patch.object(reduction, 'binary_max_stockmeyer', compare), \
             patch.object(reduction, 'drop_level', drop), \
             patch.object(reduction, 'add', lambda a, b: Cipher(a.values + b.values, a.level)):
            for max_level in (9, 16):
                state = [[Cipher(np.array([tap, -tap]), max_level) for _ in range(3)] for tap in range(9)]
                for round_plan in plan['rounds']:
                    if state[0][0].level < 8:
                        # Refresh the complete live set; no groups are discarded.
                        state = [[Cipher(ct.values.copy(), max_level) for ct in group] for group in state]
                    expected_groups = len(round_plan['pairs']) + len(round_plan['carry'])
                    flat = call_maxpool_round_stage(
                        flatten_maxpool_state(state, 3), plan, plan['rounds'].index(round_plan),
                        {'p0': None, 'p1': None}, 3,
                    )
                    state = [flat[i:i + 3] for i in range(0, len(flat), 3)]
                    self.assertEqual(len(state), expected_groups)
                    self.assertEqual(len({ct.level for group in state for ct in group}), 1)
                for ct in state[0]:
                    np.testing.assert_array_equal(ct.values / 16, [8, 0])

    def test_round_rejects_missing_live_group(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8],
        )
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            call_maxpool_round_stage([Cipher([1])] * 8, plan, 0, {}, 1)
        with self.assertRaisesRegex(ValueError, 'every live group'):
            reduction.call_poly_maxpool_reduction_round(
                [[Cipher([1])], [Cipher([2])], [Cipher([3])]],
                {'pairs': [(0, 1)], 'carry': []}, {}, [4, 4], [1, 1], 1,
            )

    def test_polynomial_reference_and_carries(self):
        channels, size, slots = 3, 8, 32
        half = size // 2
        image = np.random.default_rng(7).uniform(-0.8, 0.8, (channels, size, size))
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [size, size],
        )
        specs = taps.build_interleaved_maxpool_tap_masks(plan, [half, half], channels, slots, 1.0)
        inputs = []
        for ch in range(channels):
            for r in range(2):
                for c in range(2):
                    values = np.zeros(slots)
                    values[:half * half] = image[ch, r::2, c::2].ravel()
                    inputs.append(Cipher(values))
        masks = [[[spec['mask'] for spec in tap] for _ in range(channels)] for tap in specs]
        p0 = np.array([0, 7.30445164958251, 0, -34.6825871108659, 0, 59.8596518298826, 0, -31.8755225906466])
        p1 = np.array([0, 2.40085652217597, 0, -2.63125454261783, 0, 1.54912674773593, 0, -0.331172956504304])
        weights = []
        for r in range(4):
            bound = (1 if r == 0 else 1.05) * 2**r
            weights.append({
                'p0': np.repeat((p0 / (2 * bound)**np.arange(8))[:, None], channels, axis=1),
                'p1': np.repeat(p1[:, None], channels, axis=1),
            })

        def binary(a, b, op):
            self.assertEqual(a.level, b.level)
            return Cipher(op(a.values, b.values), a.level)

        def multiply(a, b):
            return binary(a, b, np.multiply) if isinstance(b, Cipher) else Cipher(a.values * b, a.level)

        def drop(ct, count):
            self.assertGreaterEqual(count, 0)
            return Cipher(ct.values, ct.level - count)

        # Simulate the polynomial evaluator at its established 3-level cost;
        # binary Max, carry routing and tap composition execute unchanged.
        def poly(_self, cts, coefficients):
            return [Cipher(np.polynomial.polynomial.polyval(ct.values, coefficients[:, i]), ct.level - 3)
                    for i, ct in enumerate(cts)]

        with ExitStack() as stack:
            for module in (taps, reduction):
                for name, fn in {
                    'add': lambda a, b: binary(a, b, np.add),
                    'mult': multiply,
                    'rescale': lambda ct: Cipher(ct.values, ct.level - 1),
                }.items():
                    stack.enter_context(patch.object(module, name, fn))
            stack.enter_context(patch.object(taps, 'rotate_cols', lambda ct, steps: [Cipher(np.roll(ct.values, -steps[0]), ct.level)]))
            stack.enter_context(patch.object(reduction, 'sub', lambda a, b: binary(a, b, np.subtract)))
            stack.enter_context(patch.object(reduction, 'drop_level', drop))
            stack.enter_context(patch.object(reduction, 'relin', lambda ct: ct))
            stack.enter_context(patch.object(reduction.Poly2D, 'call_stockmeyer_feature2d', poly))
            outputs = call_interleaved_deferred_half_maxpool_dag(inputs, plan, specs, masks, weights, channels)

        rr, cc = np.meshgrid(np.arange(half) * 2, np.arange(half) * 2, indexing='ij')
        reference = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                valid = (rr + dr >= 0) & (rr + dr < size) & (cc + dc >= 0) & (cc + dc < size)
                reference.append(image[:, np.where(valid, rr + dr, rr), np.where(valid, cc + dc, cc)])
        for r, round_plan in enumerate(plan['rounds']):
            next_values = []
            for i, j in round_plan['pairs']:
                a, b = reference[i], reference[j]
                sign = np.polynomial.polynomial.polyval(
                    np.polynomial.polynomial.polyval(a - b, weights[r]['p0'][:, 0]), p1,
                )
                next_values.append(a + b + (a - b) * sign)
            next_values.extend(2 * reference[i] for i in round_plan['carry'])
            reference = next_values
        for ch, output in enumerate(outputs):
            self.assertEqual(output.level, 3)  # 32 - normalization - 4*7
            np.testing.assert_allclose(output.values[:half * half].reshape(half, half) / 16, reference[0][ch] / 16)
            np.testing.assert_array_equal(output.values[half * half:], 0)

    def test_low_level_is_rejected(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [8, 8],
        )
        with self.assertRaisesRegex(ValueError, 'Insufficient levels'):
            call_interleaved_deferred_half_maxpool_dag(
                [Cipher(np.zeros(32), 9) for _ in range(4)], plan,
                [{}] * 9, [], [{}] * 4, 1,
            )


if __name__ == '__main__':
    unittest.main()
