"""Validate tap routing against Feature2DEncrypted::encode_interleaved."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.model_compiler.maxpool_plan import build_maxpool_plan
from inference.model_generator.layers import poly_maxpool_taps as taps


class InterleavedMaxPoolTapsTest(unittest.TestCase):
    def check_shape(self, size, channels, slots):
        half = size // 2
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]},
            [size, size],
        )
        scale = 0.25
        specs = taps.build_interleaved_maxpool_tap_masks(
            plan, [half, half], channels, slots, scale,
        )
        image = np.random.default_rng(43).normal(size=(channels, size, size))

        class Cipher:
            def __init__(self, values, level=9):
                self.values, self.level = values, level
                self.metadata = SimpleNamespace(level=level)

        inputs = []
        for channel in range(channels):
            for row in range(2):
                for col in range(2):
                    values = np.zeros(slots)
                    values[:half * half] = image[channel, row::2, col::2].ravel()
                    inputs.append(Cipher(values))
        weights = [[[spec['mask'] for spec in tap] for _ in range(channels)] for tap in specs]
        with patch.object(taps, 'rotate_cols', lambda ct, steps: [Cipher(np.roll(ct.values, -steps[0]), ct.level)]), \
             patch.object(taps, 'mult', lambda ct, pt: Cipher(ct.values * pt, ct.level)), \
             patch.object(taps, 'rescale', lambda ct: Cipher(ct.values, ct.level - 1)), \
             patch.object(taps, 'add', lambda a, b: Cipher(a.values + b.values, a.level)):
            outputs = taps.call_interleaved_normalized_maxpool_taps(inputs, specs, weights, channels)
        for index, output in enumerate(outputs):
            dr, dc = index // 3 - 1, index % 3 - 1
            rr, cc = np.meshgrid(np.arange(half) * 2, np.arange(half) * 2, indexing='ij')
            valid = (rr + dr >= 0) & (rr + dr < size) & (cc + dc >= 0) & (cc + dc < size)
            source_r = np.where(valid, rr + dr, rr)
            source_c = np.where(valid, cc + dc, cc)
            for channel, ct in enumerate(output):
                expected = image[channel, source_r, source_c] * scale
                np.testing.assert_allclose(ct.values[:half * half].reshape(half, half), expected)
                np.testing.assert_array_equal(ct.values[half * half:], 0)
                self.assertEqual(ct.level, 8)

    def test_small_multiple_channels(self):
        self.check_shape(8, 3, 32)

    def test_retina_spatial_layout(self):
        self.check_shape(256, 2, 32768)

    def test_unsupported_layout(self):
        plan = build_maxpool_plan(
            {'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [256, 256],
        )
        with self.assertRaises(NotImplementedError):
            taps.build_interleaved_maxpool_tap_masks(plan, [64, 64], 64, 32768, 1.0)


if __name__ == '__main__':
    unittest.main()
