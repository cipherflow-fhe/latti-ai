# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/model_compiler'))
import pipeline
from inference.model_generator.layers import avgpool2d_layer as pool
from inference.model_generator.layers import multiplexed_conv2d_pack_layer as conv


class RectangularSpatialLayoutTest(unittest.TestCase):
    def test_large_images_fill_slots_by_extending_width(self):
        for shape, slots, expected in [
            ([512, 512], 32768, [128, 256]),
            ([512, 512], 4096, [64, 64]),
            ([256, 512], 32768, [128, 256]),
            ([32, 32], 32768, [32, 32]),
        ]:
            with self.subTest(shape=shape, slots=slots):
                config = SimpleNamespace(model_type='')
                graph = SimpleNamespace(get_leading_feature_nodes=lambda: [
                    SimpleNamespace(shape=shape, data_type='feature')])
                with patch.object(pipeline, 'config', config):
                    pipeline.set_block_shape(SimpleNamespace(poly_modulus_degree=slots * 2), graph)
                self.assertEqual(config.block_shape, expected)

    def test_adaptive_pool_with_rectangular_shape_and_unequal_skips(self):
        shape, skip, stride = [4, 8], [2, 4], [2, 4]
        rng = np.random.default_rng(42)
        values = rng.normal(size=(8, *shape))
        packed = np.zeros((shape[0] * skip[0], shape[1] * skip[1]))
        for channel in range(8):
            row, col = divmod(channel, skip[1])
            packed[row::skip[0], col::skip[1]] = values[channel]
        layer = pool.Avgpool2DLayer(stride, shape, channel=8, skip=skip)
        with patch.object(pool, 'rotate_cols', lambda x, steps: [np.roll(x, -int(steps))]), \
                patch.object(pool, 'add', np.add):
            actual = layer.run_adaptive_avgpool([packed.ravel()], packed.size * 2)[0].reshape(packed.shape)
        for channel in range(8):
            row, col = divmod(channel, skip[1])
            expected = values[channel].reshape(2, 2, 2, 4).sum(axis=(1, 3))
            np.testing.assert_allclose(actual[row::skip[0] * stride[0], col::skip[1] * stride[1]], expected)

    def test_rectangular_kernel_rotates_height_and_width_independently(self):
        shape, skip = [4, 8], [1, 2]
        packed = np.arange(np.prod(shape) * np.prod(skip))
        for kernel in ([1, 3], [3, 1]):
            layer = conv.MultiplexedConv2DPackedLayer(1, 1, shape, kernel, [1, 1], skip, 2, 1, 1)
            with patch.object(conv, 'rotate_cols', lambda x, steps: [np.roll(x, -s) for s in steps]):
                actual = layer.gen_rotated_x([packed])[0]
            expected = [np.roll(packed, -((h - kernel[0] // 2) * shape[1] * skip[0] * skip[1]
                                        + (w - kernel[1] // 2) * skip[1]))
                        for h in range(kernel[0]) for w in range(kernel[1])]
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
