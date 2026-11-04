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
from inference.model_generator.layers import multiplexed_conv2d_pack_layer_depthwise as depthwise


class RectangularSpatialLayoutTest(unittest.TestCase):
    def test_depthwise_stride_preserves_channels_with_unequal_skips(self):
        shape, stride = [4, 8], [2, 2]
        for skip in ([1, 2], [2, 1], [2, 4]):
            channels_per_ct = 2 * int(np.prod(skip))
            channels = 2 * channels_per_ct + 1
            slots = channels_per_ct * int(np.prod(shape))
            values = np.arange(channels * np.prod(shape)).reshape(channels, *shape) + 1

            def pack(data, spatial_skip, per_ct):
                result = [np.zeros(slots) for _ in range((channels + per_ct - 1) // per_ct)]
                for c in range(channels):
                    block, phase = divmod(c % per_ct, int(np.prod(spatial_skip)))
                    for row in range(data.shape[1]):
                        for col in range(data.shape[2]):
                            index = (block * np.prod(shape) * np.prod(skip) +
                                     (row * spatial_skip[0] + phase // spatial_skip[1]) * shape[1] * skip[1] +
                                     col * spatial_skip[1] + phase % spatial_skip[1])
                            result[c // per_ct][index] = data[c, row, col]
                return result

            x = pack(values, skip, channels_per_ct)
            masks = []
            for c in range(channels):
                selected = np.zeros_like(values)
                selected[c, ::stride[0], ::stride[1]] = 1
                masks.append(pack(selected, skip, channels_per_ct)[c // channels_per_ct])
            expected = pack(values[:, ::2, ::2], np.multiply(skip, stride), channels_per_ct * 4)
            layer = depthwise.MultiplexedConv2DPackedLayerDepthwise(
                channels, channels, shape, [1, 1], stride, skip, channels_per_ct, len(x), len(expected))
            with patch.object(depthwise, 'rotate_cols', lambda x, steps: [np.roll(x, -int(steps[0]))]), \
                    patch.object(depthwise, 'ct_pt_mult_accumulate', lambda xs, ws: sum(a * b for a, b in zip(xs, ws))), \
                    patch.object(depthwise, 'rescale', lambda x: x), \
                    patch.object(depthwise, '_multiply_mask_param', np.multiply), \
                    patch.object(depthwise, 'add', np.add):
                actual = layer.call(x, [[np.ones(slots)] for _ in x], [np.zeros(slots) for _ in expected], masks)
            np.testing.assert_array_equal(actual, expected)

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
