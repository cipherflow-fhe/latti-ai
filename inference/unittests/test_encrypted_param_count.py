"""Storage counts must match the actual encrypted parameter node layouts."""
import unittest
from collections import Counter, defaultdict

from inference.model_generator.layers.multiplexed_conv2d_pack_layer import MultiplexedConv2DPackedLayer
from inference.model_generator.layers.conv2d_packed_layer import Conv2DPackedLayer
from inference.model_generator.layers.dense_packed_layer import DensePackedLayer
from inference.model_generator.layers.poly_relu2d import PolyRelu2D
from inference.model_generator.layers.mult_scaler import MultScalarLayer


def flatten(nodes):
    if isinstance(nodes, list):
        for node in nodes:
            yield from flatten(node)
    else:
        yield nodes


class EncryptedParamCountTest(unittest.TestCase):
    def assert_counts(self, counts, groups):
        actual = defaultdict(Counter)
        for kind, nodes in groups.items():
            for node in flatten(nodes):
                actual[node.level][kind] += 1
        self.assertEqual(counts, dict(actual))

    def test_conv_layouts(self):
        for stride, skip in [([1, 1], [1, 1]), ([2, 2], [1, 1]), ([1, 1], [2, 2])]:
            for ci in [3, 16, 64]:
                layer = MultiplexedConv2DPackedLayer(32, ci, [8, 8], [3, 3], stride, skip, 32,
                                                       (ci + 31) // 32, 1)
                for level in [2, 5, 9]:
                    weights, bias, _ = layer.make_param_ct_nodes('test', level)
                    self.assert_counts(layer.get_encrypted_param_count(level), {'weight': weights, 'bias': bias})
        layer = Conv2DPackedLayer(32, 16, [8, 8], [3, 3], [1, 1], [1, 1], 32, 1, 1)
        weights, bias = layer.make_param_ct_nodes('test', 5)
        self.assert_counts(layer.get_encrypted_param_count(5), {'weight': weights, 'bias': bias})

    def test_poly_and_scalar(self):
        for order in [2, 3, 7, 15]:
            layer = PolyRelu2D([8, 8], order, [1, 1], 32)
            for level in [6, 9]:
                nodes = layer.make_param_ct_nodes('test', 3, level)
                self.assert_counts(layer.get_encrypted_param_count(3, level), {'coefficient': nodes})
        layer = MultScalarLayer()
        self.assert_counts(layer.get_encrypted_param_count(3, 5),
                           {'scalar': layer.make_param_ct_nodes('test', 3, 5)})

    def test_dense_layouts(self):
        layer = DensePackedLayer(10, 64, [1, 1], [1, 1], 32, 2, 1)
        weights, bias = layer.make_param_ct_nodes_skip_0d('test', 5)
        self.assert_counts(layer.get_encrypted_param_count_skip_0d(5), {'weight': weights, 'bias': bias})
        for shape, skip, fill in [([8, 8], [2, 2], [1, 1]), ([8, 1], [4, 1], [2, 1])]:
            layer = DensePackedLayer(10, 1024, shape, skip, 32, 1, 1, invalid_fill=fill)
            weights, bias = layer.make_param_ct_nodes_multiplexed('test', 65536, 5)
            self.assert_counts(layer.get_encrypted_param_count_multiplexed(65536, 5),
                               {'weight': weights, 'bias': bias})
            if shape[1] == 1:
                weights, bias = layer.make_param_ct_nodes_1d_multiplexed('test', 65536, 5)
                self.assert_counts(layer.get_encrypted_param_count_1d_multiplexed(65536, 5),
                                   {'weight': weights, 'bias': bias})


if __name__ == '__main__':
    unittest.main()
