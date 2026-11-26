"""Numerical slot layouts for sparse Dense tile seeds."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from inference.model_generator.parameter_materialization import ParameterMaterializer
from inference.model_generator.h5_parameter_values import H5ParameterValues


class DenseTileMaterializationTest(unittest.TestCase):
    def test_h5_gathers_use_bounded_slices_with_numpy_layout(self):
        with tempfile.TemporaryDirectory() as tmp, h5py.File(Path(tmp) / 'weights.h5', 'w') as file:
            expected = np.arange(24 * 17, dtype=np.float64).reshape(24, 17)
            for name, array in [('flat', expected.reshape(-1)), ('matrix', expected)]:
                dataset = file.create_dataset(name, data=array)
                with patch('inference.model_generator.h5_parameter_values.CHUNK_VALUES', 16):
                    view = H5ParameterValues(dataset).reshape(24, 17)
                    rows = np.array([23, 0, 23, 8, 1])
                    cols = np.array([16, 7, 0, 3, 15])
                    np.testing.assert_array_equal(view[rows, cols], expected[rows, cols])
                    np.testing.assert_array_equal(view[7][cols], expected[7][cols])
                    self.assertEqual(view[3, 4], expected[3, 4])
                    self.assertEqual(view.reshape(-1)[407], expected.reshape(-1)[407])
                    np.testing.assert_array_equal(view[np.array([], dtype=int), 1], [])

    def test_single_channel_ciphertexts_use_the_compiled_sparse_tiles(self):
        materializer = object.__new__(ParameterMaterializer)
        feature = {'dim': 0, 'channel': 3, 'pack_num': 1, 'skip': 16}
        layout = {'input_shape': [1, 1], 'skip': [1, 16], 'invalid_fill': [1, 16], 'weight': True, 'bias': True}
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        item = {'slots': 16, 'size': [1, 1], 'param_kind': 'weight', 'layout': layout}
        actual = materializer._dense(item, feature, weights, 0)
        np.testing.assert_array_equal(actual, [1, 2, 3, 4, 5, 6] + [0] * 10)
        bias = {**item, 'size': [1], 'param_kind': 'bias'}
        actual = materializer._dense(bias, feature, np.array([7.0, 8.0]), 0)
        np.testing.assert_array_equal(actual, [7, 8] + [0] * 14)

    def test_uncompressed_single_channel_dense_remains_at_slot_zero(self):
        materializer = object.__new__(ParameterMaterializer)
        feature = {'dim': 0, 'channel': 3, 'pack_num': 1, 'skip': 16}
        item = {'slots': 16, 'size': [2, 3], 'param_kind': 'weight'}
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(materializer._dense(item, feature, weights, 4), [5] + [0] * 15)


if __name__ == '__main__':
    unittest.main()
