"""Bounded weight conversion must preserve transposes, QKV slices and scaling."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import onnx
from onnx import helper, numpy_helper

from training.nn_tools.onnx_h5 import H5Values, InitializerValues


class StreamingHighLevelTest(unittest.TestCase):
    def test_external_transpose_qkv_and_absorption_across_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = np.arange(15 * 21, dtype=np.float32).reshape(15, 21) / 13
            graph = helper.make_graph([], 'g', [], [], [numpy_helper.from_array(array, 'weight')])
            onnx.save_model(
                helper.make_model(graph),
                root / 'model.onnx',
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location='weights.bin',
                size_threshold=0,
            )
            source = InitializerValues(onnx.load(root / 'model.onnx', load_external_data=False), root / 'model.onnx')
            with h5py.File(root / 'out.h5', 'w') as file, patch('training.nn_tools.onnx_h5.CHUNK_VALUES', 16):
                out = H5Values(file)
                divisor = np.arange(1, 16, dtype=np.float64)
                cases = [
                    ('plain', array.shape, {}, array.astype('float64')),
                    ('transpose', array.T.shape, {'transpose': True}, array.astype('float64').T),
                    (
                        'q',
                        (7, 15),
                        {'transpose': True, 'columns': (7, 14), 'multipliers': (0.125, 0.3)},
                        (array.astype('float64')[:, 7:14] * 0.125 * 0.3).T,
                    ),
                    ('absorbed', array.shape, {'divisors': (divisor,)}, array.astype('float64') / divisor[:, None]),
                ]
                original = np.fromfile
                reads = []

                def read(*args, **kwargs):
                    reads.append(kwargs['count'])
                    return original(*args, **kwargs)

                with patch('training.nn_tools.onnx_h5.np.fromfile', side_effect=read):
                    for name, shape, options, expected in cases:
                        out.write_initializer(name, source, 'weight', shape, **options)
                        np.testing.assert_array_equal(out[name], expected)
                        self.assertEqual(
                            out.value_hash(name), hashlib.sha256(expected.astype('<f8').tobytes()).hexdigest()
                        )
                self.assertGreater(len(reads), 1)
                self.assertLessEqual(max(reads), 16)
                out.write_initializer('plain', source, 'weight', array.shape)
                with self.assertRaisesRegex(ValueError, 'conflict'):
                    out.write_initializer('plain', source, 'weight', array.shape, multipliers=(2,))
                self.assertFalse(source.tensors['weight'].HasField('raw_data'))


if __name__ == '__main__':
    unittest.main()
