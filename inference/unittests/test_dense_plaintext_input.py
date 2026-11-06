# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from inference.model_generator.layers import dense_packed_layer as dense


class DensePlaintextInputTest(unittest.TestCase):
    def test_bsgs_weights_match_matrix_product_with_multiple_packs_and_tail(self):
        rng = np.random.default_rng(2026)
        for pack, skip in [(8, 1), (16, 2)]:
            ci, co = pack + 3, pack + 1
            weights = rng.normal(size=(co, ci))
            bias = rng.normal(size=co)
            values = rng.normal(size=ci)
            bs = math.ceil(math.sqrt(pack))
            inputs, parameter_weights, parameter_biases = [], [], []
            for start in range(0, ci, pack):
                x = rng.normal(size=pack * skip)
                x[::skip] = 0
                x[:len(values[start:start + pack]) * skip:skip] = values[start:start + pack]
                inputs.append(x)
            for out in range(math.ceil(co / pack)):
                row = []
                for ct_in in range(len(inputs)):
                    for d in range(pack):
                        g, b = divmod(d, bs)
                        w = np.zeros(pack * skip)
                        for j in range(pack):
                            oc = out * pack + (j - g * bs) % pack
                            ic = ct_in * pack + (j + b) % pack
                            if oc < co and ic < ci:
                                w[j * skip] = weights[oc, ic]
                        row.append(w)
                parameter_weights.append(row)
                bv = np.zeros(pack * skip)
                bv[:len(bias[out * pack:(out + 1) * pack]) * skip:skip] = bias[out * pack:(out + 1) * pack]
                parameter_biases.append(bv)
            layer = dense.DensePackedLayer(co, ci, [1, 1], [1, 1], pack, len(inputs), len(parameter_weights))
            with ExitStack() as stack:
                for name, fn in {
                    'materialize_encrypted_param': lambda w, trigger: w,
                    'mult': np.multiply, 'add': np.add, 'rescale': lambda v: v,
                    'rotate_cols': lambda v, steps: [np.roll(v, -s) for s in steps],
                    'add_encrypted_param_bias': np.add,
                }.items():
                    stack.enter_context(patch.object(dense, name, fn))
                result = layer.call_param_ct_skip_0d(inputs, parameter_weights, parameter_biases, skip,
                                                     input_is_plaintext=True)
            actual = np.concatenate([v[::skip] for v in result])[:co]
            np.testing.assert_allclose(actual, weights @ values + bias, atol=1e-12)
            for v in result:
                if skip > 1:
                    np.testing.assert_array_equal(v[1::skip], 0)


if __name__ == '__main__':
    unittest.main()
