# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts


gamma_op_class = 'ParUpperDiagonalPolyActRNGamma'
poly_op_class = 'ParUpperDiagonalPolyActRNPoly'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class _ParUpperDiagonalPolyBase:
    def _init_layout(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self.n_prepad = shape[0]
        self.total_cols = shape[1]
        assert head_shape[0] == self.n_prepad
        self.m_prepad = head_shape[1]
        self.H_prepad = n_heads
        self.n_slot = n_slot

        self.H = _next_pow2(self.H_prepad)
        self.m = _next_pow2(self.m_prepad)
        self.n = _next_pow2(self.n_prepad)
        assert self.n >= self.m
        assert self.n % self.m == 0

        self.packed_extent = self.H_prepad * self.m_prepad
        self.segment_len = self.H * self.n
        assert n_slot % self.segment_len == 0
        self.c = n_slot // self.segment_len
        assert self.m % self.c == 0
        self.cts_per_mb = self.m // self.c
        self.n_mb = math.ceil(self.total_cols / self.packed_extent)
        self.total_cts = self.n_mb * self.cts_per_mb

    def _ct_index(self, mb: int, ct_local: int) -> int:
        return mb * self.cts_per_mb + ct_local


class ParUpperDiagonalPolyActRNGamma(_ParUpperDiagonalPolyBase):
    """Python model-generator counterpart of C++ ParUpperDiagonalPolyActRNGamma."""

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, gamma_pt: list) -> list:
        assert len(input_cts) == self.total_cts
        return [rescale(mult(x, gamma_pt[idx])) for idx, x in enumerate(input_cts)]

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        gamma_pt = [None] * self.total_cts
        for mb in range(self.n_mb):
            for ct_local in range(self.cts_per_mb):
                idx = self._ct_index(mb, ct_local)
                node = CkksPlaintextRingtNode(f'encode_pt_gamma_{mb}_{ct_local}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': gamma_op_class,
                        'type': 'gamma_pt',
                        'mb': mb,
                        'ct_local': ct_local,
                        'g': 0,
                    },
                )
                gamma_pt[idx] = node
        return self.call(input_cts, gamma_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {'weight': self.total_cts, 'bias': 0, 'mask': 0}
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        ops[level]['mult_plain'] = self.total_cts
        ops[level]['rescale'] = self.total_cts
        return dict(ops)


class ParUpperDiagonalPolyActRNPoly(_ParUpperDiagonalPolyBase):
    """Python model-generator counterpart of C++ ParUpperDiagonalPolyActRNPoly."""

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, degree: int):
        assert degree in (2, 4)
        self.degree = degree
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(
        self,
        input_cts: list,
        c0_add_pt: list,
        c1_pt: list,
        c2_pt: list,
        c3_pt: list | None = None,
        c4_pt: list | None = None,
    ) -> list:
        assert len(input_cts) == self.total_cts
        if self.degree == 4:
            assert c3_pt is not None and c4_pt is not None

        result = [None] * self.total_cts
        for idx, x in enumerate(input_cts):
            x_sq = rescale(relin(mult(x, x)))

            c2x2 = rescale(mult(x_sq, c2_pt[idx]))
            c1x = rescale(mult(x, c1_pt[idx]))
            low = add(drop_level(c1x), c2x2)
            low = add(low, c0_add_pt[idx])

            if self.degree == 2:
                result[idx] = low
                continue

            c4x2 = rescale(mult(x_sq, c4_pt[idx]))
            c3x = rescale(mult(x, c3_pt[idx]))
            high = add(drop_level(c3x), c4x2)

            x2_high = rescale(relin(mult(drop_level(x_sq), high)))
            result[idx] = add(drop_level(low), x2_high)
        return result

    def _make_coeff_pts(self, data_source, coeff_idx: int) -> list:
        pts = [None] * self.total_cts
        for mb in range(self.n_mb):
            for ct_local in range(self.cts_per_mb):
                idx = self._ct_index(mb, ct_local)
                node = CkksPlaintextRingtNode(f'encode_pt_poly_c{coeff_idx}_{mb}_{ct_local}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': poly_op_class,
                        'type': 'coeff_pt',
                        'coeff_idx': coeff_idx,
                        'mb': mb,
                        'ct_local': ct_local,
                        'g': 0,
                    },
                )
                pts[idx] = node
        return pts

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        c0_add_pt = self._make_coeff_pts(data_source, 0)
        c1_pt = self._make_coeff_pts(data_source, 1)
        c2_pt = self._make_coeff_pts(data_source, 2)
        if self.degree == 4:
            c3_pt = self._make_coeff_pts(data_source, 3)
            c4_pt = self._make_coeff_pts(data_source, 4)
        else:
            c3_pt = None
            c4_pt = None
        return self.call(input_cts, c0_add_pt, c1_pt, c2_pt, c3_pt, c4_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {
            'weight': self.degree * self.total_cts,
            'bias': self.total_cts,
            'mask': 0,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        T = self.total_cts
        if self.degree == 2:
            ops[level]['mult'] = T
            ops[level]['mult_plain'] = 2 * T
            ops[level]['add'] = 2 * T
            ops[level]['rescale'] = 3 * T
        else:
            ops[level]['mult'] = 2 * T
            ops[level]['mult_plain'] = 4 * T
            ops[level]['add'] = 4 * T
            ops[level]['rescale'] = 6 * T
        return dict(ops)


ParUpperDiagonalPolyActGamma = ParUpperDiagonalPolyActRNGamma
ParUpperDiagonalPolyActPoly = ParUpperDiagonalPolyActRNPoly
