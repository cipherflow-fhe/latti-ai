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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class _ParUpperDiagonalBase:
    op_class = ''

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

    def _make_pt(self, data_source, pt_idx: int, mb: int = 0, ct_local: int = 0):
        node = CkksPlaintextRingtNode(f'encode_pt_{self.op_class}_{pt_idx}_{mb}_{ct_local}')
        custom_compute(
            inputs=[data_source],
            output=node,
            type='encode_pt',
            attributes={
                'op_class': self.op_class,
                'type': 'pt',
                'pt_idx': pt_idx,
                'mb': mb,
                'ct_local': ct_local,
                'g': 0,
            },
        )
        return node

    def _make_pt_per_ct(self, data_source, pt_idx: int) -> list:
        pts = [None] * self.total_cts
        for mb in range(self.n_mb):
            for ct_local in range(self.cts_per_mb):
                pts[self._ct_index(mb, ct_local)] = self._make_pt(data_source, pt_idx, mb, ct_local)
        return pts

    def _reduce_cols_in_ct(self, ct, h0_mask_pt):
        result = ct
        step = 1
        while step < self.c:
            result = add(result, rotate_cols(result, [step * self.segment_len])[0])
            step <<= 1
        step = 1
        while step < self.H:
            result = add(result, rotate_cols(result, [step])[0])
            step <<= 1

        masked = rescale(mult(result, h0_mask_pt))
        replicated = masked
        step = 1
        while step < self.H:
            replicated = add(replicated, rotate_cols(replicated, [-step])[0])
            step <<= 1
        return replicated

    def _empty_op_count(self):
        return defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})

    def _reduce_cols_op_count_per_ct(self) -> dict[str, int]:
        log_c = int(math.log2(self.c)) if self.c > 1 else 0
        log_h = int(math.log2(self.H)) if self.H > 1 else 0
        reduce_steps = log_c + 2 * log_h
        return {'rotate': reduce_steps, 'mult_plain': 1, 'mult': 0, 'add': reduce_steps, 'rescale': 1}


class ParUpperDiagonalLNStats(_ParUpperDiagonalBase):
    op_class = 'ParUpperDiagonalLNStats'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, h0_mask_pt, inv_n_pt: list, iv_pt: list, eps_add_pt: list) -> list:
        assert len(input_cts) == self.total_cts

        partial_sum_x = [self._reduce_cols_in_ct(ct, h0_mask_pt) for ct in input_cts]
        x_sq = [rescale(relin(mult(ct, ct))) for ct in input_cts]

        sum_x = partial_sum_x[0]
        for idx in range(1, self.total_cts):
            sum_x = add(sum_x, partial_sum_x[idx])

        partial_sum_x_sq = [self._reduce_cols_in_ct(ct, h0_mask_pt) for ct in x_sq]
        sum_x_sq = partial_sum_x_sq[0]
        for idx in range(1, self.total_cts):
            sum_x_sq = add(sum_x_sq, partial_sum_x_sq[idx])

        mean_cts = [rescale(mult(sum_x, inv_n_pt[idx])) for idx in range(self.total_cts)]

        result = [None] * self.total_cts
        for idx in range(self.total_cts):
            mean_sq = rescale(relin(mult(mean_cts[idx], mean_cts[idx])))
            E_x_sq = rescale(mult(sum_x_sq, inv_n_pt[idx]))
            var = sub(E_x_sq, mean_sq)
            a_ct = rescale(mult(var, iv_pt[idx]))
            result[idx] = add(a_ct, eps_add_pt[idx])
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        h0_mask_pt = self._make_pt(data_source, 0)
        inv_n_pt = self._make_pt_per_ct(data_source, 1)
        iv_pt = self._make_pt_per_ct(data_source, 2)
        eps_add_pt = self._make_pt_per_ct(data_source, 3)
        return self.call(input_cts, h0_mask_pt, inv_n_pt, iv_pt, eps_add_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        T = self.total_cts
        reduce_ops = self._reduce_cols_op_count_per_ct()
        ops[level]['rotate'] += 2 * T * reduce_ops['rotate']
        ops[level]['mult_plain'] += 2 * T * reduce_ops['mult_plain']
        ops[level]['add'] += 2 * T * reduce_ops['add'] + 2 * max(0, T - 1)
        ops[level]['rescale'] += 2 * T * reduce_ops['rescale']

        ops[level]['mult'] += 2 * T
        ops[level]['mult_plain'] += 3 * T
        ops[level]['add'] += 2 * T
        ops[level]['rescale'] += 5 * T
        return dict(ops)


class ParUpperDiagonalLNXCentered(_ParUpperDiagonalBase):
    op_class = 'ParUpperDiagonalLNXCentered'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, h0_mask_pt, inv_n_pt: list) -> list:
        assert len(input_cts) == self.total_cts

        partial_sum_x = [self._reduce_cols_in_ct(ct, h0_mask_pt) for ct in input_cts]
        sum_x = partial_sum_x[0]
        for idx in range(1, self.total_cts):
            sum_x = add(sum_x, partial_sum_x[idx])

        mean_cts = [rescale(mult(sum_x, inv_n_pt[idx])) for idx in range(self.total_cts)]
        return [sub(drop_level(input_cts[idx], 2), mean_cts[idx]) for idx in range(self.total_cts)]

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        h0_mask_pt = self._make_pt(data_source, 0)
        inv_n_pt = self._make_pt_per_ct(data_source, 1)
        return self.call(input_cts, h0_mask_pt, inv_n_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        T = self.total_cts
        reduce_ops = self._reduce_cols_op_count_per_ct()
        ops[level]['rotate'] += T * reduce_ops['rotate']
        ops[level]['mult_plain'] += T * reduce_ops['mult_plain']
        ops[level]['add'] += T * reduce_ops['add'] + max(0, T - 1)
        ops[level]['rescale'] += T * reduce_ops['rescale']

        ops[level]['mult_plain'] += T
        ops[level]['add'] += T
        ops[level]['rescale'] += T
        return dict(ops)


class ParUpperDiagonalLNMinimaxInit(_ParUpperDiagonalBase):
    op_class = 'ParUpperDiagonalLNMinimaxInit'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, a_cts: list, c0_add_pt: list, c1_pt: list, c2_norm_pt: list) -> list:
        assert len(a_cts) == self.total_cts
        result = [None] * self.total_cts
        for idx, a in enumerate(a_cts):
            a_sq = rescale(relin(mult(a, a)))
            c2a2 = rescale(mult(a_sq, c2_norm_pt[idx]))
            c1a = rescale(mult(a, c1_pt[idx]))
            y0 = add(drop_level(c1a), c2a2)
            result[idx] = add(y0, c0_add_pt[idx])
        return result

    def call_custom_compute(self, a_cts: list, data_source) -> list:
        c0_add_pt = self._make_pt_per_ct(data_source, 0)
        c1_pt = self._make_pt_per_ct(data_source, 1)
        c2_norm_pt = self._make_pt_per_ct(data_source, 2)
        return self.call(a_cts, c0_add_pt, c1_pt, c2_norm_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        T = self.total_cts
        ops[level]['mult'] = T
        ops[level]['mult_plain'] = 2 * T
        ops[level]['add'] = 2 * T
        ops[level]['rescale'] = 3 * T
        return dict(ops)


class ParUpperDiagonalLNGoldschmidt(_ParUpperDiagonalBase):
    op_class = 'ParUpperDiagonalLNGoldschmidt'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, y_cts: list, a_cts: list, three_pt: list, half_norm_pt: list) -> list:
        assert len(y_cts) == self.total_cts
        assert len(a_cts) == self.total_cts
        result = [None] * self.total_cts
        for idx, y in enumerate(y_cts):
            a = a_cts[idx]
            if a.level > y.level:
                a = drop_level(a, a.level - y.level)

            ya = rescale(relin(mult(y, a)))
            yy = rescale(relin(mult(y, y)))
            ya_yy = rescale(relin(mult(ya, yy)))

            three_y = rescale(mult(y, three_pt[idx]))
            diff = sub(drop_level(three_y), ya_yy)
            result[idx] = rescale(mult(diff, half_norm_pt[idx]))
        return result

    def call_custom_compute(self, y_cts: list, a_cts: list, data_source) -> list:
        three_pt = self._make_pt_per_ct(data_source, 0)
        half_norm_pt = self._make_pt_per_ct(data_source, 1)
        return self.call(y_cts, a_cts, three_pt, half_norm_pt)

    def get_fhe_op_count(self, level: int, a_level: int | None = None) -> dict:
        ops = self._empty_op_count()
        T = self.total_cts
        ops[level]['mult'] = 3 * T
        ops[level]['mult_plain'] = 2 * T
        ops[level]['add'] = T
        ops[level]['rescale'] = 5 * T
        return dict(ops)


class ParUpperDiagonalLNAffine(_ParUpperDiagonalBase):
    op_class = 'ParUpperDiagonalLNAffine'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, x_centered: list, y_cts: list, gamma_pt: list, beta_add_pt: list) -> list:
        assert len(x_centered) == self.total_cts
        assert len(y_cts) == self.total_cts
        result = [None] * self.total_cts
        for idx, x_ct in enumerate(x_centered):
            yw = rescale(mult(y_cts[idx], gamma_pt[idx]))
            xc = x_ct
            if xc.level > yw.level:
                xc = drop_level(xc, xc.level - yw.level)
            out = rescale(relin(mult(xc, yw)))
            result[idx] = add(out, beta_add_pt[idx])
        return result

    def call_custom_compute(self, x_centered: list, y_cts: list, data_source) -> list:
        gamma_pt = self._make_pt_per_ct(data_source, 0)
        beta_add_pt = self._make_pt_per_ct(data_source, 1)
        return self.call(x_centered, y_cts, gamma_pt, beta_add_pt)

    def get_fhe_op_count(self, level: int, x_centered_level: int | None = None) -> dict:
        ops = self._empty_op_count()
        T = self.total_cts
        ops[level]['mult'] = T
        ops[level]['mult_plain'] = T
        ops[level]['add'] = T
        ops[level]['rescale'] = 2 * T
        return dict(ops)
