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

import sys
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _pow2_step_count(n: int) -> int:
    count = 0
    step = 1
    while step < n:
        count += 1
        step <<= 1
    return count


def _new_ops():
    return defaultdict(
        lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'relin': 0, 'add': 0, 'rescale': 0, 'drop_level': 0}
    )


def _add_drop_level_ops(ops, level: int, n_ct: int, drop: int):
    for i in range(drop):
        ops[level - i]['drop_level'] += n_ct


class _ParBlockColMajorLNBase:
    def _init_layout(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        assert shape[1] % n_heads == 0
        self.m = shape[0]
        self.total_dim = shape[1]
        self.cols_per_head = shape[1] // n_heads
        self.d = block_size
        self.n_heads = n_heads
        self.n_slot = n_slot
        self.n_h_padded = _next_pow2(n_heads)

        if n_slot >= self.n_h_padded * self.d * self.d:
            self.S = self.n_h_padded
            self.chunk_size = self.n_h_padded * self.d * self.d
            self.G = 1
        else:
            self.S = n_slot // (self.d * self.d)
            self.chunk_size = n_slot
            self.G = self.n_h_padded // self.S

        assert n_slot % self.chunk_size == 0
        self.num_chunks = n_slot // self.chunk_size
        self.num_block_rows = math.ceil(self.m / self.d)
        self.num_block_cols = math.ceil(self.cols_per_head / self.d)
        self.total_cts = self.num_block_rows * self.num_block_cols * self.G

    def _intra_block_col_sum(self, ct):
        result = ct
        step = 1
        while step < self.d:
            result = add(result, rotate_cols(result, [step * self.d * self.S])[0])
            step *= 2
        return result

    def _cross_head_mask_replicate(self, ct, h0_mask_pt):
        result = ct
        step = 1
        while step < self.S:
            result = add(result, rotate_cols(result, [step])[0])
            step *= 2

        result = rescale(mult(result, h0_mask_pt))

        step = 1
        while step < self.S:
            result = add(result, rotate_cols(result, [-step])[0])
            step *= 2
        return result

    def _sum_block_cols_and_groups(self, cts: list) -> list:
        result = [None] * self.num_block_rows
        for bi in range(self.num_block_rows):
            row_sum = None
            for bj in range(self.num_block_cols):
                for g in range(self.G):
                    ct_idx = (bi + self.num_block_rows * bj) * self.G + g
                    row_sum = cts[ct_idx] if row_sum is None else add(row_sum, cts[ct_idx])
            result[bi] = row_sum
        return result

    def _make_pt(self, data_source, op_class: str, pt_idx: int, bi: int = 0, bj: int = 0, g: int = 0):
        node = CkksPlaintextRingtNode(f'encode_pt_{op_class}_{pt_idx}_{bi}_{bj}_{g}')
        custom_compute(
            inputs=[data_source],
            output=node,
            type='encode_pt',
            attributes={
                'op_class': op_class,
                'type': 'pt',
                'pt_idx': pt_idx,
                'bi': bi,
                'bj': bj,
                'g': g,
            },
        )
        return node


class ParBlockColMajorLNStats(_ParBlockColMajorLNBase):
    op_class = 'ParBlockColMajorLNStats'

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        self._init_layout(shape, block_size, n_heads, n_slot)

    def call(self, input_cts: list, h0_mask_pt, inv_n_pt, iv_pt, eps_add_pt) -> list:
        assert len(input_cts) == self.total_cts

        col_sum_per_block = [None] * self.total_cts
        x_sq = [None] * self.total_cts
        for ct_idx, ct in enumerate(input_cts):
            col_sum_per_block[ct_idx] = self._intra_block_col_sum(ct)
            x_sq[ct_idx] = rescale(relin(mult(ct, ct)))

        col_sum_x = self._sum_block_cols_and_groups(col_sum_per_block)
        sum_x = [self._cross_head_mask_replicate(ct, h0_mask_pt) for ct in col_sum_x]

        col_sum_x_sq_per_block = [self._intra_block_col_sum(ct) for ct in x_sq]
        col_sum_x_sq = self._sum_block_cols_and_groups(col_sum_x_sq_per_block)
        sum_x_sq_row = [self._cross_head_mask_replicate(ct, h0_mask_pt) for ct in col_sum_x_sq]

        mean_cts = [rescale(mult(ct, inv_n_pt)) for ct in sum_x]

        result = [None] * self.num_block_rows
        for bi in range(self.num_block_rows):
            mean_sq = rescale(relin(mult(mean_cts[bi], mean_cts[bi])))
            E_x_sq = rescale(mult(sum_x_sq_row[bi], inv_n_pt))
            var_ct = sub(E_x_sq, mean_sq)
            a_ct = rescale(mult(var_ct, iv_pt))
            result[bi] = add(a_ct, eps_add_pt)
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        h0_mask_pt = self._make_pt(data_source, self.op_class, 0)
        inv_n_pt = self._make_pt(data_source, self.op_class, 1)
        iv_pt = self._make_pt(data_source, self.op_class, 2)
        eps_add_pt = self._make_pt(data_source, self.op_class, 3)
        return self.call(input_cts, h0_mask_pt, inv_n_pt, iv_pt, eps_add_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = _new_ops()
        n = self.total_cts
        R = self.num_block_rows
        col_steps = _pow2_step_count(self.d)
        head_steps = _pow2_step_count(self.S)
        cross_sum_adds = R * max(0, self.num_block_cols * self.G - 1)

        ops[level]['rotate'] += n * col_steps + R * head_steps
        ops[level]['add'] += n * col_steps + cross_sum_adds + R * head_steps
        ops[level]['mult'] += n
        ops[level]['relin'] += n
        ops[level]['mult_plain'] += R
        ops[level]['rescale'] += n + R

        ops[level - 1]['rotate'] += n * col_steps + 2 * R * head_steps
        ops[level - 1]['add'] += n * col_steps + cross_sum_adds + 2 * R * head_steps
        ops[level - 1]['mult_plain'] += 2 * R
        ops[level - 1]['rescale'] += 2 * R

        ops[level - 2]['rotate'] += R * head_steps
        ops[level - 2]['add'] += R * head_steps
        ops[level - 2]['mult'] += R
        ops[level - 2]['relin'] += R
        ops[level - 2]['mult_plain'] += R
        ops[level - 2]['rescale'] += 2 * R

        ops[level - 3]['add'] += R
        ops[level - 3]['mult_plain'] += R
        ops[level - 3]['rescale'] += R

        ops[level - 4]['add'] += R
        return dict(ops)


class ParBlockColMajorLNXCentered(_ParBlockColMajorLNBase):
    op_class = 'ParBlockColMajorLNXCentered'

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        self._init_layout(shape, block_size, n_heads, n_slot)

    def call(self, input_cts: list, h0_mask_pt, inv_n_pt) -> list:
        assert len(input_cts) == self.total_cts

        col_sum_per_block = [self._intra_block_col_sum(ct) for ct in input_cts]
        col_sum_x = self._sum_block_cols_and_groups(col_sum_per_block)
        sum_x = [self._cross_head_mask_replicate(ct, h0_mask_pt) for ct in col_sum_x]
        mean_cts = [rescale(mult(ct, inv_n_pt)) for ct in sum_x]

        result = [None] * self.total_cts
        for ct_idx, ct in enumerate(input_cts):
            block_idx = ct_idx // self.G
            bi = block_idx % self.num_block_rows
            result[ct_idx] = sub(drop_level(ct, 2), mean_cts[bi])
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        h0_mask_pt = self._make_pt(data_source, self.op_class, 0)
        inv_n_pt = self._make_pt(data_source, self.op_class, 1)
        return self.call(input_cts, h0_mask_pt, inv_n_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = _new_ops()
        n = self.total_cts
        R = self.num_block_rows
        col_steps = _pow2_step_count(self.d)
        head_steps = _pow2_step_count(self.S)
        cross_sum_adds = R * max(0, self.num_block_cols * self.G - 1)

        ops[level]['rotate'] += n * col_steps + R * head_steps
        ops[level]['add'] += n * col_steps + cross_sum_adds + R * head_steps
        ops[level]['mult_plain'] += R
        ops[level]['rescale'] += R
        ops[level]['drop_level'] += n

        ops[level - 1]['rotate'] += R * head_steps
        ops[level - 1]['add'] += R * head_steps
        ops[level - 1]['mult_plain'] += R
        ops[level - 1]['rescale'] += R
        ops[level - 1]['drop_level'] += n

        ops[level - 2]['add'] += n
        return dict(ops)


class ParBlockColMajorLNMinimaxInit:
    op_class = 'ParBlockColMajorLNMinimaxInit'

    def __init__(self, block_size: int, n_slot: int):
        self.d = block_size
        self.n_slot = n_slot
        self.chunk_size = block_size * block_size

    def _make_pt(self, data_source, pt_idx: int):
        node = CkksPlaintextRingtNode(f'encode_pt_{self.op_class}_{pt_idx}')
        custom_compute(
            inputs=[data_source],
            output=node,
            type='encode_pt',
            attributes={'op_class': self.op_class, 'type': 'pt', 'pt_idx': pt_idx, 'bi': 0, 'bj': 0, 'g': 0},
        )
        return node

    def call(self, a_cts: list, c0_add_pt, c1_pt, c2_norm_pt) -> list:
        result = [None] * len(a_cts)
        for bi, a in enumerate(a_cts):
            a_sq = rescale(relin(mult(a, a)))
            c2a2 = rescale(mult(a_sq, c2_norm_pt))
            c1a = rescale(mult(a, c1_pt))
            c1a_drop = drop_level(c1a)
            result[bi] = add(add(c1a_drop, c2a2), c0_add_pt)
        return result

    def call_custom_compute(self, a_cts: list, data_source) -> list:
        c0_add_pt = self._make_pt(data_source, 0)
        c1_pt = self._make_pt(data_source, 1)
        c2_norm_pt = self._make_pt(data_source, 2)
        return self.call(a_cts, c0_add_pt, c1_pt, c2_norm_pt)

    def get_fhe_op_count(self, n_ct: int, level: int) -> dict:
        ops = _new_ops()

        ops[level]['mult'] += n_ct
        ops[level]['relin'] += n_ct
        ops[level]['mult_plain'] += n_ct
        ops[level]['rescale'] += 2 * n_ct

        ops[level - 1]['mult_plain'] += n_ct
        ops[level - 1]['rescale'] += n_ct
        ops[level - 1]['drop_level'] += n_ct

        ops[level - 2]['add'] += 2 * n_ct
        return dict(ops)


class ParBlockColMajorLNGoldschmidt:
    op_class = 'ParBlockColMajorLNGoldschmidt'

    def __init__(self, block_size: int, n_slot: int):
        self.d = block_size
        self.n_slot = n_slot
        self.chunk_size = block_size * block_size

    def _make_pt(self, data_source, pt_idx: int):
        node = CkksPlaintextRingtNode(f'encode_pt_{self.op_class}_{pt_idx}')
        custom_compute(
            inputs=[data_source],
            output=node,
            type='encode_pt',
            attributes={'op_class': self.op_class, 'type': 'pt', 'pt_idx': pt_idx, 'bi': 0, 'bj': 0, 'g': 0},
        )
        return node

    def call(self, y_cts: list, a_cts: list, three_pt, half_norm_pt) -> list:
        result = [None] * len(y_cts)
        for bi, y in enumerate(y_cts):
            a = a_cts[bi]
            if a.level > y.level:
                a = drop_level(a, a.level - y.level)

            ya = rescale(relin(mult(y, a)))
            yy = rescale(relin(mult(y, y)))
            ya_yy = rescale(relin(mult(ya, yy)))

            three_y = rescale(mult(y, three_pt))
            three_y_drop = drop_level(three_y)
            diff = sub(three_y_drop, ya_yy)
            result[bi] = rescale(mult(diff, half_norm_pt))
        return result

    def call_custom_compute(self, y_cts: list, a_cts: list, data_source) -> list:
        three_pt = self._make_pt(data_source, 0)
        half_norm_pt = self._make_pt(data_source, 1)
        return self.call(y_cts, a_cts, three_pt, half_norm_pt)

    def get_fhe_op_count(self, n_ct: int, level: int, a_level: int | None = None) -> dict:
        ops = _new_ops()
        if a_level is not None and a_level > level:
            _add_drop_level_ops(ops, a_level, n_ct, a_level - level)

        ops[level]['mult'] += 2 * n_ct
        ops[level]['relin'] += 2 * n_ct
        ops[level]['mult_plain'] += n_ct
        ops[level]['rescale'] += 3 * n_ct

        ops[level - 1]['mult'] += n_ct
        ops[level - 1]['relin'] += n_ct
        ops[level - 1]['rescale'] += n_ct
        ops[level - 1]['drop_level'] += n_ct

        ops[level - 2]['add'] += n_ct
        ops[level - 2]['mult_plain'] += n_ct
        ops[level - 2]['rescale'] += n_ct
        return dict(ops)


class ParBlockColMajorLNAffine(_ParBlockColMajorLNBase):
    op_class = 'ParBlockColMajorLNAffine'

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        self._init_layout(shape, block_size, n_heads, n_slot)

    def call(self, x_centered: list, y_cts: list, gamma_pt: list, beta_add_pt: list) -> list:
        assert len(x_centered) == self.total_cts
        result = [None] * self.total_cts
        for ct_idx, x_ct in enumerate(x_centered):
            block_idx = ct_idx // self.G
            g = ct_idx % self.G
            bi = block_idx % self.num_block_rows
            bj = block_idx // self.num_block_rows

            yw = rescale(mult(y_cts[bi], gamma_pt[bj][g]))
            xc = x_ct
            if xc.level > yw.level:
                xc = drop_level(xc, xc.level - yw.level)
            out = rescale(relin(mult(xc, yw)))
            result[ct_idx] = add(out, beta_add_pt[bi][bj][g])
        return result

    def call_custom_compute(self, x_centered: list, y_cts: list, data_source) -> list:
        gamma_pt = []
        for bj in range(self.num_block_cols):
            gamma_row = []
            for g in range(self.G):
                gamma_row.append(self._make_pt(data_source, self.op_class, 0, 0, bj, g))
            gamma_pt.append(gamma_row)

        beta_add_pt = []
        for bi in range(self.num_block_rows):
            beta_bi = []
            for bj in range(self.num_block_cols):
                beta_bibj = []
                for g in range(self.G):
                    beta_bibj.append(self._make_pt(data_source, self.op_class, 1, bi, bj, g))
                beta_bi.append(beta_bibj)
            beta_add_pt.append(beta_bi)

        return self.call(x_centered, y_cts, gamma_pt, beta_add_pt)

    def get_fhe_op_count(self, level: int, x_centered_level: int | None = None) -> dict:
        ops = _new_ops()
        n = self.total_cts

        ops[level]['mult_plain'] += n
        ops[level]['rescale'] += n

        if x_centered_level is not None and x_centered_level > level - 1:
            _add_drop_level_ops(ops, x_centered_level, n, x_centered_level - (level - 1))

        ops[level - 1]['mult'] += n
        ops[level - 1]['relin'] += n
        ops[level - 1]['rescale'] += n

        ops[level - 2]['add'] += n
        return dict(ops)
