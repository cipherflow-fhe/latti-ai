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


class _ParUpperDiagonalSoftmaxBase:
    op_class = ''

    def _init_layout(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self.n_prepad = int(shape[0])
        self.total_cols = int(shape[1])
        assert self.n_prepad > 0
        assert self.total_cols > 0
        assert head_shape[0] == self.n_prepad
        self.m_prepad = int(head_shape[1])
        self.H_prepad = int(n_heads)
        self.n_slot = int(n_slot)
        assert self.m_prepad > 0
        assert self.H_prepad > 0
        assert self.total_cols == self.H_prepad * self.m_prepad
        assert self.n_prepad == self.m_prepad

        self.H = _next_pow2(self.H_prepad)
        self.m = _next_pow2(self.m_prepad)
        self.n = _next_pow2(self.n_prepad)
        assert self.n >= self.m
        assert self.n % self.m == 0

        self.segment_len = self.H * self.n
        assert self.n_slot % self.segment_len == 0
        self.c = self.n_slot // self.segment_len
        assert self.c > 0
        assert self.m % self.c == 0
        self.total_cts = self.m // self.c

    def _make_pt_per_ct(self, data_source, pt_type: str) -> list:
        pts = [None] * self.total_cts
        source_id = getattr(data_source, 'id', self.op_class)
        for ct_idx in range(self.total_cts):
            node = CkksPlaintextRingtNode(f'encode_pt_{source_id}_{pt_type}_{ct_idx}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={
                    'op_class': self.op_class,
                    'type': pt_type,
                    'ct_idx': ct_idx,
                    'mb': 0,
                    'ct_local': ct_idx,
                    'g': 0,
                },
            )
            pts[ct_idx] = node
        return pts

    def _reduce_local_diags(self, ct):
        result = ct
        step = 1
        while step < self.c:
            result = add(result, rotate_cols(result, [step * self.segment_len])[0])
            step <<= 1
        return result

    def _empty_op_count(self):
        return defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'relin': 0, 'add': 0, 'rescale': 0})


class ParUpperDiagonalAddPt(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalAddPt'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, pt: list) -> list:
        assert len(input_cts) == self.total_cts
        assert len(pt) == self.total_cts
        return [add(input_cts[idx], pt[idx]) for idx in range(self.total_cts)]

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        return self.call(input_cts, self._make_pt_per_ct(data_source, 'pt'))

    def get_fhe_op_count(self, level: int | None = None) -> dict:
        ops = self._empty_op_count()
        ops[level if level is not None else 0]['add'] = self.total_cts
        return dict(ops) if level is not None else dict(ops[0])


class ParUpperDiagonalMultipleSquare(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalMultipleSquare'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, mask_pt: list) -> list:
        assert len(input_cts) == self.total_cts
        assert len(mask_pt) == self.total_cts
        result = [None] * self.total_cts
        for idx, ct in enumerate(input_cts):
            sq = rescale(relin(mult(ct, ct)))
            result[idx] = rescale(mult(sq, mask_pt[idx]))
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        return self.call(input_cts, self._make_pt_per_ct(data_source, 'mask_pt'))

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        n = self.total_cts
        ops[level]['mult'] += n
        ops[level]['relin'] += n
        ops[level]['rescale'] += n
        ops[level - 1]['mult_plain'] += n
        ops[level - 1]['rescale'] += n
        return dict(ops)


class ParUpperDiagonalHeadColSum(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalHeadColSum'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, input_cts: list, mask_pt: list) -> list:
        assert len(input_cts) == self.total_cts
        assert len(mask_pt) == self.total_cts

        partial = [self._reduce_local_diags(ct) for ct in input_cts]
        head_sum = partial[0]
        for idx in range(1, self.total_cts):
            head_sum = add(head_sum, partial[idx])

        return [rescale(mult(head_sum, mask_pt[idx])) for idx in range(self.total_cts)]

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        return self.call(input_cts, self._make_pt_per_ct(data_source, 'mask_pt'))

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        n = self.total_cts
        log_c = int(math.log2(self.c)) if self.c > 1 else 0
        ops[level]['rotate'] += n * log_c
        ops[level]['add'] += n * log_c + max(0, n - 1)
        ops[level]['mult_plain'] += n
        ops[level]['rescale'] += n
        return dict(ops)


class ParUpperDiagonalInverseInit(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalInverseInit'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, b_cts: list, two_pt: list) -> list:
        assert len(b_cts) == self.total_cts
        assert len(two_pt) == self.total_cts
        result = [None] * self.total_cts
        for idx, b in enumerate(b_cts):
            zero = sub(b, b)
            result[idx] = add(sub(zero, b), two_pt[idx])
        return result

    def call_custom_compute(self, b_cts: list, data_source) -> list:
        return self.call(b_cts, self._make_pt_per_ct(data_source, 'two_pt'))

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        n = self.total_cts
        ops[level]['add'] += 3 * n
        return dict(ops)


class ParUpperDiagonalInverseIter(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalInverseIter'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, a_cts: list, b_cts: list, one_pt: list, two_pt: list) -> list:
        assert len(a_cts) == self.total_cts
        assert len(b_cts) == self.total_cts
        assert len(one_pt) == self.total_cts
        assert len(two_pt) == self.total_cts
        result = [None] * self.total_cts
        for idx in range(self.total_cts):
            ba = rescale(relin(mult(b_cts[idx], a_cts[idx])))
            one_a = rescale(mult(a_cts[idx], one_pt[idx]))
            product = rescale(relin(mult(one_a, ba)))
            two_a = rescale(mult(a_cts[idx], two_pt[idx]))
            result[idx] = sub(drop_level(two_a), product)
        return result

    def call_custom_compute(self, a_cts: list, b_cts: list, data_source) -> list:
        one_pt = self._make_pt_per_ct(data_source, 'one_pt')
        two_pt = self._make_pt_per_ct(data_source, 'two_pt')
        return self.call(a_cts, b_cts, one_pt, two_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        n = self.total_cts
        ops[level]['mult'] += n
        ops[level]['relin'] += n
        ops[level]['mult_plain'] += 2 * n
        ops[level]['rescale'] += 3 * n
        ops[level - 1]['mult'] += n
        ops[level - 1]['relin'] += n
        ops[level - 1]['rescale'] += n
        ops[level - 1]['drop_level'] += n
        ops[level - 2]['add'] += n
        return dict(ops)


class ParUpperDiagonalGELU(_ParUpperDiagonalSoftmaxBase):
    op_class = 'ParUpperDiagonalGELU'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def call(self, a_cts: list, b_cts: list, mask_pt: list) -> list:
        assert len(a_cts) == self.total_cts
        assert len(b_cts) == self.total_cts
        assert len(mask_pt) == self.total_cts
        result = [None] * self.total_cts
        for idx in range(self.total_cts):
            product = rescale(relin(mult(a_cts[idx], b_cts[idx])))
            result[idx] = rescale(mult(product, mask_pt[idx]))
        return result

    def call_custom_compute(self, a_cts: list, b_cts: list, data_source) -> list:
        return self.call(a_cts, b_cts, self._make_pt_per_ct(data_source, 'mask_pt'))

    def get_fhe_op_count(self, level: int) -> dict:
        ops = self._empty_op_count()
        n = self.total_cts
        ops[level]['mult'] += n
        ops[level]['relin'] += n
        ops[level]['rescale'] += n
        ops[level - 1]['mult_plain'] += n
        ops[level - 1]['rescale'] += n
        return dict(ops)
