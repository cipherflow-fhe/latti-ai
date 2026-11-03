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
from inference.model_generator.layers.matrix_param_ops import custom_compute, mult, add


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class _ParBlockColMajorSoftmaxBase:
    op_class = ''

    def _init_layout(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self.n_prepad, self.total_cols = map(int, shape)
        self.m_prepad = int(head_shape[1])
        self.H_prepad, self.n_slot = int(n_heads), int(n_slot)
        self.d = int(block_size)
        assert self.d > 0 and self.d & (self.d-1) == 0 and n_slot >= self.d*self.d
        hp = _next_pow2(self.H_prepad)
        self.u = min(hp, n_slot//(self.d*self.d))
        if self.u == 1: hp = self.H_prepad
        self.G = hp//self.u
        self.Br = math.ceil(self.n_prepad/self.d)
        self.Bc = math.ceil(self.m_prepad/self.d)
        self.cts_per_mb = self.Br*self.Bc*self.G
        self.n_mb = math.ceil(self.total_cols/(self.H_prepad*self.m_prepad))
        self.total_cts = self.n_mb*self.cts_per_mb

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

    def _empty_op_count(self):
        return defaultdict(
            lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'relin': 0, 'add': 0, 'rescale': 0, 'drop_level': 0}
        )


class ParBlockColMajorMultipleSquare(_ParBlockColMajorSoftmaxBase):
    op_class = 'ParBlockColMajorMultipleSquare'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

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


class ParBlockColMajorHeadColSum(_ParBlockColMajorSoftmaxBase):
    op_class = 'ParBlockColMajorHeadColSum'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

    def call(self, input_cts, input_mask_pt, mask_pt):
        assert len(input_cts) == self.total_cts
        partial = []
        for idx, ct in enumerate(input_cts):
            ct = rescale(mult(ct, input_mask_pt[idx]))
            step = 1
            while step < self.d:
                ct = add(ct, rotate_cols(ct, [step*self.d*self.u])[0])
                step <<= 1
            partial.append(ct)
        result = [None]*self.total_cts
        for mb in range(self.n_mb):
            for bi in range(self.Br):
                for g in range(self.G):
                    base = mb*self.cts_per_mb+bi*self.G+g
                    value = partial[base]
                    for bj in range(1,self.Bc):
                        value = add(value, partial[base+bj*self.Br*self.G])
                    for bj in range(self.Bc):
                        idx = base+bj*self.Br*self.G
                        result[idx] = rescale(mult(value,mask_pt[idx]))
        return result

    def call_custom_compute(self, input_cts, data_source):
        return self.call(input_cts,self._make_pt_per_ct(data_source,'input_mask_pt'),
                         self._make_pt_per_ct(data_source,'mask_pt'))

    def get_fhe_op_count(self, level):
        ops = self._empty_op_count()
        n = self.total_cts
        ops[level]['mult_plain'] = ops[level]['rescale'] = n
        ops[level-1]['rotate'] = n*int(math.log2(self.d))
        ops[level-1]['add'] = n*int(math.log2(self.d)) + self.n_mb*self.Br*self.G*(self.Bc-1)
        ops[level-1]['mult_plain'] = ops[level-1]['rescale'] = n
        return dict(ops)


class ParBlockColMajorInverseInit(_ParBlockColMajorSoftmaxBase):
    op_class = 'ParBlockColMajorInverseInit'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

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


class ParBlockColMajorInverseIter(_ParBlockColMajorSoftmaxBase):
    op_class = 'ParBlockColMajorInverseIter'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

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


class ParBlockColMajorCtMul(_ParBlockColMajorSoftmaxBase):
    op_class = 'ParBlockColMajorCtMul'

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

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
