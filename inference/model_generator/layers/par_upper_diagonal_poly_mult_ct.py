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


op_class = 'ParUpperDiagonalPolyMultCt'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ParUpperDiagonalPolyMultCt:
    """Python model-generator counterpart of C++ ParUpperDiagonalPolyMultCt."""

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self._init_layout(shape, head_shape, n_heads, n_slot)

    def _init_layout(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self.n_prepad = int(shape[0])
        self.total_cols = int(shape[1])
        assert head_shape[0] == self.n_prepad
        self.m_prepad = int(head_shape[1])
        self.H_prepad = int(n_heads)
        self.n_slot = int(n_slot)

        self.H = _next_pow2(self.H_prepad)
        self.m = _next_pow2(self.m_prepad)
        self.n = _next_pow2(self.n_prepad)
        assert self.n >= self.m
        assert self.n % self.m == 0

        self.packed_extent = self.H_prepad * self.m_prepad
        self.segment_len = self.H * self.n
        assert self.n_slot % self.segment_len == 0
        self.c = self.n_slot // self.segment_len
        assert self.c > 0
        assert self.m % self.c == 0
        self.cts_per_mb = self.m // self.c
        self.n_mb = math.ceil(self.total_cols / self.packed_extent)
        self.total_cts = self.n_mb * self.cts_per_mb

    def _ct_index(self, mb: int, ct_local: int) -> int:
        return mb * self.cts_per_mb + ct_local

    def call(self, half_tanh_cts: list, x_cts: list, one_pt: list, half_pt: list) -> list:
        assert len(half_tanh_cts) == self.total_cts
        assert len(x_cts) == self.total_cts
        assert len(one_pt) == self.total_cts
        assert len(half_pt) == self.total_cts

        result = [None] * self.total_cts
        for idx in range(self.total_cts):
            x_scaled = rescale(mult(x_cts[idx], one_pt[idx]))
            half_tanh_drop = drop_level(half_tanh_cts[idx])
            half_plus = add(half_tanh_drop, half_pt[idx])
            result[idx] = rescale(relin(mult(x_scaled, half_plus)))
        return result

    def _make_pt_per_ct(self, data_source, pt_type: str) -> list:
        pts = [None] * self.total_cts
        for mb in range(self.n_mb):
            for ct_local in range(self.cts_per_mb):
                idx = self._ct_index(mb, ct_local)
                node = CkksPlaintextRingtNode(f'encode_pt_{op_class}_{pt_type}_{mb}_{ct_local}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': pt_type,
                        'mb': mb,
                        'ct_local': ct_local,
                        'ct_idx': idx,
                        'g': 0,
                    },
                )
                pts[idx] = node
        return pts

    def call_custom_compute(self, half_tanh_cts: list, x_cts: list, data_source) -> list:
        one_pt = self._make_pt_per_ct(data_source, 'one_pt')
        half_pt = self._make_pt_per_ct(data_source, 'half_pt')
        return self.call(half_tanh_cts, x_cts, one_pt, half_pt)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'relin': 0, 'add': 0, 'rescale': 0})
        n = self.total_cts
        ops[level]['mult_plain'] += n
        ops[level]['rescale'] += n
        ops[level]['drop_level'] += n
        ops[level - 1]['add'] += n
        ops[level - 1]['mult'] += n
        ops[level - 1]['relin'] += n
        ops[level - 1]['rescale'] += n
        return dict(ops)
