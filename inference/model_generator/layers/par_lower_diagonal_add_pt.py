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


op_class = 'ParLowerDiagonalAddPt'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ParLowerDiagonalAddPt:
    """Python model-generator counterpart of C++ ParLowerDiagonalAddPt."""

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int):
        self.total_rows = shape[0]
        self.n_prepad = shape[1]
        self.m_prepad = head_shape[0]
        self.H_prepad = n_heads
        assert head_shape[1] == self.n_prepad

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
        self.n_mb = math.ceil(self.total_rows / self.packed_extent)
        self.total_cts = self.n_mb * self.cts_per_mb

    def call(self, A_cts: list, pt_nodes: list) -> list:
        assert len(A_cts) == self.total_cts
        assert len(pt_nodes) == self.total_cts
        return [add(ct, pt) for ct, pt in zip(A_cts, pt_nodes)]

    def call_custom_compute(self, A_cts: list, data_source) -> list:
        pt_nodes = []
        for mb in range(self.n_mb):
            for ct_local in range(self.cts_per_mb):
                node = CkksPlaintextRingtNode(f'encode_pt_add_{mb}_{ct_local}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'add_pt',
                        'mb': mb,
                        'ct_local': ct_local,
                    },
                )
                pt_nodes.append(node)
        return self.call(A_cts, pt_nodes)

    def get_fhe_op_count(self, level: int | None = None) -> dict:
        """Count the plaintext additions performed by ``call``."""
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        ops[level if level is not None else 0]['add'] = self.total_cts
        return dict(ops) if level is not None else dict(ops[0])
