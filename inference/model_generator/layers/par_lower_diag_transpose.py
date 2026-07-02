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
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts


op_class = 'ParLowerDiagTranspose'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ParLowerDiagTranspose:
    """Python model-generator counterpart of C++ ParLowerDiagTranspose."""

    def __init__(self, shape: tuple, n_heads: int, head_dim: int, n_slot: int):
        self.shape = shape
        self.H_prepad = n_heads
        self.m = head_dim
        self.n_prepad = shape[1]
        assert self.H_prepad > 0
        assert self.m > 0 and (self.m & (self.m - 1)) == 0
        assert shape[0] == self.m
        assert self.n_prepad > 0

        self.H = _next_pow2(self.H_prepad)
        self.n = _next_pow2(self.n_prepad)
        assert self.n % self.m == 0

        self.n_slot = n_slot
        self.segment_len = self.H * self.n
        assert self.segment_len > 0
        assert n_slot % self.segment_len == 0
        self.c = n_slot // self.segment_len
        assert self.c > 0 and (self.c & (self.c - 1)) == 0
        assert self.m % self.c == 0
        self.m_c = self.m // self.c

    def _apply_mask(self, ct, mask_pt):
        return rescale(mult(ct, mask_pt))

    def _run_core(self, input_cts: list, transpose_mask_pt: list) -> list:
        assert len(input_cts) == self.m_c

        ct_ell_0 = [None] * self.m_c
        ct_ell_1 = [None] * self.m_c

        for j in range(self.m_c):
            for k in range(self.c):
                source_diag_idx = self.c * j + k
                out_diag_idx = (self.m - (source_diag_idx % self.m)) % self.m
                ell = out_diag_idx // self.c
                out_local_idx = out_diag_idx % self.c
                rot = (k - out_local_idx) * self.segment_len + out_diag_idx * self.H
                ct_rot = input_cts[j] if rot == 0 else rotate_cols(input_cts[j], [rot])[0]

                term0 = self._apply_mask(ct_rot, transpose_mask_pt[out_diag_idx][0])
                ct_ell_0[ell] = term0 if ct_ell_0[ell] is None else add(ct_ell_0[ell], term0)

                term1 = self._apply_mask(ct_rot, transpose_mask_pt[out_diag_idx][1])
                ct_ell_1[ell] = term1 if ct_ell_1[ell] is None else add(ct_ell_1[ell], term1)

        result = []
        for ell in range(self.m_c):
            assert ct_ell_0[ell] is not None
            ct = ct_ell_0[ell]
            if ct_ell_1[ell] is not None:
                ct = add(ct, rotate_cols(ct_ell_1[ell], [-self.segment_len])[0])
            result.append(ct)
        return result

    def call(self, input_cts: list, transpose_mask_pt: list) -> list:
        return self._run_core(input_cts, transpose_mask_pt)

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        transpose_mask_pt = []
        for out_diag_idx in range(self.m):
            row = []
            for mask_idx in range(2):
                node = CkksPlaintextRingtNode(f'encode_pt_transpose_mask_{out_diag_idx}_{mask_idx}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'transpose_mask_pt',
                        'out_diag_idx': out_diag_idx,
                        'mask_idx': mask_idx,
                    },
                )
                row.append(node)
            transpose_mask_pt.append(row)
        return self.call(input_cts, transpose_mask_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {'weight': 0, 'bias': 0, 'mask': 2 * self.m}
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level."""
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        seen_per_ell = [0] * self.m_c
        for j in range(self.m_c):
            for k in range(self.c):
                source_diag_idx = self.c * j + k
                out_diag_idx = (self.m - (source_diag_idx % self.m)) % self.m
                ell = out_diag_idx // self.c
                out_local_idx = out_diag_idx % self.c
                rot = (k - out_local_idx) * self.segment_len + out_diag_idx * self.H
                if rot != 0:
                    ops[lv]['rotate'] += 1

                # Two masks are applied for each source diagonal.
                ops[lv]['mult_plain'] += 2
                ops[lv]['rescale'] += 2

                # Accumulation into ct_ell_0/ct_ell_1 after the first term for each ell.
                if seen_per_ell[ell] > 0:
                    ops[lv - 1]['add'] += 2
                seen_per_ell[ell] += 1

        # Combine the two ell streams; ct_ell_1 is always present for each ell.
        ops[lv - 1]['rotate'] += self.m_c
        ops[lv - 1]['add'] += self.m_c

        return dict(ops)
