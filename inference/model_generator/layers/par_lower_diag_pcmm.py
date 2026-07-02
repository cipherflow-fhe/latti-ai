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


op_class = 'ParLowerDiagPCMM'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ParLowerDiagPCMM:
    """Python model-generator counterpart of C++ ParLowerDiagPCMM."""

    def __init__(
        self,
        shape_X_T: tuple,
        W_T_shape: tuple,
        n_heads: int,
        head_dim: int,
        n_slot: int,
        has_bias: bool = False,
    ):
        self.H_prepad = n_heads
        self.m = head_dim
        assert self.H_prepad > 0
        assert self.m > 0 and (self.m & (self.m - 1)) == 0

        self.d_prepad = self.H_prepad * self.m
        self.in_rows = shape_X_T[0]
        self.n_prepad = shape_X_T[1]

        self.H = _next_pow2(self.H_prepad)
        self.n = _next_pow2(self.n_prepad)
        self.d = self.H * self.m
        assert self.n >= self.m
        assert self.n % self.m == 0

        self.n_slot = n_slot
        self.segment_len = self.H * self.n
        assert n_slot % self.segment_len == 0
        self.c = n_slot // self.segment_len
        assert self.m % self.c == 0
        self.m_c = self.m // self.c

        self.W_T_rows = W_T_shape[0]
        self.W_T_cols = W_T_shape[1]
        assert self.in_rows == self.W_T_cols
        self.out_rows = self.W_T_rows

        self.K_row = math.ceil(self.W_T_rows / self.d_prepad)
        self.K_col = math.ceil(self.W_T_cols / self.d_prepad)
        if self.K_row == self.K_col:
            self.mode = 'SQUARE'
            assert self.K_row == 1
        elif self.K_row > self.K_col:
            self.mode = 'EXPAND'
            assert self.K_col == 1
        else:
            self.mode = 'REDUCE'
            assert self.K_row == 1
        self.K = max(self.K_row, self.K_col)
        self.has_bias = has_bias
        self.n_out_mbs = self.K if self.mode == 'EXPAND' else 1

    def _run_core(self, input_cts: list, mb_indices: list, pt_A: list, mask_wrap_pt: list) -> list:
        reduced = [None] * self.m_c

        for local_mb, weight_mb in enumerate(mb_indices):
            input_offset = local_mb * self.m_c

            ct_Br = []
            for j in range(self.m_c):
                row = [input_cts[input_offset + j]]
                for ell in range(1, self.c):
                    row.append(rotate_cols(input_cts[input_offset + j], [self.segment_len * ell])[0])
                ct_Br.append(row)

            ct_ir = []
            for i in range(self.H):
                row_i = []
                for r in range(self.m_c):
                    acc = None
                    for j in range(self.m_c):
                        for ell in range(self.c):
                            pt_idx = (j * self.c + ell) * self.m_c + r
                            product = mult(ct_Br[j][ell], pt_A[weight_mb][i][pt_idx])
                            acc = product if acc is None else add(acc, product)
                    assert acc is not None
                    row_i.append(rescale(acc))
                ct_ir.append(row_i)

            ct_C = []
            for r in range(self.m_c):
                acc = drop_level(ct_ir[0][r])
                for i_idx in range(self.H - 1):
                    i = i_idx + 1
                    ct_R = rescale(mult(ct_ir[i][r], mask_wrap_pt[i_idx]))
                    ct_i_drop = drop_level(ct_ir[i][r])
                    ct_L = sub(ct_i_drop, ct_R)
                    ct_R_rot = rotate_cols(ct_R, [self.H - i])[0]
                    ct_L_rot = rotate_cols(ct_L, [-i])[0]
                    acc = add(acc, add(ct_R_rot, ct_L_rot))
                ct_C.append(acc)

            if reduced[0] is None:
                reduced = ct_C
            else:
                reduced = [add(reduced[r], ct_C[r]) for r in range(self.m_c)]

        return reduced

    def call(self, input_cts: list, pt_A: list, mask_wrap_pt: list, bias_pt: list | None = None) -> list:
        if self.mode == 'EXPAND':
            result = []
            for mb in range(self.K):
                mb_cts = self._run_core(input_cts, [mb], pt_A, mask_wrap_pt)
                if bias_pt is not None:
                    mb_cts = [add(ct, bias_pt[mb * self.m_c + r]) for r, ct in enumerate(mb_cts)]
                result.extend(mb_cts)
            return result

        result = self._run_core(input_cts, list(range(self.K)), pt_A, mask_wrap_pt)
        if bias_pt is not None:
            result = [add(ct, bias_pt[r]) for r, ct in enumerate(result)]
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        pt_A = []
        for mb in range(self.K):
            pt_mb = []
            for i in range(self.H):
                pt_i = []
                for j in range(self.m_c):
                    for ell in range(self.c):
                        for r in range(self.m_c):
                            node = CkksPlaintextRingtNode(f'encode_pt_A_{mb}_{i}_{j}_{ell}_{r}')
                            custom_compute(
                                inputs=[data_source],
                                output=node,
                                type='encode_pt',
                                attributes={
                                    'op_class': op_class,
                                    'type': 'pt_A',
                                    'mb': mb,
                                    'i': i,
                                    'j': j,
                                    'ell': ell,
                                    'r': r,
                                },
                            )
                            pt_i.append(node)
                pt_mb.append(pt_i)
            pt_A.append(pt_mb)

        mask_wrap_pt = []
        for i_idx in range(self.H - 1):
            node = CkksPlaintextRingtNode(f'encode_pt_mask_wrap_{i_idx}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'mask_wrap_pt', 'i_idx': i_idx},
            )
            mask_wrap_pt.append(node)

        bias_pt = None
        if self.has_bias:
            bias_pt = []
            for mb in range(self.n_out_mbs):
                for r in range(self.m_c):
                    node = CkksPlaintextRingtNode(f'encode_pt_bias_{mb}_{r}')
                    custom_compute(
                        inputs=[data_source],
                        output=node,
                        type='encode_pt',
                        attributes={'op_class': op_class, 'type': 'bias_pt', 'mb': mb, 'r': r},
                    )
                    bias_pt.append(node)

        return self.call(input_cts, pt_A, mask_wrap_pt, bias_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        bias_count = self.n_out_mbs * self.m_c if self.has_bias else 0
        counts = {
            'weight': self.K * self.H * self.m_c * self.m,
            'bias': bias_count,
            'mask': self.H - 1,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        The count follows ``_run_core``: input replication rotations and
        plaintext matrix products happen at the input level; wrap masking
        consumes one more level; rotations/additions that route wrapped pieces
        are counted at the post-mask level.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})

        n_processed_mbs = self.K
        n_core_calls = self.K if self.mode == 'EXPAND' else 1
        lv = level

        # Build ct_Br and ct_ir for each processed weight megablock.
        ops[lv]['rotate'] += n_processed_mbs * self.m_c * max(0, self.c - 1)
        ops[lv]['mult_plain'] += n_processed_mbs * self.H * self.m_c * self.m
        ops[lv]['add'] += n_processed_mbs * self.H * self.m_c * max(0, self.m - 1)
        ops[lv]['rescale'] += n_processed_mbs * self.H * self.m_c

        # Apply wrap masks to nonzero head offsets.
        wrap_terms = n_processed_mbs * self.m_c * max(0, self.H - 1)
        ops[lv - 1]['mult_plain'] += wrap_terms
        ops[lv - 1]['rescale'] += wrap_terms

        # Route wrapped pieces and accumulate within each output ciphertext.
        ops[lv - 2]['rotate'] += wrap_terms * 2
        ops[lv - 2]['add'] += wrap_terms * 3

        # REDUCE mode combines multiple processed megablocks in one core call.
        if self.mode != 'EXPAND':
            ops[lv - 2]['add'] += max(0, self.K - 1) * self.m_c

        if self.has_bias:
            ops[lv - 2]['add'] += n_core_calls * self.m_c

        return dict(ops)
