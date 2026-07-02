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
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts

op_class = 'ParBlockColMajorPolyActRNPoly'
gamma_op_class = 'ParBlockColMajorPolyActRNGamma'


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class ParBlockColMajorPolyActRNGamma:
    """Python model-generator counterpart of C++ ParBlockColMajorPolyActRNGamma."""

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int, K: int = 1):
        assert K > 0
        assert shape[1] % (K * n_heads) == 0

        self.m = shape[0]
        self.total_dim = shape[1]
        self.K = K
        self.cols_per_head = shape[1] // (K * n_heads)
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
            if self.S == 1:
                self.n_h_padded = self.n_heads
            self.G = self.n_h_padded // self.S

        assert n_slot % self.chunk_size == 0
        self.num_chunks = n_slot // self.chunk_size
        self.num_block_rows = math.ceil(self.m / self.d)
        self.num_block_cols = math.ceil(self.cols_per_head / self.d)
        self.cts_per_mb = self.num_block_rows * self.num_block_cols * self.G
        self.total_cts = self.K * self.cts_per_mb

    def call(self, input_cts: list, gamma_pt: list) -> list:
        assert len(input_cts) == self.total_cts
        result = [None] * self.total_cts
        for ct_idx, x in enumerate(input_cts):
            mb = ct_idx // self.cts_per_mb
            local_ct_idx = ct_idx % self.cts_per_mb
            block_idx = local_ct_idx // self.G
            g = local_ct_idx % self.G
            bi = block_idx % self.num_block_rows
            bj = block_idx // self.num_block_rows
            result[ct_idx] = rescale(mult(x, gamma_pt[mb][bi][bj][g]))
        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        gamma_pt = []
        for mb in range(self.K):
            gamma_mb = []
            for bi in range(self.num_block_rows):
                gamma_bi = []
                for bj in range(self.num_block_cols):
                    gamma_row = []
                    for g in range(self.G):
                        gamma_node = CkksPlaintextRingtNode(f'encode_pt_gamma_{mb}_{bi}_{bj}_{g}')
                        custom_compute(
                            inputs=[data_source],
                            output=gamma_node,
                            type='encode_pt',
                            attributes={
                                'op_class': gamma_op_class,
                                'type': 'gamma_pt',
                                'mb': mb,
                                'bi': bi,
                                'bj': bj,
                                'g': g,
                            },
                        )
                        gamma_row.append(gamma_node)
                    gamma_bi.append(gamma_row)
                gamma_mb.append(gamma_bi)
            gamma_pt.append(gamma_mb)
        return self.call(input_cts, gamma_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {'weight': self.total_cts, 'bias': 0, 'mask': 0}
        return memory_from_pt_counts(counts, bytes_per_plaintext)


class ParBlockColMajorPolyActRNPoly:
    """Python model-generator counterpart of C++ ParBlockColMajorPolyActRNPoly.

    Element-wise per-column polynomial evaluation on par_block_col_major packed
    ciphertexts:
        p_j(x) = c0_j + c1_j*x + c2_j*x^2 [+ c3_j*x^3 + c4_j*x^4]

    Level consumption: 2 levels for degree=2, 3 levels for degree=4.
    """

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int, degree: int, K: int = 1):
        """
        Args:
            shape:      (m, total_dim), where total_dim = K * n_heads * cols_per_head.
            block_size: d, block size.
            n_heads:    number of interleaved heads.
            n_slot:     N/2 (number of CKKS slots per ciphertext).
            degree:     polynomial degree, only 2 or 4.
            K:          megablock count.
        """
        assert degree in (2, 4)
        assert K > 0
        assert shape[1] % (K * n_heads) == 0

        self.m = shape[0]
        self.total_dim = shape[1]
        self.K = K
        self.cols_per_head = shape[1] // (K * n_heads)
        self.d = block_size
        self.n_heads = n_heads
        self.n_slot = n_slot
        self.degree = degree

        self.n_h_padded = _next_pow2(n_heads)

        if n_slot >= self.n_h_padded * self.d * self.d:
            self.S = self.n_h_padded
            self.chunk_size = self.n_h_padded * self.d * self.d
            self.G = 1
        else:
            self.S = n_slot // (self.d * self.d)
            self.chunk_size = n_slot
            if self.S == 1:
                self.n_h_padded = self.n_heads
            self.G = self.n_h_padded // self.S

        assert n_slot % self.chunk_size == 0
        self.num_chunks = n_slot // self.chunk_size
        self.num_block_rows = math.ceil(self.m / self.d)
        self.num_block_cols = math.ceil(self.cols_per_head / self.d)
        self.cts_per_mb = self.num_block_rows * self.num_block_cols * self.G
        self.total_cts = self.K * self.cts_per_mb

    def _indices_from_ct_idx(self, ct_idx: int):
        mb = ct_idx // self.cts_per_mb
        local_ct_idx = ct_idx % self.cts_per_mb
        block_idx = local_ct_idx // self.G
        g = local_ct_idx % self.G
        bj = block_idx // self.num_block_rows
        bi = block_idx % self.num_block_rows
        return mb, bi, bj, g

    def call(
        self,
        input_cts: list,
        c2_pt: list,
        c1_pt: list,
        c0_add_pt: list,
        c4_pt: list | None = None,
        c3_pt: list | None = None,
    ) -> list:
        """Build the polynomial-evaluation DAG using pre-declared plaintext nodes."""
        assert len(input_cts) == self.total_cts
        if self.degree == 4:
            assert c4_pt is not None and c3_pt is not None

        result = [None] * self.total_cts

        for ct_idx, x in enumerate(input_cts):
            mb, bi, bj, g = self._indices_from_ct_idx(ct_idx)

            x_sq = rescale(relin(mult(x, x)))

            c2x2 = rescale(mult(x_sq, c2_pt[mb][bi][bj][g]))
            c1x = rescale(mult(x, c1_pt[mb][bi][bj][g]))
            c1x_drop = drop_level(c1x)

            low = add(c1x_drop, c2x2)
            low = add(low, c0_add_pt[mb][bi][bj][g])

            if self.degree == 2:
                result[ct_idx] = low
                continue

            c4x2 = rescale(mult(x_sq, c4_pt[mb][bi][bj][g]))
            c3x = rescale(mult(x, c3_pt[mb][bi][bj][g]))
            c3x_drop = drop_level(c3x)
            high = add(c3x_drop, c4x2)

            x_sq_drop = drop_level(x_sq)
            x2_high = rescale(relin(mult(x_sq_drop, high)))
            low_drop = drop_level(low)
            result[ct_idx] = add(low_drop, x2_high)

        return result

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        """Build the DAG with plaintext coefficients generated by encode_pt."""
        c2_pt = []
        c1_pt = []
        c4_pt = []
        c3_pt = []

        for mb in range(self.K):
            c2_mb = []
            c1_mb = []
            c4_mb = []
            c3_mb = []
            for bi in range(self.num_block_rows):
                c2_bi = []
                c1_bi = []
                c4_bi = []
                c3_bi = []
                for bj in range(self.num_block_cols):
                    c2_row = []
                    c1_row = []
                    c4_row = []
                    c3_row = []
                    for g in range(self.G):
                        c2_node = CkksPlaintextRingtNode(f'encode_pt_poly_c2_{mb}_{bi}_{bj}_{g}')
                        custom_compute(
                            inputs=[data_source],
                            output=c2_node,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'coeff_pt',
                                'coeff_idx': 2,
                                'mb': mb,
                                'bi': bi,
                                'bj': bj,
                                'g': g,
                            },
                        )
                        c2_row.append(c2_node)

                        c1_node = CkksPlaintextRingtNode(f'encode_pt_poly_c1_{mb}_{bi}_{bj}_{g}')
                        custom_compute(
                            inputs=[data_source],
                            output=c1_node,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'coeff_pt',
                                'coeff_idx': 1,
                                'mb': mb,
                                'bi': bi,
                                'bj': bj,
                                'g': g,
                            },
                        )
                        c1_row.append(c1_node)

                        if self.degree == 4:
                            c4_node = CkksPlaintextRingtNode(f'encode_pt_poly_c4_{mb}_{bi}_{bj}_{g}')
                            custom_compute(
                                inputs=[data_source],
                                output=c4_node,
                                type='encode_pt',
                                attributes={
                                    'op_class': op_class,
                                    'type': 'coeff_pt',
                                    'coeff_idx': 4,
                                    'mb': mb,
                                    'bi': bi,
                                    'bj': bj,
                                    'g': g,
                                },
                            )
                            c4_row.append(c4_node)

                            c3_node = CkksPlaintextRingtNode(f'encode_pt_poly_c3_{mb}_{bi}_{bj}_{g}')
                            custom_compute(
                                inputs=[data_source],
                                output=c3_node,
                                type='encode_pt',
                                attributes={
                                    'op_class': op_class,
                                    'type': 'coeff_pt',
                                    'coeff_idx': 3,
                                    'mb': mb,
                                    'bi': bi,
                                    'bj': bj,
                                    'g': g,
                                },
                            )
                            c3_row.append(c3_node)

                    c2_bi.append(c2_row)
                    c1_bi.append(c1_row)
                    if self.degree == 4:
                        c4_bi.append(c4_row)
                        c3_bi.append(c3_row)
                c2_mb.append(c2_bi)
                c1_mb.append(c1_bi)
                if self.degree == 4:
                    c4_mb.append(c4_bi)
                    c3_mb.append(c3_bi)
            c2_pt.append(c2_mb)
            c1_pt.append(c1_mb)
            if self.degree == 4:
                c4_pt.append(c4_mb)
                c3_pt.append(c3_mb)

        c0_add_pt = []
        for mb in range(self.K):
            c0_mb = []
            for bi in range(self.num_block_rows):
                c0_bi = []
                for bj in range(self.num_block_cols):
                    c0_bibj = []
                    for g in range(self.G):
                        c0_node = CkksPlaintextRingtNode(f'encode_pt_poly_c0_{mb}_{bi}_{bj}_{g}')
                        custom_compute(
                            inputs=[data_source],
                            output=c0_node,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'coeff_pt',
                                'coeff_idx': 0,
                                'mb': mb,
                                'bi': bi,
                                'bj': bj,
                                'g': g,
                            },
                        )
                        c0_bibj.append(c0_node)
                    c0_bi.append(c0_bibj)
                c0_mb.append(c0_bi)
            c0_add_pt.append(c0_mb)

        return self.call(
            input_cts, c2_pt, c1_pt, c0_add_pt, c4_pt if self.degree == 4 else None, c3_pt if self.degree == 4 else None
        )

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {
            'weight': self.degree * self.total_cts,
            'bias': self.total_cts,
            'mask': 0,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        ops = defaultdict(
            lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'relin': 0, 'add': 0, 'rescale': 0, 'drop_level': 0}
        )
        n = self.total_cts

        ops[level]['mult'] += n
        ops[level]['relin'] += n
        ops[level]['rescale'] += n
        ops[level]['mult_plain'] += n
        ops[level]['rescale'] += n

        ops[level - 1]['mult_plain'] += n
        ops[level - 1]['rescale'] += n
        ops[level - 1]['drop_level'] += n

        ops[level - 2]['add'] += 2 * n

        if self.degree == 4:
            ops[level]['mult_plain'] += n
            ops[level]['rescale'] += n
            ops[level - 1]['mult_plain'] += n
            ops[level - 1]['rescale'] += n
            ops[level - 1]['drop_level'] += n
            ops[level - 2]['add'] += n
            ops[level - 1]['drop_level'] += n
            ops[level - 2]['mult'] += n
            ops[level - 2]['relin'] += n
            ops[level - 2]['rescale'] += n
            ops[level - 2]['drop_level'] += n
            ops[level - 3]['add'] += n

        return dict(ops)
