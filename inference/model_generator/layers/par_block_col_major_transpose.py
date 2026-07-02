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

op_class = 'ParBlockColMajorTranspose'


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class ParBlockColMajorTranspose:
    """Python model-generator counterpart of C++ ParBlockColMajorTranspose.

    In-place block transpose for matrices in par_block_col_major format
    (multi-head interleaved).  Each d*d block is transposed via (2d-1)
    diagonal rotations, and the block grid is permuted:
    block (bi, bj) -> block (bj, bi) in the result.

    Level consumption: 1 level (transpose_on_ct rescale).

    Slot layout per chunk: slot[(i + d*j) * S + h] = head h's element
    at (row=i, col=j) in block.  Chunks are tiled across n_slot.
    """

    def __init__(self, shape: tuple, block_size: int, n_heads: int, n_slot: int):
        """
        Args:
            shape:      (m, n_per_head) -- per-head shape of the input matrix.
            block_size: d, block size (must satisfy d^2 * S | n_slot).
            n_heads:    number of interleaved heads.
            n_slot:     N/2 (number of CKKS slots per ciphertext).
        """
        m, n = shape
        self.m = m
        self.n = n
        self.d = block_size
        self.n_heads = n_heads
        self.n_slot = n_slot

        self.n_h_padded = _next_pow2(n_heads)

        # Chunk sizing -- mirrors C++ constructor logic
        if n_slot >= self.n_h_padded * self.d * self.d:
            self.S = self.n_h_padded  # n_blocks_per_chunk
            self.chunk_size = self.n_h_padded * self.d * self.d
            self.G = 1  # n_cts_per_block_idx
        else:
            self.S = n_slot // (self.d * self.d)
            self.chunk_size = n_slot
            if self.S == 1:
                self.n_h_padded = self.n_heads
            self.G = self.n_h_padded // self.S

        assert n_slot % self.chunk_size == 0
        self.num_chunks = n_slot // self.chunk_size

        self.num_block_rows = math.ceil(m / block_size)
        self.num_block_cols = math.ceil(n / block_size)

        # Total CTs: num_block_rows * num_block_cols * G
        self.num_blocks = self.num_block_rows * self.num_block_cols

        n_t = 2 * self.d - 1
        self.bsgs_bs_t = math.ceil(math.sqrt(n_t))
        self.bsgs_gs_t = math.ceil(n_t / self.bsgs_bs_t)

    # ------------------------------------------------------------------ #
    #  Index helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_index(bi: int, bj: int, num_block_rows: int) -> int:
        """Column-major block index: bi + num_block_rows * bj."""
        return bi + num_block_rows * bj

    # ------------------------------------------------------------------ #
    #  Primitives -- mirror C++ transpose_on_ct                           #
    # ------------------------------------------------------------------ #

    def _transpose_on_ct(self, ct, transpose_diag_pt: list):
        """BSGS transpose_on_ct. Level L -> L-1."""
        d = self.d
        S = self.S
        chunk_size = self.chunk_size
        bsgs_bs, bsgs_gs = self.bsgs_bs_t, self.bsgs_gs_t
        n_t = 2 * d - 1
        unit = (d - 1) * S

        if bsgs_bs > 1:
            baby_rots = [ct] + rotate_cols(ct, [b * unit for b in range(1, bsgs_bs)])
        else:
            baby_rots = [ct]

        total = None
        for g in range(bsgs_gs):
            x_ct_list, w_pt_list = [], []
            b_end = min(bsgs_bs, n_t - g * bsgs_bs)
            for b in range(b_end):
                diag_idx = g * bsgs_bs + b
                x_ct_list.append(baby_rots[b])
                w_pt_list.append(transpose_diag_pt[diag_idx])
            inner = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            giant_rot = ((d - 1) * g * bsgs_bs - (d - 1) ** 2) * S
            giant_rot = ((giant_rot % chunk_size) + chunk_size) % chunk_size
            if giant_rot != 0:
                inner = rotate_cols(inner, [giant_rot])[0]
            total = inner if total is None else add(total, inner)
        return rescale(total)

    # ------------------------------------------------------------------ #
    #  Core compute -- mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, input_cts: list, transpose_diag_pt: list) -> list:
        """Build computation DAG for matrix transpose.

        Block (bi, bj) in input -> block (bj, bi) in output,
        with each block internally transposed via transpose_on_ct.

        Input block layout: column-major, num_block_rows x num_block_cols, G CTs per block.
        Output block layout: column-major, num_block_rows_T x num_block_cols_T, G CTs per block.

        Returns list of num_blocks * G CkksCiphertextNode at level L-1.
        """
        num_block_rows_T = self.num_block_cols  # transposed rows = original cols
        num_result_vecs = self.num_blocks * self.G
        result_cts = [None] * num_result_vecs

        for bj in range(self.num_block_cols):
            for bi in range(self.num_block_rows):
                for g in range(self.G):
                    # Source: block (bi, bj) in original, group g
                    src_block = self._block_index(bi, bj, self.num_block_rows)
                    src_idx = src_block * self.G + g
                    # Destination: block (bj, bi) in transposed, group g
                    dst_block = self._block_index(bj, bi, num_block_rows_T)
                    dst_idx = dst_block * self.G + g

                    result_cts[dst_idx] = self._transpose_on_ct(input_cts[src_idx], transpose_diag_pt)

        return result_cts

    # ------------------------------------------------------------------ #
    #  Public call -- pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, input_cts: list, transpose_diag_pt: list) -> list:
        """Build computation DAG using pre-declared plaintext nodes.

        Args:
            input_cts:          input CkksCiphertextNode list.
            transpose_diag_pt:  from make_pt_nodes(), (2d-1) nodes.
        Returns:
            list of output CkksCiphertextNode.
        """
        return self._run_core(input_cts, transpose_diag_pt)

    # ------------------------------------------------------------------ #
    #  Custom-compute -- inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute.

        Plaintext nodes are bound to C++ generate_transpose_diag_pt() output
        through the custom_compute mechanism.
        """
        d = self.d
        transpose_diag_pt = []
        for k_idx in range(2 * d - 1):
            node = CkksPlaintextRingtNode(f'encode_pt_transpose_diag_{k_idx}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={
                    'op_class': op_class,
                    'type': 'transpose_diag_pt',
                    'i': k_idx,
                },
            )
            transpose_diag_pt.append(node)

        return self.call(input_cts, transpose_diag_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        counts = {'weight': 0, 'bias': 0, 'mask': 2 * self.d - 1}
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per (bi, bj, g) CT, _transpose_on_ct iterates over 2d-1 diagonals.
        k=0 gives rot_amount=0 and is skipped; (2d-2) actual rotations.

          Level L — transpose_on_ct:
            rotate:     (2*d - 2)
            mult_plain: (2*d - 1)
            add:        (2*d - 2)   [within ct_pt_mult_accumulate]
            rescale:    1

          Total CTs = num_blocks * G.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level
        d = self.d
        G = self.G
        total_cts = self.num_blocks * G

        # Level L — transpose_on_ct (BSGS):
        transpose_rots = self.bsgs_bs_t - 1 + self.bsgs_gs_t  # all giant groups may have nonzero rot
        ops[lv]['rotate'] += total_cts * transpose_rots
        ops[lv]['mult_plain'] += total_cts * (2 * d - 1)
        ops[lv]['add'] += total_cts * (2 * d - 2)
        ops[lv]['rescale'] += total_cts

        return dict(ops)
