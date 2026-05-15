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

op_class = 'BlockColMajorTranspose'


class BlockColMajorTranspose:
    """Python model-generator counterpart of C++ BlockColMajorTranspose.

    In-place block transpose for matrices in block_col_major format.
    Each d*d block is transposed via (2d-1) diagonal rotations, and
    the block grid is permuted: block (bi, bj) -> block (bj, bi) in the result.

    Level consumption: 1 level (transpose_on_ct rescale).

    Slot layout per chunk: slot[i + d*j] = element at (row=i, col=j) in block.
    Chunks are tiled across n_slot.
    """

    def __init__(self, shape: tuple, block_size: int, n_slot: int):
        """
        Args:
            shape:      (m, n) -- shape of the input matrix.
            block_size: d, block size (must satisfy d^2 | n_slot).
            n_slot:     N/2 (number of CKKS slots per ciphertext).
        """
        m, n = shape
        self.m = m
        self.n = n
        self.d = block_size
        self.n_slot = n_slot

        self.chunk_size = block_size * block_size
        assert n_slot % self.chunk_size == 0, 'n_slot must be divisible by d^2'
        self.num_chunks = n_slot // self.chunk_size

        self.num_block_rows = math.ceil(m / block_size)
        self.num_block_cols = math.ceil(n / block_size)

        # Total CTs: num_block_rows * num_block_cols (same for input and output)
        self.num_blocks = self.num_block_rows * self.num_block_cols

    # ------------------------------------------------------------------ #
    #  Index helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_index(bi: int, bj: int, num_block_rows: int) -> int:
        """Column-major block index: bi + num_block_rows * bj."""
        return bi + num_block_rows * bj

    # ------------------------------------------------------------------ #
    #  Plaintext node creation                                            #
    # ------------------------------------------------------------------ #

    def make_pt_nodes(self, layer_id):
        """Create pre-declared plaintext nodes for transpose diagonals.

        Returns:
            transpose_diag_pt: list of (2d-1) CkksPlaintextRingtNode,
                               indexed by k_idx = 0..2d-2 (maps to k = k_idx-(d-1)).
        """
        d = self.d
        transpose_diag_pt = [CkksPlaintextRingtNode(f'transpose_diag_{layer_id}_{k_idx}') for k_idx in range(2 * d - 1)]
        return transpose_diag_pt

    # ------------------------------------------------------------------ #
    #  Primitives -- mirror C++ transpose_on_ct                           #
    # ------------------------------------------------------------------ #

    def _transpose_on_ct(self, ct, transpose_diag_pt: list):
        """(2d-1) rotations + (2d-1) pt_muls + accumulate + rescale.

        Mirrors C++ transpose_on_ct().  Level L -> L-1.

        Args:
            ct:                 input CkksCiphertextNode at level L.
            transpose_diag_pt:  list of (2d-1) CkksPlaintextRingtNode.
        """
        d, chunk_size = self.d, self.chunk_size
        x_ct_list = []
        w_pt_list = []

        diag_idx = 0
        for k in range(-(d - 1), d):
            rot_amount = ((d - 1) * k % chunk_size + chunk_size) % chunk_size
            rotated = ct if rot_amount == 0 else rotate_cols(ct, [rot_amount])[0]
            x_ct_list.append(rotated)
            w_pt_list.append(transpose_diag_pt[diag_idx])
            diag_idx += 1

        result = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
        return rescale(result)

    # ------------------------------------------------------------------ #
    #  Core compute -- mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, input_cts: list, transpose_diag_pt: list) -> list:
        """Build computation DAG for matrix transpose.

        Block (bi, bj) in input -> block (bj, bi) in output,
        with each block internally transposed via transpose_on_ct.

        Input block layout: column-major, num_block_rows x num_block_cols.
        Output block layout: column-major, num_block_rows_T x num_block_cols_T,
        where num_block_rows_T = num_block_cols, num_block_cols_T = num_block_rows.

        Returns list of num_blocks CkksCiphertextNode at level L-1.
        """
        num_block_rows_T = self.num_block_cols  # transposed rows = original cols
        result_cts = [None] * self.num_blocks

        for bj in range(self.num_block_cols):
            for bi in range(self.num_block_rows):
                # Source: block (bi, bj) in original
                src_idx = self._block_index(bi, bj, self.num_block_rows)
                # Destination: block (bj, bi) in transposed
                dst_idx = self._block_index(bj, bi, num_block_rows_T)

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

        Plaintext nodes are bound to C++ precompute_diagonals() output through
        the custom_compute mechanism.  Functionally equivalent to call().
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
                    'k_idx': k_idx,
                },
            )
            transpose_diag_pt.append(node)

        return self.call(input_cts, transpose_diag_pt)

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per block (total num_blocks = num_block_rows * num_block_cols):

          Level L -- transpose_on_ct:
            rotate:     (2d-2)    [k=0 has rot_amount=0, skipped]
            mult_plain: (2d-1)
            add:        (2d-2)    [within ct_pt_mult_accumulate]
            rescale:    1
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level
        d = self.d
        n = self.num_blocks

        # Level L: transpose_on_ct per block
        ops[lv]['rotate'] += n * (2 * d - 2)
        ops[lv]['mult_plain'] += n * (2 * d - 1)
        ops[lv]['add'] += n * (2 * d - 2)
        ops[lv]['rescale'] += n

        return dict(ops)
