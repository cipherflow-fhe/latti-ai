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

op_class = 'BlockColMajorCPMM'


class BlockColMajorCPMM:
    """Python model-generator counterpart of C++ BlockColMajorCPMM.

    Ciphertext-plaintext matrix multiplication C = A * B where A is encrypted
    in block_col_major format and B is a plaintext weight matrix.

    Level consumption: 1 level (block_mult_cpmm rescale).

    Slot layout per chunk: slot[i + d*j] = element at (row=i, col=j) in block.
    Chunks are tiled across n_slot.
    """

    def __init__(self, shape_A: tuple, shape_B: tuple, block_size: int, n_slot: int):
        """
        Args:
            shape_A:    (m, n) -- shape of matrix A.
            shape_B:    (n, p) -- shape of matrix B.
            block_size: d, block size (must satisfy d^2 | n_slot).
            n_slot:     N/2 (number of CKKS slots per ciphertext).
        """
        m, n = shape_A
        n2, p = shape_B
        assert n == n2, f'inner dims must match: shape_A[1]={n} != shape_B[0]={n2}'

        self.m = m
        self.n = n
        self.p = p
        self.d = block_size
        self.n_slot = n_slot

        self.chunk_size = block_size * block_size
        assert n_slot % self.chunk_size == 0, 'n_slot must be divisible by d^2'
        self.num_chunks = n_slot // self.chunk_size

        self.num_block_rows_A = math.ceil(m / block_size)
        self.num_block_cols_A = math.ceil(n / block_size)
        self.num_block_rows_B = math.ceil(n / block_size)
        self.num_block_cols_B = math.ceil(p / block_size)

        self.num_result_blocks = self.num_block_rows_A * self.num_block_cols_B

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
        """Create pre-declared plaintext nodes for B diagonals.

        Returns:
            diag_pt: diag_pt[b_idx][k], indexed by B block index and rotation k.
                     Total num_block_rows_B * num_block_cols_B * d nodes.
        """
        total_b_blocks = self.num_block_rows_B * self.num_block_cols_B
        diag_pt = [None] * total_b_blocks
        for bp in range(self.num_block_cols_B):
            for bj in range(self.num_block_rows_B):
                b_idx = self._block_index(bj, bp, self.num_block_rows_B)
                diag_pt[b_idx] = [CkksPlaintextRingtNode(f'cpmm_diag_{layer_id}_{bj}_{bp}_{k}') for k in range(self.d)]
        return diag_pt

    # ------------------------------------------------------------------ #
    #  Primitives -- mirror C++ block_mult_cpmm                           #
    # ------------------------------------------------------------------ #

    def _block_mult_cpmm(self, a_ct, diag_pts_d: list):
        """d rotations + d pt_muls + accumulate + rescale.

        Mirrors C++ block_mult_cpmm().  Level L -> L-1.
        k=0 rotation is 0 and skipped; (d-1) actual rotations.

        Args:
            a_ct:       input CkksCiphertextNode at level L.
            diag_pts_d: list of d CkksPlaintextRingtNode, one per rotation k.
        """
        d, chunk_size = self.d, self.chunk_size
        x_ct_list = []
        w_pt_list = []
        for k in range(d):
            rot_amount = (k * d) % chunk_size
            rotated = a_ct if rot_amount == 0 else rotate_cols(a_ct, [rot_amount])[0]
            x_ct_list.append(rotated)
            w_pt_list.append(diag_pts_d[k])
        result = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
        return rescale(result)

    # ------------------------------------------------------------------ #
    #  Core compute -- mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, A_cts: list, diag_pt: list) -> list:
        """Build computation DAG for C = A * B.

        Block layout is column-major: index = bi + num_block_rows * bj.
        Returns list of num_result_blocks CkksCiphertextNode at level L-1.
        """
        C_cts = [None] * self.num_result_blocks

        for bi in range(self.num_block_rows_A):
            for bp in range(self.num_block_cols_B):
                c_idx = self._block_index(bi, bp, self.num_block_rows_A)
                acc = None
                for bj in range(self.num_block_cols_A):
                    a_idx = self._block_index(bi, bj, self.num_block_rows_A)
                    b_idx = self._block_index(bj, bp, self.num_block_rows_B)
                    product = self._block_mult_cpmm(A_cts[a_idx], diag_pt[b_idx])
                    acc = product if acc is None else add(acc, product)
                C_cts[c_idx] = acc

        return C_cts

    # ------------------------------------------------------------------ #
    #  Public call -- pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, A_cts: list, diag_pt: list) -> list:
        """Build computation DAG using pre-declared plaintext nodes.

        Args:
            A_cts:    input CkksCiphertextNode list (FeatureMatEncrypted.data).
                      num_block_rows_A * num_block_cols_A entries.
            diag_pt:  from make_pt_nodes(), shape [total_b_blocks][d].
        Returns:
            list of output CkksCiphertextNode.
        """
        return self._run_core(A_cts, diag_pt)

    # ------------------------------------------------------------------ #
    #  Custom-compute -- inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, A_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute.

        Plaintext nodes are bound to C++ precompute_diagonals() output through
        the custom_compute mechanism.  Functionally equivalent to call().
        """
        d = self.d
        total_b_blocks = self.num_block_rows_B * self.num_block_cols_B
        diag_pt = [None] * total_b_blocks

        for bp in range(self.num_block_cols_B):
            for bj in range(self.num_block_rows_B):
                b_idx = self._block_index(bj, bp, self.num_block_rows_B)
                diag_pt_b = []
                for k in range(d):
                    node = CkksPlaintextRingtNode(f'encode_pt_{bj}_{bp}_{k}')
                    custom_compute(
                        inputs=[data_source],
                        output=node,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'diag_pt',
                            'bj': bj,
                            'bp': bp,
                            'k': k,
                        },
                    )
                    diag_pt_b.append(node)
                diag_pt[b_idx] = diag_pt_b

        return self.call(A_cts, diag_pt)

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per (bi, bp) output block, summed over bj inner blocks:

          Level L -- block_mult_cpmm:
            rotate:     (d-1)     [k=0 skipped, rot_amount=0]
            mult_plain: d
            add:        (d-1)     [within ct_pt_mult_accumulate]
            rescale:    1

          Over all (bi, bp) pairs and bj inner blocks:
            num_block_mults = num_block_rows_A * num_block_cols_B * num_block_cols_A

          Level L-1 -- accumulate over bj:
            add: (num_block_cols_A - 1) per output block
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level
        d = self.d
        R = self.num_block_rows_A
        C = self.num_block_cols_B
        K = self.num_block_cols_A

        n_block_mult = R * C * K

        # Level L: block_mult_cpmm
        ops[lv]['rotate'] += n_block_mult * (d - 1)
        ops[lv]['mult_plain'] += n_block_mult * d
        ops[lv]['add'] += n_block_mult * (d - 1)
        ops[lv]['rescale'] += n_block_mult
        lv -= 1

        # Level L-1: accumulate over bj
        ops[lv]['add'] += R * C * max(0, K - 1)

        return dict(ops)
