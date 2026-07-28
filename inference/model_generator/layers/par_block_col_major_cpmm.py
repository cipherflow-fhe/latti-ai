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
from inference.model_generator.layers.fhe_op_utils import naf_weight


op_class = 'ParBlockColMajorCPMM'


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class ParBlockColMajorCPMM:
    """Python model-generator counterpart of C++ ParBlockColMajorCPMM.

    Ciphertext-plaintext matrix multiplication for a matrix A in
    par_block_col_major format right-multiplied by a plaintext weight W.

    Modes (auto-detected from W_shape):
      SQUARE : W is n × n         → same number of output CTs as input
      EXPAND : W is n × (K*n)     → K times more output CTs (K separate run_core calls)
      REDUCE : W is (K*n) × n     → K input megablocks summed into one output

    Slot layout: slot[(i + d*j) * S + h] = head h's element at (row=i, col=j).
    Level consumption: 2 levels (1 for block_mult_cpmm + 1 for head-sum mask).
    """

    def __init__(
        self, shape_A: tuple, W_shape: tuple, block_size: int, n_heads: int, n_slot: int, has_bias: bool = False
    ):
        """
        Args:
            shape_A:    (m, n_per_head) — per-head row/col of A.
            W_shape:    (W_rows, W_cols) — shape of the full weight matrix.
            block_size: d, block size for diagonal encoding.
            n_heads:    number of interleaved heads.
            n_slot:     N/2 (number of CKKS slots per ciphertext).
            has_bias:   whether to add bias after matrix multiplication.
        """
        self.m = shape_A[0]
        self.n_per_head = shape_A[1]
        self.d = block_size
        self.n_heads = n_heads
        self.n_slot = n_slot
        self.n_total_per_mb = n_heads * self.n_per_head

        assert self.n_per_head <= self.d, 'per-head width must fit in one block column'

        self.n_h_padded = _next_pow2(n_heads)

        # Chunk sizing — mirrors C++ constructor logic
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
        self.num_block_rows_A = math.ceil(self.m / self.d)

        self.bsgs_bs = math.ceil(math.sqrt(self.d))
        self.bsgs_gs = math.ceil(self.d / self.bsgs_bs)

        # Mode detection — mirrors C++ constructor.
        # Pad both W dimensions to multiples of n_total, then compare to determine mode.
        # Exactly one of K_row, K_col must be 1; K = max(K_row, K_col).
        W_rows, W_cols = W_shape
        n = self.n_total_per_mb
        K_row = math.ceil(W_rows / n)
        K_col = math.ceil(W_cols / n)

        if K_row == K_col:
            self.mode = 'SQUARE'
            assert K_row == 1, 'SQUARE mode requires K_row == K_col == 1'
        elif K_col > K_row:
            self.mode = 'EXPAND'
            assert K_row == 1, 'EXPAND mode requires K_row == 1'
        else:
            self.mode = 'REDUCE'
            assert K_col == 1, 'REDUCE mode requires K_col == 1'
        self.K = max(K_row, K_col)

        self.has_bias = has_bias
        self.n_out_mbs = self.K if self.mode == 'EXPAND' else 1

    # ------------------------------------------------------------------ #
    #  Plaintext node creation                                            #
    # ------------------------------------------------------------------ #

    def make_pt_nodes(self, layer_id):
        """Create pre-declared plaintext nodes for diagonals and the h=0 mask.

        Returns:
            diag_pt:    diag_pt[mb][g][bp][k], total K*G*n_heads*d nodes.
            mask_h0_pt: single CkksPlaintextRingtNode for the head-zero mask.
        """
        diag_pt = []
        for mb in range(self.K):
            diag_pt_mb = []
            for g in range(self.G):
                diag_pt_mbg = []
                for bp in range(self.n_heads):
                    diag_pt_mbgbp = [
                        CkksPlaintextRingtNode(f'cpmm_diag_{layer_id}_{mb}_{g}_{bp}_{k}') for k in range(self.d)
                    ]
                    diag_pt_mbg.append(diag_pt_mbgbp)
                diag_pt_mb.append(diag_pt_mbg)
            diag_pt.append(diag_pt_mb)

        mask_h0_pt = CkksPlaintextRingtNode(f'cpmm_mask_h0_{layer_id}')

        bias_pt = None
        if self.has_bias:
            bias_pt = [
                CkksPlaintextRingtNode(f'cpmm_bias_{layer_id}_{mb}_{bi}_{g}')
                for mb in range(self.n_out_mbs)
                for bi in range(self.num_block_rows_A)
                for g in range(self.G)
            ]

        return diag_pt, mask_h0_pt, bias_pt

    # ------------------------------------------------------------------ #
    #  Primitives — mirror C++ block_mult_cpmm and head_sum              #
    # ------------------------------------------------------------------ #

    def _block_mult_cpmm(self, a_ct, diag_pts_d: list):
        """BSGS block_mult_cpmm. Level L → L-1."""
        d, S = self.d, self.S
        bsgs_bs, bsgs_gs = self.bsgs_bs, self.bsgs_gs
        unit = d * S

        if bsgs_bs > 1:
            baby_rots = [a_ct] + rotate_cols(a_ct, [b * unit for b in range(1, bsgs_bs)])
        else:
            baby_rots = [a_ct]

        total = None
        for g in range(bsgs_gs):
            x_ct_list = []
            w_pt_list = []
            b_end = min(bsgs_bs, d - g * bsgs_bs)
            for b in range(b_end):
                k = g * bsgs_bs + b
                x_ct_list.append(baby_rots[b])
                w_pt_list.append(diag_pts_d[k])
            inner = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            if g > 0:
                inner = rotate_cols(inner, [g * bsgs_bs * unit])[0]
            total = inner if total is None else add(total, inner)
        return rescale(total)

    def _head_sum(self, ct):
        """Tree-reduce sum across S head slots.

        Mirrors C++ head_sum().  log2(S) rotations + log2(S) adds, no level consumed.
        After this, slot h=0 at every (i,j) holds the correct partial sum.
        """
        S = self.S
        if S == 1:
            return ct
        result = ct
        step = 1
        while step < S:
            rotated = rotate_cols(result, [step])[0]
            result = add(result, rotated)
            step *= 2
        return result

    # ------------------------------------------------------------------ #
    #  Core compute — mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, A_cts: list, mb_indices: list, diag_pt, mask_h0_pt) -> list:
        """Process mb_indices megablocks and return output ciphertexts.

        Mirrors C++ run_core().  A_cts is indexed as:
            A_cts[mb_i * cts_per_mb + bi * G + g]
        where cts_per_mb = num_block_rows_A * G.

        Returns list of num_block_rows_A * G CkksCiphertextNode at level L-2.
        """
        K_eff = len(mb_indices)
        cts_per_mb = self.num_block_rows_A * self.G
        S, chunk_size = self.S, self.chunk_size

        C_cts = [None] * (self.num_block_rows_A * self.G)

        for bi in range(self.num_block_rows_A):
            for bp in range(self.n_heads):
                # Step 1+2: block_mult_cpmm + head_sum, sum across all (mb, g)
                full_sum = None
                for mb_i, mb in enumerate(mb_indices):
                    for g in range(self.G):
                        a_idx = mb_i * cts_per_mb + bi * self.G + g
                        tmp = self._block_mult_cpmm(A_cts[a_idx], diag_pt[mb][g][bp])
                        tmp = self._head_sum(tmp)
                        if full_sum is None:
                            full_sum = tmp
                        else:
                            full_sum = add(full_sum, tmp)

                # Step 3: mask h=0  (L-1 → L-2)
                full_sum = mult(full_sum, mask_h0_pt)
                full_sum = rescale(full_sum)

                # Step 4: rotate result to target head slot
                g_out = bp // S
                h_local = bp % S
                if h_local > 0:
                    full_sum = rotate_cols(full_sum, [chunk_size - h_local])[0]

                # Step 5: accumulate into output CT
                out_idx = bi * self.G + g_out
                first_bp_for_g_out = g_out * S
                if bp == first_bp_for_g_out:
                    C_cts[out_idx] = full_sum
                else:
                    C_cts[out_idx] = add(C_cts[out_idx], full_sum)

        return C_cts

    # ------------------------------------------------------------------ #
    #  Public call — pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, A_cts: list, diag_pt, mask_h0_pt, bias_pt=None) -> list:
        """Build computation DAG using pre-declared plaintext nodes.

        Mirrors C++ run().

        Args:
            A_cts:       input CkksCiphertextNode list (FeatureMatEncrypted.data).
                         EXPAND/SQUARE: num_block_rows_A * G entries.
                         REDUCE:        K * num_block_rows_A * G entries.
            diag_pt:     from make_pt_nodes(), shape [K][G][n_heads][d].
            mask_h0_pt:  from make_pt_nodes().
            bias_pt:     from make_pt_nodes(), list of n_out_mbs * R * G nodes (or None).
        Returns:
            list of output CkksCiphertextNode.
        """
        if self.mode == 'EXPAND':
            result = []
            for mb in range(self.K):
                mb_cts = self._run_core(A_cts, [mb], diag_pt, mask_h0_pt)
                result.extend(mb_cts)
        else:
            result = self._run_core(A_cts, list(range(self.K)), diag_pt, mask_h0_pt)

        # Add bias if present (plaintext addition, no level consumed)
        if bias_pt is not None:
            result = [add(ct, bp) for ct, bp in zip(result, bias_pt)]

        return result

    # ------------------------------------------------------------------ #
    #  Custom-compute — inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, A_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute.

        Plaintext nodes are bound to C++ precompute_diagonals() output through
        the custom_compute mechanism.  Functionally equivalent to call().
        """
        diag_pt = []
        for mb in range(self.K):
            diag_pt_mb = []
            for g in range(self.G):
                diag_pt_mbg = []
                for bp in range(self.n_heads):
                    diag_pt_mbgbp = []
                    for k in range(self.d):
                        w_pt = CkksPlaintextRingtNode(f'encode_pt_{mb}_{g}_{bp}_{k}')
                        custom_compute(
                            inputs=[data_source],
                            output=w_pt,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'diag_pt',
                                'mb': mb,
                                'g': g,
                                'bp': bp,
                                'k': k,
                            },
                        )
                        diag_pt_mbgbp.append(w_pt)
                    diag_pt_mbg.append(diag_pt_mbgbp)
                diag_pt_mb.append(diag_pt_mbg)
            diag_pt.append(diag_pt_mb)

        mask_h0_pt = CkksPlaintextRingtNode('encode_pt_mask_h0')
        custom_compute(
            inputs=[data_source],
            output=mask_h0_pt,
            type='encode_pt',
            attributes={'op_class': op_class, 'type': 'mask_h0_pt'},
        )

        bias_pt = None
        if self.has_bias:
            bias_pt = []
            for mb in range(self.n_out_mbs):
                for bi in range(self.num_block_rows_A):
                    for g in range(self.G):
                        bp_node = CkksPlaintextRingtNode(f'encode_pt_bias_{mb}_{bi}_{g}')
                        custom_compute(
                            inputs=[data_source],
                            output=bp_node,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'bias_pt',
                                'mb': mb,
                                'bi': bi,
                                'g': g,
                            },
                        )
                        bias_pt.append(bp_node)

        return self.call(A_cts, diag_pt, mask_h0_pt, bias_pt)

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per (bi, bp) pair, per megablock-group pair (mb_i, g):

          Level L — block_mult_cpmm (BSGS):
            rotate:     K_eff * G * (bsgs_bs-1 + bsgs_gs-1)
            mult_plain: K_eff * G * d
            add:        K_eff * G * (d-1)   [within ct_pt_mult_accumulate]
            rescale:    K_eff * G

          Level L-1 — head_sum + full_sum accumulation + mask:
            rotate:     K_eff * G * log2(S) [head_sum per block_mult result]
            add:        K_eff * G * log2(S) + max(0, K_eff*G - 1)
            mult_plain: 1                   [mask_h0]
            rescale:    1

          Level L-2 — rotate to head position + accumulate into output:
            rotate:     n_heads - G         [bps with h_local != 0]
            add:        n_heads - G         [bps not first in their g_out group]

        For EXPAND mode multiply level-L and L-1 by K (K separate run_core calls),
        level-L-2 is the same per-call cost times K.
        For REDUCE/SQUARE: n_calls=1, K_eff=K.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        if self.mode == 'EXPAND':
            K_eff = 1
            n_calls = self.K
        else:
            K_eff = self.K
            n_calls = 1

        S = self.S
        G = self.G
        d = self.d
        log2_S = int(math.log2(S)) if S > 1 else 0
        R = self.num_block_rows_A
        n_heads = self.n_heads

        # Total (call, bi, bp) iterations
        total_bp = n_calls * R * n_heads

        # Level L: block_mult_cpmm (BSGS)
        bsgs_bs = self.bsgs_bs
        bsgs_gs = self.bsgs_gs
        ops[lv]['rotate'] += total_bp * K_eff * G * (bsgs_bs - 1 + bsgs_gs - 1)
        ops[lv]['mult_plain'] += total_bp * K_eff * G * d
        ops[lv]['add'] += total_bp * K_eff * G * (d - 1)
        ops[lv]['rescale'] += total_bp * K_eff * G
        lv -= 1

        # Level L-1: head_sum (per block_mult result) + full_sum accumulation + mask
        ops[lv]['rotate'] += total_bp * K_eff * G * log2_S
        ops[lv]['add'] += total_bp * (K_eff * G * log2_S + max(0, K_eff * G - 1))
        ops[lv]['mult_plain'] += total_bp
        ops[lv]['rescale'] += total_bp
        lv -= 1

        # Level L-2: rotate to h_local + accumulate into output CT
        # G bps per bi have h_local=0 (no rotate); the rest (n_heads - G) need rotate.
        # Similarly, G bps are "first" in their g_out group; the rest need an add.
        ops[lv]['rotate'] += n_calls * R * (n_heads - G)
        ops[lv]['add'] += n_calls * R * (n_heads - G)

        # Bias addition (still at level L-2, no level consumed)
        if self.has_bias:
            n_bias_cts = self.n_out_mbs * R * G
            ops[lv]['add'] += n_bias_cts

        return dict(ops)
