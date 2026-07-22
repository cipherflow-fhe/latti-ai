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

op_class = 'ParBlockColMajorCCMM'


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class ParBlockColMajorCCMM:
    """Python model-generator counterpart of C++ ParBlockColMajorCCMM.

    Ciphertext-ciphertext matrix multiplication C = A * B where both
    operands are encrypted in par_block_col_major format (multi-head interleaved).

    Level consumption: 3 levels per block_mult (sigma L->L-1, tau L->L-1,
    psi L-1->L-2, ct*ct L-2->L-3).

    Slot layout per chunk: slot[(i + d*j) * S + h] = head h's element
    at (row=i, col=j) in block.
    """

    def __init__(self, shape_A: tuple, shape_B: tuple, block_size: int, n_heads: int, n_slot: int):
        """
        Args:
            shape_A:    (m, n) -- per-head shape of matrix A.
            shape_B:    (n, p) -- per-head shape of matrix B.
            block_size: d, block size (must satisfy d^2 * S | n_slot).
            n_heads:    number of interleaved heads.
            n_slot:     N/2 (number of CKKS slots per ciphertext).
        """
        m, n = shape_A
        n2, p = shape_B
        assert n == n2, f'inner dims must match: shape_A[1]={n} != shape_B[0]={n2}'

        self.m = m
        self.n = n
        self.p = p
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

        self.num_block_rows_A = math.ceil(m / block_size)
        self.num_block_cols_A = math.ceil(n / block_size)
        self.num_block_rows_B = math.ceil(n / block_size)
        self.num_block_cols_B = math.ceil(p / block_size)

        self.num_result_blocks = self.num_block_rows_A * self.num_block_cols_B

        self.bsgs_bs_sigma = math.ceil(math.sqrt(self.d))
        self.bsgs_gs_sigma = math.ceil(self.d / self.bsgs_bs_sigma)
        n_tau = 2 * self.d - 1
        self.bsgs_bs_tau = math.ceil(math.sqrt(n_tau))
        self.bsgs_gs_tau = math.ceil(n_tau / self.bsgs_bs_tau)

    # ------------------------------------------------------------------ #
    #  Index helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_index(bi: int, bj: int, num_block_rows: int) -> int:
        """Column-major block index: bi + num_block_rows * bj."""
        return bi + num_block_rows * bj

    # ------------------------------------------------------------------ #
    #  Primitives -- mirror C++ methods                                    #
    # ------------------------------------------------------------------ #

    def _sigma(self, a_ct, sigma_pt: list):
        """BSGS sigma. Level L -> L-1."""
        d, S = self.d, self.S
        bsgs_bs, bsgs_gs = self.bsgs_bs_sigma, self.bsgs_gs_sigma
        unit = d * S

        if bsgs_bs > 1:
            baby_rots = [a_ct] + rotate_cols(a_ct, [b * unit for b in range(1, bsgs_bs)])
        else:
            baby_rots = [a_ct]

        total = None
        for g in range(bsgs_gs):
            x_list, w_list = [], []
            b_end = min(bsgs_bs, d - g * bsgs_bs)
            for b in range(b_end):
                k = g * bsgs_bs + b
                x_list.append(baby_rots[b])
                w_list.append(sigma_pt[k])
            inner = ct_pt_mult_accumulate(x_list, w_list)
            giant_rot = g * bsgs_bs * unit
            if giant_rot != 0:
                inner = rotate_cols(inner, [giant_rot])[0]
            total = inner if total is None else add(total, inner)
        return rescale(total)

    def _tau(self, b_ct, tau_pt: list):
        """BSGS tau. Level L -> L-1."""
        d, S = self.d, self.S
        chunk_size = self.chunk_size
        bsgs_bs, bsgs_gs = self.bsgs_bs_tau, self.bsgs_gs_tau
        n_tau = 2 * d - 1
        unit = S

        if bsgs_bs > 1:
            baby_rots = [b_ct] + rotate_cols(b_ct, [b * unit for b in range(1, bsgs_bs)])
        else:
            baby_rots = [b_ct]

        total = None
        for g in range(bsgs_gs):
            x_list, w_list = [], []
            b_end = min(bsgs_bs, n_tau - g * bsgs_bs)
            for b_step in range(b_end):
                j_idx = g * bsgs_bs + b_step
                x_list.append(baby_rots[b_step])
                w_list.append(tau_pt[j_idx])
            inner = ct_pt_mult_accumulate(x_list, w_list)
            giant_rot = (g * bsgs_bs - (d - 1)) * S
            giant_rot = ((giant_rot % chunk_size) + chunk_size) % chunk_size
            if giant_rot != 0:
                inner = rotate_cols(inner, [giant_rot])[0]
            total = inner if total is None else add(total, inner)
        return rescale(total)

    def _phi(self, a_sigma, i: int):
        """Rotation by d*i*S positions within chunk.  Level unchanged."""
        rot = (self.d * i * self.S) % self.chunk_size
        return a_sigma if rot == 0 else rotate_cols(a_sigma, [rot])[0]

    def _psi(self, b_tau, i: int, psi_wk_pt, psi_wkd_pt):
        """2 rotations + 2 pt_muls + add + rescale.  Level L-1 -> L-2.
        Rotation amounts scaled by S."""
        S = self.S
        chunk_size = self.chunk_size
        psi_idx = i - 1
        rot_k = ((i * S) % chunk_size + chunk_size) % chunk_size
        rot_kd = (((i - self.d) * S) % chunk_size + chunk_size) % chunk_size
        rotated_k = b_tau if rot_k == 0 else rotate_cols(b_tau, [rot_k])[0]
        rotated_kd = b_tau if rot_kd == 0 else rotate_cols(b_tau, [rot_kd])[0]
        term1 = mult(rotated_k, psi_wk_pt[psi_idx])
        term2 = mult(rotated_kd, psi_wkd_pt[psi_idx])
        return rescale(add(term1, term2))

    def _block_mult(self, a_ct, b_ct, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt):
        """Full block multiply.  Level L -> L-3."""
        # Step 1: sigma / tau  (L -> L-1)
        a_sigma = self._sigma(a_ct, sigma_pt)
        b_tau = self._tau(b_ct, tau_pt)

        # Step 2: i=0, use all-ones psi_k0 to scale b_tau  (L-1 -> L-2 -> L-3)
        b_0 = rescale(mult(b_tau, psi_k0_pt))  # pt-mul + rescale: L-1 -> L-2
        a_0 = drop_level(a_sigma)  # L-1 -> L-2
        result = rescale(relin(mult(a_0, b_0)))  # ct*ct + relin + rescale: L-2 -> L-3

        # Step 3: i=1..d-1
        for i in range(1, self.d):
            a_i = self._phi(a_sigma, i)  # rotation only, L-1
            b_i = self._psi(b_tau, i, psi_wk_pt, psi_wkd_pt)  # L-1 -> L-2
            a_i_dropped = drop_level(a_i)  # L-1 -> L-2
            prod_i = rescale(relin(mult(a_i_dropped, b_i)))  # L-2 -> L-3
            result = add(result, prod_i)

        return result  # at L-3

    # ------------------------------------------------------------------ #
    #  Core compute -- mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, A_cts: list, B_cts: list, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt) -> list:
        """Build computation DAG for C = A * B.

        Block layout is column-major: index = bi + num_block_rows * bj.
        Each block has G CTs (one per head group).
        Returns list of num_result_blocks * G CkksCiphertextNode at level L-3.
        """
        num_result_vecs = self.num_result_blocks * self.G
        C_cts = [None] * num_result_vecs

        for bi in range(self.num_block_rows_A):
            for bp in range(self.num_block_cols_B):
                c_block = self._block_index(bi, bp, self.num_block_rows_A)
                for g in range(self.G):
                    c_idx = c_block * self.G + g
                    acc = None
                    for bj in range(self.num_block_cols_A):
                        a_block = self._block_index(bi, bj, self.num_block_rows_A)
                        b_block = self._block_index(bj, bp, self.num_block_rows_B)
                        a_idx = a_block * self.G + g
                        b_idx = b_block * self.G + g
                        product = self._block_mult(
                            A_cts[a_idx],
                            B_cts[b_idx],
                            sigma_pt,
                            tau_pt,
                            psi_k0_pt,
                            psi_wk_pt,
                            psi_wkd_pt,
                        )
                        acc = product if acc is None else add(acc, product)
                    C_cts[c_idx] = acc

        return C_cts

    # ------------------------------------------------------------------ #
    #  Public call -- pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, A_cts: list, B_cts: list, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt) -> list:
        """Build computation DAG using pre-declared plaintext nodes."""
        return self._run_core(A_cts, B_cts, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt)

    # ------------------------------------------------------------------ #
    #  Custom-compute -- inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, A_cts: list, B_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute.

        Plaintext nodes are bound to C++ generate_*_pt() output through
        the custom_compute mechanism.
        """
        d = self.d

        # sigma: d nodes
        sigma_pt = []
        for k in range(d):
            node = CkksPlaintextRingtNode(f'encode_pt_sigma_{k}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'sigma_pt', 'k': k},
            )
            sigma_pt.append(node)

        # tau: (2d-1) nodes
        tau_pt = []
        for idx in range(2 * d - 1):
            node = CkksPlaintextRingtNode(f'encode_pt_tau_{idx}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'tau_pt', 'offset_idx': idx},
            )
            tau_pt.append(node)

        # psi_k0: single all-ones node
        psi_k0_pt = CkksPlaintextRingtNode('encode_pt_psi_k0')
        custom_compute(
            inputs=[data_source],
            output=psi_k0_pt,
            type='encode_pt',
            attributes={'op_class': op_class, 'type': 'psi_k0_pt'},
        )

        # psi_wk / psi_wkd: (d-1) nodes each
        psi_wk_pt, psi_wkd_pt = [], []
        for i in range(1, d):
            node_k = CkksPlaintextRingtNode(f'encode_pt_psi_wk_{i}')
            custom_compute(
                inputs=[data_source],
                output=node_k,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'psi_wk_pt', 'i': i},
            )
            psi_wk_pt.append(node_k)

            node_kd = CkksPlaintextRingtNode(f'encode_pt_psi_wkd_{i}')
            custom_compute(
                inputs=[data_source],
                output=node_kd,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'psi_wkd_pt', 'i': i},
            )
            psi_wkd_pt.append(node_kd)

        return self.call(A_cts, B_cts, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt)

    def call_encode_ringt(self, A_cts: list, B_cts: list, layer_id: str, q_level: float, psi_scale: float):
        sources = []
        sigma_pt = []
        for k in range(self.d):
            source = CustomDataNode(type='parccmm_encode_ringt_data_source', id=f'{layer_id}_s_{k}')
            sigma_pt.append(encode_ringt(source, q_level, output_id=f'encode_ringt_{layer_id}_s_{k}'))
            sources.append(source)
        tau_pt = []
        for idx in range(2 * self.d - 1):
            source = CustomDataNode(type='parccmm_encode_ringt_data_source', id=f'{layer_id}_t_{idx}')
            tau_pt.append(encode_ringt(source, q_level, output_id=f'encode_ringt_{layer_id}_t_{idx}'))
            sources.append(source)
        source = CustomDataNode(type='parccmm_encode_ringt_data_source', id=f'{layer_id}_p0')
        psi_k0_pt = encode_ringt(source, psi_scale, output_id=f'encode_ringt_{layer_id}_p0')
        sources.append(source)
        psi_wk_pt, psi_wkd_pt = [], []
        for i in range(1, self.d):
            source = CustomDataNode(type='parccmm_encode_ringt_data_source', id=f'{layer_id}_pk_{i}')
            psi_wk_pt.append(encode_ringt(source, psi_scale, output_id=f'encode_ringt_{layer_id}_pk_{i}'))
            sources.append(source)
            source = CustomDataNode(type='parccmm_encode_ringt_data_source', id=f'{layer_id}_pkd_{i}')
            psi_wkd_pt.append(encode_ringt(source, psi_scale, output_id=f'encode_ringt_{layer_id}_pkd_{i}'))
            sources.append(source)
        return self.call(A_cts, B_cts, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt), sources

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per (bi, bp, bj, g) call to _block_mult (level L -> L-3):

          Level L — sigma + tau:
            rotate:     (d-1) + (2d-2) = 3*(d-1)   [k=0/offset=0 skipped]
            mult_plain: d + (2d-1) = 3*d - 1
            add:        (d-1) + (2d-2) = 3*(d-1)
            rescale:    2

          Level L-1 — phi + psi + psi_k0:
            rotate:     (d-1) [phi] + 2*(d-1) [psi] = 3*(d-1)
            mult_plain: 2*(d-1) [psi] + 1 [psi_k0] = 2*d - 1
            add:        (d-1) [within psi]
            rescale:    (d-1) [psi] + 1 [psi_k0] = d

          Level L-2 — ct-ct mult + accumulate within block:
            mult:       d    [one per i=0..d-1]
            add:        (d-1) [accumulate prod_i into result]
            rescale:    d

          Over all (bi, bp, bj, g):
            n_block_mult = num_block_rows_A * num_block_cols_B * num_block_cols_A * G
            n_output     = num_block_rows_A * num_block_cols_B * G

          Level L-3 — accumulate over bj into output:
            add: n_output * max(0, num_block_cols_A - 1)
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level
        d = self.d
        G = self.G
        R = self.num_block_rows_A
        C = self.num_block_cols_B
        K = self.num_block_cols_A

        n_block_mult = R * C * K * G
        n_output = R * C * G

        # Level L: sigma (BSGS) + tau (BSGS)
        sigma_rots = self.bsgs_bs_sigma - 1 + self.bsgs_gs_sigma - 1
        tau_rots = self.bsgs_bs_tau - 1 + self.bsgs_gs_tau  # all giant groups may have nonzero rot
        ops[lv]['rotate'] += n_block_mult * (sigma_rots + tau_rots)
        ops[lv]['mult_plain'] += n_block_mult * (3 * d - 1)
        ops[lv]['add'] += n_block_mult * 3 * (d - 1)
        ops[lv]['rescale'] += n_block_mult * 2
        lv -= 1

        # Level L-1: phi + psi + psi_k0
        ops[lv]['rotate'] += n_block_mult * 3 * (d - 1)
        ops[lv]['mult_plain'] += n_block_mult * (2 * d - 1)
        ops[lv]['add'] += n_block_mult * (d - 1)
        ops[lv]['rescale'] += n_block_mult * d
        lv -= 1

        # Level L-2: ct-ct mult + within-block accumulate
        ops[lv]['mult'] += n_block_mult * d
        ops[lv]['add'] += n_block_mult * (d - 1)
        ops[lv]['rescale'] += n_block_mult * d
        lv -= 1

        # Level L-3: accumulate over bj
        ops[lv]['add'] += n_output * max(0, K - 1)

        return dict(ops)
