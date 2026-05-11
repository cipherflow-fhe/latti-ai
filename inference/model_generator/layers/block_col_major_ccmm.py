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

op_class = 'BlockColMajorCCMM'


class BlockColMajorCCMM:
    """Python model-generator counterpart of C++ BlockColMajorCCMM.

    Ciphertext-ciphertext matrix multiplication C = A * B where both
    operands are encrypted in block_col_major format (column-major block order).

    Level consumption: 3 levels per block_mult (sigma L→L-1, tau L→L-1,
    psi L-1→L-2, ct*ct L-2→L-3).

    Slot layout per chunk: slot[i + d*j] = element at (row=i, col=j) in block.
    Chunks are tiled across n_slot.
    """

    def __init__(self, shape_A: tuple, shape_B: tuple, block_size: int, n_slot: int):
        """
        Args:
            shape_A:    (m, n) — shape of matrix A.
            shape_B:    (n, p) — shape of matrix B.
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

        # Result matrix C has num_block_rows_A * num_block_cols_B blocks
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
        """Create pre-declared plaintext nodes for sigma, tau, and psi diagonals.

        Returns:
            sigma_pt:   list of d nodes, indexed by k_idx.
            tau_pt:     list of (2d-1) nodes, indexed by offset_idx (0..2d-2).
            psi_k0_pt:  single all-ones node for psi i=0.
            psi_wk_pt:  list of (d-1) nodes for psi i=1..d-1.
            psi_wkd_pt: list of (d-1) nodes for psi i=1..d-1.
        """
        d = self.d
        sigma_pt = [CkksPlaintextRingtNode(f'ccmm_sigma_{layer_id}_{k}') for k in range(d)]
        tau_pt = [CkksPlaintextRingtNode(f'ccmm_tau_{layer_id}_{idx}') for idx in range(2 * d - 1)]
        psi_k0_pt = CkksPlaintextRingtNode(f'ccmm_psi_k0_{layer_id}')
        psi_wk_pt = [CkksPlaintextRingtNode(f'ccmm_psi_wk_{layer_id}_{i}') for i in range(1, d)]
        psi_wkd_pt = [CkksPlaintextRingtNode(f'ccmm_psi_wkd_{layer_id}_{i}') for i in range(1, d)]
        return sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt

    # ------------------------------------------------------------------ #
    #  Primitives — mirror C++ methods                                    #
    # ------------------------------------------------------------------ #

    def _sigma(self, a_ct, sigma_pt: list):
        """d rotations + d pt_muls + accumulate + rescale.  Level L → L-1."""
        d, chunk_size = self.d, self.chunk_size
        x_list, w_list = [], []
        for k in range(d):
            rot = (d * k) % chunk_size
            rotated = a_ct if rot == 0 else rotate_cols(a_ct, [rot])[0]
            x_list.append(rotated)
            w_list.append(sigma_pt[k])
        return rescale(ct_pt_mult_accumulate(x_list, w_list))

    def _tau(self, b_ct, tau_pt: list):
        """(2d-1) rotations + pt_muls + accumulate + rescale.  Level L → L-1."""
        d, chunk_size = self.d, self.chunk_size
        x_list, w_list = [], []
        for idx, offset in enumerate(range(-(d - 1), d)):
            rot = ((offset % chunk_size) + chunk_size) % chunk_size
            rotated = b_ct if rot == 0 else rotate_cols(b_ct, [rot])[0]
            x_list.append(rotated)
            w_list.append(tau_pt[idx])
        return rescale(ct_pt_mult_accumulate(x_list, w_list))

    def _phi(self, a_sigma, i: int):
        """Rotation by d*i positions within chunk.  Level unchanged."""
        rot = (self.d * i) % self.chunk_size
        return a_sigma if rot == 0 else rotate_cols(a_sigma, [rot])[0]

    def _psi(self, b_tau, i: int, psi_wk_pt, psi_wkd_pt):
        """2 rotations + 2 pt_muls + add + rescale.  Level L-1 → L-2."""
        d, chunk_size = self.d, self.chunk_size
        psi_idx = i - 1
        rot_k = (i % chunk_size + chunk_size) % chunk_size
        rot_kd = ((i - d) % chunk_size + chunk_size) % chunk_size
        rotated_k = b_tau if rot_k == 0 else rotate_cols(b_tau, [rot_k])[0]
        rotated_kd = b_tau if rot_kd == 0 else rotate_cols(b_tau, [rot_kd])[0]
        term1 = mult(rotated_k, psi_wk_pt[psi_idx])
        term2 = mult(rotated_kd, psi_wkd_pt[psi_idx])
        return rescale(add(term1, term2))

    def _block_mult(self, a_ct, b_ct, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt):
        """Full block multiply.  Level L → L-3."""
        # Step 1: sigma / tau  (L → L-1)
        a_sigma = self._sigma(a_ct, sigma_pt)
        b_tau = self._tau(b_ct, tau_pt)

        # Step 2: i=0, use all-ones psi_k0 to scale b_tau  (L-1 → L-2 → L-3)
        b_0 = rescale(mult(b_tau, psi_k0_pt))  # pt-mul + rescale: L-1 → L-2
        a_0 = drop_level(a_sigma)  # L-1 → L-2
        result = rescale(relin(mult(a_0, b_0)))  # ct*ct + relin + rescale: L-2 → L-3

        # Step 3: i=1..d-1
        for i in range(1, self.d):
            a_i = self._phi(a_sigma, i)  # rotation only, L-1
            b_i = self._psi(b_tau, i, psi_wk_pt, psi_wkd_pt)  # L-1 → L-2
            a_i_dropped = drop_level(a_i)  # L-1 → L-2
            prod_i = rescale(relin(mult(a_i_dropped, b_i)))  # L-2 → L-3
            result = add(result, prod_i)

        return result  # at L-3

    # ------------------------------------------------------------------ #
    #  Core compute — mirrors C++ run_core                               #
    # ------------------------------------------------------------------ #

    def _run_core(self, A_cts: list, B_cts: list, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt) -> list:
        """Build computation DAG for C = A * B.

        Block layout is column-major: index = bi + num_block_rows * bj.
        Returns list of num_result_blocks CkksCiphertextNode at level L-3.
        """
        C_cts = [None] * self.num_result_blocks

        for bi in range(self.num_block_rows_A):
            for bp in range(self.num_block_cols_B):
                c_idx = self._block_index(bi, bp, self.num_block_rows_A)
                acc = None
                for bj in range(self.num_block_cols_A):
                    a_idx = self._block_index(bi, bj, self.num_block_rows_A)
                    b_idx = self._block_index(bj, bp, self.num_block_rows_B)
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
    #  Public call — pre-declared weights                                 #
    # ------------------------------------------------------------------ #

    def call(self, A_cts: list, B_cts: list, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt) -> list:
        """Build computation DAG using pre-declared plaintext nodes."""
        return self._run_core(A_cts, B_cts, sigma_pt, tau_pt, psi_k0_pt, psi_wk_pt, psi_wkd_pt)

    # ------------------------------------------------------------------ #
    #  Custom-compute — inline weight declaration (lazy path)            #
    # ------------------------------------------------------------------ #

    def call_custom_compute(self, A_cts: list, B_cts: list, data_source) -> list:
        """Build computation DAG with inline plaintext nodes via custom_compute.

        Plaintext nodes are bound to C++ precompute_diagonals() output through
        the custom_compute mechanism.
        """
        d = self.d

        def _pt(key, attrs_extra=None):
            attrs = {'op_class': op_class, 'type': key}
            if attrs_extra:
                attrs.update(attrs_extra)
            node = CkksPlaintextRingtNode(f'encode_pt_{key}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes=attrs,
            )
            return node

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

        # tau: (2d-1) nodes, offset_idx = 0..2d-2 maps to offset -(d-1)..d-1
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

        # psi_wk / psi_wkd: (d-1) nodes each, i=1..d-1
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

    # ------------------------------------------------------------------ #
    #  FHE operation count                                                #
    # ------------------------------------------------------------------ #

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level.

        Per (bi, bp) output block, summed over bj inner blocks:

          Level L — sigma + tau (each is independent per block_mult):
            sigma: d rotations + d pt_muls + (d-1) adds + 1 rescale
            tau:   (2d-1) rots + (2d-1) pt_muls + (2d-2) adds + 1 rescale

          Level L-1 — psi (for i=1..d-1) + i=0 pt-mul:
            psi i=0: 1 pt_mul + 1 rescale  (psi_k0 * b_tau)
            psi i>=1: 2 rots + 2 pt_muls + 1 add + 1 rescale  per i
            phi i>=1: 1 rot per i  (rotation of a_sigma, no level consumed at L-1)
            drop_level: d times (a_sigma for each i)

          Level L-2 — d ct*ct multiplications + relin + rescale:
            mult:     d
            relin:    d
            rescale:  d
            add:      d-1  (accumulate products)

          Over all (bi, bp) pairs and bj inner blocks:
            per block_mult: above costs once
            num_block_mults = num_block_rows_A * num_block_cols_B * num_block_cols_A
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0, 'relin': 0})
        lv = level
        d = self.d
        R = self.num_block_rows_A
        C = self.num_block_cols_B
        K = self.num_block_cols_A

        n_block_mult = R * C * K

        # Level L: sigma + tau per block_mult
        ops[lv]['rotate'] += n_block_mult * (d + (2 * d - 1))
        ops[lv]['mult_plain'] += n_block_mult * (d + (2 * d - 1))
        ops[lv]['add'] += n_block_mult * ((d - 1) + (2 * d - 2))
        ops[lv]['rescale'] += n_block_mult * 2
        lv -= 1

        # Level L-1: psi_k0 + psi i=1..d-1 + phi rotations (no level consumed)
        ops[lv]['mult_plain'] += n_block_mult * 1  # psi_k0 * b_tau
        ops[lv]['rescale'] += n_block_mult * 1
        ops[lv]['rotate'] += n_block_mult * (2 * (d - 1) + (d - 1))  # psi rots + phi rots
        ops[lv]['mult_plain'] += n_block_mult * 2 * (d - 1)  # psi i>=1
        ops[lv]['add'] += n_block_mult * (d - 1)  # psi i>=1 accumulate
        ops[lv]['rescale'] += n_block_mult * (d - 1)  # psi i>=1
        lv -= 1

        # Level L-2: d ct*ct + relin + rescale + accumulate
        ops[lv]['mult'] += n_block_mult * d
        ops[lv]['relin'] += n_block_mult * d
        ops[lv]['rescale'] += n_block_mult * d
        ops[lv]['add'] += n_block_mult * (d - 1)  # accumulate i=0..d-1 products

        # Accumulate over bj (num_block_cols_A-1 adds per output block, at L-3)
        lv -= 1
        ops[lv]['add'] += R * C * (K - 1)

        return dict(ops)
