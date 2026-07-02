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


op_class = 'ParLowerDiagCCMM'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class ParLowerDiagCCMM:
    """Python model-generator counterpart of C++ ParLowerDiagCCMM."""

    def __init__(self, shape_A: tuple, shape_B: tuple, n_heads: int, head_dim: int, n_slot: int):
        self.shape_A = shape_A
        self.shape_B = shape_B
        self.H_prepad = n_heads
        self.m = head_dim
        self.n_prepad = shape_B[1]

        assert self.H_prepad > 0
        assert self.m > 0 and (self.m & (self.m - 1)) == 0
        assert self.n_prepad > 0

        self.H = _next_pow2(self.H_prepad)
        self.n = _next_pow2(self.n_prepad)
        assert self.n >= self.m
        assert self.n % self.m == 0

        self.n_slot = n_slot
        self.segment_len = self.H * self.n
        assert self.segment_len > 0
        assert n_slot % self.segment_len == 0
        self.c = n_slot // self.segment_len
        assert self.c > 0 and (self.c & (self.c - 1)) == 0
        assert self.m % self.c == 0
        assert self.n % self.c == 0
        self.m_c = self.m // self.c
        self.n_c = self.n // self.c

        matches_kqt = (
            shape_A[0] == self.n_prepad
            and shape_A[1] == self.m
            and shape_B[0] == self.m
            and shape_B[1] == self.n_prepad
        )
        matches_ordinary = (
            shape_A[0] == self.m
            and shape_A[1] == self.n_prepad
            and shape_B[0] == self.n_prepad
            and shape_B[1] == self.n_prepad
        )
        if matches_kqt and matches_ordinary:
            self.is_kqt = False
            self.output_shape = (self.m, self.n_prepad)
        elif matches_kqt:
            self.is_kqt = True
            self.output_shape = (self.n_prepad, self.n_prepad)
        elif matches_ordinary:
            self.is_kqt = False
            self.output_shape = (self.m, self.n_prepad)
        else:
            raise ValueError('ParLowerDiagCCMM supports only A(n,m)@B(m,n) and A(m,n)@B(n,n)')

    def _replicate_lower_diag(self, B_cts: list, replication_mask_pt: list, ell: int):
        ct_idx = ell // self.c
        mask_idx = ell % self.c
        replicated = rescale(mult(B_cts[ct_idx], replication_mask_pt[mask_idx]))
        step = 1
        while step < self.c:
            replicated = add(replicated, rotate_cols(replicated, [self.segment_len * step])[0])
            step <<= 1
        return replicated

    @staticmethod
    def _multiply_cts(a_level_l, b_level_l_minus_1):
        return rescale(relin(mult(drop_level(a_level_l), b_level_l_minus_1)))

    @staticmethod
    def _apply_route_mask(product_level_l_minus_2, mask_pt):
        return rescale(mult(product_level_l_minus_2, mask_pt))

    def _run_core_ordinary(self, A_cts: list, B_cts: list, replication_mask_pt: list, ordinary_route_pt: list) -> list:
        assert len(A_cts) == self.m_c
        assert len(B_cts) == self.n_c

        replicated_B = [self._replicate_lower_diag(B_cts, replication_mask_pt, ell) for ell in range(self.n)]

        ct_C_j_ell = []
        for j in range(self.m_c):
            row = [self._multiply_cts(A_cts[j], replicated_B[0])]
            for ell in range(1, self.n):
                ell_m = ell % self.m
                ell_c = ell_m % self.c
                source_shift = (ell_m - ell_c) // self.c
                source_j = (j + self.m_c - source_shift) % self.m_c
                rot = (ell - self.n * ell_c) * self.H
                A_rot = A_cts[source_j] if rot == 0 else rotate_cols(A_cts[source_j], [rot])[0]
                row.append(self._multiply_cts(A_rot, replicated_B[ell]))
            ct_C_j_ell.append(row)

        C_cts = []
        for j in range(self.m_c):
            ct_C_prime = None
            ct_C_double_prime = None
            prev_j = (j + self.m_c - 1) % self.m_c

            for ell in range(self.n):
                term0 = self._apply_route_mask(ct_C_j_ell[prev_j][ell], ordinary_route_pt[ell][0])
                term1 = self._apply_route_mask(ct_C_j_ell[j][ell], ordinary_route_pt[ell][1])
                term2 = self._apply_route_mask(ct_C_j_ell[prev_j][ell], ordinary_route_pt[ell][2])
                term3 = self._apply_route_mask(ct_C_j_ell[j][ell], ordinary_route_pt[ell][3])

                prime_term = add(term0, term1)
                ct_C_prime = prime_term if ct_C_prime is None else add(ct_C_prime, prime_term)

                double_prime_term = add(term2, term3)
                ct_C_double_prime = (
                    double_prime_term if ct_C_double_prime is None else add(ct_C_double_prime, double_prime_term)
                )

            assert ct_C_prime is not None
            ct = ct_C_prime
            if ct_C_double_prime is not None:
                ct = add(ct, rotate_cols(ct_C_double_prime, [-self.segment_len])[0])
            C_cts.append(ct)
        return C_cts

    def _run_core_kqt(self, A_cts: list, B_cts: list, replication_mask_pt: list, kqt_route_pt: list) -> list:
        assert len(A_cts) == self.m_c
        assert len(B_cts) == self.m_c

        replicated_B = [self._replicate_lower_diag(B_cts, replication_mask_pt, ell) for ell in range(self.m)]

        ct_C_p_ell = []
        for p in range(self.n_c):
            q_p = p // self.m_c
            u_p = p % self.m_c
            row = []
            for ell in range(self.m):
                b_ell = ell % self.c
                R_p_ell = q_p * self.m + ell
                assert R_p_ell < self.n
                rot = (R_p_ell - self.n * b_ell) * self.H
                A_rot = A_cts[u_p] if rot == 0 else rotate_cols(A_cts[u_p], [rot])[0]
                row.append(self._multiply_cts(A_rot, replicated_B[ell]))
            ct_C_p_ell.append(row)

        C_cts = []
        for j in range(self.n_c):
            ct_C_prime = None
            ct_C_double_prime = None
            for ell in range(self.m):
                a_ell = ell // self.c
                p_prev = (j + self.n_c - 1 - a_ell) % self.n_c
                p_curr = (j + self.n_c - a_ell) % self.n_c

                term0 = self._apply_route_mask(ct_C_p_ell[p_prev][ell], kqt_route_pt[j][ell][0])
                term1 = self._apply_route_mask(ct_C_p_ell[p_curr][ell], kqt_route_pt[j][ell][1])
                term2 = self._apply_route_mask(ct_C_p_ell[p_prev][ell], kqt_route_pt[j][ell][2])
                term3 = self._apply_route_mask(ct_C_p_ell[p_curr][ell], kqt_route_pt[j][ell][3])

                prime_term = add(term0, term1)
                ct_C_prime = prime_term if ct_C_prime is None else add(ct_C_prime, prime_term)

                double_prime_term = add(term2, term3)
                ct_C_double_prime = (
                    double_prime_term if ct_C_double_prime is None else add(ct_C_double_prime, double_prime_term)
                )

            assert ct_C_prime is not None
            ct = ct_C_prime
            if ct_C_double_prime is not None:
                ct = add(ct, rotate_cols(ct_C_double_prime, [-self.segment_len])[0])
            C_cts.append(ct)
        return C_cts

    def call(
        self,
        A_cts: list,
        B_cts: list,
        replication_mask_pt: list,
        ordinary_route_pt: list | None = None,
        kqt_route_pt: list | None = None,
    ) -> list:
        if self.is_kqt:
            assert kqt_route_pt is not None
            return self._run_core_kqt(A_cts, B_cts, replication_mask_pt, kqt_route_pt)
        assert ordinary_route_pt is not None
        return self._run_core_ordinary(A_cts, B_cts, replication_mask_pt, ordinary_route_pt)

    def call_custom_compute(self, A_cts: list, B_cts: list, data_source) -> list:
        replication_count = self.m if self.is_kqt else self.n
        replication_mask_pt = []
        for ell in range(min(self.c, replication_count)):
            node = CkksPlaintextRingtNode(f'encode_pt_replication_mask_{ell}')
            custom_compute(
                inputs=[data_source],
                output=node,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'replication_mask_pt', 'ell': ell},
            )
            replication_mask_pt.append(node)

        if self.is_kqt:
            kqt_route_pt = []
            for j in range(self.n_c):
                route_j = []
                for ell in range(self.m):
                    route_ell = []
                    for mask_idx in range(4):
                        node = CkksPlaintextRingtNode(f'encode_pt_kqt_route_{j}_{ell}_{mask_idx}')
                        custom_compute(
                            inputs=[data_source],
                            output=node,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'kqt_route_pt',
                                'j': j,
                                'ell': ell,
                                'mask_idx': mask_idx,
                            },
                        )
                        route_ell.append(node)
                    route_j.append(route_ell)
                kqt_route_pt.append(route_j)
            return self.call(A_cts, B_cts, replication_mask_pt, kqt_route_pt=kqt_route_pt)

        ordinary_route_pt = []
        for ell in range(self.n):
            route_ell = []
            for mask_idx in range(4):
                node = CkksPlaintextRingtNode(f'encode_pt_ordinary_route_{ell}_{mask_idx}')
                custom_compute(
                    inputs=[data_source],
                    output=node,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'ordinary_route_pt',
                        'ell': ell,
                        'mask_idx': mask_idx,
                    },
                )
                route_ell.append(node)
            ordinary_route_pt.append(route_ell)
        return self.call(A_cts, B_cts, replication_mask_pt, ordinary_route_pt=ordinary_route_pt)

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        replication_count = self.m if self.is_kqt else self.n
        replication_masks = min(self.c, replication_count)
        route_masks = self.n_c * self.m * 4 if self.is_kqt else self.n * 4
        counts = {'weight': 0, 'bias': 0, 'mask': replication_masks + route_masks}
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_fhe_op_count(self, level: int) -> dict:
        """Count FHE primitive operations grouped by level."""
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        replication_count = self.m if self.is_kqt else self.n
        replicate_steps = 0
        step = 1
        while step < self.c:
            replicate_steps += 1
            step <<= 1

        # Replicate lower-diagonal B ciphertexts.
        ops[lv]['mult_plain'] += replication_count
        ops[lv]['rescale'] += replication_count
        ops[lv]['rotate'] += replication_count * replicate_steps
        ops[lv]['add'] += replication_count * replicate_steps

        if self.is_kqt:
            self._add_kqt_op_count(ops, lv)
        else:
            self._add_ordinary_op_count(ops, lv)

        return dict(ops)

    def _add_ordinary_op_count(self, ops, lv: int) -> None:
        # Build ct_C_j_ell.
        n_products = self.m_c * self.n
        for j in range(self.m_c):
            for ell in range(1, self.n):
                ell_m = ell % self.m
                ell_c = ell_m % self.c
                rot = (ell - self.n * ell_c) * self.H
                if rot != 0:
                    ops[lv]['rotate'] += 1
        ops[lv - 1]['mult'] += n_products
        ops[lv - 1]['rescale'] += n_products

        # Apply ordinary route masks and accumulate routed terms.
        route_terms = self.m_c * self.n
        ops[lv - 2]['mult_plain'] += route_terms * 4
        ops[lv - 2]['rescale'] += route_terms * 4
        ops[lv - 3]['add'] += route_terms * 2
        ops[lv - 3]['add'] += self.m_c * max(0, self.n - 1) * 2
        ops[lv - 3]['rotate'] += self.m_c
        ops[lv - 3]['add'] += self.m_c

    def _add_kqt_op_count(self, ops, lv: int) -> None:
        # Build ct_C_p_ell.
        n_products = self.n_c * self.m
        for p in range(self.n_c):
            q_p = p // self.m_c
            for ell in range(self.m):
                b_ell = ell % self.c
                R_p_ell = q_p * self.m + ell
                rot = (R_p_ell - self.n * b_ell) * self.H
                if rot != 0:
                    ops[lv]['rotate'] += 1
        ops[lv - 1]['mult'] += n_products
        ops[lv - 1]['rescale'] += n_products

        # Apply KQT route masks and accumulate routed terms.
        route_terms = self.n_c * self.m
        ops[lv - 2]['mult_plain'] += route_terms * 4
        ops[lv - 2]['rescale'] += route_terms * 4
        ops[lv - 3]['add'] += route_terms * 2
        ops[lv - 3]['add'] += self.n_c * max(0, self.m - 1) * 2
        ops[lv - 3]['rotate'] += self.n_c
        ops[lv - 3]['add'] += self.n_c
