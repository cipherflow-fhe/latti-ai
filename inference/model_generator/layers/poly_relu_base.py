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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *

import numpy as np


class PolyReluBase:
    """Shared BSGS infrastructure for all PolyRelu variants (0D / 1D / 2D).

    Subclasses must set self.order in their __init__.
    """

    @staticmethod
    def _compute_power_info(order):
        """Optimal power decomposition tree. Matches C++ compute_all_powers."""
        info = {1: (0, 0, 0)}  # n -> (depth, decomp_a, decomp_b)
        for n in range(2, order + 1):
            best_depth = float('inf')
            best_a, best_b = 1, n - 1
            for a in range(1, n // 2 + 1):
                b = n - a
                depth = max(info[a][0], info[b][0]) + 1
                if depth < best_depth:
                    best_depth = depth
                    best_a, best_b = a, b
                elif depth == best_depth and abs(a - b) < abs(best_a - best_b):
                    best_a, best_b = a, b
            info[n] = (best_depth, best_a, best_b)
        return info

    @staticmethod
    def _determine_required_powers(order, baby_steps, giant_steps, power_info):
        """Required powers + dependencies. Matches C++ determine_required_powers_bsgs."""
        required = set()
        for i in range(1, baby_steps + 1):
            required.add(i)
        for g in range(1, giant_steps):
            gp = g * baby_steps
            if gp <= order:
                required.add(gp)

        to_compute = set(required)

        def add_deps(n):
            if n <= 1:
                return
            _, a, b = power_info[n]
            if a > 1:
                to_compute.add(a)
                add_deps(a)
            if b > 1:
                to_compute.add(b)
                add_deps(b)

        for p in list(required):
            add_deps(p)
        return required, to_compute

    @staticmethod
    def compute_bsgs_level_cost(order):
        """Level cost of BSGS algorithm. Matches C++ bsgs_output_level logic."""
        if order <= 1:
            return 1
        baby_steps = int(np.ceil(np.sqrt(order + 1)))
        giant_steps = int(np.ceil((order + 1) / baby_steps))
        power_info = PolyReluBase._compute_power_info(order)
        required, to_compute = PolyReluBase._determine_required_powers(order, baby_steps, giant_steps, power_info)
        power_depth = {1: 0}
        for n in sorted(to_compute):
            if n <= 1:
                continue
            _, a, b = power_info[n]
            power_depth[n] = max(power_depth[a], power_depth[b]) + 1
        max_depth = max(power_depth[p] for p in required)
        return max_depth + 1

    def _run_bsgs_core(self, x: list, get_weight):
        """Core BSGS computation graph. Matches C++ run_core_bsgs.

        Args:
            x:          list of CkksCiphertextNode inputs
            get_weight: callable(coeff_idx, x_idx) -> plaintext weight node
        """
        order = self.order
        baby_steps = int(np.ceil(np.sqrt(order + 1)))
        giant_steps = int(np.ceil((order + 1) / baby_steps))

        power_info = self._compute_power_info(order)
        required, to_compute = self._determine_required_powers(order, baby_steps, giant_steps, power_info)

        level_cost = self.compute_bsgs_level_cost(order)
        for i, ct in enumerate(x):
            if ct.level < level_cost:
                raise ValueError(
                    f'{type(self).__name__} (order={order}): Input ciphertext {i} has '
                    f'insufficient level: {ct.level} < {level_cost}. '
                    f'BSGS algorithm will consume {level_cost} levels.'
                )

        result = [None] * len(x)

        for x_idx in range(len(x)):
            # 1. Compute powers using optimal decomposition
            x_powers = {1: x[x_idx]}
            for i in sorted(to_compute):
                if i <= 1:
                    continue
                _, a, b = power_info[i]
                xa = x_powers[a]
                xb = x_powers[b]
                tgt = min(xa.level, xb.level)
                if xa.level > tgt:
                    xa = drop_level(xa, xa.level - tgt)
                if xb.level > tgt:
                    xb = drop_level(xb, xb.level - tgt)
                x_powers[i] = rescale(relin(mult(xa, xb)))

            # Compute bsgs_output_level
            max_depth = 0
            max_power_level = x[x_idx].level
            for p in required:
                d = power_info[p][0]
                if d > max_depth:
                    max_depth = d
                    max_power_level = x_powers[p].level
            bsgs_output_level = max_power_level - 1

            bp_out_level = [bsgs_output_level] * giant_steps
            for g in range(1, giant_steps):
                if g * baby_steps <= order:
                    bp_out_level[g] = bsgs_output_level + 1

            # 2. Build baby polynomials
            baby_polys = [None] * giant_steps
            baby_poly_has_terms = [False] * giant_steps
            coeff0_pts = [None] * giant_steps

            for g in range(giant_steps):
                target_level = bp_out_level[g]

                for b in range(baby_steps):
                    idx = g * baby_steps + b
                    if idx > order:
                        break

                    w_pt = get_weight(idx, x_idx)

                    if b == 0:
                        coeff0_pts[g] = w_pt
                        continue

                    baby_poly_has_terms[g] = True
                    x_copy = x_powers[b]
                    if x_copy.level > target_level + 1:
                        x_copy = drop_level(x_copy, x_copy.level - (target_level + 1))
                    term = rescale(mult(x_copy, w_pt))

                    if baby_polys[g] is None:
                        baby_polys[g] = term
                    else:
                        baby_polys[g] = add(baby_polys[g], term)

                if coeff0_pts[g] is not None and baby_poly_has_terms[g]:
                    baby_polys[g] = add(baby_polys[g], coeff0_pts[g])

            # 3. Combine: result = P0 + P1*x^baby + P2*x^(2*baby) + ...
            result[x_idx] = baby_polys[0]

            for g in range(1, giant_steps):
                giant_power = g * baby_steps
                if giant_power > order:
                    break

                x_giant = x_powers[giant_power]
                mult_level = bsgs_output_level + 1
                if x_giant.level > mult_level:
                    x_giant = drop_level(x_giant, x_giant.level - mult_level)

                if baby_poly_has_terms[g]:
                    bp = baby_polys[g]
                    if bp.level > mult_level:
                        bp = drop_level(bp, bp.level - mult_level)
                    term = rescale(relin(mult(bp, x_giant)))
                else:
                    if coeff0_pts[g] is not None:
                        term = rescale(mult(x_giant, coeff0_pts[g]))
                    else:
                        continue

                result[x_idx] = add(result[x_idx], term)

        return result

    def make_pt_nodes(self, layer_id, n_pack_in_channel):
        """Return weight_pt placeholder list with shape [order+1][n_pack_in_channel]."""
        return [
            [CkksPlaintextRingtNode(f'poly_reluw_{layer_id}_{i}_{j}') for j in range(n_pack_in_channel)]
            for i in range(self.order + 1)
        ]
