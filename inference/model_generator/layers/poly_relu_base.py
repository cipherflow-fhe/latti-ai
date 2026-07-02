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
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts


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

    def get_fhe_op_count(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in _run_bsgs_core() for n_ct input ciphertexts, grouped by level.

        Per ciphertext:
          Power construction (to_compute powers via mult_relin + rescale):
            len(to_compute) - 1  mult (relin) + rescale each  (power 1 is free)
          Baby polynomial build per giant step g (giant_steps total):
            (baby_steps - 1) mult_plain + rescale  (baby_step terms, b>=1)
            (baby_steps - 2) add  (accumulate baby_poly)
            1 add  (coeff0_pt)
          Giant combine per g>0 where giant_power <= order:
            1 mult (relin, bp * x_giant) + 1 rescale + 1 add  (result += term)
            or 1 mult_plain (coeff0 only) + 1 rescale + 1 add
        drop_level calls are free (no crypto cost).

        Level tracking follows BSGS depth: power construction consumes levels,
        then baby polys and giant combine share levels. The function uses
        compute_bsgs_level_cost() to determine total levels consumed and
        distributes ops across levels accordingly.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})

        order = self.order
        baby_steps = int(np.ceil(np.sqrt(order + 1)))
        giant_steps = int(np.ceil((order + 1) / baby_steps))

        power_info = self._compute_power_info(order)
        required, to_compute = self._determine_required_powers(order, baby_steps, giant_steps, power_info)

        # Build power depth map to know which level each power is computed at
        power_depth = {1: 0}
        for n in sorted(to_compute):
            if n <= 1:
                continue
            _, a, b = power_info[n]
            power_depth[n] = max(power_depth[a], power_depth[b]) + 1

        # Powers: each needs 1 mult_relin + 1 rescale (excluding power 1)
        # Group by depth: power at depth d is computed at level (level - d + 1)... but for
        # simplicity (as in the flat version), we track per-ct total and distribute to the
        # appropriate level based on depth.
        # Powers at depth d consume level (level - d) → (level - d - 1)
        # We aggregate by input level minus depth.
        n_powers = len(to_compute) - 1  # subtract power 1

        # Baby polys and giant combine operate at bsgs_output_level = max_power_level - 1
        # max_depth of required powers
        max_depth = max(power_depth[p] for p in required)
        bsgs_output_level = level - max_depth - 1

        # Power construction: ops at varying levels from level-1 down to level-max_depth
        # Group mult+rescale by depth
        for n in sorted(to_compute):
            if n <= 1:
                continue
            d = power_depth[n]
            lv_power = level - d  # mult at level-d, rescale from level-d → level-d-1
            ops[lv_power]['mult'] += n_ct
            ops[lv_power]['rescale'] += n_ct

        # Baby polys per giant step (at bsgs_output_level + 1 → bsgs_output_level)
        mult_plain_baby = 0
        rescale_baby = 0
        add_baby = 0
        for g in range(giant_steps):
            n_terms = 0
            has_coeff0 = False
            for b in range(baby_steps):
                idx = g * baby_steps + b
                if idx > order:
                    break
                if b == 0:
                    has_coeff0 = True
                else:
                    n_terms += 1
            mult_plain_baby += n_terms
            rescale_baby += n_terms
            if n_terms > 1:
                add_baby += n_terms - 1
            if has_coeff0 and n_terms > 0:
                add_baby += 1

        baby_level = bsgs_output_level + 1  # baby mult_plain+rescale go at bsgs_output_level+1
        ops[baby_level]['mult_plain'] += mult_plain_baby * n_ct
        ops[baby_level]['rescale'] += rescale_baby * n_ct
        ops[bsgs_output_level]['add'] += add_baby * n_ct

        # Giant combine (at bsgs_output_level + 1 → bsgs_output_level)
        mult_giant = 0
        rescale_giant = 0
        add_giant = 0
        for g in range(1, giant_steps):
            if g * baby_steps > order:
                break
            mult_giant += 1
            rescale_giant += 1
            add_giant += 1

        ops[bsgs_output_level + 1]['mult'] += mult_giant * n_ct
        ops[bsgs_output_level + 1]['rescale'] += rescale_giant * n_ct
        ops[bsgs_output_level]['add'] += add_giant * n_ct

        return dict(ops)

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

    def get_memory_bsgs(self, n_ct: int, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return plaintext memory for BSGS polyrelu weights.

        Eager BSGS preparation stores weight_pt[coeff_idx][ct_idx] with
        coeff_idx in [0, order] and ct_idx in input ciphertexts.
        """
        counts = {
            'weight': (self.order + 1) * int(n_ct),
            'bias': 0,
            'mask': 0,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_memory_horner(self, n_ct: int, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return plaintext memory for Horner polyrelu weights."""
        counts = {
            'weight': (self.order + 1) * int(n_ct),
            'bias': 0,
            'mask': 0,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)
