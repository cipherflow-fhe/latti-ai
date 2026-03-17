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
import os

# Add mega_ag_generator to path for importing frontend module
script_dir = os.path.dirname(os.path.abspath(__file__))
mega_ag_generator_dir = os.path.join(script_dir, '../../lattisense')
sys.path.insert(0, mega_ag_generator_dir)

from frontend.custom_task import *

import numpy as np

op_class = 'PolyReluLayer'


class PolyReluLayer:
    def __init__(
        self, input_shape, order, skip, n_channel_per_ct, upsample_factor: list = [1, 1], block_expansion: list = [1, 1]
    ):
        self.input_shape = input_shape
        self.order = order
        self.skip = skip
        self.n_channel_per_ct = n_channel_per_ct
        self.upsample_factor = upsample_factor
        self.block_expansion = block_expansion

        if input_shape[0] & (input_shape[0] - 1) != 0 or input_shape[1] & (input_shape[1] - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')
        self.pre_skip = [1, 1]
        self.pre_skip[0] = self.skip[0] * self.upsample_factor[0]
        self.pre_skip[1] = self.skip[1] * self.upsample_factor[1]
        self.block_shape = [1, 1]
        self.block_shape[0] = int(self.input_shape[0] * self.skip[0] / self.block_expansion[0])
        self.block_shape[1] = int(self.input_shape[1] * self.skip[1] / self.block_expansion[1])
        if self.block_shape[0] & (self.block_shape[0] - 1) != 0 or self.block_shape[1] & (self.block_shape[1] - 1) != 0:
            raise ValueError(f'block_shape must be powers of 2, got: [{self.block_shape[0]}, {self.block_shape[1]}]')

    @classmethod
    def create_for_feature0d(cls, order, skip, n_channel_per_ct):
        """Create a PolyReluLayer for Feature0D (1D channel-only, no spatial dimensions).

        Args:
            order: polynomial order
            skip: 1D skip value (scalar), channel ch at slot ch * skip
            n_channel_per_ct: number of channels packed per ciphertext
        """
        obj = cls.__new__(cls)
        obj.input_shape = [1, 1]
        obj.order = order
        obj.skip = [skip, 1]
        obj.n_channel_per_ct = n_channel_per_ct
        obj.upsample_factor = [1, 1]
        obj.block_expansion = [1, 1]
        obj.pre_skip = [skip, 1]
        obj.block_shape = [skip, 1]
        return obj

    def call(self, x: list[CkksCiphertextNode], weight_pt):
        x = list(x)  # shallow copy to avoid mutating caller's list
        result = list()
        if self.order != 4:
            for i in range(len(x)):
                res = rescale(mult(x[i], weight_pt[self.order][i]))
                for order_idx in range(self.order - 1, 0, -1):
                    res = add(res, weight_pt[order_idx][i])
                    if res.level > x[i].level:
                        res = drop_level(res, res.level - x[i].level)
                    if res.level < x[i].level:
                        x[i] = drop_level(x[i], x[i].level - res.level)
                    res = rescale(relin(mult(res, x[i])))
                res = add(res, weight_pt[0][i])
                result.append(res)
        else:
            result = [0 for i in range(len(x))]
            for x_idx in range(len(x)):
                if len(weight_pt) <= 1:
                    res = rescale(mult(x[x_idx], weight_pt[1][x_idx]))
                    res = add(res, weight_pt[0][x_idx])
                    result[x_idx] = res
                else:
                    baby_steps = int(np.ceil(np.sqrt(self.order + 1)))
                    giant_steps = int(np.ceil((self.order + 1) / baby_steps))

                    x_powers = [0 for i in range(baby_steps + 1)]
                    x_powers[1] = x[x_idx]
                    for i in range(2, baby_steps + 1):
                        if i % 2 == 0:
                            half = int(i / 2)
                            x_powers[i] = rescale(relin(mult(x_powers[half], x_powers[half])))
                        else:
                            y = x_powers[1]
                            if x_powers[1].level > x_powers[i - 1].level:
                                y = drop_level(x_powers[1], x_powers[1].level - x_powers[i - 1].level)
                            x_powers[i] = rescale(relin(mult(x_powers[i - 1], y)))
                    x_giant = x_powers[baby_steps]
                    current_giant_power = x_giant

                    for giant_step in range(0, giant_steps):
                        for baby_step in range(0, baby_steps):
                            coeff_idx = giant_step * baby_steps + baby_step
                            if coeff_idx > self.order:
                                continue
                            if baby_step == 0:
                                coeff0_pt = weight_pt[coeff_idx][x_idx]
                            else:
                                x_copy = x_powers[baby_step]

                                if x_copy.level > x_powers[1].level - 2 + giant_step:
                                    x_copy = drop_level(x_copy, (x_copy.level - (x_powers[1].level - 2 + giant_step)))

                                term = rescale(mult(x_copy, weight_pt[coeff_idx][x_idx]))
                            if baby_step == 0:
                                continue
                            elif baby_step == 1:
                                baby_poly = add(term, coeff0_pt)
                            else:
                                baby_poly = add(baby_poly, term)

                        if giant_step > 0:
                            baby_poly = rescale(relin(mult(baby_poly, current_giant_power)))
                        if giant_step == 0:
                            result[x_idx] = baby_poly
                        else:
                            result[x_idx] = add(result[x_idx], baby_poly)

                        if giant_step < giant_steps - 1 and giant_step > 1:
                            current_giant_power = rescale(relin(mult(current_giant_power, x_giant)))
        return result

    def call_custom_compute(self, x: list[CkksCiphertextNode], poly_data_source, layer_id: str = ''):
        x = list(x)  # shallow copy to avoid mutating caller's list
        result = list()
        ct_counter = 0  # Ciphertext node counter

        # Check if input ciphertext level is sufficient
        # BSGS algorithm for order=4 consumes 3 levels (refer to C++ implementation)
        # Horner algorithm for other orders consumes order-1 levels
        min_required_level = 3 if self.order == 4 else max(self.order - 1, 1)
        for i, ct in enumerate(x):
            if ct.level < min_required_level:
                raise ValueError(
                    f'PolyReluLayer (order={self.order}): Input ciphertext {i} has insufficient level: '
                    f'{ct.level} < {min_required_level}. '
                    f'Algorithm will consume {min_required_level} levels, causing negative level. '
                    f'Please use bootstrap before this layer or adjust the network depth.'
                )

        if self.order != 4:
            for i in range(len(x)):
                res = x[i]
                for order_idx in range(self.order - 1, 0, -1):
                    w_pt = CkksPlaintextRingtNode(f'encode_pt_{order_idx}_{i}')
                    custom_compute(
                        inputs=[poly_data_source],
                        output=w_pt,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'weight_pt',
                            'i': order_idx,
                            'j': i,
                        },
                    )
                    res = add(res, w_pt)
                    if res.level > x[i].level:
                        res = drop_level(res, res.level - x[i].level)
                    if res.level < x[i].level:
                        x[i] = drop_level(x[i], x[i].level - res.level)
                    res = rescale(relin(mult(res, x[i])))
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{0}_{i}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt',
                        'i': 0,
                        'j': i,
                    },
                )
                res = add(res, w_pt)
                result.append(res)
        else:
            result = [0 for i in range(len(x))]
            for x_idx in range(len(x)):
                if self.order <= 1:
                    w_pt_1 = CkksPlaintextRingtNode(f'encode_pt_{1}_{x_idx}')
                    custom_compute(
                        inputs=[poly_data_source],
                        output=w_pt_1,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'weight_pt',
                            'i': 1,
                            'j': x_idx,
                        },
                    )
                    res = rescale(mult(x[x_idx], w_pt_1))
                    w_pt_0 = CkksPlaintextRingtNode(f'encode_pt_{0}_{x_idx}')
                    custom_compute(
                        inputs=[poly_data_source],
                        output=w_pt_0,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'weight_pt',
                            'i': 0,
                            'j': x_idx,
                        },
                    )
                    res = add(res, w_pt_0)
                    result[x_idx] = res
                else:
                    baby_steps = int(np.ceil(np.sqrt(self.order + 1)))
                    giant_steps = int(np.ceil((self.order + 1) / baby_steps))

                    # Save initial level, corresponding to the level parameter in C++
                    initial_level = x[x_idx].level

                    # Pre-compute powers of x: x, x^2, ..., x^baby_steps
                    x_powers = [0 for i in range(baby_steps + 1)]
                    x_powers[1] = x[x_idx]
                    for i in range(2, baby_steps + 1):
                        if i % 2 == 0:
                            half = int(i / 2)
                            tp = relin(mult(x_powers[half], x_powers[half]))
                            x_powers[i] = rescale(tp)
                        else:
                            y = x_powers[1]
                            if x_powers[1].level > x_powers[i - 1].level:
                                y = drop_level(x_powers[1], x_powers[1].level - x_powers[i - 1].level)
                            x_powers[i] = rescale(relin(mult(x_powers[i - 1], y)))
                    x_giant = x_powers[baby_steps]
                    current_giant_power = x_giant

                    # Giant steps loop
                    for giant_step in range(0, giant_steps):
                        # Polynomial computation within baby steps
                        baby_poly = None
                        coeff0_pt = None
                        has_coeff0 = False

                        for baby_step in range(0, baby_steps):
                            coeff_idx = giant_step * baby_steps + baby_step
                            if coeff_idx > self.order:
                                continue

                            if baby_step == 0:
                                # Save coeff0, but don't create term
                                has_coeff0 = True
                                w_pt = CkksPlaintextRingtNode(f'encode_pt_{coeff_idx}_{x_idx}')
                                custom_compute(
                                    inputs=[poly_data_source],
                                    output=w_pt,
                                    type='encode_pt',
                                    attributes={
                                        'op_class': op_class,
                                        'type': 'weight_pt',
                                        'i': coeff_idx,
                                        'j': x_idx,
                                    },
                                )
                                coeff0_pt = w_pt
                            else:
                                # Adjust x_copy level, using fixed initial_level
                                x_copy = x_powers[baby_step]
                                target_level = initial_level - 2 + giant_step

                                # If x_copy level is higher than target level, reduce to target level
                                # Equivalent to C++ while loop logic: while (x_copy.get_level() > level - 2 + giant_step)
                                if x_copy.level > target_level:
                                    x_copy = drop_level(x_copy, x_copy.level - target_level)

                                # Create weight and compute term
                                w_pt = CkksPlaintextRingtNode(f'encode_pt_{coeff_idx}_{x_idx}')
                                custom_compute(
                                    inputs=[poly_data_source],
                                    output=w_pt,
                                    type='encode_pt',
                                    attributes={
                                        'op_class': op_class,
                                        'type': 'weight_pt',
                                        'i': coeff_idx,
                                        'j': x_idx,
                                    },
                                )
                                term = rescale(mult(x_copy, w_pt))

                            # Accumulate baby polynomial
                            if baby_step == 0:
                                continue
                            elif baby_step == 1:
                                # When adding term for the first time, also add coeff0
                                if has_coeff0 and coeff0_pt is not None:
                                    baby_poly = add(term, coeff0_pt)
                                else:
                                    baby_poly = term
                            else:
                                baby_poly = add(baby_poly, term)

                        # Multiply baby polynomial by current giant step power
                        if giant_step > 0:
                            baby_poly = rescale(relin(mult(baby_poly, current_giant_power)))

                        if giant_step == 0:
                            result[x_idx] = baby_poly
                        else:
                            result[x_idx] = add(result[x_idx], baby_poly)

                        # Update giant step power
                        if giant_step < giant_steps - 1 and giant_step > 1:
                            current_giant_power = rescale(relin(mult(current_giant_power, x_giant)))
        return result

    # =========================================================================
    # General BSGS with optimal power decomposition (matches C++ run_core_bsgs)
    # =========================================================================

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
        power_info = PolyReluLayer._compute_power_info(order)
        required, to_compute = PolyReluLayer._determine_required_powers(order, baby_steps, giant_steps, power_info)
        power_depth = {1: 0}
        for n in sorted(to_compute):
            if n <= 1:
                continue
            _, a, b = power_info[n]
            power_depth[n] = max(power_depth[a], power_depth[b]) + 1
        max_depth = max(power_depth[p] for p in required)
        return max_depth + 1

    def _run_bsgs_core(self, x: list[CkksCiphertextNode], get_weight):
        """Core BSGS algorithm. Matches C++ run_core_bsgs for any order.

        Args:
            x: list of input ciphertext nodes
            get_weight: callable(coeff_idx, x_idx) -> plaintext weight node
        """
        order = self.order
        baby_steps = int(np.ceil(np.sqrt(order + 1)))
        giant_steps = int(np.ceil((order + 1) / baby_steps))

        power_info = self._compute_power_info(order)
        required, to_compute = self._determine_required_powers(order, baby_steps, giant_steps, power_info)

        # Validate input levels
        level_cost = self.compute_bsgs_level_cost(order)
        for i, ct in enumerate(x):
            if ct.level < level_cost:
                raise ValueError(
                    f'PolyReluLayer (order={order}): Input ciphertext {i} has insufficient level: '
                    f'{ct.level} < {level_cost}. '
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

            # Baby polynomial output levels
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

                # Add constant term
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

    def call_bsgs(self, x: list[CkksCiphertextNode], weight_pt):
        """BSGS with pre-computed weight plaintexts (eager mode)."""
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_lazy(self, x: list[CkksCiphertextNode], poly_data_source, layer_id: str = ''):
        """BSGS with on-demand weight generation via custom_compute (lazy/mega mode)."""
        weight_cache = {}

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{idx}_{x_idx}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt',
                        'i': idx,
                        'j': x_idx,
                    },
                )
                weight_cache[key] = w_pt
            return weight_cache[key]

        return self._run_bsgs_core(x, get_weight)

    def call_bsgs_feature0d(self, x: list[CkksCiphertextNode], weight_pt):
        """BSGS for Feature0D with pre-computed weight plaintexts (eager mode).

        Identical algorithm to call_bsgs — only weight packing differs,
        which is handled by the caller when building weight_pt.
        """
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_feature0d_lazy(self, x: list[CkksCiphertextNode], poly_data_source, layer_id: str = ''):
        """BSGS for Feature0D with on-demand weight generation (lazy mode)."""
        weight_cache = {}

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                w_pt = CkksPlaintextRingtNode(f'encode_pt_0d_{idx}_{x_idx}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt_feature0d',
                        'i': idx,
                        'j': x_idx,
                    },
                )
                weight_cache[key] = w_pt
            return weight_cache[key]

        return self._run_bsgs_core(x, get_weight)
