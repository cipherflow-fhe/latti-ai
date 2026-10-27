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
from inference.model_generator.layers.poly_relu_base import PolyReluBase

import numpy as np

op_class = 'PolyRelu2D'


def _bsgs_coeff_scales(param, order: int, level: int) -> dict[int, float]:
    """Mirror C++ PolyReluBase::compute_coefficient_scales_bsgs.

    Returns the encoding scale used by the C++ encoder for each coefficient index,
    so the lazy encode_pt node metadata matches what the backend actually produces.
    """
    S = float(param.default_scale)
    q = [float(v) for v in param.q]
    modulus = q[: level + 1]

    # powers[n] = (depth, level, scale)
    powers: dict[int, tuple[int, int, float]] = {1: (0, level, S)}
    for n in range(2, order + 1):
        best_depth = None
        best_a, best_b = 1, n - 1
        for a in range(1, n // 2 + 1):
            b = n - a
            da, _, sa = powers[a]
            db, _, sb = powers[b]
            depth = max(da, db) + 1
            if best_depth is None or depth < best_depth or (
                depth == best_depth and abs(a - b) < abs(best_a - best_b)
            ):
                best_depth = depth
                best_a, best_b = a, b
        _, la, sa = powers[best_a]
        _, lb, sb = powers[best_b]
        result_level = min(la, lb) - 1
        result_scale = sa * sb
        if result_level >= 0 and result_level + 1 < len(modulus):
            result_scale /= modulus[result_level + 1]
        powers[n] = (best_depth, result_level, result_scale)

    baby_steps = int(np.ceil(np.sqrt(order + 1)))
    giant_steps = int(np.ceil((order + 1) / baby_steps))

    required = set(range(1, baby_steps + 1))
    for g in range(1, giant_steps):
        giant_power = g * baby_steps
        if giant_power <= order:
            required.add(giant_power)

    max_depth = 0
    max_power_level = level
    for p in sorted(required):
        depth, p_level, _ = powers[p]
        if depth > max_depth:
            max_depth = depth
            max_power_level = p_level
    bsgs_output_level = max_power_level - 1

    coeff_scale: dict[int, float] = {}
    for g in range(giant_steps):
        if g == 0:
            baby_level = bsgs_output_level
            baby_scale = S
        else:
            giant_power = g * baby_steps
            if giant_power > order:
                break
            _, _, gp_scale = powers[giant_power]
            baby_level = bsgs_output_level + 1
            baby_scale = S * q[baby_level] / gp_scale

        start_idx = g * baby_steps
        end_idx = min(start_idx + baby_steps - 1, order)
        for idx in range(start_idx, end_idx + 1):
            baby_step = idx - start_idx
            if baby_step == 0:
                coeff_scale[idx] = baby_scale
            else:
                _, _, x_scale = powers[baby_step]
                coeff_scale[idx] = baby_scale * q[baby_level + 1] / x_scale
    return coeff_scale


class PolyRelu2D(PolyReluBase):
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

    def get_fhe_op_count_call(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call() for n_ct input ciphertexts, grouped by level.

        order != 4 (Horner evaluation), per ct:
          - (order-1) mult (relin) + rescale  (one per intermediate step)
          - (order-1) add  (add weight_pt[order_idx] before each mult)
          - 1 add  (final coeff0)
          Each rescale reduces level by 1; ops distributed level → level-1 → ... → level-(order-1).

        order == 4 (fixed BSGS), per ct:
          Power construction (powers 2..baby_steps), simple doubling/chaining:
            (baby_steps - 1) mult (relin) + rescale
          Giant power updates (current_giant_power *= x_giant for giant_step in 2..giant_steps-2):
            max(0, giant_steps - 3) mult (relin) + rescale
          Baby poly per giant step (baby_steps-1 non-zero terms each):
            giant_steps * (baby_steps - 1) mult_plain + rescale
            giant_steps * (baby_steps - 2) add  (accumulate terms b>=2)
            giant_steps * 1 add  (add coeff0_pt at b==1)
          Giant combine (giant_step > 0):
            (giant_steps - 1) mult (relin) + rescale + add
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})

        order = self.order
        n_ct = int(n_ct)

        if order != 4:
            # Horner evaluation: each step does add + mult + rescale at current level
            # Level goes from `level` down by 1 each rescale
            # Final add (coeff0) is at level-(order-1) (after last rescale)
            lv = level
            for step in range(order - 1):
                ops[lv]['mult'] += n_ct
                ops[lv]['rescale'] += n_ct
                ops[lv]['add'] += n_ct  # add weight_pt before mult
                lv -= 1
            ops[lv]['add'] += n_ct  # final coeff0 add
        else:
            baby_steps = int(np.ceil(np.sqrt(order + 1)))
            giant_steps = int(np.ceil((order + 1) / baby_steps))

            # Power construction: baby_steps-1 mult+rescale, each at successive levels
            lv = level
            for _ in range(baby_steps - 1):
                ops[lv]['mult'] += n_ct
                ops[lv]['rescale'] += n_ct
                lv -= 1

            # Giant power updates: at current lv
            giant_power_updates = max(0, giant_steps - 3)
            for _ in range(giant_power_updates):
                ops[lv]['mult'] += n_ct
                ops[lv]['rescale'] += n_ct
                lv -= 1

            # Baby poly and giant combine share the same level pair
            # Baby poly: mult_plain+rescale at lv, add at lv-1
            # Giant combine: mult+rescale at lv, add at lv-1
            mult_plain_baby = giant_steps * (baby_steps - 1)
            rescale_baby = giant_steps * (baby_steps - 1)
            add_baby = giant_steps * (baby_steps - 2 + 1)

            mult_giant_comb = giant_steps - 1
            rescale_giant_comb = giant_steps - 1
            add_giant_comb = giant_steps - 1

            ops[lv]['mult_plain'] += mult_plain_baby * n_ct
            ops[lv]['rescale'] += rescale_baby * n_ct
            ops[lv]['mult'] += mult_giant_comb * n_ct
            ops[lv]['rescale'] += rescale_giant_comb * n_ct
            lv -= 1
            ops[lv]['add'] += (add_baby + add_giant_comb) * n_ct

        return dict(ops)

    def call(self, x: list[CkksCiphertextNode], weight_pt):
        x = list(x)  # shallow copy to avoid mutating caller's list
        result = list()
        if self.order != 4:
            for i in range(len(x)):
                res = rescale(mult(x[i], weight_pt[self.order][i]))
                for order_idx in range(self.order - 1, 0, -1):
                    res = add(res, weight_pt[order_idx][i])
                    if res.metadata.level > x[i].metadata.level:
                        res = drop_level(res, res.metadata.level - x[i].metadata.level)
                    if res.metadata.level < x[i].metadata.level:
                        x[i] = drop_level(x[i], x[i].metadata.level - res.metadata.level)
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
                            if x_powers[1].metadata.level > x_powers[i - 1].metadata.level:
                                y = drop_level(x_powers[1], x_powers[1].metadata.level - x_powers[i - 1].metadata.level)
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

                                if x_copy.metadata.level > x_powers[1].metadata.level - 2 + giant_step:
                                    x_copy = drop_level(x_copy, (x_copy.metadata.level - (x_powers[1].metadata.level - 2 + giant_step)))

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

        param = get_fhe_param()
        lid = layer_id or getattr(poly_data_source, 'id', '')
        _scale_cache: dict[int, dict[int, float]] = {}

        def coeff_scales(level: int) -> dict[int, float]:
            if level not in _scale_cache:
                _scale_cache[level] = _bsgs_coeff_scales(param, self.order, level)
            return _scale_cache[level]

        min_required_level = 3 if self.order == 4 else max(self.order - 1, 1)
        for i, ct in enumerate(x):
            if ct.metadata.level < min_required_level:
                raise ValueError(
                    f'PolyReluLayer (order={self.order}): Input ciphertext {i} has insufficient level: '
                    f'{ct.metadata.level} < {min_required_level}. '
                    f'Algorithm will consume {min_required_level} levels, causing negative level. '
                    f'Please use bootstrap before this layer or adjust the network depth.'
                )

        if self.order != 4:
            for i in range(len(x)):
                scales = coeff_scales(x[i].metadata.level)
                res = x[i]
                for order_idx in range(self.order - 1, 0, -1):
                    w_pt = CkksPlaintextNode(
                        f'encode_pt_{lid}_{order_idx}_{i}', is_ringt=True, scale=scales[order_idx]
                    )
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
                    if res.metadata.level > x[i].metadata.level:
                        res = drop_level(res, res.metadata.level - x[i].metadata.level)
                    if res.metadata.level < x[i].metadata.level:
                        x[i] = drop_level(x[i], x[i].metadata.level - res.metadata.level)
                    res = rescale(relin(mult(res, x[i])))
                w_pt = CkksPlaintextNode(f'encode_pt_{lid}_{0}_{i}', is_ringt=True, scale=scales[0])
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
                    w_pt_1 = CkksPlaintextNode(f'encode_pt_{1}_{x_idx}', is_ringt=True)
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
                    w_pt_0 = CkksPlaintextNode(f'encode_pt_{0}_{x_idx}', is_ringt=True)
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

                    initial_level = x[x_idx].metadata.level
                    scales = coeff_scales(initial_level)

                    x_powers = [0 for i in range(baby_steps + 1)]
                    x_powers[1] = x[x_idx]
                    for i in range(2, baby_steps + 1):
                        if i % 2 == 0:
                            half = int(i / 2)
                            tp = relin(mult(x_powers[half], x_powers[half]))
                            x_powers[i] = rescale(tp)
                        else:
                            y = x_powers[1]
                            if x_powers[1].metadata.level > x_powers[i - 1].metadata.level:
                                y = drop_level(x_powers[1], x_powers[1].metadata.level - x_powers[i - 1].metadata.level)
                            x_powers[i] = rescale(relin(mult(x_powers[i - 1], y)))
                    x_giant = x_powers[baby_steps]
                    current_giant_power = x_giant

                    for giant_step in range(0, giant_steps):
                        baby_poly = None
                        coeff0_pt = None
                        has_coeff0 = False

                        for baby_step in range(0, baby_steps):
                            coeff_idx = giant_step * baby_steps + baby_step
                            if coeff_idx > self.order:
                                continue

                            if baby_step == 0:
                                has_coeff0 = True
                                w_pt = CkksPlaintextNode(
                                    f'encode_pt_{lid}_{coeff_idx}_{x_idx}', is_ringt=True, scale=scales[coeff_idx]
                                )
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
                                x_copy = x_powers[baby_step]
                                target_level = initial_level - 2 + giant_step

                                if x_copy.metadata.level > target_level:
                                    x_copy = drop_level(x_copy, x_copy.metadata.level - target_level)

                                w_pt = CkksPlaintextNode(
                                    f'encode_pt_{lid}_{coeff_idx}_{x_idx}', is_ringt=True, scale=scales[coeff_idx]
                                )
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

                            if baby_step == 0:
                                continue
                            elif baby_step == 1:
                                if has_coeff0 and coeff0_pt is not None:
                                    baby_poly = add(term, coeff0_pt)
                                else:
                                    baby_poly = term
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

    def get_fhe_op_count_bsgs_feature2d(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_bsgs_feature2d() for n_ct input ciphertexts, grouped by level.

        call_bsgs_feature2d() delegates entirely to _run_bsgs_core(), so the op
        count is identical to PolyReluBase.get_fhe_op_count().
        See PolyReluBase.get_fhe_op_count() for the detailed breakdown.
        """
        return self.get_fhe_op_count(n_ct, level)

    def call_bsgs_feature2d(self, x: list[CkksCiphertextNode], weight_pt):
        """BSGS with pre-computed weight plaintexts (eager mode)."""
        return self._run_bsgs_core(x, lambda idx, x_idx: weight_pt[idx][x_idx])

    def call_bsgs_lazy(self, x: list[CkksCiphertextNode], poly_data_source, layer_id: str = ''):
        """BSGS with on-demand weight generation via custom_compute (lazy/mega mode)."""
        param = get_fhe_param()
        lid = layer_id or getattr(poly_data_source, 'id', '')
        weight_cache = {}
        _scale_cache: dict[int, dict[int, float]] = {}

        def coeff_scales(level: int) -> dict[int, float]:
            if level not in _scale_cache:
                _scale_cache[level] = _bsgs_coeff_scales(param, self.order, level)
            return _scale_cache[level]

        def get_weight(idx, x_idx):
            key = (idx, x_idx)
            if key not in weight_cache:
                scales = coeff_scales(x[x_idx].metadata.level)
                w_pt = CkksPlaintextNode(
                    f'encode_pt_{lid}_{idx}_{x_idx}', is_ringt=True, scale=scales[idx]
                )
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


# Backward-compatible alias
PolyRelu = PolyRelu2D
