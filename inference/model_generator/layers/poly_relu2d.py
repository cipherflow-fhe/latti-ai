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

op_class = 'PolyReluLayer'


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

                    initial_level = x[x_idx].level

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
                                x_copy = x_powers[baby_step]
                                target_level = initial_level - 2 + giant_step

                                if x_copy.level > target_level:
                                    x_copy = drop_level(x_copy, x_copy.level - target_level)

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


# Backward-compatible alias
PolyRelu = PolyRelu2D
