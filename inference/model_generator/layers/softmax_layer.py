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

op_class = 'SoftmaxLayer'


def compute_poly_depth(degree: int) -> int:
    if degree <= 1:
        return 0
    import math
    return int(math.ceil(math.log2(degree)))


def compute_standard_poly_coeffs(func, degree: int, a: float, b: float):
    from numpy.polynomial import chebyshev as C, polynomial as P
    n = degree + 1
    x_samples = np.linspace(a, b, max(n * 20, 200))
    y_samples = np.array([func(x) for x in x_samples])
    cheb_fit = C.Chebyshev.fit(x_samples, y_samples, degree, domain=[a, b])
    poly_fit = cheb_fit.convert(kind=P.Polynomial)
    coeffs = list(poly_fit.coef)
    while len(coeffs) < n:
        coeffs.append(0.0)
    return coeffs[:n]


class SoftmaxLayer:
    def __init__(self, exp_range, exp_degree, inv_range, inv_degree, n_channel, n_channel_per_ct, temperature=1.0):
        self.exp_range = exp_range
        self.exp_degree = exp_degree
        self.inv_range = inv_range
        self.inv_degree = inv_degree
        self.n_channel = n_channel
        self.n_channel_per_ct = n_channel_per_ct
        self.temperature = temperature

        self.n_slots = min(n_channel, n_channel_per_ct)

        self.exp_depth = compute_poly_depth(self.exp_degree)
        self.inv_depth = compute_poly_depth(self.inv_degree)
        self.total_depth = self.exp_depth + self.inv_depth + 4
        if self.temperature != 1.0:
            self.total_depth += 1

        self.plaintext_nodes = []

    def _sum_slots(self, x: CkksCiphertextNode, n_slots: int, prefix: str):
        import inference.lattisense.frontend.custom_task as ct
        total_slots = ct.g_param.n // 2

        pt_mask = CkksPlaintextRingtNode(f'{prefix}_mask')
        self.plaintext_nodes.append(pt_mask)
        masked_x = mult(x, pt_mask, output_id=f'{prefix}_masked')
        masked_x = rescale(masked_x, output_id=f'{prefix}_rescale_mask')

        res = masked_x
        rot_idx = 0
        step = 1
        while step < total_slots:
            rotated = rotate_cols(res, step, output_id=f'{prefix}_rot_{rot_idx}')[0]
            res = add(res, rotated, output_id=f'{prefix}_sum_{rot_idx}')
            step <<= 1
            rot_idx += 1

        return res

    def _eval_poly(self, x: CkksCiphertextNode, func, range_tuple, degree: int, prefix: str):
        a, b = range_tuple
        alpha_node = CkksPlaintextRingtNode(f'{prefix}_alpha')
        beta_node = CkksPlaintextRingtNode(f'{prefix}_beta')
        self.plaintext_nodes.append(alpha_node)
        self.plaintext_nodes.append(beta_node)

        x_sub = mult(x, alpha_node, output_id=f'{prefix}_sub_mult')
        x_sub = rescale(x_sub, output_id=f'{prefix}_sub_rescale')
        x_sub = add(x_sub, beta_node, output_id=f'{prefix}_sub_add')

        coeffs = [CkksPlaintextRingtNode(f'{prefix}_c{i}') for i in range(degree + 1)]
        for c in coeffs:
            self.plaintext_nodes.append(c)
        
        if degree == 0:
            return coeffs[0]
        
        if degree == 1:
            result = mult(x_sub, coeffs[1], output_id=f'{prefix}_mult_1')
            result = rescale(result, output_id=f'{prefix}_rescale_1')
            result = add(result, coeffs[0], output_id=f'{prefix}_add_0')
            return result
        
        log_degree = int(np.ceil(np.log2(degree)))
        
        x_powers = {1: x_sub}
        current = x_sub
        for i in range(1, log_degree):
            next_power = 1 << i
            current = mult_relin(current, current, output_id=f'{prefix}_x_pow_{next_power}')
            current = rescale(current, output_id=f'{prefix}_x_pow_rescale_{next_power}')
            x_powers[next_power] = current
        
        def eval_range(start, end, level):
            if start > end:
                return None
            
            if start == end:
                return coeffs[start]
            
            mid = 1
            while mid * 2 <= (end - start):
                mid *= 2
            
            high = eval_range(start + mid, end, level - 1)
            low = eval_range(start, start + mid - 1, level)
            
            if high is None:
                return low
            if low is None:
                return high
            
            x_pow = x_powers.get(mid, x_sub)
            
            if isinstance(high, (CkksCiphertextNode, CkksCiphertext3Node)):
                if isinstance(x_pow, (CkksCiphertextNode, CkksCiphertext3Node)):
                    while x_pow.level > high.level:
                        x_pow = drop_level(x_pow, 1, output_id=f'{prefix}_pow_drop_{mid}')
                    while high.level > x_pow.level:
                        high = drop_level(high, 1, output_id=f'{prefix}_high_drop_{mid}')
                    result = mult_relin(high, x_pow, output_id=f'{prefix}_mul_{start}_{end}')
                else:
                    result = mult(high, x_pow, output_id=f'{prefix}_mul_{start}_{end}')
                result = rescale(result, output_id=f'{prefix}_rescale_{start}_{end}')
            else:
                if isinstance(x_pow, (CkksCiphertextNode, CkksCiphertext3Node)):
                    result = mult(x_pow, high, output_id=f'{prefix}_mul_{start}_{end}')
                    result = rescale(result, output_id=f'{prefix}_rescale_{start}_{end}')
                else:
                    result = mult(high, x_pow, output_id=f'{prefix}_mul_{start}_{end}')
                    result = rescale(result, output_id=f'{prefix}_rescale_{start}_{end}')
            
            if isinstance(result, (CkksCiphertextNode, CkksCiphertext3Node)) and \
               isinstance(low, (CkksCiphertextNode, CkksCiphertext3Node)):
                while result.level > low.level:
                    result = drop_level(result, 1, output_id=f'{prefix}_res_drop_{start}_{end}')
                while low.level > result.level:
                    low = drop_level(low, 1, output_id=f'{prefix}_low_drop_{start}_{end}')
            
            result = add(result, low, output_id=f'{prefix}_add_{start}_{end}')
            return result
        
        result = eval_range(0, degree, x_sub.level)
        return result if result is not None else coeffs[0]

    def _eval_exp_poly(self, x: CkksCiphertextNode, prefix: str):
        return self._eval_poly(x, np.exp, self.exp_range, self.exp_degree, prefix)

    def _eval_inv_poly(self, x: CkksCiphertextNode, prefix: str):
        return self._eval_poly(x, lambda v: 1.0/v, self.inv_range, self.inv_degree, prefix)

    def call(self, x: list[CkksCiphertextNode]):
        result = []

        for i, ct in enumerate(x):
            prefix = f'sm_{i}'

            if ct.level < self.total_depth:
                raise RuntimeError(f"SoftmaxLayer: input level too low. Need at least {self.total_depth}, got {ct.level}")

            if self.temperature != 1.0:
                temp_node = CkksPlaintextRingtNode(f'{prefix}_temp')
                self.plaintext_nodes.append(temp_node)
                scaled_ct = mult(ct, temp_node, output_id=f'{prefix}_temp_mult')
                scaled_ct = rescale(scaled_ct, output_id=f'{prefix}_temp_rescale')
            else:
                scaled_ct = ct

            exp_ct = self._eval_exp_poly(scaled_ct, f'{prefix}_exp')
            sum_exp = self._sum_slots(exp_ct, self.n_slots, prefix=f'{prefix}_sum')
            inv_sum = self._eval_inv_poly(sum_exp, f'{prefix}_inv')

            exp_x_aligned = exp_ct
            while exp_x_aligned.level > inv_sum.level:
                exp_x_aligned = drop_level(exp_x_aligned, 1, output_id=f'{prefix}_exp_drop')

            out3 = mult_relin(exp_x_aligned, inv_sum, output_id=f'{prefix}_mult')
            if out3.level > 0:
                out = rescale(out3, output_id=f'{prefix}_rescale')
            else:
                out = out3

            result.append(out)

        return result

    def call_custom_compute(self, x: list[CkksCiphertextNode], data_source):
        return self.call(x)

    def get_plaintext_values(self, n_total_slots: int):
        values = []

        if self.temperature != 1.0:
            values.append([1.0 / self.temperature] * n_total_slots)

        a_exp, b_exp = self.exp_range
        exp_func_mapped = lambda x_prime, a=a_exp, b=b_exp: np.exp((b - a) / 2 * x_prime + (a + b) / 2)
        exp_coeffs = compute_standard_poly_coeffs(exp_func_mapped, self.exp_degree, -1.0, 1.0)
        exp_alpha = 2.0 / (b_exp - a_exp)
        exp_beta = -(a_exp + b_exp) / (b_exp - a_exp)

        a_inv, b_inv = self.inv_range
        inv_func_mapped = lambda x_prime, a=a_inv, b=b_inv: 1.0 / ((b - a) / 2 * x_prime + (a + b) / 2)
        inv_coeffs = compute_standard_poly_coeffs(inv_func_mapped, self.inv_degree, -1.0, 1.0)
        inv_alpha = 2.0 / (b_inv - a_inv)
        inv_beta = -(a_inv + b_inv) / (b_inv - a_inv)

        mask_values = [0.0] * n_total_slots
        for i in range(self.n_slots):
            mask_values[i] = 1.0

        values.append([exp_alpha] * n_total_slots)
        values.append([exp_beta] * n_total_slots)
        for c in exp_coeffs:
            values.append([c] * n_total_slots)
        values.append(mask_values)
        values.append([inv_alpha] * n_total_slots)
        values.append([inv_beta] * n_total_slots)
        for c in inv_coeffs:
            values.append([c] * n_total_slots)

        return values
