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


class SoftmaxLayer:
    def __init__(self, n_classes: int):
        if n_classes <= 0:
            raise ValueError(f'n_classes must be positive, got {n_classes}')
        if (n_classes & (n_classes - 1)) != 0:
            raise ValueError(
                f'SoftmaxLayer currently requires power-of-two class count for rotation-sum, got {n_classes}'
            )
        self.n_classes = n_classes
        self.rotate_steps = self._build_rotate_steps(n_classes)

    @staticmethod
    def _build_rotate_steps(n_classes: int):
        steps = []
        step = 1
        while step < n_classes:
            steps.append(step)
            step <<= 1
        return steps

    def _repeated_block_sum(self, x: CkksCiphertextNode, prefix: str):
        total = x
        for step in self.rotate_steps:
            total = add(
                total,
                rotate_cols(total, step, output_id=f'{prefix}_rot_{step}')[0],
                output_id=f'{prefix}_sum_{step}',
            )
        return total

    @staticmethod
    def _eval_exp_poly_v1(
        x: CkksCiphertextNode,
        c5: CkksPlaintextMulNode,
        c4: CkksPlaintextNode,
        c3: CkksPlaintextNode,
        c2: CkksPlaintextNode,
        c1: CkksPlaintextNode,
        c0: CkksPlaintextNode,
        prefix: str,
    ):
        x_lm1 = drop_level(x, 1, output_id=f'{prefix}_x_lm1')
        x_lm2 = drop_level(x, 2, output_id=f'{prefix}_x_lm2')
        x_lm3 = drop_level(x, 3, output_id=f'{prefix}_x_lm3')
        x_lm4 = drop_level(x, 4, output_id=f'{prefix}_x_lm4')

        acc = rescale(mult(x, c5, output_id=f'{prefix}_mul_c5'), output_id=f'{prefix}_acc_1')
        acc = add(acc, c4, output_id=f'{prefix}_acc_2')
        acc = rescale(mult_relin(acc, x_lm1, output_id=f'{prefix}_mul_x3'), output_id=f'{prefix}_acc_3')
        acc = add(acc, c3, output_id=f'{prefix}_acc_4')
        acc = rescale(mult_relin(acc, x_lm2, output_id=f'{prefix}_mul_x2'), output_id=f'{prefix}_acc_5')
        acc = add(acc, c2, output_id=f'{prefix}_acc_6')
        acc = rescale(mult_relin(acc, x_lm3, output_id=f'{prefix}_mul_x1'), output_id=f'{prefix}_acc_7')
        acc = add(acc, c1, output_id=f'{prefix}_acc_8')
        acc = rescale(mult_relin(acc, x_lm4, output_id=f'{prefix}_mul_x0'), output_id=f'{prefix}_acc_9')
        acc = add(acc, c0, output_id=f'{prefix}_poly')

        exp_half = rescale(mult_relin(acc, acc, output_id=f'{prefix}_square_1'), output_id=f'{prefix}_half')
        return rescale(mult_relin(exp_half, exp_half, output_id=f'{prefix}_square_2'), output_id=f'{prefix}_out')

    @staticmethod
    def _eval_recip_poly_v1(
        x: CkksCiphertextNode,
        c3: CkksPlaintextMulNode,
        c2: CkksPlaintextNode,
        c1: CkksPlaintextNode,
        c0: CkksPlaintextNode,
        prefix: str,
    ):
        x_lm1 = drop_level(x, 1, output_id=f'{prefix}_x_lm1')
        x_lm2 = drop_level(x, 2, output_id=f'{prefix}_x_lm2')

        acc = rescale(mult(x, c3, output_id=f'{prefix}_mul_c3'), output_id=f'{prefix}_acc_1')
        acc = add(acc, c2, output_id=f'{prefix}_acc_2')
        acc = rescale(mult_relin(acc, x_lm1, output_id=f'{prefix}_mul_x1'), output_id=f'{prefix}_acc_3')
        acc = add(acc, c1, output_id=f'{prefix}_acc_4')
        acc = rescale(mult_relin(acc, x_lm2, output_id=f'{prefix}_mul_x0'), output_id=f'{prefix}_acc_5')
        return add(acc, c0, output_id=f'{prefix}_out')

    def call(
        self,
        logits: CkksCiphertextNode,
        pt_quarter: CkksPlaintextRingtNode,
        pt_inv_classes: CkksPlaintextRingtNode,
        exp_coeffs: list,
        recip_coeffs: list,
        output_prefix: str,
    ):
        if len(exp_coeffs) != 6:
            raise ValueError(f'exp_coeffs must have 6 items, got {len(exp_coeffs)}')
        if len(recip_coeffs) != 4:
            raise ValueError(f'recip_coeffs must have 4 items, got {len(recip_coeffs)}')

        logits_quarter = rescale(
            mult(logits, pt_quarter, output_id=f'{output_prefix}_logits_quarter_mul'),
            output_id=f'{output_prefix}_logits_quarter',
        )
        quarter_sum = self._repeated_block_sum(logits_quarter, f'{output_prefix}_quarter_sum')

        mean_quarter = rescale(
            mult(quarter_sum, pt_inv_classes, output_id=f'{output_prefix}_mean_quarter_mul'),
            output_id=f'{output_prefix}_mean_quarter',
        )
        logits_quarter_lm1 = drop_level(logits_quarter, 1, output_id=f'{output_prefix}_logits_quarter_lm1')
        centered_quarter = sub(logits_quarter_lm1, mean_quarter, output_id=f'{output_prefix}_centered_quarter')

        exp_logits = self._eval_exp_poly_v1(
            centered_quarter,
            exp_coeffs[5],
            exp_coeffs[4],
            exp_coeffs[3],
            exp_coeffs[2],
            exp_coeffs[1],
            exp_coeffs[0],
            prefix=f'{output_prefix}_exp',
        )
        denom = self._repeated_block_sum(exp_logits, f'{output_prefix}_denom')
        inv_denom = self._eval_recip_poly_v1(
            denom,
            recip_coeffs[3],
            recip_coeffs[2],
            recip_coeffs[1],
            recip_coeffs[0],
            prefix=f'{output_prefix}_recip',
        )

        exp_logits_lm3 = drop_level(exp_logits, 3, output_id=f'{output_prefix}_exp_lm3')
        return rescale(
            mult_relin(exp_logits_lm3, inv_denom, output_id=f'{output_prefix}_softmax_mul'),
            output_id=f'{output_prefix}_softmax',
        )
