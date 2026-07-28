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

from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import inference.lattisense.frontend.custom_task as custom_task


@dataclass
class MatPolyPowerInfo:
    depth: int
    level: int
    scale: float
    decomp_a: int
    decomp_b: int
    computed: bool


def _ceil_log2_int(n: int) -> int:
    depth = 0
    span = 1
    while span < n:
        span <<= 1
        depth += 1
    return depth


def _next_power_of_two_int(n: int) -> int:
    span = 1
    while span < n:
        span <<= 1
    return span


def _stockmeyer_baby_poly_count(order: int) -> int:
    return (order + 4) // 4


def _validate_horner_baby_steps(baby_steps: int):
    if baby_steps not in (4, 8):
        raise ValueError('MatPolyBase Stockmeyer Horner supports only baby_steps 4 or 8')


def _log2_horner_baby_steps(baby_steps: int) -> int:
    _validate_horner_baby_steps(baby_steps)
    return 2 if baby_steps == 4 else 3


def _stockmeyer_horner_baby_poly_count(order: int, baby_steps: int) -> int:
    _validate_horner_baby_steps(baby_steps)
    return (order + baby_steps) // baby_steps


class MatPolyBase:
    """Python model-generator counterpart of C++ MatPolyBase planning logic."""

    def __init__(self, order: int):
        self.order = int(order)
        self.N = 0
        self.level = 0

        self.modulus: list[int] = []

        self.stockmeyer_baby_steps = 4
        self.stockmeyer_n_baby_polys = 0
        self.stockmeyer_powers: dict[int, MatPolyPowerInfo] = {}
        self.stockmeyer_baby_poly_output_scale: list[float] = []
        self.stockmeyer_baby_poly_output_level: list[int] = []
        self.stockmeyer_output_level = 0
        self.cached_stockmeyer_coeff_scale: dict[int, float] = {}
        self.cached_stockmeyer_level_order: dict[int, int] = {}
        self.stockmeyer_initialized = False

        self.stockmeyer_horner_baby_steps = 8
        self.stockmeyer_horner_n_baby_polys = 0
        self.stockmeyer_horner_powers: dict[int, MatPolyPowerInfo] = {}
        self.stockmeyer_horner_baby_poly_output_scale: list[float] = []
        self.stockmeyer_horner_baby_poly_output_level: list[int] = []
        self.stockmeyer_horner_output_level = 0
        self.cached_stockmeyer_horner_coeff_scale: dict[int, float] = {}
        self.cached_stockmeyer_horner_level_order: dict[int, int] = {}
        self.stockmeyer_horner_initialized = False

    @staticmethod
    def compute_stockmeyer_level_cost(order: int) -> int:
        if order <= 0 or order >= 64:
            raise ValueError('MatPolyBase Stockmeyer supports only order < 64')
        if order < 2:
            return 1

        n_baby_polys = _stockmeyer_baby_poly_count(order)
        return 2 + _ceil_log2_int(n_baby_polys)

    @staticmethod
    def compute_stockmeyer_horner_level_cost(order: int, baby_steps: int = 8) -> int:
        _validate_horner_baby_steps(baby_steps)
        if order <= 0 or order >= 64:
            raise ValueError('MatPolyBase Stockmeyer Horner supports only order < 64')

        n_baby_polys = _stockmeyer_horner_baby_poly_count(order, baby_steps)
        return (n_baby_polys - 1) + _log2_horner_baby_steps(baby_steps)

    def _param(self):
        param = custom_task.g_param
        if param is None:
            raise RuntimeError('Please call set_fhe_param() before using MatPolyBase.')
        return param

    def _default_scale(self) -> float:
        return float(self._param().scale)

    def _q(self, level: int) -> float:
        return float(self._param().q[level])

    def _set_level(self, level: int):
        self.level = int(level)
        self.N = int(self._param().n)
        self.modulus = [self._param().q[i] for i in range(self.level + 1)]

    def init_stockmeyer(self, level: int):
        level = int(level)
        if self.stockmeyer_initialized and self.level == level:
            return

        if self.order <= 0 or self.order >= 64:
            raise ValueError('MatPolyBase Stockmeyer supports only order < 64')

        level_cost = self.compute_stockmeyer_level_cost(self.order)
        if level < level_cost:
            raise ValueError(f'MatPolyBase Stockmeyer input level is too low for order {self.order}')

        self._set_level(level)
        self.stockmeyer_baby_steps = 4
        self.stockmeyer_n_baby_polys = _stockmeyer_baby_poly_count(self.order)
        self.stockmeyer_output_level = self.level - level_cost

        self.compute_stockmeyer_power_info()
        self.compute_coefficient_scales_stockmeyer(
            self.cached_stockmeyer_coeff_scale,
            self.cached_stockmeyer_level_order,
        )
        self.stockmeyer_initialized = True

    def init_stockmeyer_horner(self, level: int, baby_steps: int = 8):
        _validate_horner_baby_steps(baby_steps)
        level = int(level)
        if (
            self.stockmeyer_horner_initialized
            and self.level == level
            and self.stockmeyer_horner_baby_steps == baby_steps
        ):
            return

        if self.order <= 0 or self.order >= 64:
            raise ValueError('MatPolyBase Stockmeyer Horner supports only order < 64')

        level_cost = self.compute_stockmeyer_horner_level_cost(self.order, baby_steps)
        if level < level_cost:
            raise ValueError(f'MatPolyBase Stockmeyer Horner input level is too low for order {self.order}')

        self._set_level(level)
        self.stockmeyer_horner_baby_steps = baby_steps
        self.stockmeyer_horner_n_baby_polys = _stockmeyer_horner_baby_poly_count(self.order, baby_steps)
        self.stockmeyer_horner_output_level = self.level - level_cost

        self.compute_stockmeyer_horner_power_info()
        self.compute_coefficient_scales_stockmeyer_horner(
            self.cached_stockmeyer_horner_coeff_scale,
            self.cached_stockmeyer_horner_level_order,
        )
        self.stockmeyer_horner_initialized = True

    def compute_stockmeyer_power_info(self):
        scale = self._default_scale()
        self.stockmeyer_powers.clear()
        self.stockmeyer_powers[1] = MatPolyPowerInfo(0, self.level, scale, 0, 0, True)

        def add_square_power(power: int, half_power: int):
            half = self.stockmeyer_powers[half_power]
            result_level = half.level - 1
            if result_level < 0 or half.level >= len(self.modulus):
                raise ValueError('MatPolyBase Stockmeyer power level is out of range')
            result_scale = (half.scale / float(self.modulus[half.level])) * half.scale
            self.stockmeyer_powers[power] = MatPolyPowerInfo(
                half.depth + 1,
                result_level,
                result_scale,
                half_power,
                half_power,
                True,
            )

        max_power = 2 if self.order >= 2 else 1
        if self.stockmeyer_n_baby_polys > 1:
            tree_span = _next_power_of_two_int(self.stockmeyer_n_baby_polys)
            max_power = max(max_power, 2 * tree_span)

        power = 2
        while power <= max_power:
            add_square_power(power, power // 2)
            power <<= 1

    def compute_stockmeyer_horner_power_info(self):
        scale = self._default_scale()
        self.stockmeyer_horner_powers.clear()
        self.stockmeyer_horner_powers[1] = MatPolyPowerInfo(0, self.level, scale, 0, 0, True)

        def add_square_power(power: int, half_power: int):
            half = self.stockmeyer_horner_powers[half_power]
            result_level = half.level - 1
            if result_level < 0 or half.level >= len(self.modulus):
                raise ValueError('MatPolyBase Stockmeyer Horner power level is out of range')
            result_scale = (half.scale / float(self.modulus[half.level])) * half.scale
            self.stockmeyer_horner_powers[power] = MatPolyPowerInfo(
                half.depth + 1,
                result_level,
                result_scale,
                half_power,
                half_power,
                True,
            )

        power = 2
        while power <= self.stockmeyer_horner_baby_steps:
            add_square_power(power, power // 2)
            power <<= 1

    def compute_coefficient_scales_stockmeyer(self, coeff_scale: dict[int, float], level_order: dict[int, int]):
        coeff_scale.clear()
        level_order.clear()

        scale = self._default_scale()
        a1 = self.stockmeyer_powers[1].scale
        self.stockmeyer_baby_poly_output_scale = [0.0] * self.stockmeyer_n_baby_polys
        self.stockmeyer_baby_poly_output_level = [-1] * self.stockmeyer_n_baby_polys

        lout = self.stockmeyer_output_level
        tree_span = _next_power_of_two_int(self.stockmeyer_n_baby_polys)

        def assign_targets(start: int, span: int, actual_count: int, target_level: int, target_scale: float):
            if actual_count <= 0:
                return
            if span == 1:
                self.stockmeyer_baby_poly_output_level[start] = target_level
                self.stockmeyer_baby_poly_output_scale[start] = target_scale
                return

            half = span // 2
            left_count = min(actual_count, half)
            right_count = actual_count - left_count
            assign_targets(start, half, left_count, target_level, target_scale)

            if right_count > 0:
                combine_power = self.stockmeyer_baby_steps * half
                power_info = self.stockmeyer_powers[combine_power]
                right_level = target_level + 1
                right_scale = (target_scale / power_info.scale) * self._q(right_level)
                assign_targets(start + half, half, right_count, right_level, right_scale)

        assign_targets(0, tree_span, self.stockmeyer_n_baby_polys, lout, scale)

        for j in range(self.stockmeyer_n_baby_polys):
            target_level = self.stockmeyer_baby_poly_output_level[j]
            target_scale = self.stockmeyer_baby_poly_output_scale[j]
            if target_level < 0 or target_level > self.level:
                raise ValueError('MatPolyBase Stockmeyer baby polynomial target level is out of range')

            base = j * self.stockmeyer_baby_steps
            if base <= self.order:
                level_order[base] = target_level
                coeff_scale[base] = target_scale
            if base + 1 <= self.order:
                if target_level + 1 > self.level:
                    raise ValueError('MatPolyBase Stockmeyer linear coefficient target level is out of range')
                level_order[base + 1] = target_level + 1
                coeff_scale[base + 1] = (target_scale / a1) * self._q(target_level + 1)
            if base + 2 <= self.order:
                if target_level + 1 > self.level:
                    raise ValueError('MatPolyBase Stockmeyer quadratic coefficient target level is out of range')
                a2 = self.stockmeyer_powers[2].scale
                level_order[base + 2] = target_level + 1
                coeff_scale[base + 2] = (target_scale / a2) * self._q(target_level + 1)
            if base + 3 <= self.order:
                if target_level + 2 > self.level:
                    raise ValueError('MatPolyBase Stockmeyer cubic coefficient target level is out of range')
                a2 = self.stockmeyer_powers[2].scale
                level_order[base + 3] = target_level + 2
                coeff_scale[base + 3] = ((target_scale / a2) * self._q(target_level + 1) / a1) * self._q(
                    target_level + 2
                )

    def compute_coefficient_scales_stockmeyer_horner(self, coeff_scale: dict[int, float], level_order: dict[int, int]):
        coeff_scale.clear()
        level_order.clear()

        scale = self._default_scale()
        a1 = self.stockmeyer_horner_powers[1].scale
        a2 = self.stockmeyer_horner_powers[2].scale
        a4 = self.stockmeyer_horner_powers[4].scale
        ab = self.stockmeyer_horner_powers[self.stockmeyer_horner_baby_steps].scale

        self.stockmeyer_horner_baby_poly_output_scale = [0.0] * self.stockmeyer_horner_n_baby_polys
        self.stockmeyer_horner_baby_poly_output_level = [-1] * self.stockmeyer_horner_n_baby_polys

        lout = self.stockmeyer_horner_output_level
        self.stockmeyer_horner_baby_poly_output_level[0] = lout
        self.stockmeyer_horner_baby_poly_output_scale[0] = scale

        for j in range(1, self.stockmeyer_horner_n_baby_polys):
            target_level = lout + j
            if target_level < 0 or target_level > self.level:
                raise ValueError('MatPolyBase Stockmeyer Horner baby polynomial target level is out of range')
            self.stockmeyer_horner_baby_poly_output_level[j] = target_level
            self.stockmeyer_horner_baby_poly_output_scale[j] = (
                self.stockmeyer_horner_baby_poly_output_scale[j - 1] / ab
            ) * self._q(target_level)

        def set_coeff(coeff_idx: int, coeff_level: int, value: float):
            if coeff_level < 0 or coeff_level > self.level:
                raise ValueError('MatPolyBase Stockmeyer Horner coefficient target level is out of range')
            level_order[coeff_idx] = coeff_level
            coeff_scale[coeff_idx] = value

        for j in range(self.stockmeyer_horner_n_baby_polys):
            target_level = self.stockmeyer_horner_baby_poly_output_level[j]
            target_scale = self.stockmeyer_horner_baby_poly_output_scale[j]
            if target_level < 0 or target_level > self.level:
                raise ValueError('MatPolyBase Stockmeyer Horner baby polynomial target level is out of range')

            base = j * self.stockmeyer_horner_baby_steps
            if base <= self.order:
                set_coeff(base, target_level, target_scale)
            if base + 1 <= self.order:
                set_coeff(base + 1, target_level + 1, (target_scale / a1) * self._q(target_level + 1))
            if base + 2 <= self.order:
                set_coeff(base + 2, target_level + 1, (target_scale / a2) * self._q(target_level + 1))
            if base + 3 <= self.order:
                set_coeff(
                    base + 3,
                    target_level + 2,
                    ((target_scale / a2) * self._q(target_level + 1) / a1) * self._q(target_level + 2),
                )

            if self.stockmeyer_horner_baby_steps == 4:
                continue

            if base + 4 <= self.order:
                set_coeff(base + 4, target_level + 1, (target_scale / a4) * self._q(target_level + 1))
            if base + 5 <= self.order:
                set_coeff(
                    base + 5,
                    target_level + 2,
                    ((target_scale / a4) * self._q(target_level + 1) / a1) * self._q(target_level + 2),
                )
            if base + 6 <= self.order:
                set_coeff(
                    base + 6,
                    target_level + 2,
                    ((target_scale / a4) * self._q(target_level + 1) / a2) * self._q(target_level + 2),
                )
            if base + 7 <= self.order:
                set_coeff(
                    base + 7,
                    target_level + 3,
                    (((target_scale / a4) * self._q(target_level + 1) / a2) * self._q(target_level + 2) / a1)
                    * self._q(target_level + 3),
                )
