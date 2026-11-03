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
import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.matrix_param_ops import custom_compute, mult, add
from inference.model_generator.layers.mat_poly_base import MatPolyBase


op_class = 'ParBlockColMajorPoly'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass
class _StockmeyerNode:
    ct: CkksCiphertextNode | None = None
    const_coeff_idx: int = -1
    target_level: int = -1
    target_scale: float = 0.0
    has_ct: bool = False
    const_only: bool = False


class ParBlockColMajorPoly(MatPolyBase):
    """Python model-generator counterpart of C++ ParBlockColMajorPoly."""

    def __init__(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, order: int, block_size: int):
        super().__init__(order)
        assert self.order > 0
        self._init_layout(shape, head_shape, n_heads, n_slot, block_size)

    def _init_layout(self, shape: tuple, head_shape: tuple, n_heads: int, n_slot: int, block_size: int):
        self.n_prepad, self.total_cols = map(int, shape)
        self.m_prepad = int(head_shape[1])
        self.H_prepad, self.n_slot = int(n_heads), int(n_slot)
        self.d = int(block_size)
        assert self.d > 0 and self.d & (self.d-1) == 0 and n_slot >= self.d*self.d
        hp = _next_pow2(self.H_prepad)
        self.u = min(hp, n_slot//(self.d*self.d))
        if self.u == 1: hp = self.H_prepad
        self.G = hp//self.u
        self.Br = math.ceil(self.n_prepad/self.d)
        self.Bc = math.ceil(self.m_prepad/self.d)
        self.cts_per_mb = self.Br*self.Bc*self.G
        self.n_mb = math.ceil(self.total_cols/(self.H_prepad*self.m_prepad))
        self.total_cts = self.n_mb*self.cts_per_mb

    def _ct_index(self, mb: int, ct_local: int) -> int:
        return mb * self.cts_per_mb + ct_local

    def _drop_to_level(self, ct, target_level: int, label: str):
        if target_level < 0:
            raise ValueError(f'ParBlockColMajorPoly Stockmeyer {label}: negative target level')
        if ct.level < target_level:
            raise ValueError(
                f'ParBlockColMajorPoly Stockmeyer {label}: cannot raise level from {ct.level} to {target_level}'
            )
        if ct.level > target_level:
            return drop_level(ct, ct.level - target_level)
        return ct

    def call(self, input_cts: list, stockmeyer_weight_pt: list) -> list:
        return self.call_stockmeyer(input_cts, stockmeyer_weight_pt)

    def call_stockmeyer(self, input_cts: list, stockmeyer_weight_pt: list) -> list:
        assert len(input_cts) == self.total_cts
        assert len(stockmeyer_weight_pt) == self.order + 1
        for row in stockmeyer_weight_pt:
            assert len(row) == self.total_cts
        if not input_cts:
            return []
        for ct in input_cts:
            assert ct.level == input_cts[0].level

        self.init_stockmeyer(input_cts[0].level)
        if self.order <= 0 or self.order >= 64:
            raise ValueError('ParBlockColMajorPoly Stockmeyer supports only 1 <= order < 64')

        result = [None] * len(input_cts)

        for x_idx, x in enumerate(input_cts):
            x_powers = {1: x}

            for power, info in sorted(self.stockmeyer_powers.items()):
                if power <= 1:
                    continue
                half = x_powers[info.decomp_a]
                x_powers[power] = rescale(relin(mult(half, half)))

            def get_coeff(coeff_idx: int):
                if self.cached_stockmeyer_level_order.get(coeff_idx) is None:
                    raise ValueError(f'ParBlockColMajorPoly Stockmeyer: missing coefficient target {coeff_idx}')
                return stockmeyer_weight_pt[coeff_idx][x_idx]

            def add_const_coeff(acc, coeff_idx: int):
                return add(acc, get_coeff(coeff_idx))

            def multiply_plain_term(power_ct, coeff_idx: int, mult_level: int, label: str):
                power_copy = self._drop_to_level(power_ct, mult_level, label + '_power')
                return rescale(mult(power_copy, get_coeff(coeff_idx)))

            def clone_node(node: _StockmeyerNode) -> _StockmeyerNode:
                return _StockmeyerNode(
                    ct=node.ct,
                    const_coeff_idx=node.const_coeff_idx,
                    target_level=node.target_level,
                    target_scale=node.target_scale,
                    has_ct=node.has_ct,
                    const_only=node.const_only,
                )

            def eval_baby_node(baby_idx: int) -> _StockmeyerNode:
                base = baby_idx * self.stockmeyer_baby_steps
                if base > self.order:
                    raise ValueError(f'ParBlockColMajorPoly Stockmeyer: missing baby polynomial {baby_idx}')

                target_level = self.stockmeyer_baby_poly_output_level[baby_idx]
                target_scale = self.stockmeyer_baby_poly_output_scale[baby_idx]
                node = _StockmeyerNode(target_level=target_level, target_scale=target_scale)

                acc = None

                def add_term(term):
                    nonlocal acc
                    acc = term if acc is None else add(acc, term)

                if base + 1 <= self.order:
                    add_term(
                        multiply_plain_term(
                            x_powers[1],
                            base + 1,
                            target_level + 1,
                            f'P{baby_idx}_c1x',
                        )
                    )

                if base + 2 <= self.order:
                    add_term(
                        multiply_plain_term(
                            x_powers[2],
                            base + 2,
                            target_level + 1,
                            f'P{baby_idx}_c2x2',
                        )
                    )

                if base + 3 <= self.order:
                    c3x = multiply_plain_term(
                        x_powers[1],
                        base + 3,
                        target_level + 2,
                        f'P{baby_idx}_c3x',
                    )
                    x2_for_c3 = self._drop_to_level(x_powers[2], target_level + 1, f'P{baby_idx}_x2_for_c3')
                    add_term(rescale(relin(mult(c3x, x2_for_c3))))

                if acc is not None:
                    node.ct = add_const_coeff(acc, base)
                    node.has_ct = True
                    return node

                node.const_coeff_idx = base
                node.const_only = True
                return node

            def combine_with_power(left: _StockmeyerNode, right: _StockmeyerNode, power: int, label: str):
                if not left.has_ct or left.ct is None:
                    raise ValueError(f'ParBlockColMajorPoly Stockmeyer: left node is not ciphertext in {label}')

                combined_node = _StockmeyerNode(
                    target_level=left.target_level,
                    target_scale=left.target_scale,
                )
                mult_level = left.target_level + 1
                if right.target_level != mult_level:
                    raise ValueError(f'ParBlockColMajorPoly Stockmeyer: right node target level mismatch in {label}')

                left_copy = self._drop_to_level(left.ct, left.target_level, label + '_left')
                power_copy = self._drop_to_level(x_powers[power], mult_level, label + f'_x{power}')

                if right.has_ct:
                    assert right.ct is not None
                    right_copy = self._drop_to_level(right.ct, mult_level, label + '_right')
                    term = rescale(relin(mult(right_copy, power_copy)))
                elif right.const_only:
                    term = rescale(mult(power_copy, get_coeff(right.const_coeff_idx)))
                else:
                    raise ValueError(f'ParBlockColMajorPoly Stockmeyer: empty right node in {label}')

                combined_node.ct = add(left_copy, term)
                combined_node.has_ct = True
                return combined_node

            nodes = [eval_baby_node(j) for j in range(self.stockmeyer_n_baby_polys)]

            combine_power = self.stockmeyer_baby_steps
            combine_round = 0
            while len(nodes) > 1:
                next_nodes = []
                for i in range(0, len(nodes), 2):
                    if i + 1 >= len(nodes):
                        next_nodes.append(clone_node(nodes[i]))
                        continue
                    next_nodes.append(
                        combine_with_power(
                            nodes[i],
                            nodes[i + 1],
                            combine_power,
                            f'combine_{combine_round}_{i // 2}',
                        )
                    )
                nodes = next_nodes
                combine_power *= 2
                combine_round += 1

            if not nodes or not nodes[0].has_ct or nodes[0].ct is None:
                raise ValueError('ParBlockColMajorPoly Stockmeyer: result is not a ciphertext')
            result[x_idx] = nodes[0].ct

        return result

    def _make_stockmeyer_weight_pts(self, data_source) -> list:
        weight_pt = []
        source_id = getattr(data_source, 'id', op_class)
        for coeff_idx in range(self.order + 1):
            row = [None] * self.total_cts
            for mb in range(self.n_mb):
                for ct_local in range(self.cts_per_mb):
                    ct_idx = self._ct_index(mb, ct_local)
                    node = CkksPlaintextRingtNode(f'encode_pt_{source_id}_c{coeff_idx}_{mb}_{ct_local}')
                    custom_compute(
                        inputs=[data_source],
                        output=node,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'stockmeyer_coeff_pt',
                            'coeff_idx': coeff_idx,
                            'ct_idx': ct_idx,
                            'mb': mb,
                            'ct_local': ct_local,
                        },
                    )
                    row[ct_idx] = node
            weight_pt.append(row)
        return weight_pt

    def call_custom_compute(self, input_cts: list, data_source) -> list:
        if input_cts:
            self.init_stockmeyer(input_cts[0].level)
        return self.call_stockmeyer(input_cts, self._make_stockmeyer_weight_pts(data_source))

    def get_fhe_op_count(self, n_ct: int | None = None, level: int | None = None) -> dict:
        """Count the Stockmeyer DAG without requiring frontend encoding parameters."""
        n = self.total_cts if n_ct is None else n_ct
        level = self.level if level is None else int(level)
        cost = self.compute_stockmeyer_level_cost(self.order)
        if level < cost:
            raise ValueError('PCM polynomial input level is too low')
        count = (self.order + 4) // 4
        span = _next_pow2(count)
        targets = [0] * count

        def assign(start, width, actual, target):
            if width == 1:
                targets[start] = target
                return
            half = width // 2
            assign(start, half, min(actual, half), target)
            if actual > half:
                assign(start + half, half, actual - half, target + 1)

        assign(0, span, count, level - cost)
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0,
                                   'relin': 0, 'add': 0, 'rescale': 0})

        def product(lv, plain=False):
            ops[lv]['mult_plain' if plain else 'mult'] += n
            ops[lv]['rescale'] += n
            if not plain:
                ops[lv]['relin'] += n

        power, lv = 2, level
        max_power = max(2 if self.order >= 2 else 1, 2 * span if count > 1 else 1)
        while power <= max_power:
            product(lv)
            power *= 2
            lv -= 1
        nodes = []
        for baby, target in enumerate(targets):
            terms = min(3, self.order - 4 * baby)
            for off in range(1, terms + 1):
                product(target + (2 if off == 3 else 1), plain=True)
                if off == 3:
                    product(target + 1)
            if terms:
                ops[target]['add'] += n * terms  # term joins and constant
            nodes.append((target, terms > 0))
        while len(nodes) > 1:
            combined = []
            for i in range(0, len(nodes), 2):
                target, has_ct = nodes[i]
                if i + 1 < len(nodes):
                    product(target + 1, plain=not nodes[i + 1][1])
                    ops[target]['add'] += n
                    has_ct = True
                combined.append((target, has_ct))
            nodes = combined
        return dict(ops)
