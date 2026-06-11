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


class ComputeDistanceApproxLayer:
    def __init__(self, dim: int, norm2_min: float, norm2_max: float, nr_iterations: int):
        if dim <= 0:
            raise ValueError('dim must be positive')
        if norm2_min <= 0.0:
            raise ValueError('norm2_min must be positive')
        if norm2_max < norm2_min:
            raise ValueError('norm2_max must be greater than or equal to norm2_min')
        if nr_iterations < 1:
            raise ValueError('nr_iterations must be at least 1')
        self.dim = dim
        self.norm2_min = norm2_min
        self.norm2_max = norm2_max
        self.nr_iterations = nr_iterations

    @staticmethod
    def align_level(x, target_level: int):
        if x.level < target_level:
            raise ValueError(f'cannot align level {x.level} to higher level {target_level}')
        if x.level == target_level:
            return x
        return drop_level(x, x.level - target_level)

    @staticmethod
    def align_pair(x, y):
        target_level = min(x.level, y.level)
        return ComputeDistanceApproxLayer.align_level(x, target_level), ComputeDistanceApproxLayer.align_level(y, target_level)

    @staticmethod
    def sum_slots(x, dim: int):
        result = x
        step = 1
        while step < dim:
            rotated = rotate_cols(result, [step])[0]
            result = add(result, rotated)
            step <<= 1
        return result

    @staticmethod
    def make_pt_nodes(layer_id: str, level: int, nr_iterations: int, norm2_level: int):
        pts = {
            'gallery': [CkksPlaintextNode(f'distance_gallery_{layer_id}_0', level=level)],
            'x0_cubic': [CkksPlaintextNode(f'distance_x0_cubic_{layer_id}_0', level=norm2_level)],
            'x0_linear': [CkksPlaintextNode(f'distance_x0_linear_{layer_id}_0', level=norm2_level - 1)],
            'neg_half': [],
            'one_point_five': [],
            'neg_one': [],
            'two': [],
        }

        x_level = norm2_level - 1
        for i in range(1, nr_iterations):
            x_work_level = x_level
            if x_work_level - 3 <= 0:
                raise ValueError('compute_distance_approx task lowering does not support bootstrap constants yet')
            if x_work_level > norm2_level:
                x_work_level = norm2_level
            pts['neg_half'].append(CkksPlaintextNode(f'distance_neg_half_{layer_id}_{i}', level=x_work_level - 2))
            pts['one_point_five'].append(CkksPlaintextNode(f'distance_one_point_five_{layer_id}_{i}', level=x_work_level - 3))
            x_level = x_work_level - 4

        pts['neg_one'].append(CkksPlaintextNode(f'distance_neg_one_{layer_id}_0', level=x_level - 1))
        pts['two'].append(CkksPlaintextNode(f'distance_two_{layer_id}_0', level=x_level - 2))
        return pts

    def bootstrap_if_needed(self, x):
        if x.level - 3 > 0:
            return x
        raise ValueError('compute_distance_approx task lowering does not support bootstrap constants yet')

    def inverse_sqrt_one_iteration(self, norm2, pts):
        term = rescale(mult(norm2, pts['x0_cubic'][0]))
        return add(term, pts['x0_linear'][0])

    def inverse_sqrt_next_iteration(self, norm2, x, neg_half_pt, one_point_five_pt):
        x_work = self.bootstrap_if_needed(x)
        if x_work.level > norm2.level:
            x_work = drop_level(x_work, x_work.level - norm2.level)

        x2 = rescale(mult_relin(x_work, x_work))
        norm2_aligned = self.align_level(norm2, x2.level)
        ax2 = rescale(mult_relin(norm2_aligned, x2))
        factor = rescale(mult(ax2, neg_half_pt))
        factor = add(factor, one_point_five_pt)
        x_aligned = self.align_level(x_work, factor.level)
        return rescale(mult_relin(x_aligned, factor))

    def inverse_sqrt_iterations(self, norm2, pts):
        x = self.inverse_sqrt_one_iteration(norm2, pts)
        for i in range(1, self.nr_iterations):
            x = self.inverse_sqrt_next_iteration(norm2, x, pts['neg_half'][i - 1], pts['one_point_five'][i - 1])
        return x

    def call(self, x, pts):
        if len(x) != 1:
            raise ValueError('compute_distance_approx expects one packed 0D ciphertext')
        query = x[0]
        if query.level != pts['gallery'][0].level:
            raise ValueError('gallery plaintext level must match query ciphertext level')

        dot2 = rescale(mult(query, pts['gallery'][0]))
        dot2 = self.sum_slots(dot2, self.dim)

        norm2 = rescale(mult_relin(query, query))
        norm2 = self.sum_slots(norm2, self.dim)

        rsqrt = self.inverse_sqrt_iterations(norm2, pts)
        dot2 = self.align_level(dot2, rsqrt.level)
        cos2 = rescale(mult_relin(dot2, rsqrt))

        dist2 = rescale(mult(cos2, pts['neg_one'][0]))
        dist2 = add(dist2, pts['two'][0])
        return [dist2]
