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
from inference.model_generator.layers.encrypted_param_ops import (
    accumulate_encrypted_param_terms,
    multiply_with_encrypted_param,
    require_no_plaintext_input_rotation,
)
from inference.model_generator.layers.fhe_op_utils import naf_weight


op_class = 'MultiplexedConv2DPackedLayer'


class MultiplexedConv2DPackedLayer:
    rotate_num = 0
    add_num = 0
    mult_num = 0
    rescale_num = 0
    drop_level_num = 0

    def __init__(
        self,
        n_out_channel,
        n_in_channel,
        input_shape,
        kernel_shape,
        stride,
        skip,
        n_channel_per_ct,
        n_packed_in_channel,
        n_packed_out_channel,
        external_upsample_factor: list = [1, 1],
    ):
        self.n_out_channel: int = n_out_channel
        self.n_in_channel: int = n_in_channel
        self.input_shape: list[int] = input_shape
        self.kernel_shape: list[int] = kernel_shape
        self.stride: list[int] = stride
        self.skip: list[int] = skip

        if input_shape[0] & (input_shape[0] - 1) != 0 or input_shape[1] & (input_shape[1] - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f'stride must be powers of 2, got: [{stride[0]}, {stride[1]}]')
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')

        self.n_channel_per_ct: int = n_channel_per_ct
        self.n_packed_in_channel: int = n_packed_in_channel
        self.n_packed_out_channel: int = n_packed_out_channel
        padding_shape = [kernel_shape[0] // 2, kernel_shape[1] // 2]
        self.input_shape_ct = [input_shape[0] * skip[0], input_shape[1] * skip[1]]
        self.input_rotate_units = [skip[0] * self.input_shape_ct[1], skip[1] * 1]
        self.input_rotate_ranges = [padding_shape[1], padding_shape[0]]
        self.n_block_per_ct: int = int(np.ceil(n_channel_per_ct / (skip[0] * skip[1])))
        self.external_upsample_factor: list = external_upsample_factor
        self.zero_inserted_skip: list = [1, 1]
        self.zero_inserted_skip[0] = self.skip[0] * self.stride[0] / self.external_upsample_factor[0]
        self.zero_inserted_skip[1] = self.skip[1] * self.stride[1] / self.external_upsample_factor[1]

    def get_fhe_op_count(self, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   rotate_block + rotate_kernel + rotate_sum_slot + mult_plain + add_accum + add_sum_slot + rescale_base,
            level-1: add(bias)  [stride=1,skip=1]  or  mult(mask)+rescale  [stride>1],
            level-2: rotate_stride + add_bias  [stride>1 only],
          }

        step 1 - block direction: populate_rotations_1_side over n_packed_in_channel cts:
          unit = input_shape[0]*skip[0]*input_shape[1]*skip[1] (power of 2)
          steps: [1*unit, ..., (n_block_per_ct-1)*unit]
          naf_weight(i*unit) = naf_weight(i) since unit is power of 2
          primitive rotates = n_packed_in_channel * sum(naf_weight(i) for i in 1..n_block_per_ct-1)

        step 2 - kernel direction: gen_rotated_x over n_packed_in_channel*n_block_per_ct cts:
          input_rotate_units[0] = skip[0]*input_shape[1]*skip[1] (power of 2)
          input_rotate_units[1] = skip[1] (power of 2)
          row direction: populate_rotations_2_sides(c, kh, unit_0), fc0=kh//2
            primitive rotates per ct = sum(naf_weight(i) for i in range(-fc0,kh-fc0) if i!=0)
          col direction: kh calls of populate_rotations_2_sides(r, kw, unit_1), fc1=kw//2
            primitive rotates per ct = kh * sum(naf_weight(j) for j in range(-fc1,kw-fc1) if j!=0)

        step 3 - per output group (size_0 = ceil(n_out_channel / n_block_per_ct)):
          mult_plain: size_1 * kernel_size  (size_1 = n_packed_in_channel * n_block_per_ct)
          add: same - 1 (accumulate)
          rescale: 1  [level → level-1]
          sum_slot steps are powers of 2 -> floor(log2(skip)) rotates/adds each

        stride=1,skip=1 path (at level-1): +1 add (bias) per output group
        stride>1 path (at level-1): simulate rot_step per (ct_idx, i) with naf_weight;
          valid_n mult (mask) + valid_n rescale;  [level-1 → level-2]
          (at level-2): rotate_stride + add_bias
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        n_pack_in_channel = int(np.ceil(self.n_in_channel / self.n_channel_per_ct))
        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        size_0 = int(np.ceil(self.n_out_channel / self.n_block_per_ct))
        size_1 = n_pack_in_channel * self.n_block_per_ct
        kh, kw = self.kernel_shape

        # step 1: block rotations (unit is power of 2, naf_weight(i*unit) = naf_weight(i))
        ops[lv]['rotate'] += self.n_packed_in_channel * sum(naf_weight(i) for i in range(1, self.n_block_per_ct))

        # step 2: kernel rotations (units are powers of 2)
        fc0 = kh // 2
        fc1 = kw // 2
        rots_row = sum(naf_weight(i) for i in range(-fc0, kh - fc0) if i != 0)
        rots_col = kh * sum(naf_weight(j) for j in range(-fc1, kw - fc1) if j != 0)
        ops[lv]['rotate'] += (self.n_packed_in_channel * self.n_block_per_ct) * (rots_row + rots_col)

        log2_skip0 = int(np.floor(np.log2(self.skip[0]))) if self.skip[0] > 1 else 0
        log2_skip1 = int(np.floor(np.log2(self.skip[1]))) if self.skip[1] > 1 else 0
        ops[lv]['rotate'] += size_0 * (log2_skip0 + log2_skip1)

        ops[lv]['mult_plain'] += size_0 * size_1 * kernel_size
        ops[lv]['add'] += size_0 * (size_1 * kernel_size - 1)  # accumulate
        ops[lv]['add'] += size_0 * (log2_skip0 + log2_skip1)  # sum_slot

        if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
            ops[lv]['rescale'] += size_0
            lv -= 1
            ops[lv]['add'] += size_0  # bias
        else:
            ops[lv]['rescale'] += size_0
            lv -= 1

            # Simulate rot_step for each (ct_idx, i)
            rotate_stride = 0
            for ct_idx in range(size_0):
                valid_n = min(self.n_block_per_ct, self.n_out_channel - ct_idx * self.n_block_per_ct)
                steps = []
                for i in range(valid_n):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
                    )
                    n_block_residue = (
                        np.floor(n_block / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (n_block % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = (
                        -n_block_residue
                        - n_skip
                        - n_block % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    steps.append(int(rot_step))
                rotate_stride += sum(naf_weight(s) for s in steps)
            n_packed_out = self.n_packed_out_channel
            ops[lv]['mult_plain'] += self.n_out_channel
            ops[lv]['rescale'] += self.n_out_channel
            lv -= 1

            ops[lv]['rotate'] += rotate_stride
            ops[lv]['add'] += n_packed_out * 2  # bias add + accumulate per output ct

        return dict(ops)

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[CkksCiphertextNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        if steps:
            result += rotate_cols(x, steps)
        return result

    @staticmethod
    def populate_rotations_2_sides(x: CkksCiphertextNode, n_rotation: int, unit: int):
        filter_center = int(np.floor(n_rotation / 2))
        steps = []
        for i in range(-filter_center, n_rotation - filter_center):
            if i != 0:
                steps.append(i * unit)
        r_temp = rotate_cols(x, steps) if steps else []
        result: list[CkksCiphertextNode] = list()
        result += list(r_temp[0:filter_center])
        result.append(x)
        result += r_temp[filter_center::]
        return result

    @staticmethod
    def rotation_steps_2_sides(n_rotation: int, unit: int) -> list[int]:
        filter_center = int(np.floor(n_rotation / 2))
        return [i * unit for i in range(-filter_center, n_rotation - filter_center)]

    def kernel_rotation_steps(self) -> list[int]:
        steps = []
        row_steps = self.rotation_steps_2_sides(self.kernel_shape[0], self.input_rotate_units[0])
        col_steps = self.rotation_steps_2_sides(self.kernel_shape[1], self.input_rotate_units[1])
        for row_step in row_steps:
            for col_step in col_steps:
                steps.append(row_step + col_step)
        return steps

    def gen_rotated_x(self, x: list[CkksCiphertextNode]):
        rotated_x: list[list[CkksCiphertextNode]] = list()
        for c in x:
            row: list[CkksCiphertextNode] = list()
            rotations = self.populate_rotations_2_sides((c), self.kernel_shape[0], self.input_rotate_units[0])
            for r in rotations:
                temp = self.populate_rotations_2_sides((r), self.kernel_shape[1], self.input_rotate_units[1])
                row += temp
            rotated_x.append(row)
        return rotated_x

    def sum_slot(self, x: CkksCiphertextNode, m: int, p: int):
        result = x
        for j in range(1, int(np.floor(np.log2(m))) + 1):
            res = rotate_cols(result, [int(np.power(2, j - 1) * p)])
            result = add(result, res[0])

        for j in range(int(np.floor(np.log2(m))) - 1):
            if int(np.floor(m / np.power(2, j))) % 2 == 1:
                res = rotate_cols(result, [int(np.floor(m / np.power(2, j + 1))) * np.power(2, j + 1) * p])
                result = add(result, res[0])
        return result

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source) -> list[CkksCiphertextNode]:
        # 1. Block direction rotation
        block_rotations: list[CkksCiphertextNode] = list()
        for x_ct in x:
            block_rotations += MultiplexedConv2DPackedLayer.populate_rotations_1_side(
                x_ct, self.n_block_per_ct - 1, self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
            )
        # 2. Kernel direction rotation
        kernel_rotations = self.gen_rotated_x(block_rotations)
        # 3. Result computation and organization
        res: list = list()
        result_ct = list()

        n_pack_in_channel = int(np.ceil(self.n_in_channel / self.n_channel_per_ct))
        size_0 = int(np.ceil(self.n_out_channel / self.n_block_per_ct))
        size_1 = int(n_pack_in_channel * self.n_block_per_ct)
        size_2 = int(self.kernel_shape[0] * self.kernel_shape[1])
        for ct_idx in range(size_0):
            partial_sum: DataNode | None = None
            x_ct_list = list()
            w_pt_list = list()
            for j in range(size_1):
                for k in range(size_2):
                    w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{j}_{k}')
                    custom_compute(
                        inputs=[conv_data_source],  # All nodes reference the same data source
                        output=w_pt,
                        type='encode_pt',
                        attributes={'op_class': op_class, 'type': 'weight_pt', 'i': ct_idx, 'j': j, 'k': k},
                    )
                    x_ct_list.append(kernel_rotations[j][k])
                    w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(partial_sum)
            s = self.sum_slot(s, self.skip[0], self.skip[1] * self.input_shape[1])
            s = self.sum_slot(s, self.skip[1], 1)
            if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
                res.append(s)
            else:
                valid_n = min(self.n_block_per_ct, self.n_out_channel - ct_idx * self.n_block_per_ct)
                for i in range(valid_n):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
                    )
                    n_block_residue = (
                        np.floor(n_block / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (n_block % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = int(
                        -n_block_residue
                        - n_skip
                        - n_block % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    m_pt = CkksPlaintextRingtNode(f'encode_pt_mask_{ct_idx}_{i}')
                    custom_compute(
                        inputs=[conv_data_source],
                        output=m_pt,
                        type='encode_pt',
                        attributes={'op_class': op_class, 'type': 'mask_pt', 'i': i},
                    )
                    c_m = mult(s, m_pt)
                    c_m = rescale(c_m)
                    result_ct.append(rotate_cols(c_m, [rot_step])[0])

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        for i in range(len(res)):
            bias_idx = i
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{bias_idx}')
            custom_compute(
                inputs=[conv_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': bias_idx},
            )
            res[i] = add(res[i], b_pt)
        return res

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt, mask_pt).

        weight_pt[i][j][k]: i in size_0, j in size_1, k in kernel_size
        bias_pt[i]: i in n_packed_out_channel
        mask_pt[i]: i in min(n_block_per_ct, n_out_channel)  (empty list if no mask needed)
        """
        import math as _math

        n_pack_in_channel = _math.ceil(self.n_in_channel / self.n_channel_per_ct)
        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        size_0 = _math.ceil(self.n_out_channel / self.n_block_per_ct)
        size_1 = n_pack_in_channel * self.n_block_per_ct

        weight_pt = [
            [
                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(kernel_size)]
                for j in range(size_1)
            ]
            for i in range(size_0)
        ]
        n_bias = _math.ceil(self.n_out_channel / (self.stride[0] * self.stride[1] * self.n_channel_per_ct))
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_bias)]
        if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
            mask_pt = []
        else:
            n_mask = min(self.n_block_per_ct, self.n_out_channel)
            mask_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_mask)]
        return weight_pt, bias_pt, mask_pt

    def make_param_ct_nodes(self, layer_id, level: int):
        """Return encrypted weight/bias nodes plus encrypted structural masks."""
        import math as _math

        n_pack_in_channel = _math.ceil(self.n_in_channel / self.n_channel_per_ct)
        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        size_0 = _math.ceil(self.n_out_channel / self.n_block_per_ct)
        size_1 = n_pack_in_channel * self.n_block_per_ct

        weight_ct = [
            [
                [CkksCiphertextNode(f'convw_{layer_id}_{i}_{j}_{k}', level=level) for k in range(kernel_size)]
                for j in range(size_1)
            ]
            for i in range(size_0)
        ]
        n_bias = _math.ceil(self.n_out_channel / (self.stride[0] * self.stride[1] * self.n_channel_per_ct))
        bias_level = level - 1
        if not (self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1):
            bias_level -= 1
        bias_ct = [CkksCiphertextNode(f'convb_{layer_id}_{i}', level=bias_level) for i in range(n_bias)]
        if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
            mask_ct = []
        else:
            n_mask = min(self.n_block_per_ct, self.n_out_channel)
            mask_ct = [CkksCiphertextNode(f'convm_{layer_id}_{i}', level=level - 1) for i in range(n_mask)]
        return weight_ct, bias_ct, mask_ct

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]:
        # 1. block direction rotation
        block_rotations: list[CkksCiphertextNode] = list()
        for x_ct in x:
            block_rotations += MultiplexedConv2DPackedLayer.populate_rotations_1_side(
                x_ct, self.n_block_per_ct - 1, self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
            )
        # 2. Kernel direction rotation
        kernel_rotations = self.gen_rotated_x(block_rotations)
        # 3. Result computation and organization
        res: list = list()
        result_ct = list()
        for ct_idx in range(len(weight_pt)):
            partial_sum: DataNode | None = None
            x_ct_list = list()
            w_pt_list = list()
            for j in range(len(weight_pt[ct_idx])):
                for k in range(len(weight_pt[ct_idx][j])):
                    x_ct = kernel_rotations[j][k]
                    w_pt = weight_pt[ct_idx][j][k]
                    x_ct_list.append(x_ct)
                    w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(partial_sum)
            s = self.sum_slot(s, self.skip[0], self.skip[1] * self.input_shape[1])
            s = self.sum_slot(s, self.skip[1], 1)
            if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
                res.append(s)
            else:
                valid_n = min(self.n_block_per_ct, self.n_out_channel - ct_idx * self.n_block_per_ct)
                for i in range(valid_n):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
                    )
                    n_block_residue = (
                        np.floor(n_block / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (n_block % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = int(
                        -n_block_residue
                        - n_skip
                        - n_block % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    c_m = multiply_with_encrypted_param(s, mast_pt[i])
                    c_m = rescale(c_m)
                    result_ct.append(rotate_cols(c_m, [rot_step])[0])

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        for i in range(len(res)):
            res[i] = add(res[i], bias_pt[i])
        return res

    def call_param_ct(
        self, x: list[DataNode], weight_ct, bias_ct, mast_pt, input_is_plaintext: bool = False
    ) -> list[CkksCiphertextNode]:
        if input_is_plaintext:
            return self.call_param_ct_plaintext_input(x, weight_ct, bias_ct, mast_pt)

        rotations_needed = self.n_block_per_ct > 1 or self.kernel_shape[0] > 1 or self.kernel_shape[1] > 1
        require_no_plaintext_input_rotation(op_class, input_is_plaintext, rotations_needed)

        block_rotations: list[DataNode] = list()
        for x_node in x:
            block_rotations += MultiplexedConv2DPackedLayer.populate_rotations_1_side(
                x_node, self.n_block_per_ct - 1, self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
            )
        kernel_rotations = self.gen_rotated_x(block_rotations)
        res: list = list()
        result_ct = list()
        for ct_idx in range(len(weight_ct)):
            x_terms = []
            w_terms = []
            for j in range(len(weight_ct[ct_idx])):
                for k in range(len(weight_ct[ct_idx][j])):
                    x_terms.append(kernel_rotations[j][k])
                    w_terms.append(weight_ct[ct_idx][j][k])
            s = accumulate_encrypted_param_terms(x_terms, w_terms)
            s = rescale(s)
            s = self.sum_slot(s, self.skip[0], self.skip[1] * self.input_shape[1])
            s = self.sum_slot(s, self.skip[1], 1)
            if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
                res.append(s)
            else:
                valid_n = min(self.n_block_per_ct, self.n_out_channel - ct_idx * self.n_block_per_ct)
                for i in range(valid_n):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
                    )
                    n_block_residue = (
                        np.floor(n_block / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (n_block % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = int(
                        -n_block_residue
                        - n_skip
                        - n_block % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    c_m = multiply_with_encrypted_param(s, mast_pt[i])
                    c_m = rescale(c_m)
                    result_ct.append(rotate_cols(c_m, [rot_step])[0])

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        for i in range(len(res)):
            res[i] = add(res[i], bias_ct[i])
        return res

    def call_param_ct_plaintext_input(self, x: list[DataNode], weight_ct, bias_ct, mast_pt) -> list[CkksCiphertextNode]:
        block_unit = self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
        kernel_steps = self.kernel_rotation_steps()
        res: list = list()
        result_ct = list()

        for ct_idx in range(len(weight_ct)):
            partial_sum = None
            for j in range(len(weight_ct[ct_idx])):
                base_x = x[j // self.n_block_per_ct]
                block_step = (j % self.n_block_per_ct) * block_unit
                for k in range(len(weight_ct[ct_idx][j])):
                    total_step = int(block_step + kernel_steps[k])
                    w = weight_ct[ct_idx][j][k]
                    if total_step != 0:
                        w = rotate_cols(w, [-total_step])[0]
                    term = mult(w, base_x)
                    if total_step != 0:
                        term = rotate_cols(term, [total_step])[0]
                    partial_sum = term if partial_sum is None else add(partial_sum, term)
            if partial_sum is None:
                raise ValueError('Encrypted multiplexed conv accumulation produced no terms')
            s = rescale(partial_sum)
            s = self.sum_slot(s, self.skip[0], self.skip[1] * self.input_shape[1])
            s = self.sum_slot(s, self.skip[1], 1)
            if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
                res.append(s)
            else:
                valid_n = min(self.n_block_per_ct, self.n_out_channel - ct_idx * self.n_block_per_ct)
                for i in range(valid_n):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
                    )
                    n_block_residue = (
                        np.floor(n_block / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (n_block % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = int(
                        -n_block_residue
                        - n_skip
                        - n_block % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    c_m = mult(s, mast_pt[i])
                    c_m = rescale(c_m)
                    result_ct.append(rotate_cols(c_m, [rot_step])[0])

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.external_upsample_factor[0] * self.external_upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        for i in range(len(res)):
            res[i] = add(res[i], bias_ct[i])
        return res

    def make_pt_nodes_reduct_rot(self, layer_id):
        """Return (weight_pt, bias_pt, mask_pt) for call_reduct_rot().

        Mirrors C++ prepare_weight_for_reduct_rot():
          weight_pt[i][j][k]: i in n_packed_out_channel * skip_out_prod,
                               j in n_packed_in_channel * n_actual_blocks,
                               k in kernel_size
          bias_pt[i]: i in n_packed_out_channel
          mask_pt[i][j]: i in n_weight_pt, j in mask_size (empty if stride=skip=1)
        """
        import math as _math

        n_pack_in_channel = _math.ceil(self.n_in_channel / self.n_channel_per_ct)
        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        skip_out_prod = int(self.zero_inserted_skip[0] * self.zero_inserted_skip[1])
        n_weight_pt = self.n_packed_out_channel * skip_out_prod
        size_1 = n_pack_in_channel * self.n_actual_blocks
        # Bias count: one entry per output CT after accumulation.
        # n_channel_per_ct_out = n_block_per_ct * skip_out_prod (reduct_rot grouping).
        n_channel_per_ct_out = self.n_block_per_ct * skip_out_prod
        n_bias = max(1, _math.ceil(self.n_out_channel / n_channel_per_ct_out))

        weight_pt = [
            [
                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(kernel_size)]
                for j in range(size_1)
            ]
            for i in range(n_weight_pt)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_bias)]
        if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
            mask_pt = []
        else:
            # Compute per-ct_idx valid mask count: only entries where channel_out < n_out_channel.
            # channel_out(ct_idx, i) = output_ct_group * n_channel_per_ct_out + i * skip_out_prod + sub_pos
            skip_out_prod_l = int(self.zero_inserted_skip[0] * self.zero_inserted_skip[1])
            n_channel_per_ct_out_l = self.n_block_per_ct * skip_out_prod_l
            mask_pt = []
            for ct_idx_l in range(n_weight_pt):
                sub_pos_l = ct_idx_l % skip_out_prod_l
                output_ct_group_l = ct_idx_l // skip_out_prod_l
                base = output_ct_group_l * n_channel_per_ct_out_l + sub_pos_l
                if base >= self.n_out_channel:
                    valid_i = 0
                else:
                    valid_i = min(
                        self.n_block_per_ct,
                        int(np.ceil((self.n_out_channel - base) / skip_out_prod_l)),
                    )
                mask_pt.append([CkksPlaintextRingtNode(f'convm_{layer_id}_{ct_idx_l}_{j}') for j in range(valid_i)])
        return weight_pt, bias_pt, mask_pt

    def call_reduct_rot(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, mask_pt) -> list[CkksCiphertextNode]:
        """Corresponds to C++ run_core_for_reduct_rot."""
        # 1. Block direction rotation with loop-packing replication (same as call/call_custom_compute).
        input_ct_size = self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
        n_actual_blocks = self.n_actual_blocks
        n_rot_factor = self.n_block_per_ct // n_actual_blocks if n_actual_blocks < self.n_block_per_ct else 1
        n_rep_iters = int(np.floor(np.log2(n_rot_factor))) if n_rot_factor > 1 else 0

        x_rep = list(x)
        for x_id in range(len(x)):
            for r in range(n_rep_iters):
                x_rep[x_id] = add(
                    x_rep[x_id],
                    rotate_cols(x_rep[x_id], [-(2**r) * n_actual_blocks * input_ct_size])[0],
                )

        block_rotations: list[CkksCiphertextNode] = list()
        for x_ct in x_rep:
            block_rotations += MultiplexedConv2DPackedLayer.populate_rotations_1_side(
                x_ct, n_actual_blocks - 1, input_ct_size
            )
        # 2. Kernel direction rotation
        kernel_rotations = self.gen_rotated_x(block_rotations)

        # 3. Multiply-accumulate, rescale, post-process
        res: list = list()
        result_ct: list = list()
        skip_out_prod = int(self.zero_inserted_skip[0] * self.zero_inserted_skip[1])
        n_channel_per_ct_out = self.n_block_per_ct * skip_out_prod

        for ct_idx in range(len(weight_pt)):
            sub_pos = ct_idx % skip_out_prod
            output_ct_group = ct_idx // skip_out_prod

            x_ct_list = list()
            w_pt_list = list()
            for j in range(len(weight_pt[ct_idx])):
                for k in range(len(weight_pt[ct_idx][j])):
                    x_ct_list.append(kernel_rotations[j][k])
                    w_pt_list.append(weight_pt[ct_idx][j][k])
            s = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(s)

            if self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1:
                # No mask needed: bias indexed directly by output_ct_group (= ct_idx when skip_out_prod=1)
                res.append(s)
            else:
                valid_i = len(mask_pt[ct_idx]) if mask_pt else 0
                if valid_i == 0:
                    # No valid output channels for this ct_idx; skip to avoid dangling nodes.
                    continue
                s = self.sum_slot(s, self.skip[0], self.skip[1] * self.input_shape[1])
                s = self.sum_slot(s, self.skip[1], 1)
                # Only compute rotations for the valid i range (where channel_out < n_out_channel).
                # mask_pt[ct_idx] already has exactly that many entries.
                valid_i = len(mask_pt[ct_idx])
                steps = []
                for i in range(valid_i):
                    channel_local = i * skip_out_prod + sub_pos
                    n_block_residue = (
                        np.floor(channel_local / (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                        * self.skip[0]
                        * self.skip[1]
                        * self.input_shape[0]
                        * self.input_shape[1]
                    )
                    n_skip = (
                        np.floor(
                            (channel_local % (self.zero_inserted_skip[0] * self.zero_inserted_skip[1]))
                            / self.zero_inserted_skip[1]
                        )
                        * self.input_shape[1]
                        * self.skip[1]
                    )
                    rot_step = (
                        -n_block_residue
                        - n_skip
                        - channel_local % self.zero_inserted_skip[1]
                        + i * self.skip[0] * self.skip[1] * self.input_shape[0] * self.input_shape[1]
                    )
                    steps.append(int(rot_step))
                s_rots = rotate_cols(s, steps)
                # mask_pt[ct_idx] only contains entries where channel_out < n_out_channel
                for i in range(valid_i):
                    c_m_s = mult(s_rots[i], mask_pt[ct_idx][i])
                    result_ct.append(rescale(c_m_s))

        # 4. Accumulate n_channel_per_ct_out results per output CT, then add bias
        if not (self.stride[0] == 1 and self.stride[1] == 1 and self.skip[0] == 1 and self.skip[1] == 1):
            sp = None
            for i in range(len(result_ct)):
                p = i % n_channel_per_ct_out
                c_m_s = result_ct[i]
                if p == 0:
                    sp = c_m_s
                else:
                    sp = add(sp, c_m_s)
                if (i + 1) % n_channel_per_ct_out == 0 or i == len(result_ct) - 1:
                    res.append(sp)
        for i in range(len(res)):
            res[i] = add(res[i], bias_pt[i])
        return res
