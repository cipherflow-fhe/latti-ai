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
    require_no_plaintext_input_rotation,
)
from inference.model_generator.layers.fhe_op_utils import naf_weight


op_class = 'Conv2DPackedDepthwiseLayer'


class Conv2DPackedDepthwiseLayer:
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
        pack,
        n_packed_in_channel,
        n_packed_out_channel,
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

        self.pack: int = pack
        self.n_packed_in_channel: int = n_packed_in_channel
        self.n_packed_out_channel: int = n_packed_out_channel
        padding_shape = [kernel_shape[0] // 2, kernel_shape[1] // 2]
        self.input_shape_ct = [input_shape[0] * skip[0], input_shape[1] * skip[1]]
        self.input_rotate_units = [skip[0] * self.input_shape_ct[1], skip[0] * 1]
        self.input_rotate_ranges = [padding_shape[1], padding_shape[0]]

    def get_fhe_op_count(self, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   rotate_kernel + mult_plain + add_accum + rescale_base,
            level-1: add(bias)  [stride=1]  or  mult_plain(select)+rescale  [stride>1],
            level-2: rotate_stride + add(bias+accum)  [stride>1 only],
          }

        'rotate' is the total primitive RotateColUnit count via NAF decomposition.

        input_rotate_units[0] = skip[0] * input_shape_ct[1] = skip[0]*input_shape[1]*skip[1] (power of 2)
        input_rotate_units[1] = skip[0]                                                       (power of 2)
        Because both units are powers of 2: naf_weight(i * unit) == naf_weight(i).

        gen_rotated_x over n_packed_out_channel cts:
          row direction: populate_rotations_2_sides(c, range_h, unit_0)
            steps: [-range_h*unit_0, ..., -1*unit_0, 1*unit_0, ..., range_h*unit_0]
            primitive rotates per ct = 2 * sum(naf_weight(i) for i in 1..range_h)
          col direction: (2*range_h+1) calls of populate_rotations_2_sides(r, range_w, unit_1)
            steps per call: [-range_w*unit_1, ..., range_w*unit_1] (excluding 0)
            primitive rotates per ct = (2*range_h+1) * 2 * sum(naf_weight(i) for i in 1..range_w)

        per output packed channel (= n_packed_out_channel):
          terms = kernel_h * kernel_w
          mult_plain: terms, add(accum): terms-1, rescale: 1  [level → level-1]
          stride=1: add(bias): 1  [level-1]
          stride>1: mult_plain(select): valid_n_total + rescale: valid_n_total  [level-1 → level-2]
                    rotate_stride + add(bias+accum)  [level-2]
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        range_h, range_w = self.input_rotate_ranges
        kh, kw = self.kernel_shape

        rots_row = 2 * sum(naf_weight(i) for i in range(1, range_h + 1))
        rots_col = (2 * range_h + 1) * 2 * sum(naf_weight(i) for i in range(1, range_w + 1))
        ops[lv]['rotate'] += self.n_packed_in_channel * (rots_row + rots_col)

        terms_per_out = kh * kw
        ops[lv]['mult_plain'] += self.n_packed_out_channel * terms_per_out
        ops[lv]['add'] += self.n_packed_out_channel * (terms_per_out - 1)  # accumulate
        ops[lv]['rescale'] += self.n_packed_out_channel
        lv -= 1

        if self.stride[0] == 1:
            ops[lv]['add'] += self.n_packed_out_channel  # bias
        else:
            # Simulate rot_step for each (ct_idx, i)
            rotate_stride = 0
            for ct_idx in range(self.n_packed_in_channel):
                steps = []
                for i in range(0, min(self.n_channel_per_ct, self.n_out_channel), self.skip[0]):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        r_n_block = int(
                            (ct_idx * self.n_channel_per_ct + i)
                            / int(self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1])
                        )
                        r_n_block_residue = (ct_idx * self.n_channel_per_ct + i) % int(
                            self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1]
                        )
                        r_n_stride_skip = int(np.floor(r_n_block_residue / (self.stride[0] * self.skip[0])))
                        r_n_stride_skip_residue = r_n_block_residue % int(self.stride[0] * self.skip[0])
                        n_block = int(np.floor((ct_idx * self.n_channel_per_ct + i) / int(self.skip[0] * self.skip[1])))
                        n_block_residue = int(
                            np.floor((ct_idx * self.n_channel_per_ct + i)) % int(self.skip[0] * self.skip[1])
                        )
                        n_stride_skip = int(np.floor(n_block_residue / self.skip[0]))
                        n_stride_skip_residue = n_block_residue % self.skip[0]
                        rot_step = (
                            (r_n_block - n_block)
                            * self.skip[0]
                            * self.skip[1]
                            * self.input_shape[0]
                            * self.input_shape[1]
                            + (r_n_stride_skip - n_stride_skip) * self.skip[0] * self.input_shape[0]
                            + (r_n_stride_skip_residue - n_stride_skip_residue)
                        )
                        steps.append(-rot_step)
                rotate_stride += sum(naf_weight(s) for s in steps)
            n_packed_out = self.n_packed_out_channel
            valid_n_total = self.n_out_channel
            ops[lv]['mult_plain'] += valid_n_total
            ops[lv]['rescale'] += valid_n_total
            lv -= 1

            ops[lv]['rotate'] += rotate_stride
            ops[lv]['add'] += n_packed_out + (valid_n_total - n_packed_out)  # bias + accumulate

        return dict(ops)

    @staticmethod
    def populate_rotations_1_side(x: DataNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[DataNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        if steps:
            result += rotate_cols(x, steps)
        return result

    @staticmethod
    def populate_rotations_2_sides(x: DataNode, n_rotation: int, unit: int):
        post_steps = []
        nega_steps = []
        for i in range(1, n_rotation + 1):
            post_steps.append(i * unit)
            nega_steps.append(-i * unit)
        steps = nega_steps + post_steps
        r_temp = rotate_cols(x, steps) if steps else []
        result: list[DataNode] = list()

        # Reverse negatives when inserting
        result += list(reversed(r_temp[0 : len(nega_steps)]))
        result.append(x)
        result += r_temp[len(nega_steps) : :]
        return result

    @staticmethod
    def rotation_steps_2_sides(n_rotation: int, unit: int) -> list[int]:
        result = []
        for i in range(n_rotation, 0, -1):
            result.append(-i * unit)
        result.append(0)
        for i in range(1, n_rotation + 1):
            result.append(i * unit)
        return result

    def kernel_rotation_steps(self) -> list[int]:
        steps = []
        row_steps = self.rotation_steps_2_sides(self.input_rotate_ranges[0], self.input_rotate_units[0])
        col_steps = self.rotation_steps_2_sides(self.input_rotate_ranges[1], self.input_rotate_units[1])
        for row_step in row_steps:
            for col_step in col_steps:
                steps.append(row_step + col_step)
        return steps

    def gen_rotated_x(self, x: list[DataNode]):
        rotated_x: list[list[DataNode]] = list()
        for c in x:
            row: list[DataNode] = list()
            rotations = self.populate_rotations_2_sides((c), self.input_rotate_ranges[0], self.input_rotate_units[0])
            for r in rotations:
                temp = self.populate_rotations_2_sides((r), self.input_rotate_ranges[1], self.input_rotate_units[1])
                row += temp
            rotated_x.append(row)
        return rotated_x

    def call_custom_compute(self, x: list[DataNode], conv_data_source) -> list[DataNode]:
        rotated_x = x
        rotated_x_2d = self.gen_rotated_x(rotated_x)
        result = list()
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            partial_sum: DataNode | None = None
            x_ct_list = []
            w_pt_list = []
            for i in range(self.kernel_shape[0]):
                for j in range(self.kernel_shape[1]):
                    index = i * self.kernel_shape[1] + j
                    x_ct = rotated_x_2d[packed_out_channel_idx][index]
                    w_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_channel_idx}_{index}')
                    custom_compute(
                        inputs=[conv_data_source],
                        output=w_pt,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'weight_pt',
                            'i': packed_out_channel_idx,
                            'k': index,
                        },
                    )
                    x_ct_list.append(x_ct)
                    w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_channel_idx}')
            custom_compute(
                inputs=[conv_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': packed_out_channel_idx},
            )
            result_ct = add(partial_sum, b_pt)
            result.append(result_ct)
        return result

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt) with shapes matching call()."""
        index = self.kernel_shape[0] * self.kernel_shape[1]
        weight_pt = [
            [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{i}') for i in range(index)]
            for n in range(self.n_packed_out_channel)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(self.n_packed_out_channel)]
        return weight_pt, bias_pt

    def make_param_ct_nodes(self, layer_id, level: int):
        """Return encrypted parameter nodes with the same IDs/shapes as make_pt_nodes()."""
        index = self.kernel_shape[0] * self.kernel_shape[1]
        weight_ct = [
            [CkksCiphertextNode(f'convw_{layer_id}_{n}_{i}', level=level) for i in range(index)]
            for n in range(self.n_packed_out_channel)
        ]
        bias_ct = [
            CkksCiphertextNode(f'convb_{layer_id}_{i}', level=level - 1) for i in range(self.n_packed_out_channel)
        ]
        return weight_ct, bias_ct

    def call(self, x: list[DataNode], weight_pt, bias_pt) -> list[DataNode]:
        rotated_x = x
        rotated_x_2d = self.gen_rotated_x(rotated_x)
        result = list()
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            partial_sum: DataNode | None = None
            x_ct_list = []
            w_pt_list = []
            for i in range(self.kernel_shape[0]):
                for j in range(self.kernel_shape[1]):
                    index = i * self.kernel_shape[1] + j
                    x_ct = rotated_x_2d[packed_out_channel_idx][index]
                    w_pt = weight_pt[packed_out_channel_idx][index]
                    x_ct_list.append(x_ct)
                    w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b = bias_pt[packed_out_channel_idx]
            result_ct = add(partial_sum, b)
            result.append(result_ct)
        return result

    def call_param_ct(self, x: list[DataNode], weight_ct, bias_ct, input_is_plaintext: bool = False):
        if input_is_plaintext:
            return self.call_param_ct_plaintext_input(x, weight_ct, bias_ct)

        rotations_needed = self.input_rotate_ranges[0] > 0 or self.input_rotate_ranges[1] > 0
        require_no_plaintext_input_rotation(op_class, input_is_plaintext, rotations_needed)

        rotated_x_2d = self.gen_rotated_x(x)
        result = []
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            x_terms = []
            w_terms = []
            for i in range(self.kernel_shape[0]):
                for j in range(self.kernel_shape[1]):
                    index = i * self.kernel_shape[1] + j
                    x_terms.append(rotated_x_2d[packed_out_channel_idx][index])
                    w_terms.append(weight_ct[packed_out_channel_idx][index])
            partial_sum = accumulate_encrypted_param_terms(x_terms, w_terms)
            partial_sum = rescale(partial_sum)
            result.append(add(partial_sum, bias_ct[packed_out_channel_idx]))
        return result

    def call_param_ct_plaintext_input(self, x: list[DataNode], weight_ct, bias_ct):
        kernel_steps = self.kernel_rotation_steps()
        result = []
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            partial_sum = None
            base_x = x[packed_out_channel_idx]
            for index in range(self.kernel_shape[0] * self.kernel_shape[1]):
                total_step = int(kernel_steps[index])
                w = weight_ct[packed_out_channel_idx][index]
                if total_step != 0:
                    w = rotate_cols(w, [-total_step])[0]
                term = mult(w, base_x)
                if total_step != 0:
                    term = rotate_cols(term, [total_step])[0]
                partial_sum = term if partial_sum is None else add(partial_sum, term)
            if partial_sum is None:
                raise ValueError('Encrypted depthwise conv accumulation produced no terms')
            partial_sum = rescale(partial_sum)
            result.append(add(partial_sum, bias_ct[packed_out_channel_idx]))
        return result
