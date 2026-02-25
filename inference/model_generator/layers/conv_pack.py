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

op_class = 'Conv2DPackedLayer'


class Conv2DPackedLayer:
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
            raise ValueError(f"input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]")
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f"stride must be powers of 2, got: [{stride[0]}, {stride[1]}]")
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f"skip must be powers of 2, got: [{skip[0]}, {skip[1]}]")

        self.pack: int = pack
        self.n_packed_in_channel: int = n_packed_in_channel
        self.n_packed_out_channel: int = n_packed_out_channel
        padding_shape = [kernel_shape[0] // 2, kernel_shape[1] // 2]
        self.input_shape_ct = [input_shape[0] * skip[0], input_shape[1] * skip[1]]
        self.input_rotate_units = [skip[0] * self.input_shape_ct[1], skip[0] * 1]
        self.input_rotate_ranges = [padding_shape[1], padding_shape[0]]

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[CkksCiphertextNode]:
        result: list[CkksCiphertextNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

    @staticmethod
    def populate_rotations_2_sides(x: CkksCiphertextNode, n_rotation: int, unit: int):
        post_steps = []
        nega_steps = []
        for i in range(1, n_rotation + 1):
            post_steps.append(i * unit)
            nega_steps.append(-i * unit)
        steps = nega_steps + post_steps
        r_temp = rotate_cols(x, steps)
        result: list[CkksCiphertextNode] = list()

        # Reverse negatives when inserting
        result += list(reversed(r_temp[0 : len(nega_steps)]))
        result.append(x)
        result += r_temp[len(nega_steps) : :]
        return result

    def gen_rotated_x(self, x: list[CkksCiphertextNode]):
        rotated_x: list[list[CkksCiphertextNode]] = list()
        for c in x:
            row: list[CkksCiphertextNode] = list()
            rotations = self.populate_rotations_2_sides((c), self.input_rotate_ranges[0], self.input_rotate_units[0])
            for r in rotations:
                temp = self.populate_rotations_2_sides((r), self.input_rotate_ranges[1], self.input_rotate_units[1])
                row += temp
            rotated_x.append(row)
        return rotated_x

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source) -> list[CkksCiphertextNode]:
        rotated_x: list[CkksCiphertextNode] = list()
        for x_ct in x:
            rotated_x += Conv2DPackedLayer.populate_rotations_1_side(
                x_ct, self.pack - 1, self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
            )
        rotated_x_2d = self.gen_rotated_x(rotated_x)
        result = list()

        for packed_out_channel_idx in range(self.n_packed_out_channel):
            partial_sum: DataNode | None = None
            x_ct_list = []
            w_pt_list = []
            for in_channel_idx in range(self.n_packed_in_channel * self.pack):
                for i in range(self.kernel_shape[0]):
                    for j in range(self.kernel_shape[1]):
                        index = i * self.kernel_shape[1] + j
                        x_ct = rotated_x_2d[in_channel_idx][index]
                        w_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_channel_idx}_{in_channel_idx}_{index}')
                        custom_compute(
                            inputs=[conv_data_source],
                            output=w_pt,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'weight_pt',
                                'i': packed_out_channel_idx,
                                'j': in_channel_idx,
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

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt) -> list[CkksCiphertextNode]:
        rotated_x: list[CkksCiphertextNode] = list()
        for x_ct in x:
            rotated_x += Conv2DPackedLayer.populate_rotations_1_side(
                x_ct, self.pack - 1, self.input_shape[0] * self.skip[0] * self.input_shape[1] * self.skip[1]
            )
        rotated_x_2d = self.gen_rotated_x(rotated_x)
        result = list()

        for packed_out_channel_idx in range(self.n_packed_out_channel):
            partial_sum: DataNode | None = None
            x_ct_list = []
            w_pt_list = []
            for in_channel_idx in range(self.n_packed_in_channel * self.pack):
                for i in range(self.kernel_shape[0]):
                    for j in range(self.kernel_shape[1]):
                        index = i * self.kernel_shape[1] + j
                        x_ct = rotated_x_2d[in_channel_idx][index]
                        w_pt = weight_pt[packed_out_channel_idx][in_channel_idx][index]
                        x_ct_list.append(x_ct)
                        w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b = bias_pt[packed_out_channel_idx]
            result_ct = add(partial_sum, b)
            result.append(result_ct)
        return result
