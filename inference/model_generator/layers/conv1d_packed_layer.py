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

op_class = 'Conv1DPackedLayer'


class Conv1DPackedLayer:
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
        self.input_shape: int = input_shape
        self.kernel_shape: int = kernel_shape
        self.stride: int = stride
        self.skip: int = skip
        self.pack: int = pack
        self.n_packed_in_channel: int = n_packed_in_channel
        self.n_packed_out_channel: int = n_packed_out_channel
        self.input_shape_ct: int = input_shape * skip

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[CkksCiphertextNode]:
        result: list[CkksCiphertextNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

    @staticmethod
    def populate_rotations_2_sides(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[CkksCiphertextNode]:
        filter_center = n_rotation // 2
        steps = []
        for i in range(-filter_center, n_rotation - filter_center):
            if i != 0:
                steps.append(i * unit)
        r_temp = rotate_cols(x, steps)
        result: list[CkksCiphertextNode] = list()
        result += list(r_temp[0:filter_center])
        result.append(x)
        result += r_temp[filter_center::]
        return result

    def gen_rotated_x(self, x: list[CkksCiphertextNode]):
        rotated_x: list[list[CkksCiphertextNode]] = list()
        for c in x:
            row = self.populate_rotations_2_sides(c, self.kernel_shape, self.skip)
            rotated_x.append(row)
        return rotated_x

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source) -> list[CkksCiphertextNode]:
        rot_num = min(self.n_in_channel, self.pack)

        # 1. Channel packing rotations
        rotated_x: list[CkksCiphertextNode] = list()
        for x_ct in x:
            rotated_x += Conv1DPackedLayer.populate_rotations_1_side(x_ct, rot_num - 1, self.input_shape_ct)

        # 2. Kernel direction rotations
        rotated_x_2d = self.gen_rotated_x(rotated_x)

        # 3. Multiply-accumulate
        result = list()
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            x_ct_list = []
            w_pt_list = []
            for in_channel_idx in range(self.n_packed_in_channel * rot_num):
                for kernel_idx in range(self.kernel_shape):
                    x_ct = rotated_x_2d[in_channel_idx][kernel_idx]
                    w_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_channel_idx}_{in_channel_idx}_{kernel_idx}')
                    custom_compute(
                        inputs=[conv_data_source],
                        output=w_pt,
                        type='encode_pt',
                        attributes={
                            'op_class': op_class,
                            'type': 'weight_pt',
                            'i': packed_out_channel_idx,
                            'j': in_channel_idx,
                            'k': kernel_idx,
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
        rot_num = min(self.n_in_channel, self.pack)

        # 1. Channel packing rotations
        rotated_x: list[CkksCiphertextNode] = list()
        for x_ct in x:
            rotated_x += Conv1DPackedLayer.populate_rotations_1_side(x_ct, rot_num - 1, self.input_shape_ct)

        # 2. Kernel direction rotations
        rotated_x_2d = self.gen_rotated_x(rotated_x)

        # 3. Multiply-accumulate
        result = list()
        for packed_out_channel_idx in range(self.n_packed_out_channel):
            x_ct_list = []
            w_pt_list = []
            for in_channel_idx in range(self.n_packed_in_channel * rot_num):
                for kernel_idx in range(self.kernel_shape):
                    x_ct = rotated_x_2d[in_channel_idx][kernel_idx]
                    w_pt = weight_pt[packed_out_channel_idx][in_channel_idx][kernel_idx]
                    x_ct_list.append(x_ct)
                    w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b = bias_pt[packed_out_channel_idx]
            result_ct = add(partial_sum, b)
            result.append(result_ct)
        return result
