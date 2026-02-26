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

import numpy as np

op_class = 'DensePackedLayer'


class DensePackedLayer:
    def __init__(self, n_out_channel, n_in_channel, input_shape, skip, pack, n_packed_in_feature, n_packed_out_feature):
        self.n_out_channel: int = n_out_channel
        self.n_in_channel: int = n_in_channel
        self.input_shape: list[int] = input_shape
        self.skip: list[int] = skip

        if int(input_shape[0]) & (int(input_shape[0]) - 1) != 0 or int(input_shape[1]) & (int(input_shape[1]) - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if int(skip[0]) & (int(skip[0]) - 1) != 0 or int(skip[1]) & (int(skip[1]) - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')

        self.pack: int = pack
        self.n_packed_in_feature: int = n_packed_in_feature
        self.n_packed_out_feature: int = n_packed_out_feature

        self.mark: int = 0

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[DataNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt):
        input_rotated_x: list[CkksCiphertextNode] = list()
        input_shape_ct = list()
        input_shape_ct.append(int(self.input_shape[0] * self.skip[0]))
        input_shape_ct.append(int(self.input_shape[1] * self.skip[1]))

        for y in x:
            input_rotated_x += self.populate_rotations_1_side(y, self.pack - 1, input_shape_ct[0] * input_shape_ct[1])
        result: list[CkksCiphertextNode] = list()
        for packed_out_feature_idx in range(self.n_packed_out_feature):
            partial_sum: CkksCiphertextNode | None = None
            x_ct_list = []
            w_pt_list = []
            for in_feature_idx in range(len(input_rotated_x)):
                x_ct = input_rotated_x[in_feature_idx]
                w_pt = weight_pt[packed_out_feature_idx][in_feature_idx]
                x_ct_list.append(x_ct)
                w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b_pt = bias_pt[packed_out_feature_idx]
            partial_sum = add(partial_sum, b_pt)
            n_term = input_shape_ct[0] * input_shape_ct[1]
            if n_term == 1:
                return partial_sum
            y: DataNode = None
            while n_term > 1:
                if n_term == input_shape_ct[0] * input_shape_ct[1]:
                    rotated = rotate_cols(partial_sum, [int(n_term / 2)])[0]
                    y = add(partial_sum, rotated)
                else:
                    rotated = rotate_cols(y, [int(n_term / 2)])[0]
                    y = add(y, rotated)
                n_term = int(n_term / 2)
            result.append(y)
        return result

    def call_mult_pack(self, x: list[DataNode], weight_pt, bias_pt, n):
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        x_size = len(x)
        n_pack = int(np.ceil(n / 2 / self.input_shape[0] / self.input_shape[1]))
        n_block_input = int(np.ceil(n_pack / (self.skip[0] * self.skip[1])))
        n_num_pre_ct = int(np.ceil(n / 2 / input_ct_shape[0] / input_ct_shape[1]))
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        rotated_tmp = []
        for x_id in range(0, x_size):
            r_tmp = self.populate_rotations_1_side(x[x_id], n_block_input - 1, input_ct_shape[0] * input_ct_shape[1])
            rotated_tmp.append(r_tmp)
        input_rotated_x = []
        for rr in rotated_tmp:
            for ri in rr:
                input_rotated_x.append(ri)
        result = []

        for packed_out_feature_idx in range(0, n_packed_out_feature_for_mult_pack):
            for in_feature_idx in range(0, len(weight_pt[packed_out_feature_idx])):
                x_ct = input_rotated_x[in_feature_idx]
                w_pt = weight_pt[packed_out_feature_idx][in_feature_idx]
                p = mult(x_ct, w_pt)
                if in_feature_idx == 0:
                    s = p
                else:
                    s = add(s, p)
            s = rescale(s)
            b_pt = bias_pt[packed_out_feature_idx]
            s = add(s, b_pt)
            n_term = input_ct_shape[0] * input_ct_shape[1]
            while n_term > 1:
                rotated = rotate_cols(s, int(n_term / 2))
                s = add(s, rotated[0])
                n_term /= 2
            result.append(s)
        return result

    def call_custom_compute(self, x: list[CkksCiphertextNode], dense_data_source):
        input_rotated_x: list[CkksCiphertextNode] = list()
        input_shape_ct = list()
        input_shape_ct.append(int(self.input_shape[0] * self.skip[0]))
        input_shape_ct.append(int(self.input_shape[1] * self.skip[1]))

        for y in x:
            input_rotated_x += self.populate_rotations_1_side(y, self.pack - 1, input_shape_ct[0] * input_shape_ct[1])
        result: list[CkksCiphertextNode] = list()
        for packed_out_feature_idx in range(self.n_packed_out_feature):
            partial_sum: CkksCiphertextNode | None = None
            x_ct_list = []
            w_pt_list = []
            for in_feature_idx in range(len(input_rotated_x)):
                x_ct = input_rotated_x[in_feature_idx]
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_feature_idx}_{in_feature_idx}')
                custom_compute(
                    inputs=[dense_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt',
                        'i': packed_out_feature_idx,
                        'j': in_feature_idx,
                    },
                )
                x_ct_list.append(x_ct)
                w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            partial_sum = rescale(partial_sum)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_feature_idx}')
            custom_compute(
                inputs=[dense_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': packed_out_feature_idx},
            )
            partial_sum = add(partial_sum, b_pt)
            n_term = input_shape_ct[0] * input_shape_ct[1]
            if n_term == 1:
                return partial_sum
            y: DataNode = None
            while n_term > 1:
                if n_term == input_shape_ct[0] * input_shape_ct[1]:
                    rotated = rotate_cols(partial_sum, [int(n_term / 2)])[0]
                    y = add(partial_sum, rotated)
                else:
                    rotated = rotate_cols(y, [int(n_term / 2)])[0]
                    y = add(y, rotated)
                n_term = int(n_term / 2)
            result.append(y)
        return result

    def call_mult_pack_custom_compute(self, x: list[DataNode], dense_data_source, n):
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        x_size = len(x)
        n_pack = int(np.ceil(n / 2 / self.input_shape[0] / self.input_shape[1]))
        n_block_input = int(np.ceil(n_pack / (self.skip[0] * self.skip[1])))
        n_num_pre_ct = int(np.ceil(n / 2 / input_ct_shape[0] / input_ct_shape[1]))
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        N_half = int(n / 2)
        cached_n_block_input = (
            int(np.ceil(self.n_in_channel * self.input_shape[0] * self.input_shape[1] / N_half)) * n_num_pre_ct
        )

        rotated_tmp = []
        for x_id in range(0, x_size):
            r_tmp = self.populate_rotations_1_side(x[x_id], n_block_input - 1, input_ct_shape[0] * input_ct_shape[1])
            rotated_tmp.append(r_tmp)
        input_rotated_x = []
        for rr in rotated_tmp:
            for ri in rr:
                input_rotated_x.append(ri)
        result = []

        for packed_out_feature_idx in range(0, n_packed_out_feature_for_mult_pack):
            # Use cached_n_block_input instead of n_block_input
            for in_feature_idx in range(0, cached_n_block_input):
                x_ct = input_rotated_x[in_feature_idx]
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_feature_idx}_{in_feature_idx}')
                custom_compute(
                    inputs=[dense_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt',
                        'i': packed_out_feature_idx,
                        'j': in_feature_idx,
                    },
                )
                p = mult(x_ct, w_pt)
                if in_feature_idx == 0:
                    s = p
                else:
                    s = add(s, p)
            s = rescale(s)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_feature_idx}')
            custom_compute(
                inputs=[dense_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': packed_out_feature_idx},
            )
            s = add(s, b_pt)
            n_term = input_ct_shape[0] * input_ct_shape[1]
            while n_term > 1:
                rotated = rotate_cols(s, int(n_term / 2))
                s = add(s, rotated[0])
                n_term /= 2
            result.append(s)
        return result
