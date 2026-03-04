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

op_class = 'MultConv2DPackedLayer'


class MultConv2DPackedLayer:
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
        upsample_factor: list = [1, 1],
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
        self.upsample_factor: list = upsample_factor
        self.zero_inserted_skip: list = [1, 1]
        self.zero_inserted_skip[0] = self.skip[0] * self.stride[0] / self.upsample_factor[0]
        self.zero_inserted_skip[1] = self.skip[1] * self.stride[1] / self.upsample_factor[1]

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[CkksCiphertextNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

    @staticmethod
    def populate_rotations_2_sides(x: CkksCiphertextNode, n_rotation: int, unit: int):
        filter_center = int(np.floor(n_rotation / 2))
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
        # 0. Create unified data source node (containing all weight/bias/mask data)
        # This node acts as a "pointer", created only once, all encode_pt nodes reference it

        # 1. Block direction rotation
        block_rotations: list[CkksCiphertextNode] = list()
        for x_ct in x:
            block_rotations += MultConv2DPackedLayer.populate_rotations_1_side(
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
                b_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}')
                custom_compute(
                    inputs=[conv_data_source],  # Reference same data source
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': ct_idx},
                )
                res.append(add(s, b_pt))
            else:
                steps = []
                for i in range(min(self.n_block_per_ct, self.n_out_channel)):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.upsample_factor[0] * self.upsample_factor[1])
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
                s_rots = rotate_cols(s, steps)
                for i in range(self.n_block_per_ct):
                    if (ct_idx * self.n_block_per_ct + i) < self.n_out_channel:
                        m_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{i}')
                        custom_compute(
                            inputs=[conv_data_source],  # Reference same data source
                            output=m_pt,
                            type='encode_pt',
                            attributes={'op_class': op_class, 'type': 'mask_pt', 'i': ct_idx, 'j': i},
                        )
                        c_m_s = mult(s_rots[i], m_pt)
                        result_ct.append(rescale(c_m_s))

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.upsample_factor[0] * self.upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
                bias_idx = int(
                    np.floor(
                        i
                        / (
                            self.stride[0]
                            * self.stride[1]
                            * self.n_channel_per_ct
                            / (self.upsample_factor[0] * self.upsample_factor[1])
                        )
                    )
                )
                b_pt = CkksPlaintextRingtNode(f'encode_pt_{bias_idx}')
                custom_compute(
                    inputs=[conv_data_source],  # Reference same data source
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': bias_idx},
                )
                sp = add(sp, b_pt)
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.upsample_factor[0] * self.upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        return res

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]:
        # 1. block direction rotation
        block_rotations: list[CkksCiphertextNode] = list()
        for x_ct in x:
            block_rotations += MultConv2DPackedLayer.populate_rotations_1_side(
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
                res.append(add(s, bias_pt[ct_idx]))
            else:
                steps = []
                for i in range(min(self.n_block_per_ct, self.n_out_channel)):
                    n_block = (ct_idx * self.n_block_per_ct + i) % (
                        self.n_channel_per_ct
                        * self.stride[0]
                        * self.stride[1]
                        / (self.upsample_factor[0] * self.upsample_factor[1])
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
                s_rots = rotate_cols(s, steps)
                for i in range(self.n_block_per_ct):
                    if (ct_idx * self.n_block_per_ct + i) < self.n_out_channel:
                        c_m_s = mult(s_rots[i], mast_pt[ct_idx][i])
                        result_ct.append(rescale(c_m_s))

        for i in range(len(result_ct)):
            n_block = i % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.upsample_factor[0] * self.upsample_factor[1])
            )
            c_m_s = result_ct[i]
            if n_block == 0:
                sp = c_m_s
                bias_idx = int(
                    np.floor(
                        i
                        / (
                            self.stride[0]
                            * self.stride[1]
                            * self.n_channel_per_ct
                            / (self.upsample_factor[0] * self.upsample_factor[1])
                        )
                    )
                )
                sp = add(sp, bias_pt[bias_idx])
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (
                self.stride[0]
                * self.stride[1]
                * self.n_channel_per_ct
                / (self.upsample_factor[0] * self.upsample_factor[1])
            ) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        return res
