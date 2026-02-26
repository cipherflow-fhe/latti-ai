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

op_class = 'MultConv2DPackedDepthwiseLayer'


class MultConv2DPackedDepthwiseLayer:
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
        self.n_block_per_ct: int = int(np.floor(n_channel_per_ct / (skip[0] * skip[1])))
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

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]:
        # 1. Kernel direction rotation
        kernel_rotations = self.gen_rotated_x(x)
        # 2. Result computation and organization
        res: list = list()
        result_ct = list()
        for ct_idx in range(len(weight_pt)):
            for j in range(len(weight_pt[ct_idx])):
                w_pt = weight_pt[ct_idx][j]
                r = mult(kernel_rotations[ct_idx][j], w_pt)
                if j == 0:
                    s = r
                else:
                    s = add(s, r)
            s = rescale(s)
            if self.stride[0] == 1:
                res.append(s)
            else:
                steps = []
                for i in range(0, min(self.n_channel_per_ct, self.n_out_channel), self.skip[0]):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        # Position of channel i after reordering
                        r_n_block = int(
                            (ct_idx * self.n_channel_per_ct + i)
                            / int(self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1])
                        )
                        r_n_block_residue = (ct_idx * self.n_channel_per_ct + i) % int(
                            self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1]
                        )
                        r_n_stride_skip = int(np.floor(r_n_block_residue / (self.stride[0] * self.skip[0])))
                        r_n_stride_skip_residue = r_n_block_residue % int(self.stride[0] * self.skip[0])
                        # Current position of channel i
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
                s_rots = rotate_cols(s, steps)
                for i in range(self.n_channel_per_ct):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        c_m_s = mult(s_rots[int(i / self.skip[0])], mast_pt[ct_idx * self.n_channel_per_ct + i])
                        result_ct.append(rescale(c_m_s))
        if self.stride[0] == 1:
            for i in range(len(res)):
                res[i] = add(res[i], bias_pt[i])
            return res

        for i in range(len(result_ct)):
            p = i % (self.stride[0] * self.stride[1] * self.n_channel_per_ct)
            c_m_s = result_ct[i]
            if p == 0:
                sp = c_m_s
                btp_idx = int(np.floor(i / (self.stride[0] * self.stride[1] * self.n_channel_per_ct)))
                sp = add(sp, bias_pt[btp_idx])
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (self.stride[0] * self.stride[1] * self.n_channel_per_ct) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        return res

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source) -> list[CkksCiphertextNode]:
        # 1. Calculate the number of input ciphertexts to process
        n_pack_in_channel = int(np.ceil(self.n_in_channel / self.n_channel_per_ct))
        # Only generate kernel rotations for needed input ciphertexts (avoid generating unused nodes)
        kernel_rotations = self.gen_rotated_x(x)

        # 2. Result computation and organization
        res: list = list()
        result_ct = list()

        k_size = self.kernel_shape[0] * self.kernel_shape[1]
        for ct_idx in range(n_pack_in_channel):
            for j in range(k_size):
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{j}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'weight_pt', 'i': ct_idx, 'j': j},
                )
                r = mult(kernel_rotations[ct_idx][j], w_pt)
                if j == 0:
                    s = r
                else:
                    s = add(s, r)
            s = rescale(s)
            if self.stride[0] == 1:
                res.append(s)
            else:
                steps = []
                for i in range(0, min(self.n_channel_per_ct, self.n_out_channel), self.skip[0]):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        # Position of channel i after reordering
                        r_n_block = int(
                            (ct_idx * self.n_channel_per_ct + i)
                            / int(self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1])
                        )
                        r_n_block_residue = (ct_idx * self.n_channel_per_ct + i) % int(
                            self.skip[0] * self.skip[1] * self.stride[0] * self.stride[1]
                        )
                        r_n_stride_skip = int(np.floor(r_n_block_residue / (self.stride[0] * self.skip[0])))
                        r_n_stride_skip_residue = r_n_block_residue % int(self.stride[0] * self.skip[0])
                        # Current position of channel i
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

                # Generate all rotations (maintain original order and count, even if duplicates)
                if steps:
                    s_rots = rotate_cols(s, steps)
                else:
                    s_rots = []

                for i in range(self.n_channel_per_ct):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        # Calculate corresponding rotation index
                        rot_idx = int(i / self.skip[0])
                        rot_ct = s_rots[rot_idx]
                        m_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{i}')
                        custom_compute(
                            inputs=[conv_data_source],
                            output=m_pt,
                            type='encode_pt',
                            attributes={'op_class': op_class, 'type': 'mask_pt', 'i': ct_idx, 'j': i},
                        )
                        c_m_s = mult(rot_ct, m_pt)
                        result_ct.append(rescale(c_m_s))
        if self.stride[0] == 1:
            for i in range(len(res)):
                b_pt = CkksPlaintextRingtNode(f'encode_pt_{i}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': i},
                )
                res[i] = add(res[i], b_pt)
            return res

        for i in range(len(result_ct)):
            p = i % (self.stride[0] * self.stride[1] * self.n_channel_per_ct)
            c_m_s = result_ct[i]
            if p == 0:
                sp = c_m_s
                btp_idx = int(np.floor(i / (self.stride[0] * self.stride[1] * self.n_channel_per_ct)))
                b_pt = CkksPlaintextRingtNode(f'encode_pt_{btp_idx}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': btp_idx},
                )
                sp = add(sp, b_pt)
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % (self.stride[0] * self.stride[1] * self.n_channel_per_ct) == 0 or i == len(result_ct) - 1:
                res.append(sp)
        return res
