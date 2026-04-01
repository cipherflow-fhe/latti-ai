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

import numpy as np

op_class = 'ParMultiplexedDWConv1DPackedLayer'


class ParMultiplexedDWConv1DPackedLayer:
    rotate_num = 0
    add_num = 0
    mult_num = 0
    rescale_num = 0
    drop_level_num = 0

    def __init__(
        self,
        n_channel,
        input_shape,
        kernel_shape,
        stride,
        skip,
        n_channel_per_ct,
        n_packed_ct,
    ):
        self.n_channel: int = n_channel
        self.input_shape: int = input_shape
        self.kernel_shape: int = kernel_shape
        self.stride: int = stride
        self.skip: int = skip
        self.n_channel_per_ct: int = n_channel_per_ct
        self.n_packed_ct: int = n_packed_ct
        self.input_shape_ct: int = input_shape * skip
        self.n_block_per_ct: int = int(np.ceil(n_channel_per_ct / skip))

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
        # 1. Kernel direction rotation
        rotated_x = self.gen_rotated_x(x)

        # 2. Mult + Add (no cross-ct loop; each ct is self-contained for DW conv)
        conv_results = list()

        for ct_idx in range(self.n_packed_ct):
            x_ct_list = []
            w_pt_list = []
            for k in range(self.kernel_shape):
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{k}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': op_class,
                        'type': 'weight_pt',
                        'i': ct_idx,
                        'j': k,
                    },
                )
                x_ct_list.append(rotated_x[ct_idx][k])
                w_pt_list.append(w_pt)

            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

            # 3. Skip reduction NOT done for DW conv — each slot already holds
            #    its own channel's result; summing would mix channels.

            # 4. Rescale
            s = rescale(partial_sum)
            conv_results.append(s)

        # 5. Add bias
        needs_rearrange = self.skip > 1 or self.stride > 1

        if not needs_rearrange:
            res = list()
            for ct_idx in range(self.n_packed_ct):
                b_pt = CkksPlaintextRingtNode(f'encode_pt_bias_{ct_idx}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': ct_idx},
                )
                res.append(add(conv_results[ct_idx], b_pt))
            return res
        else:
            # Select + rotate + merge
            skip_out = self.skip * self.stride
            output_shape = self.input_shape // self.stride
            n_packed_out = int(np.ceil(self.n_channel / self.n_channel_per_ct))

            # One select tensor per local channel (not per block), so that we can
            # pick the specific skip-offset slot j = local_ch % skip.
            n_local_ch = min(self.n_channel_per_ct, self.n_channel)
            select_pts = []
            for local_ch in range(n_local_ch):
                s_pt = CkksPlaintextRingtNode(f'encode_pt_select_{local_ch}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=s_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'select_pt', 'i': local_ch},
                )
                select_pts.append(s_pt)

            res = list()
            for po in range(n_packed_out):
                combined = None
                for ch_local in range(self.n_channel_per_ct):
                    ch = po * self.n_channel_per_ct + ch_local
                    if ch >= self.n_channel:
                        break

                    ct_idx = po  # ch // n_channel_per_ct
                    local_ch = ch_local  # ch % n_channel_per_ct
                    t = local_ch // self.skip  # block within CT
                    j = local_ch % self.skip  # channel_index within skip group

                    masked = mult(conv_results[ct_idx], select_pts[local_ch])
                    masked = rescale(masked)

                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    source_base = t * self.input_shape_ct + j
                    target_base = group * (output_shape * skip_out) + ch_offset
                    rotation = target_base - source_base

                    if rotation != 0:
                        rot = rotate_cols(masked, [-rotation])
                        masked = rot[0]

                    if combined is None:
                        combined = masked
                    else:
                        combined = add(combined, masked)

                b_pt = CkksPlaintextRingtNode(f'encode_pt_bias_{po}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': po},
                )
                combined = add(combined, b_pt)
                res.append(combined)
            return res

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, block_select_pt=None) -> list[CkksCiphertextNode]:
        # 1. Kernel direction rotation
        rotated_x = self.gen_rotated_x(x)

        # 2. Mult + Add
        conv_results = list()

        for ct_idx in range(self.n_packed_ct):
            x_ct_list = []
            w_pt_list = []
            for k in range(self.kernel_shape):
                x_ct_list.append(rotated_x[ct_idx][k])
                w_pt_list.append(weight_pt[ct_idx][k])

            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

            # 3. Skip reduction NOT done for DW conv.
            # 4. Rescale
            s = rescale(partial_sum)
            conv_results.append(s)

        # 5. Add bias
        needs_rearrange = self.skip > 1 or self.stride > 1

        if not needs_rearrange:
            res = list()
            for ct_idx in range(self.n_packed_ct):
                res.append(add(conv_results[ct_idx], bias_pt[ct_idx]))
            return res
        else:
            skip_out = self.skip * self.stride
            output_shape = self.input_shape // self.stride
            n_packed_out = int(np.ceil(self.n_channel / self.n_channel_per_ct))

            res = list()
            for po in range(n_packed_out):
                combined = None
                for ch_local in range(self.n_channel_per_ct):
                    ch = po * self.n_channel_per_ct + ch_local
                    if ch >= self.n_channel:
                        break

                    ct_idx = po  # ch // n_channel_per_ct
                    local_ch = ch_local  # ch % n_channel_per_ct
                    t = local_ch // self.skip
                    j = local_ch % self.skip

                    masked = mult(conv_results[ct_idx], block_select_pt[local_ch])
                    masked = rescale(masked)

                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    source_base = t * self.input_shape_ct + j
                    target_base = group * (output_shape * skip_out) + ch_offset
                    rotation = target_base - source_base

                    if rotation != 0:
                        rot = rotate_cols(masked, [-rotation])
                        masked = rot[0]

                    if combined is None:
                        combined = masked
                    else:
                        combined = add(combined, masked)

                combined = add(combined, bias_pt[po])
                res.append(combined)
            return res
