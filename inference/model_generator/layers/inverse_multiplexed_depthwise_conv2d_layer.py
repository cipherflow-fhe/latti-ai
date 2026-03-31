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

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *

op_class = 'InverseMultiplexedDepthwiseConv2d'


class InverseMultiplexedDepthwiseConv2DLayer:
    rotate_num = 0
    add_num = 0
    mult_num = 0
    rescale_num = 0
    drop_level_num = 0

    def __init__(self, n_channel, input_shape, padding, kernel_shape, stride, stride_next, skip, block_shape):
        self.n_out_channel: int = n_channel
        self.n_in_channel: int = n_channel
        self.input_shape: list[int] = input_shape
        self.kernel_shape: list[int] = kernel_shape
        self.stride: list[int] = stride
        self.stride_next: list[int] = stride_next
        self.skip: list[int] = skip
        self.padding: list[int] = padding
        self.block_shape: list[int] = block_shape

        if input_shape[0] & (input_shape[0] - 1) != 0 or input_shape[1] & (input_shape[1] - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f'stride must be powers of 2, got: [{stride[0]}, {stride[1]}]')
        if stride_next[0] & (stride_next[0] - 1) != 0 or stride_next[1] & (stride_next[1] - 1) != 0:
            raise ValueError(f'stride_next must be powers of 2, got: [{stride_next[0]}, {stride_next[1]}]')
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')
        if block_shape[0] & (block_shape[0] - 1) != 0 or block_shape[1] & (block_shape[1] - 1) != 0:
            raise ValueError(f'block_shape must be powers of 2, got: [{block_shape[0]}, {block_shape[1]}]')

        if self.padding[0] < 0 and self.padding[1] < 0:
            self.padding = [(kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2]

        # Stride decomposition for output_shape < block_shape
        self.orig_stride = list(stride)
        output_shape0 = input_shape[0] // stride[0]
        output_shape1 = input_shape[1] // stride[1]
        self.need_repack = (output_shape0 < block_shape[0]) or (output_shape1 < block_shape[1])
        if self.need_repack:
            self.stride = [input_shape[0] // block_shape[0], input_shape[1] // block_shape[1]]
            self.stride_next = [1, 1]

    def get_used_input_indices(self) -> set:
        """Return the set of input CT indices that are actually used in the convolution."""
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        stride_next0 = self.stride_next[0]
        stride_next1 = self.stride_next[1]
        used = set()
        # Depthwise: each output channel uses only its own input channel
        for n_ch in range(self.n_in_channel):
            base = n_ch * stride0 * stride1 * stride_next0 * stride_next1
            for r_i2 in range(stride_next0):
                for r_j2 in range(stride_next1):
                    for row_seg_idx in range(stride0):
                        for col_seg_idx in range(stride1):
                            if row_seg_idx >= self.kernel_shape[0] or col_seg_idx >= self.kernel_shape[1]:
                                continue
                            split_ks0 = (self.kernel_shape[0] - 1 - row_seg_idx) // stride0 + 1
                            split_ks1 = (self.kernel_shape[1] - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_ks0):
                                for v_s in range(split_ks1):
                                    begin_row = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * stride_next0)
                                    begin_row = (begin_row + stride0 * stride_next0) % (stride0 * stride_next0)
                                    begin_col = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * stride_next1)
                                    begin_col = (begin_col + stride1 * stride_next1) % (stride1 * stride_next1)
                                    begin_idx = begin_row * stride1 * stride_next1 + begin_col
                                    used.add(base + begin_idx)
        return used

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source, N: int) -> list[CkksCiphertextNode]:
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        stride_next0 = self.stride_next[0]
        stride_next1 = self.stride_next[1]
        kernel_shape0 = self.kernel_shape[0]
        kernel_shape1 = self.kernel_shape[1]
        block_shape1 = self.block_shape[1]

        # Depthwise: rotated_x indexed by out_channel (each uses its own input channel)
        rotated_x = [[] for i in range(self.n_out_channel)]

        for out_channel_idx in range(0, self.n_out_channel):
            base_in_ct_idx = int(out_channel_idx * stride0 * stride1 * stride_next0 * stride_next1)
            for r_i2 in range(0, stride_next0):
                for r_j2 in range(0, stride_next1):
                    for row_seg_idx in range(self.stride[0]):
                        for col_seg_idx in range(self.stride[1]):
                            split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) // stride0 + 1
                            split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_kernel_shape0):
                                for v_s in range(split_kernel_shape1):
                                    begin_row_idx = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (
                                        stride0 * stride_next0
                                    )
                                    begin_row_idx = (begin_row_idx + stride0 * stride_next0) % (stride0 * stride_next0)
                                    begin_col_idx = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (
                                        stride1 * stride_next1
                                    )
                                    begin_col_idx = (begin_col_idx + stride1 * stride_next1) % (stride1 * stride_next1)
                                    begin_idx = begin_row_idx * stride1 * stride_next1 + begin_col_idx
                                    in_ct_idx = base_in_ct_idx + begin_idx
                                    row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) // (
                                        stride0 * stride_next0
                                    )
                                    col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) // (
                                        stride1 * stride_next1
                                    )
                                    step = int(row_step * block_shape1 + col_step)
                                    if step == 0:
                                        res_temp = x[in_ct_idx]
                                    else:
                                        res_temp = rotate_cols(x[in_ct_idx], [step])[0]
                                    rotated_x[out_channel_idx].append(res_temp)

        n_channel_per_ct_out = 1
        if 2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1] < N:
            n_channel_per_ct_out = N / (2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1])
        else:
            n_channel_per_ct_out = 1

        temp_res = [0 for i in range(self.n_out_channel * self.stride_next[0] * self.stride_next[1])]

        for ct_idx in range(0, self.n_out_channel):
            for r_i2 in range(0, stride_next0):
                for r_j2 in range(0, stride_next1):
                    s = 0
                    out_ct_idx = ct_idx * stride_next0 * stride_next1 + r_i2 * stride_next1 + r_j2
                    base_idx = (r_i2 * stride_next1 + r_j2) * self.kernel_shape[0] * self.kernel_shape[1]
                    reference_level = rotated_x[ct_idx][base_idx].level
                    partial_sum: DataNode | None = None
                    x_ct_list = []
                    w_pt_list = []
                    # Depthwise: no inner j loop over n_in_channel, only kernel positions
                    for k in range(0, self.kernel_shape[0] * self.kernel_shape[1]):
                        w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{k + base_idx}')
                        custom_compute(
                            inputs=[conv_data_source],
                            output=w_pt,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'weight_pt',
                                'i': ct_idx,
                                'k': k + base_idx,
                            },
                        )
                        x_ct_list.append(rotated_x[ct_idx][k + base_idx])
                        w_pt_list.append(w_pt)
                    partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
                    s = rescale(partial_sum)
                    b_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}')
                    custom_compute(
                        inputs=[conv_data_source],
                        output=b_pt,
                        type='encode_pt',
                        attributes={'op_class': op_class, 'type': 'bias_pt', 'i': ct_idx},
                    )
                    s = add(s, b_pt)
                    temp_res[out_ct_idx] = s

        if self.need_repack:
            output_shape0 = self.input_shape[0] // self.orig_stride[0]
            output_shape1 = self.input_shape[1] // self.orig_stride[1]
            out_skip0 = self.block_shape[0] // output_shape0
            out_skip1 = self.block_shape[1] // output_shape1
            n_channel_per_block = out_skip0 * out_skip1
            n_block_per_ct = (N // 2) // (self.block_shape[0] * self.block_shape[1])
            n_channel_per_ct_out_repack = n_channel_per_block * n_block_per_ct
            n_out_ct = math.ceil(self.n_out_channel / n_channel_per_ct_out_repack)

            # Shared mask: select row%skip==0 && col%skip==0
            repack_mask = CkksPlaintextRingtNode('repack_mask')
            custom_compute(
                inputs=[conv_data_source],
                output=repack_mask,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'repack_mask'},
            )

            # Step 1: mask all channels
            for c in range(len(temp_res)):
                temp_res[c] = mult(temp_res[c], repack_mask)

            # Step 2: rotate + accumulate
            res = [None] * n_out_ct
            for out_ct_idx in range(n_out_ct):
                packed = None
                for ch_in_ct in range(n_channel_per_ct_out_repack):
                    c = out_ct_idx * n_channel_per_ct_out_repack + ch_in_ct
                    if c >= self.n_out_channel:
                        break
                    block_idx = ch_in_ct // n_channel_per_block
                    ch_in_block = ch_in_ct % n_channel_per_block
                    cx = ch_in_block // out_skip1
                    cy = ch_in_block % out_skip1

                    rot_step = -(cx * self.block_shape[1] + cy + block_idx * self.block_shape[0] * self.block_shape[1])
                    if rot_step == 0:
                        rotated = temp_res[c]
                    else:
                        rotated = rotate_cols(temp_res[c], [rot_step])[0]

                    if packed is None:
                        packed = rotated
                    else:
                        packed = add(packed, rotated)
                res[out_ct_idx] = rescale(packed)
            return res

        res = [
            0 for i in range(int(math.ceil(self.n_out_channel / n_channel_per_ct_out) * stride_next0 * stride_next1))
        ]
        if n_channel_per_ct_out == 1:
            res = temp_res
        else:
            for out_ct_idx in range(0, len(temp_res)):
                pack_out_ct_idx = int(out_ct_idx // n_channel_per_ct_out)
                channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out
                if channel_idx_in_ct == 0:
                    res[pack_out_ct_idx] = temp_res[out_ct_idx]
                else:
                    step = int(
                        -1
                        * channel_idx_in_ct
                        * self.input_shape[0]
                        // self.stride[0]
                        * self.input_shape[1]
                        // self.stride[1]
                    )
                    if step == 0:
                        s_rot = temp_res[out_ct_idx]
                    else:
                        s_rot = rotate_cols(temp_res[out_ct_idx], [step])[0]
                    res[pack_out_ct_idx] = add(res[pack_out_ct_idx], s_rot)
        return res

    def call(
        self, x: list[CkksCiphertextNode], weight_pt, bias_pt, N: int, conv_data_source=None, repack_mask_pt=None
    ) -> list[CkksCiphertextNode]:
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        stride_next0 = self.stride_next[0]
        stride_next1 = self.stride_next[1]
        kernel_shape0 = self.kernel_shape[0]
        kernel_shape1 = self.kernel_shape[1]
        block_shape1 = self.block_shape[1]

        # Depthwise: rotated_x indexed by out_channel (each uses its own input channel)
        rotated_x = [[] for i in range(self.n_out_channel)]

        for out_channel_idx in range(0, self.n_out_channel):
            base_in_ct_idx = int(out_channel_idx * stride0 * stride1 * stride_next0 * stride_next1)
            for r_i2 in range(0, stride_next0):
                for r_j2 in range(0, stride_next1):
                    for row_seg_idx in range(self.stride[0]):
                        for col_seg_idx in range(self.stride[1]):
                            split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) // stride0 + 1
                            split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_kernel_shape0):
                                for v_s in range(split_kernel_shape1):
                                    begin_row_idx = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (
                                        stride0 * stride_next0
                                    )
                                    begin_row_idx = (begin_row_idx + stride0 * stride_next0) % (stride0 * stride_next0)
                                    begin_col_idx = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (
                                        stride1 * stride_next1
                                    )
                                    begin_col_idx = (begin_col_idx + stride1 * stride_next1) % (stride1 * stride_next1)
                                    begin_idx = begin_row_idx * stride1 * stride_next1 + begin_col_idx
                                    in_ct_idx = base_in_ct_idx + begin_idx
                                    row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) // (
                                        stride0 * stride_next0
                                    )
                                    col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) // (
                                        stride1 * stride_next1
                                    )
                                    step = int(row_step * block_shape1 + col_step)
                                    if step == 0:
                                        res_temp = x[in_ct_idx]
                                    else:
                                        res_temp = rotate_cols(x[in_ct_idx], [step])[0]
                                    rotated_x[out_channel_idx].append(res_temp)

        n_channel_per_ct_out = 1
        if 2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1] < N:
            n_channel_per_ct_out = N / (2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1])
        else:
            n_channel_per_ct_out = 1

        temp_res = [0 for i in range(len(weight_pt) * self.stride_next[0] * self.stride_next[1])]

        for ct_idx in range(0, len(weight_pt)):
            for r_i2 in range(0, stride_next0):
                for r_j2 in range(0, stride_next1):
                    partial_sum: DataNode | None = None
                    x_ct_list = list()
                    w_pt_list = list()
                    out_ct_idx = ct_idx * stride_next0 * stride_next1 + r_i2 * stride_next1 + r_j2
                    base_idx = (r_i2 * stride_next1 + r_j2) * self.kernel_shape[0] * self.kernel_shape[1]
                    # Depthwise: no inner j loop, weight_pt[ct_idx] is 1D [kernel], not 2D [in_ch][kernel]
                    for k in range(0, self.kernel_shape[0] * self.kernel_shape[1]):
                        x_ct_list.append(rotated_x[ct_idx][k + base_idx])
                        w_pt_list.append(weight_pt[ct_idx][k + base_idx])
                    partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
                    s = rescale(partial_sum)
                    s = add(s, bias_pt[ct_idx])
                    temp_res[out_ct_idx] = s

        if self.need_repack:
            output_shape0 = self.input_shape[0] // self.orig_stride[0]
            output_shape1 = self.input_shape[1] // self.orig_stride[1]
            out_skip0 = self.block_shape[0] // output_shape0
            out_skip1 = self.block_shape[1] // output_shape1
            n_channel_per_block = out_skip0 * out_skip1
            n_block_per_ct = (N // 2) // (self.block_shape[0] * self.block_shape[1])
            n_channel_per_ct_out_repack = n_channel_per_block * n_block_per_ct
            n_out_ct = math.ceil(self.n_out_channel / n_channel_per_ct_out_repack)

            repack_mask = repack_mask_pt

            # Step 1: mask all channels
            for c in range(len(temp_res)):
                temp_res[c] = mult(temp_res[c], repack_mask)

            # Step 2: rotate + accumulate
            res = [None] * n_out_ct
            for out_ct_idx in range(n_out_ct):
                packed = None
                for ch_in_ct in range(n_channel_per_ct_out_repack):
                    c = out_ct_idx * n_channel_per_ct_out_repack + ch_in_ct
                    if c >= self.n_out_channel:
                        break
                    block_idx = ch_in_ct // n_channel_per_block
                    ch_in_block = ch_in_ct % n_channel_per_block
                    cx = ch_in_block // out_skip1
                    cy = ch_in_block % out_skip1

                    rot_step = -(cx * self.block_shape[1] + cy + block_idx * self.block_shape[0] * self.block_shape[1])
                    if rot_step == 0:
                        rotated = temp_res[c]
                    else:
                        rotated = rotate_cols(temp_res[c], [rot_step])[0]

                    if packed is None:
                        packed = rotated
                    else:
                        packed = add(packed, rotated)
                res[out_ct_idx] = rescale(packed)
            return res

        res = [0 for i in range(int(math.ceil(len(weight_pt) / n_channel_per_ct_out) * stride_next0 * stride_next1))]
        if n_channel_per_ct_out == 1:
            res = temp_res
        else:
            for out_ct_idx in range(0, len(temp_res)):
                pack_out_ct_idx = int(out_ct_idx // n_channel_per_ct_out)
                channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out
                if channel_idx_in_ct == 0:
                    res[pack_out_ct_idx] = temp_res[out_ct_idx]
                else:
                    step = int(
                        -1
                        * channel_idx_in_ct
                        * self.input_shape[0]
                        // self.stride[0]
                        * self.input_shape[1]
                        // self.stride[1]
                    )
                    if step == 0:
                        s_rot = temp_res[out_ct_idx]
                    else:
                        s_rot = rotate_cols(temp_res[out_ct_idx], [step])[0]
                    res[pack_out_ct_idx] = add(res[pack_out_ct_idx], s_rot)
        return res
