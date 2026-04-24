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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.fhe_op_utils import naf_weight


op_class = 'InverseMultiplexedConv2DLayer'


class InverseMultiplexedConv2DLayer:
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
        padding,
        kernel_shape,
        stride,
        block_shape,
    ):
        self.n_out_channel: int = n_out_channel
        self.n_in_channel: int = n_in_channel
        self.input_shape: list[int] = input_shape
        self.kernel_shape: list[int] = kernel_shape
        self.stride: list[int] = stride
        self.padding: list[int] = padding
        self.block_shape: list[int] = block_shape

        if input_shape[0] & (input_shape[0] - 1) != 0 or input_shape[1] & (input_shape[1] - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f'stride must be powers of 2, got: [{stride[0]}, {stride[1]}]')
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
        self.output_step = [
            input_shape[0] // (block_shape[0] * self.stride[0]),
            input_shape[1] // (block_shape[1] * self.stride[1]),
        ]

    def get_fhe_op_count(self, level: int, N: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   rotate (stage1) + mult_plain (accumulate) + add (accumulate) + rescale,
            level-1: add (bias)
                     [+ mult_plain(mask) + rotate(repack) + add(repack) + rescale(repack)  if need_repack]
                     [or + rotate(pack) + add(pack)                                         if n_channel_per_ct_out > 1],
          }

        Rotation phase (build rotated_x): simulate nested loops for each n_in_channel,
          compute step = row_step*block_shape[1] + col_step, sum naf_weight for non-zero steps.

        Accumulate phase (per out_ct_idx x r_i2 x r_j2):
          terms = n_in_channel * kernel_h * kernel_w
          mult_plain: terms, add: terms-1 (accumulate), rescale: 1  [level -> level-1]
          n_out_channel * output_step[0] * output_step[1] such groups.

        Bias add at level-1: 1 add per group.

        Repack path (at level-1): n_out_channel mult_plain (mask) + simulate rot_steps with naf_weight
          + adds + rescale per out_ct.
        No-repack packing (at level-1): step = -channel_idx * output_pixels (output_pixels is power of 2),
          naf_weight(channel_idx * output_pixels) = naf_weight(channel_idx).
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        pad0, pad1 = self.padding[0], self.padding[1]
        stride0, stride1 = self.stride[0], self.stride[1]
        output_step0, output_step1 = self.output_step[0], self.output_step[1]
        kh, kw = self.kernel_shape
        block_shape1 = self.block_shape[1]
        n_groups = self.n_out_channel * output_step0 * output_step1
        terms = self.n_in_channel * kh * kw

        # Rotation phase: simulate nested loops (no level change)
        for _ in range(self.n_in_channel):
            for r_i2 in range(output_step0):
                for r_j2 in range(output_step1):
                    for row_seg_idx in range(stride0):
                        for col_seg_idx in range(stride1):
                            split_ks0 = (kh - 1 - row_seg_idx) // stride0 + 1
                            split_ks1 = (kw - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_ks0):
                                for v_s in range(split_ks1):
                                    begin_row = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * output_step0)
                                    begin_row = (begin_row + stride0 * output_step0) % (stride0 * output_step0)
                                    begin_col = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * output_step1)
                                    begin_col = (begin_col + stride1 * output_step1) % (stride1 * output_step1)
                                    row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row) // (
                                        stride0 * output_step0
                                    )
                                    col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col) // (
                                        stride1 * output_step1
                                    )
                                    step = int(row_step * block_shape1 + col_step)
                                    if step != 0:
                                        ops[lv]['rotate'] += naf_weight(step)

        # Accumulate phase at lv: mult_plain + add(accumulate) + rescale  [lv -> lv-1]
        ops[lv]['mult_plain'] += n_groups * terms
        ops[lv]['add'] += n_groups * (terms - 1)
        ops[lv]['rescale'] += n_groups
        lv -= 1

        # Bias add at lv (= level-1)
        ops[lv]['add'] += n_groups

        # Packing / repack phase
        output_area = self.input_shape[0] / stride0 * self.input_shape[1] / stride1
        n_channel_per_ct_out = 1
        if 2 * output_area < N:
            n_channel_per_ct_out = int(N / (2 * output_area))

        if self.need_repack:
            output_shape0 = self.input_shape[0] // self.orig_stride[0]
            output_shape1 = self.input_shape[1] // self.orig_stride[1]
            out_skip0 = self.block_shape[0] // output_shape0
            out_skip1 = self.block_shape[1] // output_shape1
            n_channel_per_block = out_skip0 * out_skip1
            n_block_per_ct_repack = (N // 2) // (self.block_shape[0] * self.block_shape[1])
            n_channel_per_ct_out_repack = n_channel_per_block * n_block_per_ct_repack
            n_out_ct = math.ceil(self.n_out_channel / n_channel_per_ct_out_repack)

            # mask mult for all n_out_channel items at lv (= level-1)
            ops[lv]['mult_plain'] += self.n_out_channel

            # simulate rot_steps and accumulate adds at lv
            for out_ct_idx in range(n_out_ct):
                for ch_in_ct in range(n_channel_per_ct_out_repack):
                    c = out_ct_idx * n_channel_per_ct_out_repack + ch_in_ct
                    if c >= self.n_out_channel:
                        break
                    block_idx = ch_in_ct // n_channel_per_block
                    ch_in_block = ch_in_ct % n_channel_per_block
                    cx = ch_in_block // out_skip1
                    cy = ch_in_block % out_skip1
                    rot_step = -(cx * self.block_shape[1] + cy + block_idx * self.block_shape[0] * self.block_shape[1])
                    if rot_step != 0:
                        ops[lv]['rotate'] += naf_weight(rot_step)
                    if ch_in_ct > 0 and c < self.n_out_channel:
                        ops[lv]['add'] += 1

            ops[lv]['rescale'] += n_out_ct
            return dict(ops)

        n_temp = n_groups
        if n_channel_per_ct_out <= 1:
            return dict(ops)

        # Normal packing at lv (= level-1):
        # step = -channel_idx * output_pixels; output_pixels is power of 2
        # naf_weight(k * output_pixels) = naf_weight(k) since output_pixels is power of 2
        n_packed_normal = math.ceil(n_temp / n_channel_per_ct_out)
        ops[lv]['rotate'] += sum(
            naf_weight(out_ct_idx % n_channel_per_ct_out)
            for out_ct_idx in range(n_temp)
            if out_ct_idx % n_channel_per_ct_out != 0
        )
        ops[lv]['add'] += n_temp - n_packed_normal
        return dict(ops)

    def get_used_input_indices(self) -> set:
        """Return the set of input CT indices that are actually used in the convolution.
        Useful for filtering input_args before calling process_custom_task."""
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        output_step0 = self.output_step[0]
        output_step1 = self.output_step[1]
        used = set()
        for n_in_ch in range(self.n_in_channel):
            base = n_in_ch * stride0 * stride1 * output_step0 * output_step1
            for r_i2 in range(output_step0):
                for r_j2 in range(output_step1):
                    for row_seg_idx in range(stride0):
                        for col_seg_idx in range(stride1):
                            if row_seg_idx >= self.kernel_shape[0] or col_seg_idx >= self.kernel_shape[1]:
                                continue
                            split_ks0 = (self.kernel_shape[0] - 1 - row_seg_idx) // stride0 + 1
                            split_ks1 = (self.kernel_shape[1] - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_ks0):
                                for v_s in range(split_ks1):
                                    begin_row = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * output_step0)
                                    begin_row = (begin_row + stride0 * output_step0) % (stride0 * output_step0)
                                    begin_col = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * output_step1)
                                    begin_col = (begin_col + stride1 * output_step1) % (stride1 * output_step1)
                                    begin_idx = begin_row * stride1 * output_step1 + begin_col
                                    used.add(base + begin_idx)
        return used

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source, N: int) -> list[CkksCiphertextNode]:
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        output_step0 = self.output_step[0]
        output_step1 = self.output_step[1]
        kernel_shape0 = self.kernel_shape[0]
        kernel_shape1 = self.kernel_shape[1]
        block_shape1 = self.block_shape[1]

        rotated_x = [[] for i in range(self.n_in_channel)]

        for n_in_channel in range(0, self.n_in_channel):
            base_in_ct_idx = int(n_in_channel * stride0 * stride1 * output_step0 * output_step1)
            # Directly create kernel_shape[0] * kernel_shape[1] nodes for each (r_i2, r_j2)
            for r_i2 in range(0, output_step0):
                for r_j2 in range(0, output_step1):
                    # Create required rotation nodes for this (r_i2, r_j2) combination
                    for row_seg_idx in range(self.stride[0]):
                        for col_seg_idx in range(self.stride[1]):
                            split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) // stride0 + 1
                            split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_kernel_shape0):
                                for v_s in range(split_kernel_shape1):
                                    begin_row_idx = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (
                                        stride0 * output_step0
                                    )
                                    begin_row_idx = (begin_row_idx + stride0 * output_step0) % (stride0 * output_step0)
                                    begin_col_idx = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (
                                        stride1 * output_step1
                                    )
                                    begin_col_idx = (begin_col_idx + stride1 * output_step1) % (stride1 * output_step1)
                                    begin_idx = begin_row_idx * stride1 * output_step1 + begin_col_idx
                                    in_ct_idx = base_in_ct_idx + begin_idx
                                    row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) // (
                                        stride0 * output_step0
                                    )
                                    col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) // (
                                        stride1 * output_step1
                                    )
                                    step = int(row_step * block_shape1 + col_step)
                                    # Avoid creating unused intermediate nodes: use original ciphertext directly when step=0
                                    if step == 0:
                                        res_temp = x[in_ct_idx]
                                    else:
                                        res_temp = rotate_cols(x[in_ct_idx], [step])[0]
                                    rotated_x[n_in_channel].append(res_temp)

        n_channel_per_ct_out = 1
        if 2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1] < N:
            n_channel_per_ct_out = N / (2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1])
        else:
            n_channel_per_ct_out = 1

        temp_res = [0 for i in range(self.n_out_channel * self.output_step[0] * self.output_step[1])]

        for ct_idx in range(0, self.n_out_channel):
            for r_i2 in range(0, output_step0):
                for r_j2 in range(0, output_step1):
                    s = 0
                    out_ct_idx = ct_idx * output_step0 * output_step1 + r_i2 * output_step1 + r_j2
                    base_idx = (r_i2 * output_step1 + r_j2) * self.kernel_shape[0] * self.kernel_shape[1]
                    # Use the level of the first rotated_x as reference, ensuring all w_pt have consistent level
                    reference_level = rotated_x[0][base_idx].level
                    partial_sum: DataNode | None = None
                    x_ct_list = []
                    w_pt_list = []
                    for j in range(0, self.n_in_channel):
                        for k in range(0, self.kernel_shape[0] * self.kernel_shape[1]):
                            w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{j}_{k + base_idx}')
                            custom_compute(
                                inputs=[conv_data_source],  # All nodes reference the same data source
                                output=w_pt,
                                type='encode_pt',
                                attributes={
                                    'op_class': op_class,
                                    'type': 'weight_pt',
                                    'i': ct_idx,
                                    'j': j,
                                    'k': k + base_idx,
                                },
                            )
                            x_ct_list.append(rotated_x[j][k + base_idx])
                            w_pt_list.append(w_pt)
                    partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
                    s = rescale(partial_sum)
                    b_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}')
                    custom_compute(
                        inputs=[conv_data_source],  # Reference same data source
                        output=b_pt,
                        type='encode_pt',
                        attributes={'op_class': op_class, 'type': 'bias_pt', 'i': ct_idx},
                    )
                    s = add(s, b_pt)
                    # For the first output, consume unused input CTs so all appear in the graph
                    if ct_idx == 0 and r_i2 == 0 and r_j2 == 0:
                        used = self.get_used_input_indices()
                        total = self.n_in_channel * stride0 * stride1 * output_step0 * output_step1
                        zero_cts = []
                        for idx in range(total):
                            if idx not in used:
                                zero_cts.append(sub(x[idx], x[idx]))
                        if zero_cts:
                            sum_zero = zero_cts[0]
                            for zc in zero_cts[1:]:
                                sum_zero = add(sum_zero, zc)
                            sum_zero = sub(sum_zero, sum_zero)
                            s = add(s, drop_level(sum_zero, 1))
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

            # Shared mask: select row%out_skip==0 && col%out_skip==0
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
            0 for i in range(int(math.ceil(self.n_out_channel / n_channel_per_ct_out) * output_step0 * output_step1))
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

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt, repack_mask_pt).

        weight_pt[k][n][i]: k in n_out_channel, n in n_in_channel,
                            i in kernel_size * output_step[0] * output_step[1]
        bias_pt[i]: i in n_out_channel
        repack_mask_pt: a single CkksPlaintextRingtNode if need_repack, else None
        """
        inner = self.kernel_shape[0] * self.kernel_shape[1] * self.output_step[0] * self.output_step[1]
        weight_pt = [
            [
                [CkksPlaintextRingtNode(f'convw_{layer_id}_{k}_{n}_{i}') for i in range(inner)]
                for n in range(self.n_in_channel)
            ]
            for k in range(self.n_out_channel)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(self.n_out_channel)]
        repack_mask_pt = CkksPlaintextRingtNode(f'repack_mask_{layer_id}') if self.need_repack else None
        return weight_pt, bias_pt, repack_mask_pt

    def call(
        self, x: list[CkksCiphertextNode], weight_pt, bias_pt, N: int, repack_mask_pt=None
    ) -> list[CkksCiphertextNode]:
        pad0 = self.padding[0]
        pad1 = self.padding[1]
        stride0 = self.stride[0]
        stride1 = self.stride[1]
        output_step0 = self.output_step[0]
        output_step1 = self.output_step[1]
        kernel_shape0 = self.kernel_shape[0]
        kernel_shape1 = self.kernel_shape[1]
        block_shape1 = self.block_shape[1]

        rotated_x = [[] for i in range(self.n_in_channel)]

        for n_in_channel in range(0, self.n_in_channel):
            base_in_ct_idx = int(n_in_channel * stride0 * stride1 * output_step0 * output_step1)
            # Directly create kernel_shape[0] * kernel_shape[1] nodes for each (r_i2, r_j2)
            for r_i2 in range(0, output_step0):
                for r_j2 in range(0, output_step1):
                    # Create required rotation nodes for this (r_i2, r_j2) combination
                    for row_seg_idx in range(self.stride[0]):
                        for col_seg_idx in range(self.stride[1]):
                            split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) // stride0 + 1
                            split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) // stride1 + 1
                            for u_s in range(split_kernel_shape0):
                                for v_s in range(split_kernel_shape1):
                                    begin_row_idx = (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (
                                        stride0 * output_step0
                                    )
                                    begin_row_idx = (begin_row_idx + stride0 * output_step0) % (stride0 * output_step0)
                                    begin_col_idx = (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (
                                        stride1 * output_step1
                                    )
                                    begin_col_idx = (begin_col_idx + stride1 * output_step1) % (stride1 * output_step1)
                                    begin_idx = begin_row_idx * stride1 * output_step1 + begin_col_idx
                                    in_ct_idx = base_in_ct_idx + begin_idx
                                    row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) // (
                                        stride0 * output_step0
                                    )
                                    col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) // (
                                        stride1 * output_step1
                                    )
                                    step = int(row_step * block_shape1 + col_step)
                                    # Avoid creating unused intermediate nodes: use original ciphertext directly when step=0
                                    if step == 0:
                                        res_temp = x[in_ct_idx]
                                    else:
                                        res_temp = rotate_cols(x[in_ct_idx], [step])[0]
                                    rotated_x[n_in_channel].append(res_temp)

        n_channel_per_ct_out = 1
        if 2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1] < N:
            n_channel_per_ct_out = N / (2 * self.input_shape[0] / self.stride[0] * self.input_shape[1] / self.stride[1])
        else:
            n_channel_per_ct_out = 1

        temp_res = [0 for i in range(len(weight_pt) * self.output_step[0] * self.output_step[1])]

        for ct_idx in range(0, len(weight_pt)):
            for r_i2 in range(0, output_step0):
                for r_j2 in range(0, output_step1):
                    partial_sum: DataNode | None = None
                    x_ct_list = list()
                    w_pt_list = list()
                    # s = 0
                    out_ct_idx = ct_idx * output_step0 * output_step1 + r_i2 * output_step1 + r_j2
                    base_idx = (r_i2 * output_step1 + r_j2) * self.kernel_shape[0] * self.kernel_shape[1]
                    for j in range(0, len(weight_pt[ct_idx])):
                        for k in range(0, self.kernel_shape[0] * self.kernel_shape[1]):
                            x_ct_list.append(rotated_x[j][k + base_idx])
                            w_pt_list.append(weight_pt[ct_idx][j][k + base_idx])
                    partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
                    s = rescale(partial_sum)
                    s = add(s, bias_pt[ct_idx])
                    # For the first output, consume unused input CTs so all appear in the graph
                    if ct_idx == 0 and r_i2 == 0 and r_j2 == 0:
                        used = self.get_used_input_indices()
                        total = self.n_in_channel * stride0 * stride1 * output_step0 * output_step1
                        zero_cts = []
                        for idx in range(total):
                            if idx not in used:
                                zero_cts.append(sub(x[idx], x[idx]))
                        if zero_cts:
                            sum_zero = zero_cts[0]
                            for zc in zero_cts[1:]:
                                sum_zero = add(sum_zero, zc)
                            sum_zero = sub(sum_zero, sum_zero)
                            s = add(s, drop_level(sum_zero, 1))
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

        res = [0 for i in range(int(math.ceil(len(weight_pt) / n_channel_per_ct_out) * output_step0 * output_step1))]
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
