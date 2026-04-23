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
from inference.model_generator.layers.fhe_op_utils import naf_weight


op_class = 'MultiplexedConv2DPackedLayerDepthwise'


class MultiplexedConv2DPackedLayerDepthwise:
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

    def get_fhe_op_count(self, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   rotate_kernel + mult_plain + add_accum + rescale_base,
            level-1: add(bias)  [stride=1]  or  mult_plain(select)+rescale  [stride>1],
            level-2: rotate_stride + add(bias+accum)  [stride>1 only],
          }

        Depthwise: no block-direction rotation step (each ct is processed independently).
        gen_rotated_x over n_packed_in_channel cts:
          input_rotate_units[0] = skip[0]*input_shape[1]*skip[1] (power of 2)
          input_rotate_units[1] = skip[1] (power of 2)
          row direction: populate_rotations_2_sides(c, kh, unit_0), fc0=kh//2
            primitive rotates per ct = sum(naf_weight(i) for i in range(-fc0,kh-fc0) if i!=0)
          col direction: kh calls of populate_rotations_2_sides(r, kw, unit_1), fc1=kw//2
            primitive rotates per ct = kh * sum(naf_weight(j) for j in range(-fc1,kw-fc1) if j!=0)

        Per input ct (= n_packed_in_channel):
          mult_plain: kernel_size, add: kernel_size-1, rescale: 1  [level → level-1]

        stride=1 path (at level-1): n_packed_in_channel add (bias).
        stride>1 path (at level-1): simulate rot_step per (ct_idx, i) with naf_weight;
          valid_n mult_plain + valid_n rescale;  [level-1 → level-2]
          (at level-2): rotate_stride + accumulate + bias adds.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        kh, kw = self.kernel_shape

        # Kernel rotations: units are powers of 2
        fc0 = kh // 2
        fc1 = kw // 2
        rots_row = sum(naf_weight(i) for i in range(-fc0, kh - fc0) if i != 0)
        rots_col = kh * sum(naf_weight(j) for j in range(-fc1, kw - fc1) if j != 0)
        ops[lv]['rotate'] += self.n_packed_in_channel * (rots_row + rots_col)

        ops[lv]['mult_plain'] += self.n_packed_in_channel * kernel_size
        ops[lv]['add'] += self.n_packed_in_channel * (kernel_size - 1)
        ops[lv]['rescale'] += self.n_packed_in_channel
        lv -= 1

        if self.stride[0] == 1 and self.stride[1] == 1:
            # stride=1: just add bias per ct
            ops[lv]['add'] += self.n_packed_in_channel
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

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt, mask_pt).

        weight_pt[j][k]: j in n_packed_in_channel, k in kernel_size
        bias_pt[i]: i in n_packed_out_channel
        mask_pt[i]: i in n_out_channel (each (ct_idx, channel_in_ct) needs its own
                    source-position mask; empty list if stride==1)
        """
        kernel_size = self.kernel_shape[0] * self.kernel_shape[1]
        weight_pt = [
            [CkksPlaintextRingtNode(f'convw_{layer_id}_{j}_{k}') for k in range(kernel_size)]
            for j in range(self.n_packed_in_channel)
        ]
        import math as _math

        n_bias = _math.ceil(self.n_out_channel / (self.stride[0] * self.stride[1] * self.n_channel_per_ct))
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_bias)]
        if self.stride[0] != 1 or self.stride[1] != 1:
            mask_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(self.n_out_channel)]
        else:
            mask_pt = []
        return weight_pt, bias_pt, mask_pt

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, mast_pt) -> list[CkksCiphertextNode]:
        # 1. Kernel direction rotation
        kernel_rotations = self.gen_rotated_x(x)
        # 2. Result computation and organization
        res: list = list()
        result_ct = list()
        for ct_idx in range(len(weight_pt)):
            partial_sum: DataNode | None = None
            x_ct_list = list()
            w_pt_list = list()
            for j in range(len(weight_pt[ct_idx])):
                w_pt = weight_pt[ct_idx][j]
                x_ct_list.append(kernel_rotations[ct_idx][j])
                w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(partial_sum)
            if self.stride[0] == 1 and self.stride[1] == 1:
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
                for i in range(self.n_channel_per_ct):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        c_m = mult(s, mast_pt[ct_idx * self.n_channel_per_ct + i])
                        c_m = rescale(c_m)
                        result_ct.append(rotate_cols(c_m, [steps[int(i / self.skip[0])]])[0])
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

    def make_mask_pt_nodes(self, layer_id):
        """Create mask_pt nodes for lazy mode (offline-generated, per (ct_idx, channel_in_ct)).

        Each entry is a source-position mask (target mask rotated by -step_k); since step_k
        depends on the full channel_global, entries do not repeat across ct_idx. Returns an
        empty list when stride==1 (no mask needed).
        """
        if self.stride[0] == 1:
            return []
        return [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(self.n_out_channel)]

    def call_custom_compute(
        self, x: list[CkksCiphertextNode], conv_data_source, mask_pt_nodes=None
    ) -> list[CkksCiphertextNode]:
        # Weight/bias still go through encode_pt (lazy), but mask_pt is offline
        # (populated by prepare_weight_lazy on the C++ side) and passed in as a
        # list of static plaintext nodes shared across ct_idx.
        if mask_pt_nodes is None:
            mask_pt_nodes = []

        # 1. Calculate the number of input ciphertexts to process
        n_pack_in_channel = int(np.ceil(self.n_in_channel / self.n_channel_per_ct))
        # Only generate kernel rotations for needed input ciphertexts (avoid generating unused nodes)
        kernel_rotations = self.gen_rotated_x(x)

        # 2. Result computation and organization
        res: list = list()
        result_ct = list()

        k_size = self.kernel_shape[0] * self.kernel_shape[1]
        for ct_idx in range(n_pack_in_channel):
            partial_sum: DataNode | None = None
            x_ct_list = list()
            w_pt_list = list()
            for j in range(k_size):
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{ct_idx}_{j}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'weight_pt', 'i': ct_idx, 'j': j},
                )
                x_ct_list.append(kernel_rotations[ct_idx][j])
                w_pt_list.append(w_pt)
            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(partial_sum)
            if self.stride[0] == 1 and self.stride[1] == 1:
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

                for i in range(self.n_channel_per_ct):
                    if (ct_idx * self.n_channel_per_ct + i) < self.n_out_channel:
                        c_m = mult(s, mask_pt_nodes[ct_idx * self.n_channel_per_ct + i])
                        c_m = rescale(c_m)
                        result_ct.append(rotate_cols(c_m, [steps[int(i / self.skip[0])]])[0])
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
