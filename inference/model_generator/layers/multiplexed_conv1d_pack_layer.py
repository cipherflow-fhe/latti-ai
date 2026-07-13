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


op_class = 'MultiplexedConv1DPackedLayer'


class MultiplexedConv1DPackedLayer:
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
    ):
        self.n_out_channel: int = n_out_channel
        self.n_in_channel: int = n_in_channel
        self.input_shape: int = input_shape
        self.kernel_shape: int = kernel_shape
        self.stride: int = stride
        self.skip: int = skip
        self.n_channel_per_ct: int = n_channel_per_ct
        self.n_packed_in_channel: int = n_packed_in_channel
        self.n_packed_out_channel: int = n_packed_out_channel
        self.input_shape_ct: int = input_shape * skip
        self.n_block_per_ct: int = int(np.ceil(n_channel_per_ct / skip))

    def get_fhe_op_count(self, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   ops that run at input level (before each rescale),
            level-1: ops that run after the base rescale (rearrange path only),
          }

        gen_rotated_x over n_packed_in_channel cts:
          per ct: populate_rotations_2_sides(kernel_shape, skip) — fc=kernel_shape//2
            steps i*skip for i in range(-fc, kernel_shape-fc) if i!=0
            primitive rotates per ct = sum(naf_weight(i*skip) for those i)

        Per output group (n_out_groups = ceil(n_out_channel / n_block_per_ct)):
          block rotations for b>0: kernel_shape calls of rotate_cols(rotated_x[in_ct][k],[b*input_shape_ct])
            per (in_ct, b): kernel_shape * naf_weight(b * input_shape_ct) rotates
          mult_plain: n_packed_in_channel * n_block_per_ct * kernel_shape per group
          add (accumulate): same - 1
          sum_slot(skip, 1): floor(log2(skip)) rotate + add each (steps are powers of 2)
          rescale: 1 per group  [level → level-1 boundary]

        No-rearrange path (at level-1): n_out_groups add (bias).
        Rearrange path (at level-1, skip>1 or stride>1):
          n_out_channel mult_plain (select) + n_out_channel rescale
          + simulate rotation = target_base - source_base per (po, ch_local) with naf_weight
          + (n_out_channel - n_packed_out) add (accumulate) + n_packed_out add (bias)
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level  # tracks current level as rescales are consumed

        n_out_groups = int(np.ceil(self.n_out_channel / self.n_block_per_ct))
        n_packed_out = int(np.ceil(self.n_out_channel / self.n_channel_per_ct))
        terms_per_group = self.n_packed_in_channel * self.n_block_per_ct * self.kernel_shape

        # Kernel rotations: steps i*skip for i in range(-fc, ks-fc) if i!=0
        fc = self.kernel_shape // 2
        rots_per_ct = sum(naf_weight(i * self.skip) for i in range(-fc, self.kernel_shape - fc) if i != 0)
        ops[lv]['rotate'] += self.n_packed_in_channel * rots_per_ct

        # Block rotations: kernel_shape * naf_weight(b * input_shape_ct) per (wg, in_ct, b>0)
        ops[lv]['rotate'] += (
            n_out_groups
            * self.n_packed_in_channel
            * self.kernel_shape
            * sum(naf_weight(b * self.input_shape_ct) for b in range(1, self.n_block_per_ct))
        )

        # mult_plain + accumulate add
        ops[lv]['mult_plain'] += n_out_groups * terms_per_group
        ops[lv]['add'] += n_out_groups * (terms_per_group - 1)

        # sum_slot: floor(log2(skip)) rotate + add each
        log2_skip = int(np.floor(np.log2(self.skip))) if self.skip > 1 else 0
        ops[lv]['rotate'] += n_out_groups * log2_skip
        ops[lv]['add'] += n_out_groups * log2_skip

        # rescale (level → level-1)
        ops[lv]['rescale'] += n_out_groups
        lv -= 1

        # bias / rearrange at lv (= level-1)
        needs_rearrange = self.skip > 1 or self.stride > 1
        if not needs_rearrange:
            ops[lv]['add'] += n_out_groups  # bias
        else:
            # select mult_plain + rescale per output channel (level-1 → level-2)
            ops[lv]['mult_plain'] += self.n_out_channel
            ops[lv]['rescale'] += self.n_out_channel
            lv -= 1

            # rotate + accumulate add + bias add at level-2
            skip_out = self.skip * self.stride
            output_shape_val = self.input_shape // self.stride
            for po in range(n_packed_out):
                for ch_local in range(self.n_channel_per_ct):
                    out_ch = po * self.n_channel_per_ct + ch_local
                    if out_ch >= self.n_out_channel:
                        break
                    wg = out_ch // self.n_block_per_ct
                    t = out_ch % self.n_block_per_ct
                    if wg >= n_out_groups:
                        break
                    source_base = t * self.input_shape_ct
                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    target_base = group * (output_shape_val * skip_out) + ch_offset
                    rotation = target_base - source_base
                    if rotation != 0:
                        ops[lv]['rotate'] += naf_weight(rotation)

            ops[lv]['add'] += (self.n_out_channel - n_packed_out) + n_packed_out

        return dict(ops)

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

        # 2. Mult + Add
        n_out_groups = int(np.ceil(self.n_out_channel / self.n_block_per_ct))
        conv_results = list()

        for wg in range(n_out_groups):
            x_ct_list = []
            w_pt_list = []
            for in_ct in range(self.n_packed_in_channel):
                for b in range(self.n_block_per_ct):
                    # Block rotation
                    if b == 0:
                        block_rots = [rotated_x[in_ct][k] for k in range(self.kernel_shape)]
                    else:
                        block_rots = []
                        for k in range(self.kernel_shape):
                            rot = rotate_cols(rotated_x[in_ct][k], [b * self.input_shape_ct])
                            block_rots.append(rot[0])

                    w_idx = in_ct * self.n_block_per_ct + b
                    for k in range(self.kernel_shape):
                        w_pt = CkksPlaintextRingtNode(f'encode_pt_{wg}_{w_idx}_{k}')
                        custom_compute(
                            inputs=[conv_data_source],
                            output=w_pt,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'weight_pt',
                                'i': wg,
                                'j': w_idx,
                                'k': k,
                            },
                        )
                        x_ct_list.append(block_rots[k])
                        w_pt_list.append(w_pt)

            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

            # 3. Skip accumulation
            s = self.sum_slot(partial_sum, self.skip, 1)

            # 4. Rescale
            s = rescale(s)
            conv_results.append(s)

        # 5. Add bias
        needs_rearrange = self.skip > 1 or self.stride > 1

        if not needs_rearrange:
            res = list()
            for wg in range(n_out_groups):
                b_pt = CkksPlaintextRingtNode(f'encode_pt_bias_{wg}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=b_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'bias_pt', 'i': wg},
                )
                res.append(add(conv_results[wg], b_pt))
            return res
        else:
            # Select + rotate + merge
            skip_out = self.skip * self.stride
            output_shape = self.input_shape // self.stride
            n_packed_out = int(np.ceil(self.n_out_channel / self.n_channel_per_ct))

            # Only generate select tensors that can be referenced by an output
            # channel. Otherwise lazy graph generation leaves orphan encode_pt
            # nodes when n_out_channel < n_block_per_ct.
            n_select = min(self.n_block_per_ct, self.n_out_channel)
            select_pts = []
            for t in range(n_select):
                s_pt = CkksPlaintextRingtNode(f'encode_pt_select_{t}')
                custom_compute(
                    inputs=[conv_data_source],
                    output=s_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'select_pt', 'i': t},
                )
                select_pts.append(s_pt)

            res = list()
            for po in range(n_packed_out):
                combined = None
                for ch_local in range(self.n_channel_per_ct):
                    out_ch = po * self.n_channel_per_ct + ch_local
                    if out_ch >= self.n_out_channel:
                        break

                    wg = out_ch // self.n_block_per_ct
                    t = out_ch % self.n_block_per_ct
                    if wg >= n_out_groups:
                        break

                    masked = mult(conv_results[wg], select_pts[t])
                    masked = rescale(masked)

                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    source_base = t * self.input_shape_ct
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

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt, block_select_pt) matching call().

        weight_pt[i][j][k]: i in n_out_groups, j in n_packed_in_channel*n_block_per_ct, k in kernel_shape
        bias_pt[i]: i in ceil(n_out_channel / n_channel_per_ct)
        block_select_pt[i]: i in min(n_block_per_ct, n_out_channel)  (empty if not needs_rearrange)
        """
        import math as _math

        n_out_groups = _math.ceil(self.n_out_channel / self.n_block_per_ct)
        n_packed_out = _math.ceil(self.n_out_channel / self.n_channel_per_ct)
        weight_pt = [
            [
                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(self.kernel_shape)]
                for j in range(self.n_packed_in_channel * self.n_block_per_ct)
            ]
            for i in range(n_out_groups)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out)]
        needs_rearrange = self.skip > 1 or self.stride > 1
        if needs_rearrange:
            n_select = min(self.n_block_per_ct, self.n_out_channel)
            block_select_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_select)]
        else:
            block_select_pt = []
        return weight_pt, bias_pt, block_select_pt

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, block_select_pt=None) -> list[CkksCiphertextNode]:
        # 1. Kernel direction rotation
        rotated_x = self.gen_rotated_x(x)

        # 2. Mult + Add
        n_out_groups = len(weight_pt)
        conv_results = list()

        for wg in range(n_out_groups):
            x_ct_list = []
            w_pt_list = []
            for in_ct in range(self.n_packed_in_channel):
                for b in range(self.n_block_per_ct):
                    # Block rotation
                    if b == 0:
                        block_rots = [rotated_x[in_ct][k] for k in range(self.kernel_shape)]
                    else:
                        block_rots = []
                        for k in range(self.kernel_shape):
                            rot = rotate_cols(rotated_x[in_ct][k], [b * self.input_shape_ct])
                            block_rots.append(rot[0])

                    w_idx = in_ct * self.n_block_per_ct + b
                    for k in range(self.kernel_shape):
                        x_ct_list.append(block_rots[k])
                        w_pt_list.append(weight_pt[wg][w_idx][k])

            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

            # 3. Skip accumulation
            s = self.sum_slot(partial_sum, self.skip, 1)

            # 4. Rescale
            s = rescale(s)
            conv_results.append(s)

        # 5. Add bias
        needs_rearrange = self.skip > 1 or self.stride > 1

        if not needs_rearrange:
            res = list()
            for wg in range(n_out_groups):
                res.append(add(conv_results[wg], bias_pt[wg]))
            return res
        else:
            # Select + rotate + merge
            skip_out = self.skip * self.stride
            output_shape = self.input_shape // self.stride
            n_packed_out = int(np.ceil(self.n_out_channel / self.n_channel_per_ct))

            res = list()
            for po in range(n_packed_out):
                combined = None
                for ch_local in range(self.n_channel_per_ct):
                    out_ch = po * self.n_channel_per_ct + ch_local
                    if out_ch >= self.n_out_channel:
                        break

                    wg = out_ch // self.n_block_per_ct
                    t = out_ch % self.n_block_per_ct
                    if wg >= n_out_groups:
                        break

                    masked = mult(conv_results[wg], block_select_pt[t])
                    masked = rescale(masked)

                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    source_base = t * self.input_shape_ct
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
