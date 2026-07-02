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
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts, naf_weight


op_class = 'MultiplexedDWConv1DPackedLayer'


class MultiplexedDWConv1DPackedLayer:
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

    def get_fhe_op_count(self, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call(), grouped by level.

        Returns a dict keyed by level:
          {
            level:   ops that run at input level (before base rescale),
            level-1: ops that run after the base rescale (rearrange path only),
            level-2: ops that run after the rearrange rescale (rearrange path only),
          }

        Depthwise: each ct is processed independently (no cross-ct loop).
        gen_rotated_x over n_packed_ct cts:
          per ct: populate_rotations_2_sides(kernel_shape, skip) — fc=kernel_shape//2
            steps i*skip for i in range(-fc, kernel_shape-fc) if i!=0
            primitive rotates per ct = sum(naf_weight(i*skip) for those i)

        Per ct (n_packed_ct total):
          mult_plain: kernel_shape, add: kernel_shape-1, rescale: 1  [level → level-1]

        No-rearrange path (at level-1): n_packed_ct add (bias).
        Rearrange path (at level-1):
          n_channel mult (select) + n_channel rescale  [level-1 → level-2]
          (at level-2): simulate rotation + (n_channel - n_packed_out) add + n_packed_out add (bias)
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        n_packed_out = int(np.ceil(self.n_channel / self.n_channel_per_ct))

        # Kernel rotations at lv
        fc = self.kernel_shape // 2
        rots_per_ct = sum(naf_weight(i * self.skip) for i in range(-fc, self.kernel_shape - fc) if i != 0)
        ops[lv]['rotate'] += self.n_packed_ct * rots_per_ct

        # mult_plain + accumulate + rescale at lv
        ops[lv]['mult_plain'] += self.n_packed_ct * self.kernel_shape
        ops[lv]['add'] += self.n_packed_ct * (self.kernel_shape - 1)
        ops[lv]['rescale'] += self.n_packed_ct
        lv -= 1

        # bias / rearrange at lv (= level-1)
        needs_rearrange = self.skip > 1 or self.stride > 1
        if not needs_rearrange:
            ops[lv]['add'] += self.n_packed_ct  # bias
        else:
            # select mult_plain + rescale per channel (level-1 → level-2)
            ops[lv]['mult_plain'] += self.n_channel
            ops[lv]['rescale'] += self.n_channel
            lv -= 1

            # rotate + accumulate + bias at level-2
            skip_out = self.skip * self.stride
            output_shape_val = self.input_shape // self.stride
            for po in range(n_packed_out):
                for ch_local in range(self.n_channel_per_ct):
                    ch = po * self.n_channel_per_ct + ch_local
                    if ch >= self.n_channel:
                        break
                    t = ch_local // self.skip
                    j_val = ch_local % self.skip
                    source_base = t * self.input_shape_ct + j_val
                    group = ch_local // skip_out
                    ch_offset = ch_local % skip_out
                    target_base = group * (output_shape_val * skip_out) + ch_offset
                    rotation = target_base - source_base
                    if rotation != 0:
                        ops[lv]['rotate'] += naf_weight(rotation)
            ops[lv]['add'] += (self.n_channel - n_packed_out) + n_packed_out

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

    def make_pt_nodes(self, layer_id):
        """Return (weight_pt, bias_pt, block_select_pt) matching call().

        weight_pt[ct_idx][k]: ct_idx in n_packed_ct, k in kernel_shape
        bias_pt[i]:           i in n_packed_ct
        block_select_pt[i]:   i in n_channel_per_ct (empty if not needs_rearrange)
        """
        import math as _math

        n_packed_ct = _math.ceil(self.n_channel / self.n_channel_per_ct)
        weight_pt = [
            [CkksPlaintextRingtNode(f'convw_{layer_id}_{ct_idx}_{k}') for k in range(self.kernel_shape)]
            for ct_idx in range(n_packed_ct)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_ct)]
        needs_rearrange = self.skip > 1 or self.stride > 1
        if needs_rearrange:
            n_local_ch = min(self.n_channel_per_ct, self.n_channel)
            block_select_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_local_ch)]
        else:
            block_select_pt = []
        return weight_pt, bias_pt, block_select_pt

    def get_memory(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return generated plaintext counts and estimated bytes for this layer."""
        import math as _math

        n_packed_ct = _math.ceil(self.n_channel / self.n_channel_per_ct)
        needs_rearrange = self.skip > 1 or self.stride > 1
        n_select = min(self.n_channel_per_ct, self.n_channel) if needs_rearrange else 0
        counts = {
            'weight': n_packed_ct * self.kernel_shape,
            'bias': n_packed_ct,
            'mask': n_select,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

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
