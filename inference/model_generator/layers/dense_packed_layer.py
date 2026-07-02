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
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.fhe_op_utils import memory_from_pt_counts, naf_weight


op_class = 'DensePackedLayer'


class DensePackedLayer:
    def __init__(
        self,
        n_out_channel,
        n_in_channel,
        input_shape,
        skip,
        pack,
        n_packed_in_feature,
        n_packed_out_feature,
        invalid_fill=None,
    ):
        self.n_out_channel: int = n_out_channel
        self.n_in_channel: int = n_in_channel
        self.input_shape: list[int] = input_shape
        self.skip: list[int] = skip
        self.invalid_fill: list[int] = invalid_fill if invalid_fill is not None else [1, 1]

        if int(input_shape[0]) & (int(input_shape[0]) - 1) != 0 or int(input_shape[1]) & (int(input_shape[1]) - 1) != 0:
            raise ValueError(f'input_shape must be powers of 2, got: [{input_shape[0]}, {input_shape[1]}]')
        if int(skip[0]) & (int(skip[0]) - 1) != 0 or int(skip[1]) & (int(skip[1]) - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')

        self.pack: int = pack
        self.n_packed_in_feature: int = n_packed_in_feature
        self.n_packed_out_feature: int = n_packed_out_feature

        self.mark: int = 0

    def get_fhe_op_count_skip_0d(self, n_input_ct: int, skip_0d: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_skip_0d(), grouped by level.

        Baby-step rotations per input ct: steps b*skip_0d for b in 1..bsgs_bs-1.
        Per output packed feature, per input ct, per giant step g:
          - inner accumulate: bsgs_bs mult_plain + (bsgs_bs-1) add
          - giant-step rotate (g > 0): step = g*bsgs_bs*skip_0d, cost = naf_weight(step)
          - add partial to total (g > 0): 1 add
        After all g: 1 rescale + 1 add (bias).

        Level structure:
          level:   rotate_baby + rotate_giant + mult_plain + add_accum + rescale
          level-1: add_bias
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        bsgs_bs = int(math.ceil(math.sqrt(self.pack)))
        bsgs_gs = int(math.ceil(self.pack / bsgs_bs))

        rotate_baby = n_input_ct * sum(naf_weight(b * skip_0d) for b in range(1, bsgs_bs))
        rotate_giant = (
            self.n_packed_out_feature * n_input_ct * sum(naf_weight(g * bsgs_bs * skip_0d) for g in range(1, bsgs_gs))
        )
        rotate_total = rotate_baby + rotate_giant

        mult_plain_total = self.n_packed_out_feature * n_input_ct * bsgs_gs * bsgs_bs
        # accumulate adds: (bsgs_bs-1) inner + (bsgs_gs-1) outer per (out,in_ct)
        add_accum = self.n_packed_out_feature * n_input_ct * (bsgs_gs * (bsgs_bs - 1) + (bsgs_gs - 1))
        rescale_total = self.n_packed_out_feature

        ops[lv]['rotate'] += rotate_total
        ops[lv]['mult_plain'] += mult_plain_total
        ops[lv]['add'] += add_accum
        ops[lv]['rescale'] += rescale_total
        lv -= 1

        ops[lv]['add'] += self.n_packed_out_feature  # bias

        return dict(ops)

    def get_fhe_op_count_multiplexed(self, n_input_ct: int, n: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_multiplexed(), grouped by level.

        input_ct_shape = [input_shape[0]*skip[0], input_shape[1]*skip[1]]
        block_size = input_ct_shape[0] * input_ct_shape[1]  (power of 2)
        n_num_pre_ct = ceil(N/2 / (block_size))
        n_block_input = ceil(n_channel / (n_num_pre_ct * n_channel_per_block)) * n_num_pre_ct
        n_packed_out = ceil(n_out_channel / n_num_pre_ct)

        Rotations per input ct: steps k*block_size for k in 1..n_num_pre_ct-1.
          block_size is power of 2, so naf_weight(k*block_size) = naf_weight(k).
        Per output packed feature:
          mult_plain: n_block_input, add: n_block_input-1 (accumulate) + 1 (bias), rescale: 1
          fold: log2(block_size) rotate + log2(block_size) add

        Level structure:
          level:   rotate_expand + mult_plain + add_accum + rescale
          level-1: rotate_fold + add_fold + add_bias
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        N_half = int(n / 2)
        n_num_pre_ct = int(np.ceil(N_half / (input_ct_shape[0] * input_ct_shape[1])))
        valid_skip_0 = self.skip[0] // self.invalid_fill[0]
        valid_skip_1 = self.skip[1] // self.invalid_fill[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel = self.n_in_channel // (self.input_shape[0] * self.input_shape[1])
        n_block_input = int(np.ceil(n_channel / (n_num_pre_ct * n_channel_per_block))) * n_num_pre_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_num_pre_ct))
        block_size = input_ct_shape[0] * input_ct_shape[1]
        log2_block = int(math.log2(block_size))

        rotate_expand = n_input_ct * sum(naf_weight(k) for k in range(1, n_num_pre_ct))
        ops[lv]['rotate'] += rotate_expand
        ops[lv]['mult_plain'] += n_packed_out * n_block_input
        ops[lv]['add'] += n_packed_out * (n_block_input - 1)  # accumulate
        ops[lv]['rescale'] += n_packed_out
        lv -= 1

        ops[lv]['rotate'] += n_packed_out * log2_block  # fold
        ops[lv]['add'] += n_packed_out * log2_block  # fold adds
        ops[lv]['add'] += n_packed_out  # bias

        return dict(ops)

    def get_fhe_op_count_1d_multiplexed(self, n_input_ct: int, n: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_1d_multiplexed(), grouped by level.

        block_size = shape * skip  (power of 2)
        n_block_per_ct = N/2 // block_size
        n_valid_per_ct = n_block_per_ct * (skip // invalid_fill)
        n_actual_channels = n_in_channel // shape
        n_block_input = ceil(n_actual_channels / n_valid_per_ct) * n_block_per_ct
        n_packed_out = ceil(n_out_channel / n_block_per_ct)

        Rotations per input ct: steps k*block_size for k in 1..n_block_per_ct-1.
          block_size is power of 2, so naf_weight(k*block_size) = naf_weight(k).
        Per output group: mult_plain: n_block_input, add: n_block_input (accum + bias),
          rescale: 1, fold: log2(block_size) rotate + log2(block_size) add.

        Level structure:
          level:   rotate_expand + mult_plain + add_accum + rescale
          level-1: rotate_fold + add_fold + add_bias
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        N_half = n // 2
        shape = int(self.input_shape[0])
        skip_val = int(self.skip[0])
        invalid_fill_val = int(self.invalid_fill[0])
        block_size = shape * skip_val
        n_block_per_ct = N_half // block_size
        valid_sub = skip_val // invalid_fill_val
        n_valid_per_ct = n_block_per_ct * valid_sub
        n_actual_channels = self.n_in_channel // shape
        n_block_input = int(np.ceil(n_actual_channels / n_valid_per_ct)) * n_block_per_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_block_per_ct))
        log2_block = int(math.log2(block_size))

        rotate_expand = n_input_ct * sum(naf_weight(k) for k in range(1, n_block_per_ct))
        ops[lv]['rotate'] += rotate_expand
        ops[lv]['mult_plain'] += n_packed_out * n_block_input
        ops[lv]['add'] += n_packed_out * (n_block_input - 1)  # accumulate
        ops[lv]['rescale'] += n_packed_out
        lv -= 1

        ops[lv]['rotate'] += n_packed_out * log2_block  # fold
        ops[lv]['add'] += n_packed_out * log2_block  # fold adds
        ops[lv]['add'] += n_packed_out  # bias

        return dict(ops)

    def get_memory_skip_0d(self, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return plaintext counts and estimated bytes for call_skip_0d()."""
        counts = {
            'weight': self.n_packed_out_feature * self.n_packed_in_feature * self.pack,
            'bias': self.n_packed_out_feature,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_memory_multiplexed(self, n: int, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return plaintext counts and estimated bytes for call_multiplexed()."""
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        N_half = int(n / 2)
        n_num_pre_ct = int(np.ceil(N_half / (input_ct_shape[0] * input_ct_shape[1])))
        valid_skip_0 = self.skip[0] // self.invalid_fill[0]
        valid_skip_1 = self.skip[1] // self.invalid_fill[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel = self.n_in_channel // (self.input_shape[0] * self.input_shape[1])
        n_block_input = int(np.ceil(n_channel / (n_channel_per_block * n_num_pre_ct))) * n_num_pre_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_num_pre_ct))
        counts = {
            'weight': n_packed_out * n_block_input,
            'bias': n_packed_out,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    def get_memory_1d_multiplexed(self, n: int, bytes_per_plaintext: int = 0) -> dict[str, int]:
        """Return plaintext counts and estimated bytes for call_1d_multiplexed()."""
        N_half = n // 2
        shape = int(self.input_shape[0])
        skip_val = int(self.skip[0])
        invalid_fill_val = int(self.invalid_fill[0])
        block_size = shape * skip_val
        n_block_per_ct = N_half // block_size
        valid_sub = skip_val // invalid_fill_val
        n_valid_per_ct = n_block_per_ct * valid_sub
        n_actual_channels = self.n_in_channel // shape
        n_block_input = int(np.ceil(n_actual_channels / n_valid_per_ct)) * n_block_per_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_block_per_ct))
        counts = {
            'weight': n_packed_out * n_block_input,
            'bias': n_packed_out,
        }
        return memory_from_pt_counts(counts, bytes_per_plaintext)

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[DataNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

    def make_pt_nodes_skip_0d(self, layer_id):
        """Return (weight_pt, bias_pt) for call_skip_0d().

        weight_pt[m][i]: m in n_packed_out_feature, i in n_packed_in_feature * pack
        bias_pt[i]: i in n_packed_out_feature
        """
        weight_pt_size = self.n_packed_in_feature * self.pack
        weight_pt = [
            [CkksPlaintextRingtNode(f'densew_{layer_id}_{m}_{i}') for i in range(weight_pt_size)]
            for m in range(self.n_packed_out_feature)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(self.n_packed_out_feature)]
        return weight_pt, bias_pt

    def make_pt_nodes_multiplexed(self, layer_id, n):
        """Return (weight_pt, bias_pt) for call_multiplexed().

        weight_pt[i][j]: i in n_packed_out_feature_for_mult_pack, j in n_block_input
        bias_pt[i]: i in n_packed_out_feature_for_mult_pack
        """
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        N_half = int(n / 2)
        n_num_pre_ct = int(np.ceil(N_half / (input_ct_shape[0] * input_ct_shape[1])))
        valid_skip_0 = self.skip[0] // self.invalid_fill[0]
        valid_skip_1 = self.skip[1] // self.invalid_fill[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel = self.n_in_channel // (self.input_shape[0] * self.input_shape[1])
        n_block_input = int(np.ceil(n_channel / (n_channel_per_block * n_num_pre_ct))) * n_num_pre_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_num_pre_ct))
        weight_pt = [
            [CkksPlaintextRingtNode(f'densew_{layer_id}_{i}_{j}') for j in range(n_block_input)]
            for i in range(n_packed_out)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out)]
        return weight_pt, bias_pt

    def call_skip_0d(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, skip_0d: int):
        """Corresponds to C++ run_core_0d + run_skip_0d (BSGS approach)."""
        bsgs_bs = int(math.ceil(math.sqrt(self.pack)))
        bsgs_gs = int(math.ceil(self.pack / bsgs_bs))

        # Baby-step rotations for each input CT
        baby_rots = []
        for ct in x:
            if bsgs_bs > 1:
                steps = [b * skip_0d for b in range(1, bsgs_bs)]
                rots = [ct] + rotate_cols(ct, steps)
            else:
                rots = [ct]
            baby_rots.append(rots)

        result = []
        for out_idx in range(self.n_packed_out_feature):
            total = None
            for ct_in in range(len(x)):
                for g in range(bsgs_gs):
                    # Inner sum over baby-steps
                    x_ct_list = []
                    w_pt_list = []
                    b_end = min(bsgs_bs, self.pack - g * bsgs_bs)
                    for b in range(b_end):
                        d = g * bsgs_bs + b
                        weight_idx = ct_in * self.pack + d
                        x_ct_list.append(baby_rots[ct_in][b])
                        w_pt_list.append(weight_pt[out_idx][weight_idx])

                    inner = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

                    # Giant-step rotation (g=0 needs no rotation)
                    if g > 0:
                        inner = rotate_cols(inner, [g * bsgs_bs * skip_0d])[0]

                    if total is None:
                        total = inner
                    else:
                        total = add(total, inner)

            total = rescale(total)
            total = add(total, bias_pt[out_idx])
            result.append(total)
        return result

    def call_skip_0d_custom_compute(self, x: list[CkksCiphertextNode], dense_data_source, skip_0d: int):
        """Corresponds to C++ run_core_0d with lazy weight generation."""
        bsgs_bs = int(math.ceil(math.sqrt(self.pack)))
        bsgs_gs = int(math.ceil(self.pack / bsgs_bs))

        # Baby-step rotations for each input CT
        baby_rots = []
        for ct in x:
            if bsgs_bs > 1:
                steps = [b * skip_0d for b in range(1, bsgs_bs)]
                rots = [ct] + rotate_cols(ct, steps)
            else:
                rots = [ct]
            baby_rots.append(rots)

        result = []
        for out_idx in range(self.n_packed_out_feature):
            total = None
            for ct_in in range(len(x)):
                for g in range(bsgs_gs):
                    # Inner sum over baby-steps
                    x_ct_list = []
                    w_pt_list = []
                    b_end = min(bsgs_bs, self.pack - g * bsgs_bs)
                    for b in range(b_end):
                        d = g * bsgs_bs + b
                        weight_idx = ct_in * self.pack + d
                        w_pt = CkksPlaintextRingtNode(f'encode_pt_{out_idx}_{weight_idx}')
                        custom_compute(
                            inputs=[dense_data_source],
                            output=w_pt,
                            type='encode_pt',
                            attributes={
                                'op_class': op_class,
                                'type': 'weight_pt',
                                'i': out_idx,
                                'j': weight_idx,
                            },
                        )
                        x_ct_list.append(baby_rots[ct_in][b])
                        w_pt_list.append(w_pt)

                    inner = ct_pt_mult_accumulate(x_ct_list, w_pt_list)

                    # Giant-step rotation (g=0 needs no rotation)
                    if g > 0:
                        inner = rotate_cols(inner, [g * bsgs_bs * skip_0d])[0]

                    if total is None:
                        total = inner
                    else:
                        total = add(total, inner)

            total = rescale(total)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{out_idx}')
            custom_compute(
                inputs=[dense_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': out_idx},
            )
            total = add(total, b_pt)
            result.append(total)
        return result

    def call_multiplexed(self, x: list[DataNode], weight_pt, bias_pt, n):
        """Corresponds to C++ run_core_mult_pack + run_multiplexed."""
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        x_size = len(x)
        N_half = int(n / 2)
        n_num_pre_ct = int(np.ceil(N_half / (input_ct_shape[0] * input_ct_shape[1])))

        valid_skip_0 = self.skip[0] // self.invalid_fill[0]
        valid_skip_1 = self.skip[1] // self.invalid_fill[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel = self.n_in_channel // (self.input_shape[0] * self.input_shape[1])
        n_block_input = int(np.ceil(n_channel / (n_channel_per_block * n_num_pre_ct))) * n_num_pre_ct
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        block_size = input_ct_shape[0] * input_ct_shape[1]

        # Replicate input data across CT blocks inside the dense layer
        n_rot_factor = n_num_pre_ct // n_block_input if 0 < n_block_input < n_num_pre_ct else 1
        n_rep_iters = int(np.floor(np.log2(n_rot_factor))) if n_rot_factor > 1 else 0

        x_rep = list(x)
        for x_id in range(x_size):
            for r in range(n_rep_iters):
                x_rep[x_id] = add(x_rep[x_id], rotate_cols(x_rep[x_id], -(2**r) * n_block_input * block_size)[0])

        n_rotations_per_ct = min(n_block_input, n_num_pre_ct)
        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x_rep[x_id], n_rotations_per_ct - 1, block_size))

        result = []

        for packed_out_feature_idx in range(n_packed_out_feature_for_mult_pack):
            x_ct_list = []
            w_pt_list = []
            for in_feature_idx in range(len(weight_pt[packed_out_feature_idx])):
                group = in_feature_idx // n_num_pre_ct
                offset = in_feature_idx % n_num_pre_ct
                x_ct = rotated_cts[group][offset]
                w_pt = weight_pt[packed_out_feature_idx][in_feature_idx]
                x_ct_list.append(x_ct)
                w_pt_list.append(w_pt)

            partial_sum = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(partial_sum)
            b_pt = bias_pt[packed_out_feature_idx]
            s = add(s, b_pt)
            n_fold = block_size
            while n_fold > 1:
                rotated = rotate_cols(s, n_fold // 2)
                s = add(s, rotated[0])
                n_fold //= 2
            result.append(s)
        return result

    def make_pt_nodes_1d_multiplexed(self, layer_id, n):
        """Return (weight_pt, bias_pt) for call_1d_multiplexed().

        input_shape is 1D: [shape], skip is 1D: [skip_val].
        skip already contains invalid_fill, so:
          block_stride  = skip                              (NOT skip * invalid_fill)
          block_size    = shape * skip
          n_block_per_ct = N_half // block_size
          valid_sub      = skip // invalid_fill             (valid sub_pos per block)
          n_valid_per_ct = n_block_per_ct * valid_sub       (channels with actual data)
          n_actual_ch    = n_in_channel // shape              (flattened → channel count)
          n_block_input  = ceil(n_actual_ch / n_valid_per_ct) * n_block_per_ct
          n_packed_out   = ceil(n_out_channel / n_block_per_ct)

        weight_pt[i][j]: i in n_packed_out, j in n_block_input
        bias_pt[i]:       i in n_packed_out
        """
        N_half = n // 2
        shape = int(self.input_shape[0])
        skip_val = int(self.skip[0])
        invalid_fill_val = int(self.invalid_fill[0])
        block_stride = skip_val  # skip already contains invalid_fill
        block_size = shape * block_stride  # = shape * skip
        n_block_per_ct = N_half // block_size
        valid_sub = skip_val // invalid_fill_val  # valid sub_pos per block
        n_valid_per_ct = n_block_per_ct * valid_sub  # channels with actual data
        # n_in_channel is the flattened count (channels * spatial_length); divide by shape to get
        # the number of actual conv1d output channels, which determines the number of input CTs.
        n_actual_channels = self.n_in_channel // shape
        n_block_input = int(np.ceil(n_actual_channels / n_valid_per_ct)) * n_block_per_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_block_per_ct))

        weight_pt = [
            [CkksPlaintextRingtNode(f'densew_{layer_id}_{i}_{j}') for j in range(n_block_input)]
            for i in range(n_packed_out)
        ]
        bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out)]
        return weight_pt, bias_pt

    def call_1d_multiplexed(self, x: list, weight_pt, bias_pt, n):
        """Corresponds to C++ run_1d_multiplexed.

        x is a list of CkksCiphertextNode (one per input ciphertext).
        skip already contains invalid_fill:
          block_stride = skip  (NOT skip * invalid_fill)
          block_size   = shape * skip
        Rotation unit is block_size; fold over block_size after accumulation.
        """
        N_half = n // 2
        shape = int(self.input_shape[0])
        skip_val = int(self.skip[0])
        invalid_fill_val = int(self.invalid_fill[0])
        block_stride = skip_val  # skip already contains invalid_fill
        block_size = shape * block_stride  # = shape * skip
        n_block_per_ct = N_half // block_size
        valid_sub = skip_val // invalid_fill_val  # valid sub_pos per block
        n_valid_per_ct = n_block_per_ct * valid_sub  # channels with actual data
        # n_in_channel is flattened (channels * spatial_length); divide by shape to get actual channel count.
        n_actual_channels = self.n_in_channel // shape
        n_block_input = int(np.ceil(n_actual_channels / n_valid_per_ct)) * n_block_per_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_block_per_ct))
        x_size = len(x)

        # Rotations per input ct: n_block_per_ct rotations of block_size each
        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x[x_id], n_block_per_ct - 1, block_size))

        result = []
        for out_group in range(n_packed_out):
            x_ct_list = []
            w_pt_list = []
            for rot_idx in range(len(weight_pt[out_group])):
                group = rot_idx // n_block_per_ct
                offset = rot_idx % n_block_per_ct
                x_ct_list.append(rotated_cts[group][offset])
                w_pt_list.append(weight_pt[out_group][rot_idx])

            s = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(s)
            s = add(s, bias_pt[out_group])

            # Fold over block_size (sum spatial positions into slot 0 of each block)
            n_fold = block_size
            while n_fold > 1:
                rotated = rotate_cols(s, n_fold // 2)
                s = add(s, rotated[0])
                n_fold //= 2

            result.append(s)
        return result

    def call_1d_multiplexed_custom_compute(self, x: list, dense_data_source, n):
        """Corresponds to C++ run_1d_multiplexed with lazy weight generation."""
        N_half = n // 2
        shape = int(self.input_shape[0])
        skip_val = int(self.skip[0])
        invalid_fill_val = int(self.invalid_fill[0])
        block_stride = skip_val
        block_size = shape * block_stride
        n_block_per_ct = N_half // block_size
        valid_sub = skip_val // invalid_fill_val
        n_valid_per_ct = n_block_per_ct * valid_sub
        n_actual_channels = self.n_in_channel // shape
        n_block_input = int(np.ceil(n_actual_channels / n_valid_per_ct)) * n_block_per_ct
        n_packed_out = int(np.ceil(self.n_out_channel / n_block_per_ct))
        x_size = len(x)

        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x[x_id], n_block_per_ct - 1, block_size))

        result = []
        for out_group in range(n_packed_out):
            x_ct_list = []
            w_pt_list = []
            for rot_idx in range(n_block_input):
                group = rot_idx // n_block_per_ct
                offset = rot_idx % n_block_per_ct
                x_ct_list.append(rotated_cts[group][offset])
                w_pt = CkksPlaintextRingtNode(f'encode_pt_{out_group}_{rot_idx}')
                custom_compute(
                    inputs=[dense_data_source],
                    output=w_pt,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'weight_pt', 'i': out_group, 'j': rot_idx},
                )
                w_pt_list.append(w_pt)

            s = ct_pt_mult_accumulate(x_ct_list, w_pt_list)
            s = rescale(s)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{out_group}')
            custom_compute(
                inputs=[dense_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': out_group},
            )
            s = add(s, b_pt)

            n_fold = block_size
            while n_fold > 1:
                rotated = rotate_cols(s, n_fold // 2)
                s = add(s, rotated[0])
                n_fold //= 2

            result.append(s)
        return result

    def call_multiplexed_custom_compute(self, x: list[DataNode], dense_data_source, n):
        """Corresponds to C++ run_core_mult_pack with lazy weight generation."""
        input_ct_shape = [int(self.input_shape[0] * self.skip[0]), int(self.input_shape[1] * self.skip[1])]
        x_size = len(x)
        N_half = int(n / 2)
        n_num_pre_ct = int(np.ceil(N_half / (input_ct_shape[0] * input_ct_shape[1])))

        valid_skip_0 = self.skip[0] // self.invalid_fill[0]
        valid_skip_1 = self.skip[1] // self.invalid_fill[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel = self.n_in_channel // (self.input_shape[0] * self.input_shape[1])
        n_block_input = int(np.ceil(n_channel / (n_channel_per_block * n_num_pre_ct))) * n_num_pre_ct
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        block_size = input_ct_shape[0] * input_ct_shape[1]

        # Replicate input data across CT blocks inside the dense layer
        n_rot_factor = n_num_pre_ct // n_block_input if n_block_input > 0 and n_block_input < n_num_pre_ct else 1
        n_rep_iters = int(np.floor(np.log2(n_rot_factor))) if n_rot_factor > 1 else 0

        x_rep = list(x)
        for x_id in range(x_size):
            for r in range(n_rep_iters):
                x_rep[x_id] = add(x_rep[x_id], rotate_cols(x_rep[x_id], -(2**r) * n_block_input * block_size)[0])

        # rotated_cts[x_id][rot] = x_rep[x_id] rotated by rot * block_size
        n_rotations_per_ct = min(n_block_input, n_num_pre_ct)
        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x_rep[x_id], n_rotations_per_ct - 1, block_size))

        result = []

        for packed_out_feature_idx in range(n_packed_out_feature_for_mult_pack):
            x_ct_list = []
            w_pt_list = []
            for in_feature_idx in range(n_block_input):
                group = in_feature_idx // n_num_pre_ct
                offset = in_feature_idx % n_num_pre_ct
                x_ct = rotated_cts[group][offset]
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
            s = rescale(partial_sum)
            b_pt = CkksPlaintextRingtNode(f'encode_pt_{packed_out_feature_idx}')
            custom_compute(
                inputs=[dense_data_source],
                output=b_pt,
                type='encode_pt',
                attributes={'op_class': op_class, 'type': 'bias_pt', 'i': packed_out_feature_idx},
            )
            s = add(s, b_pt)
            n_fold = block_size
            while n_fold > 1:
                rotated = rotate_cols(s, n_fold // 2)
                s = add(s, rotated[0])
                n_fold //= 2
            result.append(s)
        return result
