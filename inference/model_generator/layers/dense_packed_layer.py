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

import math
import numpy as np

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

    @staticmethod
    def populate_rotations_1_side(x: CkksCiphertextNode, n_rotation: int, unit: int) -> list[DataNode]:
        result: list[DataNode] = [x]
        steps = []
        for i in range(1, n_rotation + 1):
            steps.append(i * unit)
        result += rotate_cols(x, steps)
        return result

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
        n_block_input = int(np.ceil(n_channel / (n_num_pre_ct * n_channel_per_block))) * n_num_pre_ct
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        # Each input ct contributes n_num_pre_ct rotations (one per block slot within the ct).
        # rotated_cts[x_id][rot] = x[x_id] rotated by rot * block_size slots.
        block_size = input_ct_shape[0] * input_ct_shape[1]
        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x[x_id], n_num_pre_ct - 1, block_size))

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
        n_block_input = int(np.ceil(n_channel / (n_num_pre_ct * n_channel_per_block))) * n_num_pre_ct
        n_packed_out_feature_for_mult_pack = int(np.ceil(self.n_out_channel / n_num_pre_ct))

        # Each input ct contributes n_num_pre_ct rotations (one per block slot within the ct).
        block_size = input_ct_shape[0] * input_ct_shape[1]
        rotated_cts = []
        for x_id in range(x_size):
            rotated_cts.append(self.populate_rotations_1_side(x[x_id], n_num_pre_ct - 1, block_size))

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
