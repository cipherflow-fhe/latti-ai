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
import numpy as np


class Avgpool2DLayer:
    def __init__(self, stride, shape, channel=1, skip=[1, 1]):
        self.stride = stride
        self.shape = shape
        self.skip = skip
        self.channel = channel

        if shape[0] & (shape[0] - 1) != 0 or shape[1] & (shape[1] - 1) != 0:
            raise ValueError(f'shape must be powers of 2, got: [{shape[0]}, {shape[1]}]')
        if stride[0] & (stride[0] - 1) != 0 or stride[1] & (stride[1] - 1) != 0:
            raise ValueError(f'stride must be powers of 2, got: [{stride[0]}, {stride[1]}]')
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f'skip must be powers of 2, got: [{skip[0]}, {skip[1]}]')

    def get_fhe_op_count(self, n_ct: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call() for n_ct input ciphertexts, grouped by level.

        call() per ct:
          - (stride[0]-1) rotations + (stride[0]-1) adds  (horizontal accumulation)
          - log2(stride[0]) rotations + log2(stride[0]) adds  (binary fold)
        Note: call() only folds along one dimension (stride[0]), matching the code.
        No rescale, so all ops are at a single level key.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        stride = self.stride[0]
        rot_add_per_ct = (stride - 1) + int(math.log2(stride))
        ops[lv]['rotate'] += n_ct * rot_add_per_ct
        ops[lv]['add'] += n_ct * rot_add_per_ct

        return dict(ops)

    def call(self, x: list[DataNode]):
        res: list[DataNode] = list()
        for i in range(len(x)):
            rr = x[i]
            for j in range(1, self.stride[0]):
                ri = rotate_cols(x[i], [j * self.shape[0]])[0]
                rr = add(rr, ri)
            step = self.stride[0]
            while step > 1:
                step = int(step)
                ri = rotate_cols(rr, [step // 2])[0]
                rr = add(rr, ri)
                step /= 2
            res.append(rr)
        return res

    def get_fhe_op_count_adaptive(self, n_ct: int, n: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in run_adaptive_avgpool() for n_ct input ciphertexts, grouped by level.

        rotate_cols(x, step) internally decomposes `step` via NAF into ±2^k sub-steps,
        each costing one RotateColUnit primitive. All steps here are powers of 2
        (stride and skip are both required to be powers of 2), so each rotate_cols
        call costs exactly 1 primitive rotate.

        Per ct:
          - log2(stride[0]) rotate_cols calls × 1 primitive each (height accumulation,
            steps: 2^i * shape[0] * skip[0] * skip[1], all powers of 2)
          - log2(stride[1]) rotate_cols calls × 1 primitive each (width accumulation,
            steps: 2^j * skip[1], all powers of 2)
          - floor(log2(n_rot)) rotate_cols calls × 1 primitive each (slot fill,
            steps: 2^r * channel * shape[0] * shape[1], all powers of 2)
        where n_rot = floor(n / 2 / (channel * shape[0] * shape[1])).
        No rescale, so all ops are at a single level key.
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        log2_stride_0 = int(math.ceil(math.log2(self.stride[0]))) if self.stride[0] > 1 else 0
        log2_stride_1 = int(math.ceil(math.log2(self.stride[1]))) if self.stride[1] > 1 else 0
        n_rot = int(np.floor(n / 2 / (self.channel * self.shape[0] * self.shape[1])))
        fill_steps = int(np.floor(np.log2(n_rot))) if n_rot > 1 else 0
        rot_add_per_ct = log2_stride_0 + log2_stride_1 + fill_steps
        ops[lv]['rotate'] += n_ct * rot_add_per_ct
        ops[lv]['add'] += n_ct * rot_add_per_ct

        return dict(ops)

    def run_adaptive_avgpool(self, x: list[DataNode], n: int):
        # n: number of valid slots in a ciphertext
        x_size = len(x)

        log2_stride_0 = int(np.ceil(np.log2(self.stride[0])))
        log2_stride_1 = int(np.ceil(np.log2(self.stride[1])))

        result = []
        for idx in range(0, x_size):
            res = x[idx]
            for i in range(log2_stride_0 - 1, 0 - 1, -1):
                ct_tmp = rotate_cols(res, (2**i) * self.shape[0] * self.skip[0] * self.skip[1])
                res = add(res, ct_tmp[0])

            for j in range(log2_stride_1 - 1, 0 - 1, -1):
                ct_tmp = rotate_cols(res, (2**j) * self.skip[1])
                res = add(res, ct_tmp[0])
            result.append(res)
        return result

    def get_fhe_op_count_interleaved(self, x_size: int, N: int, level: int) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_interleaved_avgpool(), grouped by level.

        'rotate' is the total primitive RotateColUnit count computed via NAF
        decomposition. Stage 2 step for channel index k is k * output_h * output_w;
        since output_h * output_w is a power of 2 (shape and stride are powers of 2),
        naf_weight(k * 2^m) == naf_weight(k), so the cost reduces to summing
        naf_weight(k) for k in 1..n_channel_per_ct_out-1, multiplied by n_packed_out.

        Stage 1: (stride[0]*stride[1] - 1) adds per output ct (no rotations).
        Stage 2 (channel repacking, only when n_channel_per_ct_out > 1):
          - sum(naf_weight(k) for k in 1..n_channel_per_ct_out-1) * n_packed_out rotates
          - (n_channel_per_ct_out - 1) * n_packed_out adds
        No rescale, so all ops are at a single level key.
        """
        out_size = x_size // (self.stride[0] * self.stride[1])
        adds_stage1 = out_size * (self.stride[0] * self.stride[1] - 1)

        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        output_h = self.shape[0] // self.stride[0]
        output_w = self.shape[1] // self.stride[1]
        n_channel_per_ct_out = 1
        if 2 * output_h * output_w < N:
            n_channel_per_ct_out = N // (2 * output_h * output_w)

        ops[lv]['add'] += adds_stage1

        if n_channel_per_ct_out > 1:
            n_packed_out = math.ceil(out_size / n_channel_per_ct_out)
            # naf_weight(k * output_h * output_w) == naf_weight(k) since output_h*output_w is 2^m
            rots_per_pack = sum(naf_weight(k) for k in range(1, n_channel_per_ct_out))
            rots_stage2 = n_packed_out * rots_per_pack
            adds_stage2 = n_packed_out * (n_channel_per_ct_out - 1)
            ops[lv]['rotate'] += rots_stage2
            ops[lv]['add'] += adds_stage2

        return dict(ops)

    def call_interleaved_avgpool(self, x: list, block_expansion, N: int, repack_mask_pt=None, block_shape=None):
        """
        Interleaved (split) avgpool computation graph.

        Corresponds to C++ run_split_avgpool() (avgpool2d_layer.cpp).

        When output_shape >= block_shape (no repack):
          Single-stage cross-block sum. Level cost = 0.

        When output_shape < block_shape (repack needed):
          Stage 1: Cross-block sum with first_stage_stride (= block_expansion).
          Stage 2: In-block adaptive avgpool with second_stage_stride.
          Stage 3: Repack (mask mult + rotate + accumulate + rescale). Level cost = 1.
        """
        need_repack = repack_mask_pt is not None

        if need_repack:
            first_stage_stride = list(block_expansion)
            second_stage_stride = [self.stride[i] // first_stage_stride[i] for i in range(2)]
        else:
            first_stage_stride = list(self.stride)
            second_stage_stride = [1, 1]

        # Stage 1: Interleaved cross-block sum
        x_size = len(x)
        out_size = x_size // (first_stage_stride[0] * first_stage_stride[1])
        res = [None] * out_size

        for channel_idx in range(self.channel):
            base_idx = (
                channel_idx
                * (block_expansion[0] // first_stage_stride[0])
                * (block_expansion[1] // first_stage_stride[1])
            )
            for row_idx in range(block_expansion[0]):
                for col_idx in range(block_expansion[1]):
                    ct_idx = (
                        channel_idx * block_expansion[0] * block_expansion[1] + row_idx * block_expansion[1] + col_idx
                    )
                    out_idx = (
                        base_idx
                        + (row_idx // first_stage_stride[0]) * (block_expansion[1] // first_stage_stride[1])
                        + col_idx // first_stage_stride[1]
                    )
                    if row_idx % first_stage_stride[0] == 0 and col_idx % first_stage_stride[1] == 0:
                        res[out_idx] = x[ct_idx]
                    else:
                        res[out_idx] = add(res[out_idx], x[ct_idx])

        if need_repack:
            # Stage 2: Adaptive avgpool within block (rotation sum with second_stage_stride)
            log2_ss0 = int(np.ceil(np.log2(second_stage_stride[0]))) if second_stage_stride[0] > 1 else 0
            log2_ss1 = int(np.ceil(np.log2(second_stage_stride[1]))) if second_stage_stride[1] > 1 else 0
            for idx in range(len(res)):
                r = res[idx]
                for i in range(log2_ss0 - 1, -1, -1):
                    r = add(r, rotate_cols(r, [int(2**i * block_shape[1])])[0])
                for j in range(log2_ss1 - 1, -1, -1):
                    r = add(r, rotate_cols(r, [int(2**j)])[0])
                res[idx] = r

            # Stage 3: Repack (mask + rotate + accumulate + rescale)
            out_skip0 = second_stage_stride[0]
            out_skip1 = second_stage_stride[1]
            n_channel_per_block = out_skip0 * out_skip1
            n_block_per_ct = (N // 2) // (block_shape[0] * block_shape[1])
            n_channel_per_ct_out = n_channel_per_block * n_block_per_ct
            n_out_ct = math.ceil(self.channel / n_channel_per_ct_out)

            # Step 3a: mask all channels
            for c in range(len(res)):
                res[c] = mult(res[c], repack_mask_pt)

            # Step 3b: rotate + accumulate
            repack_res = [None] * n_out_ct
            for out_ct_idx in range(n_out_ct):
                packed = None
                for ch_in_ct in range(n_channel_per_ct_out):
                    c = out_ct_idx * n_channel_per_ct_out + ch_in_ct
                    if c >= self.channel:
                        break
                    block_idx = ch_in_ct // n_channel_per_block
                    ch_in_block = ch_in_ct % n_channel_per_block
                    cx = ch_in_block // out_skip1
                    cy = ch_in_block % out_skip1
                    rot_step = -(cx * block_shape[1] + cy + block_idx * block_shape[0] * block_shape[1])
                    if rot_step == 0:
                        rotated = res[c]
                    else:
                        rotated = rotate_cols(res[c], [rot_step])[0]
                    packed = rotated if packed is None else add(packed, rotated)
                repack_res[out_ct_idx] = rescale(packed)
            return repack_res

        # No repack: existing packing logic (output >= block)
        output_h = self.shape[0] // self.stride[0]
        output_w = self.shape[1] // self.stride[1]

        n_channel_per_ct_out = 1
        if 2 * output_h * output_w < N:
            n_channel_per_ct_out = N // (2 * output_h * output_w)

        if n_channel_per_ct_out == 1:
            return res
        else:
            packed_res = [0 for i in range((out_size + n_channel_per_ct_out - 1) // n_channel_per_ct_out)]
            for out_ct_idx in range(out_size):
                pack_out_ct_idx = int(out_ct_idx // n_channel_per_ct_out)
                channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out
                if channel_idx_in_ct == 0:
                    packed_res[pack_out_ct_idx] = res[out_ct_idx]
                else:
                    step = int(-1 * channel_idx_in_ct * output_h * output_w)
                    if step == 0:
                        s_rot = res[out_ct_idx]
                    else:
                        s_rot = rotate_cols(res[out_ct_idx], [step])[0]
                    packed_res[pack_out_ct_idx] = add(packed_res[pack_out_ct_idx], s_rot)
            return packed_res

    def make_pt_nodes_multiplexed_avgpool(self, layer_id, n_channel, n):
        """Return (select_tensor_pt, n_channel_per_ct) for call_multiplexed_avgpool().

        select_tensor_pt[i]: i in min(n_channel, n_channel_per_ct * stride[0] * stride[1])
        Also returns n_channel_per_ct so caller can pass it to call_multiplexed_avgpool().
        """
        n_channel_per_ct = int(math.ceil(n / 2 / (self.shape[0] * self.shape[1])))
        out_channels_per_ct = n_channel_per_ct * self.stride[0] * self.stride[1]
        n_select_pt = min(n_channel, out_channels_per_ct)
        select_tensor_pt = [CkksPlaintextRingtNode(f'select_pt_{layer_id}_{i}') for i in range(n_select_pt)]
        return select_tensor_pt, n_channel_per_ct

    def call_custom_compute_multiplexed_avgpool(self, x: list, avg_data_source, n_channel: int, n: int) -> list:
        """Lazy path for multiplexed avgpool: generate select_tensor_pt on-demand."""
        n_channel_per_ct = int(math.ceil(n / 2 / (self.shape[0] * self.shape[1])))
        out_channels_per_ct = n_channel_per_ct * self.stride[0] * self.stride[1]
        n_select_pt = min(n_channel, out_channels_per_ct)

        select_tensor_pt = []
        for i in range(n_select_pt):
            w_pt = CkksPlaintextRingtNode(f'encode_pt_select_{i}')
            custom_compute(
                inputs=[avg_data_source],
                output=w_pt,
                type='encode_pt',
                attributes={'op_class': 'Avgpool2DLayer', 'type': 'select_pt', 'i': i},
            )
            select_tensor_pt.append(w_pt)

        return self.call_multiplexed_avgpool(x, select_tensor_pt, n_channel, n_channel_per_ct)

    def get_fhe_op_count_multiplexed(
        self, x_size: int, n_channel: int, n_channel_per_ct: int, level: int
    ) -> dict[int, dict[str, int]]:
        """Count FHE primitive operations in call_multiplexed_avgpool(), grouped by level.

        'rotate' is the total primitive RotateColUnit count.
        Stage 1 steps are powers of 2: naf_weight = 1 each.
        Stage 2: hoisted rotation — for each input ct the unique non-zero steps are
        collected, then rotate_cols is called once per unique step. The primitive
        rotate count is sum(naf_weight(s) for s in unique_non_zero_steps) per ct.
        Steps are simulated exactly as in call_multiplexed_avgpool().

        Level structure:
          level:   Stage 1 rotate + Stage 2 rotate + mult_plain + rescale
          level-1: Stage 3 add (channel repacking)
        """
        ops = defaultdict(lambda: {'rotate': 0, 'mult_plain': 0, 'mult': 0, 'add': 0, 'rescale': 0})
        lv = level

        stride = self.stride
        shape = self.shape
        skip = self.skip
        log2_stride_0 = int(math.ceil(math.log2(stride[0]))) if stride[0] > 1 else 0
        log2_stride_1 = int(math.ceil(math.log2(stride[1]))) if stride[1] > 1 else 0
        out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]

        for idx in range(x_size):
            # Stage 1: all steps are powers of 2, naf_weight = 1
            ops[lv]['rotate'] += log2_stride_0 + log2_stride_1
            ops[lv]['add'] += log2_stride_0 + log2_stride_1

            # Stage 2: simulate step computation, sum naf_weight of unique non-zero steps
            n_valid = min(n_channel_per_ct, n_channel - idx * n_channel_per_ct)
            steps = []
            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i
                rp = channel_id % out_channels_per_ct
                r_num0 = (rp // (skip[0] * skip[1] * stride[0] * stride[1])) * skip[0] * skip[1] * shape[0] * shape[1]
                r_num1 = (
                    ((rp % (skip[0] * skip[1] * stride[0] * stride[1])) // (stride[1] * skip[1])) * shape[1] * skip[1]
                )
                r_num2 = rp % (skip[1] * stride[1])
                lp = channel_id % n_channel_per_ct
                l_num0 = (lp // (skip[0] * skip[1])) * skip[0] * skip[1] * shape[0] * shape[1]
                l_num1 = ((lp % (skip[0] * skip[1])) // skip[1]) * shape[1] * skip[1]
                l_num2 = lp % skip[1]
                steps.append(-r_num0 - r_num1 - r_num2 + l_num0 + l_num1 + l_num2)

            unique_non_zero = set(s for s in steps if s != 0)
            ops[lv]['rotate'] += sum(naf_weight(s) for s in unique_non_zero)
            ops[lv]['mult_plain'] += n_valid
            ops[lv]['rescale'] += n_valid

        lv -= 1

        # Stage 3: channel repacking adds (at level-1)
        n_out_cts = math.ceil(n_channel / out_channels_per_ct)
        ops[lv]['add'] += n_channel - n_out_cts

        return dict(ops)

    def call_multiplexed_avgpool(
        self, x: list[CkksCiphertextNode], select_tensor_pt, n_channel: int, n_channel_per_ct: int
    ):
        """
        Multiplexed avgpool computation graph.

        Corresponds to C++ run_multiplexed_avgpool() (avgpool2d_layer.cpp:144-219).

        Three stages:
        1. Rotation accumulation along height and width (log2(stride) steps each)
        2. Hoisted rotation + select_tensor mask multiplication + rescale
        3. Channel repacking into output ciphertexts
        """
        x_size = len(x)
        stride = self.stride
        shape = self.shape
        skip = self.skip

        log2_stride_0 = int(math.ceil(math.log2(stride[0]))) if stride[0] > 1 else 0
        log2_stride_1 = int(math.ceil(math.log2(stride[1]))) if stride[1] > 1 else 0
        out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]

        result_tmp = []

        # Stage 1 + 2: For each input CT
        for idx in range(x_size):
            # Stage 1: Rotation accumulation (C++ lines 155-165)
            res_ct = x[idx]
            for i in range(log2_stride_0 - 1, -1, -1):
                step = int(pow(2, i) * shape[1] * skip[0] * skip[1])
                ct_tmp = rotate_cols(res_ct, [step])[0]
                res_ct = add(res_ct, ct_tmp)
            for j in range(log2_stride_1 - 1, -1, -1):
                step = int(pow(2, j) * skip[1])
                ct_tmp = rotate_cols(res_ct, [step])[0]
                res_ct = add(res_ct, ct_tmp)

            # Stage 2: Compute rotation steps only for valid channels (C++ lines 166-182)
            n_valid = min(n_channel_per_ct, n_channel - idx * n_channel_per_ct)
            steps = []
            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i
                rp = channel_id % out_channels_per_ct
                r_num0 = (rp // (skip[0] * skip[1] * stride[0] * stride[1])) * skip[0] * skip[1] * shape[0] * shape[1]
                r_num1 = (
                    ((rp % (skip[0] * skip[1] * stride[0] * stride[1])) // (stride[1] * skip[1])) * shape[1] * skip[1]
                )
                r_num2 = rp % (skip[1] * stride[1])

                lp = channel_id % n_channel_per_ct
                l_num0 = (lp // (skip[0] * skip[1])) * skip[0] * skip[1] * shape[0] * shape[1]
                l_num1 = ((lp % (skip[0] * skip[1])) // skip[1]) * shape[1] * skip[1]
                l_num2 = lp % skip[1]

                r_num = -r_num0 - r_num1 - r_num2 + l_num0 + l_num1 + l_num2
                steps.append(r_num)

            # Hoisted rotation (C++ line 183)
            unique_steps = list(set(steps))
            non_zero_steps = [s for s in unique_steps if s != 0]
            if non_zero_steps:
                rotated_list = rotate_cols(res_ct, non_zero_steps)
                s_rots = {step: rotated_list[i] for i, step in enumerate(non_zero_steps)}
                s_rots[0] = res_ct
            else:
                s_rots = {0: res_ct}

            # Mask multiplication + rescale (C++ lines 184-193)
            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i

                out_channel_pos = channel_id % out_channels_per_ct
                select_pt = select_tensor_pt[out_channel_pos]

                x_rot = s_rots[steps[i]]
                c_m_s = mult(x_rot, select_pt)
                c_m_s_rescaled = rescale(c_m_s)
                result_tmp.append(c_m_s_rescaled)

        # Stage 3: Channel repacking (C++ lines 195-209)
        res = []
        sp = None
        for i in range(n_channel):
            p = i % out_channels_per_ct
            c_m_s = result_tmp[i]
            if p == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)
            if (i + 1) % out_channels_per_ct == 0 or i == n_channel - 1:
                res.append(sp)

        return res
