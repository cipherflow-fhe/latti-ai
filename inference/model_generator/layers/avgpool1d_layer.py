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


class Avgpool1DLayer:
    def __init__(self, stride, shape, channel=1, skip=1):
        self.stride = stride
        self.shape = shape
        self.skip = skip
        self.channel = channel

        if shape & (shape - 1) != 0:
            raise ValueError(f'shape must be a power of 2, got: {shape}')
        if stride & (stride - 1) != 0:
            raise ValueError(f'stride must be a power of 2, got: {stride}')
        if skip & (skip - 1) != 0:
            raise ValueError(f'skip must be a power of 2, got: {skip}')

    def get_fhe_op_count_adaptive(self, n_ct: int, n: int) -> dict[str, int]:
        """Count FHE primitive operations in run_adaptive_avgpool() for n_ct input ciphertexts.

        rotate_cols(x, step) internally decomposes `step` via NAF into ±2^k sub-steps,
        each costing one RotateColUnit primitive. All steps here are powers of 2
        (stride and skip are both required to be powers of 2), so each rotate_cols
        call costs exactly 1 primitive rotate.

        Per ct:
          - log2(stride) rotate_cols calls × 1 primitive each = log2(stride) rotates + adds
            (stride accumulation, steps: 2^i * skip, all powers of 2)
          - floor(log2(n_rot)) rotate_cols calls × 1 primitive each (slot fill,
            steps: 2^r * channel * shape, all powers of 2)
        where n_rot = floor(n / 2 / (channel * shape)).
        """
        import math

        log2_stride = int(math.ceil(math.log2(self.stride))) if self.stride > 1 else 0
        n_rot = int(np.floor(n / 2 / (self.channel * self.shape)))
        fill_steps = int(np.floor(np.log2(n_rot))) if n_rot > 1 else 0
        rot_add_per_ct = log2_stride + fill_steps
        return {
            'rotate': n_ct * rot_add_per_ct,
            'mult_plain': 0,
            'mult': 0,
            'add': n_ct * rot_add_per_ct,
            'rescale': 0,
        }

    def run_adaptive_avgpool(self, x: list[DataNode], n: int):
        x_size = len(x)

        n_rot = int(np.floor(n / 2 / (self.channel * self.shape)))

        log2_stride = int(np.ceil(np.log2(self.stride)))

        result = []
        for idx in range(0, x_size):
            res = x[idx]
            for i in range(log2_stride - 1, 0 - 1, -1):
                ct_tmp = rotate_cols(res, (2**i) * self.skip)
                res = add(res, ct_tmp[0])

            for r in range(0, int(np.floor(np.log2(n_rot))) if n_rot > 1 else 0):
                res = add(res, rotate_cols(res, (2**r) * self.channel * self.shape)[0])
            result.append(res)
        return result

    def get_fhe_op_count_multiplexed(self, x_size: int, n_channel: int, n_channel_per_ct: int) -> dict[str, int]:
        """Count FHE primitive operations in call_multiplexed_avgpool().

        'rotate' is the total primitive RotateColUnit count.
        Stage 1 steps are powers of 2: naf_weight = 1 each.
        Stage 2: hoisted rotation — unique non-zero steps per ct are simulated
        exactly as in call_multiplexed_avgpool(); primitive rotate count is
        sum(naf_weight(s) for s in unique_non_zero_steps) per ct.
        """
        import math
        from inference.model_generator.layers.fhe_op_utils import naf_weight

        stride = self.stride
        shape = self.shape
        skip = self.skip
        log2_stride = int(math.ceil(math.log2(stride))) if stride > 1 else 0
        out_channels_per_ct = n_channel_per_ct * stride

        total_rotate = 0
        total_mult_plain = 0
        total_rescale = 0
        total_add = 0

        for idx in range(x_size):
            # Stage 1: steps are powers of 2, naf_weight = 1
            total_rotate += log2_stride
            total_add += log2_stride

            # Stage 2: simulate step computation, sum naf_weight of unique non-zero steps
            n_valid = min(n_channel_per_ct, n_channel - idx * n_channel_per_ct)
            steps = []
            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i
                rp = channel_id % out_channels_per_ct
                r_num0 = (rp // (skip * stride)) * skip * shape
                r_num1 = rp % (skip * stride)
                lp = channel_id % n_channel_per_ct
                l_num0 = (lp // skip) * skip * shape
                l_num1 = lp % skip
                steps.append(-r_num0 - r_num1 + l_num0 + l_num1)

            unique_non_zero = set(s for s in steps if s != 0)
            total_rotate += sum(naf_weight(s) for s in unique_non_zero)
            total_mult_plain += n_valid
            total_rescale += n_valid

        # Stage 3
        n_out_cts = math.ceil(n_channel / out_channels_per_ct)
        total_add += n_channel - n_out_cts

        return {
            'rotate': total_rotate,
            'mult_plain': total_mult_plain,
            'mult': 0,
            'add': total_add,
            'rescale': total_rescale,
        }

    def call_multiplexed_avgpool(
        self, x: list[CkksCiphertextNode], select_tensor_pt, n_channel: int, n_channel_per_ct: int
    ):
        import math

        x_size = len(x)
        stride = self.stride
        shape = self.shape
        skip = self.skip

        log2_stride = int(math.ceil(math.log2(stride))) if stride > 1 else 0
        out_channels_per_ct = n_channel_per_ct * stride

        result_tmp = []

        for idx in range(x_size):
            res_ct = x[idx]
            for i in range(log2_stride - 1, -1, -1):
                step = int(pow(2, i) * skip)
                ct_tmp = rotate_cols(res_ct, [step])[0]
                res_ct = add(res_ct, ct_tmp)

            n_valid = min(n_channel_per_ct, n_channel - idx * n_channel_per_ct)
            steps = []
            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i
                rp = channel_id % out_channels_per_ct
                r_num0 = (rp // (skip * stride)) * skip * shape
                r_num1 = rp % (skip * stride)

                lp = channel_id % n_channel_per_ct
                l_num0 = (lp // skip) * skip * shape
                l_num1 = lp % skip

                r_num = -r_num0 - r_num1 + l_num0 + l_num1
                steps.append(r_num)

            unique_steps = list(set(steps))
            non_zero_steps = [s for s in unique_steps if s != 0]
            if non_zero_steps:
                rotated_list = rotate_cols(res_ct, non_zero_steps)
                s_rots = {step: rotated_list[i] for i, step in enumerate(non_zero_steps)}
                s_rots[0] = res_ct
            else:
                s_rots = {0: res_ct}

            for i in range(n_valid):
                channel_id = idx * n_channel_per_ct + i

                out_channel_pos = channel_id % out_channels_per_ct
                select_pt = select_tensor_pt[out_channel_pos]

                x_rot = s_rots[steps[i]]
                c_m_s = mult(x_rot, select_pt)
                c_m_s_rescaled = rescale(c_m_s)
                result_tmp.append(c_m_s_rescaled)

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
