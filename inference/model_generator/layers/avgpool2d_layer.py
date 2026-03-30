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

    def run_adaptive_avgpool(self, x: list[DataNode], n: int):
        # n: number of valid slots in a ciphertext
        x_size = len(x)

        # If ciphertext slots are not full, need to rotate to fill them
        n_rot = int(np.floor(n / 2 / (self.channel * self.shape[0] * self.shape[1])))

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

            for r in range(0, int(np.floor(np.log2(n_rot))) if n_rot > 1 else 0):
                res = add(res, rotate_cols(res, (2**r) * self.channel * self.shape[0] * self.shape[1])[0])
            result.append(res)
        return result

    def call_interleaved_avgpool(self, x: list, block_expansion):
        """
        Interleaved (split) avgpool computation graph.

        Corresponds to C++ run_split_avgpool() (avgpool2d_layer.cpp:224-245).

        Adds stride[0]*stride[1] ciphertexts together per output position.
        No level consumption (only adds, no mult/rescale).
        """
        x_size = len(x)
        out_size = x_size // (self.stride[0] * self.stride[1])
        res = [None] * out_size

        for channel_idx in range(self.channel):
            base_idx = channel_idx * (block_expansion[0] // self.stride[0]) * (block_expansion[1] // self.stride[1])
            for row_idx in range(block_expansion[0]):
                for col_idx in range(block_expansion[1]):
                    ct_idx = (
                        channel_idx * block_expansion[0] * block_expansion[1] + row_idx * block_expansion[1] + col_idx
                    )
                    out_idx = (
                        base_idx
                        + (row_idx // self.stride[0]) * (block_expansion[1] // self.stride[1])
                        + col_idx // self.stride[1]
                    )
                    if row_idx % self.stride[0] == 0 and col_idx % self.stride[1] == 0:
                        res[out_idx] = x[ct_idx]
                    else:
                        res[out_idx] = add(res[out_idx], x[ct_idx])
        return res

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
        import math

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
