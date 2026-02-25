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
import os

# Add mega_ag_generator to path for importing frontend module
script_dir = os.path.dirname(os.path.abspath(__file__))
mega_ag_generator_dir = os.path.join(script_dir, '../../lattisense')
sys.path.insert(0, mega_ag_generator_dir)

from frontend.custom_task import *


class UpsampleNearestLayer:
    """
    UpsampleNearest (nearest neighbor upsampling) layer computation graph generator - Lazy mode

    Corresponds to C++ code: fhe_layers/upsample_nearest_layer.cpp

    Function: Implements nearest neighbor upsampling of packed ciphertexts using rotation and masking

    Implementation notes:
        - Python implementation uses custom_compute to generate select_tensor_pt on-demand
        - Corresponds to C++ lazy mode (prepare_weight_lazy + generate_select_tensor_pt_for_index)
        - C++ also supports normal mode (prepare_weight pre-generates all select_tensor_pt)
    """

    def __init__(
        self, shape: list[int], skip: list[int], upsample_factor: list[int], n_channel_per_ct: int, level: int
    ):
        """
        Initialize UpsampleNearest layer

        Args:
            shape: Input feature map shape [H, W]
            skip: Stride [skip_h, skip_w]
            upsample_factor: Upsampling factor [factor_h, factor_w]
            n_channel_per_ct: Number of channels per ciphertext
            level: Ciphertext level
        """
        self.shape = shape
        self.skip = skip
        self.upsample_factor = upsample_factor
        self.n_channel_per_ct = n_channel_per_ct
        self.level = level

        if shape[0] & (shape[0] - 1) != 0 or shape[1] & (shape[1] - 1) != 0:
            raise ValueError(f"shape must be powers of 2, got: [{shape[0]}, {shape[1]}]")
        if skip[0] & (skip[0] - 1) != 0 or skip[1] & (skip[1] - 1) != 0:
            raise ValueError(f"skip must be powers of 2, got: [{skip[0]}, {skip[1]}]")

        # Calculate the number of blocks per ciphertext
        self.n_block_per_ct = (n_channel_per_ct + skip[0] * skip[1] - 1) // (skip[0] * skip[1])

    def call_custom_compute(
        self, x: list[CkksCiphertextNode], data_source: 'CustomDataNode', n_channel: int
    ) -> list[CkksCiphertextNode]:
        """
        Generate computation graph for upsample_nearest layer using custom_compute (Lazy mode)

        Args:
            x: Input ciphertext node list
            data_source: Custom data source node (for on-demand generation of select_tensor_pt)
            n_channel: Total number of channels

        Returns:
            Output ciphertext node list

        Implementation logic corresponds to C++ run() method - Lazy mode branch (upsample_nearest_layer.cpp:66-172)
        - Uses custom_compute to generate select_tensor_pt on-demand (corresponds to C++ lines 108-110)
        - Three-stage processing: rotation+masking, repacking, nearest neighbor replication
        """
        x_size = len(x)
        if x_size == 0:
            raise ValueError(f'UpsampleNearestLayer: input x is empty')
        if n_channel == 0:
            raise ValueError(f'UpsampleNearestLayer: n_channel is 0')

        result_tmp = []

        factor_h = self.upsample_factor[0]
        factor_w = self.upsample_factor[1]

        # Calculate output packed channel count
        # n_packed_out_channel = ceil(n_channel / (n_channel_per_ct / (factor_h * factor_w)))
        out_channels_per_ct = self.n_channel_per_ct // (factor_h * factor_w)
        n_packed_out_channel = (n_channel + out_channels_per_ct - 1) // out_channels_per_ct

        # Stage 1: Rotation, mask selection, rescale (C++ lines 81-122)
        for idx in range(x_size):
            # Pre-compute rotation steps needed for all channels (C++ lines 83-100)
            steps = []
            for i in range(self.n_channel_per_ct):
                channel_id = idx * self.n_channel_per_ct + i
                if channel_id >= n_channel:
                    steps.append(0)  # Use 0-step rotation for out-of-range channels
                    continue

                # Calculate rotation steps
                rp = channel_id % out_channels_per_ct
                r_num0 = (
                    (rp // (self.skip[0] * self.skip[1] // (factor_h * factor_w)))
                    * self.skip[0]
                    * self.skip[1]
                    * self.shape[0]
                    * self.shape[1]
                )
                r_num1 = (
                    ((rp % (self.skip[0] * self.skip[1] // (factor_h * factor_w))) // (self.skip[1] // factor_w))
                    * self.shape[1]
                    * self.skip[1]
                )
                r_num2 = rp % (self.skip[1] // factor_w)

                lp = channel_id % self.n_channel_per_ct
                l_num0 = (
                    (lp // (self.skip[0] * self.skip[1])) * self.skip[0] * self.skip[1] * self.shape[0] * self.shape[1]
                )
                l_num1 = ((lp % (self.skip[0] * self.skip[1])) // self.skip[1]) * self.shape[1] * self.skip[1]
                l_num2 = lp % self.skip[1]

                r_num = -r_num0 - r_num1 - r_num2 + l_num0 + l_num1 + l_num2
                steps.append(r_num)

            # Hoisted rotation: Execute all required rotations on the entire ciphertext at once (C++ line 101)
            # Find all unique rotation steps
            unique_steps = list(set(steps))
            if 0 in unique_steps and len(unique_steps) == 1:
                # If only 0-step rotation is needed, use original ciphertext directly
                s_rots = {0: x[idx]}
            else:
                # Use rotate_cols to generate all needed rotations at once
                # Filter out 0-step rotation (0-step rotation uses original ciphertext directly)
                non_zero_steps = [s for s in unique_steps if s != 0]
                if non_zero_steps:
                    rotated_list = rotate_cols(x[idx], non_zero_steps)
                    s_rots = {step: rotated_list[i] for i, step in enumerate(non_zero_steps)}
                    s_rots[0] = x[idx]  # 0-step rotation uses original ciphertext
                else:
                    s_rots = {0: x[idx]}

            # Process each channel in loop, using pre-computed rotation results (C++ lines 103-121)
            for i in range(self.n_channel_per_ct):
                channel_id = idx * self.n_channel_per_ct + i
                if channel_id >= n_channel:
                    continue

                # Use pre-computed rotation results directly
                x_rot = s_rots[steps[i]]

                # Generate select_tensor_pt on-demand and perform mask multiplication (C++ lines 108-111 - Lazy mode)
                out_channel_pos = channel_id % out_channels_per_ct
                select_pt = CkksPlaintextRingtNode(f'select_pt_{idx}_{i}_{out_channel_pos}')
                custom_compute(
                    inputs=[data_source],
                    output=select_pt,
                    type='encode_pt',
                    attributes={
                        'op_class': 'UpsampleNearest',
                        'type': 'select_tensor_pt',
                        'i': out_channel_pos,
                    },
                )

                # Mask multiplication and rescale (C++ lines 111-119)
                c_m_s = mult(x_rot, select_pt)
                c_m_s_rescaled = rescale(c_m_s)
                result_tmp.append(c_m_s_rescaled)

        # Stage 2: Repack channels (C++ lines 126-138)
        res = []
        sp = None
        for i in range(n_channel):
            p = i % out_channels_per_ct
            c_m_s = result_tmp[i]

            if p == 0:
                sp = c_m_s
            else:
                sp = add(sp, c_m_s)

            # When packing of an output ciphertext is complete, or reached the last channel
            if (i + 1) % out_channels_per_ct == 0 or i == n_channel - 1:
                res.append(sp)

        # Stage 3: Use rotation and addition to replicate data (implements nearest neighbor upsampling) (C++ lines 150-161)
        import math

        log2_upsample_0 = int(math.ceil(math.log2(factor_h))) if factor_h > 1 else 0
        log2_upsample_1 = int(math.ceil(math.log2(factor_w))) if factor_w > 1 else 0

        result = []
        for idx in range(len(res)):
            result_ct = res[idx]

            # Replicate in height direction (C++ lines 151-155)
            for i in range(log2_upsample_0):
                step = -int(pow(2, i) * self.shape[1] * self.skip[1] * self.skip[0] // factor_h)
                ct_tmp = rotate_cols(result_ct, [step])[0]
                result_ct = add(result_ct, ct_tmp)

            # Replicate in width direction (C++ lines 156-160)
            for j in range(log2_upsample_1):
                step = -int(pow(2, j) * self.skip[1] // factor_w)
                ct_tmp = rotate_cols(result_ct, [step])[0]
                result_ct = add(result_ct, ct_tmp)

            result.append(result_ct)

        if len(result) == 0:
            raise ValueError(
                f'UpsampleNearestLayer: generated empty result! '
                f'x_size={x_size}, n_channel={n_channel}, n_channel_per_ct={self.n_channel_per_ct}, '
                f'factor={self.upsample_factor}, out_channels_per_ct={out_channels_per_ct}'
            )

        return result
