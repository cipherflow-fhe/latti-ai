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


class ConcatLayer:
    """
    Concat (channel concatenation) layer computation graph generator

    Corresponds to C++ code: fhe_layers/concat_layer.cpp

    Function: Concatenates multiple feature maps along the channel dimension
    """

    def __init__(self):
        """
        Initialize Concat layer
        """
        pass

    def call(self, x1: list[CkksCiphertextNode], x2: list[CkksCiphertextNode]) -> list[CkksCiphertextNode]:
        """
        Concatenate two input feature maps (along channel dimension)

        Args:
            x1: First input ciphertext node list
            x2: Second input ciphertext node list

        Returns:
            Concatenated ciphertext node list

        Logic explanation:
            Corresponds to C++ run_multiple_inputs method
            - First adds all ciphertexts from x1, then all ciphertexts from x2
            - Order: result = x1 + x2 (consistent with parameter names and C++ inputs[0], inputs[1] order)
            - No homomorphic operations needed, just list merging
        """
        result = []
        for elem in x1:
            result.append(elem)
        for elem in x2:
            result.append(elem)
        return result

    def call_multiple_inputs(self, inputs: list[list[CkksCiphertextNode]]) -> list[CkksCiphertextNode]:
        """
        Concatenate multiple input feature maps (along channel dimension).
        Fast path: simply merge node lists when all inputs have n_channel divisible by n_channel_per_ct.

        Args:
            inputs: List of multiple input ciphertext node lists

        Returns:
            Concatenated ciphertext node list
        """
        if not inputs:
            raise ValueError('Empty input list in ConcatLayer.call_multiple_inputs')

        if len(inputs) == 1:
            return inputs[0]

        # Concatenate all inputs in order (fast path)
        result = []
        for input_nodes in inputs:
            for elem in input_nodes:
                result.append(elem)

        return result

    def call_multiple_inputs_uneven(
        self,
        inputs: list[list[CkksCiphertextNode]],
        input_n_channels: list[int],
        n_channel_per_ct: int,
        shape: list[int],
        skip: list[int],
        mask_pts: list[CkksPlaintextRingtNode],
    ) -> list[CkksCiphertextNode]:
        """
        Concatenate multiple input feature maps with per-channel mask+rotate+add
        when n_channel % n_channel_per_ct != 0.

        Corresponds to C++ run_multiple_inputs_uneven method.
        Similar to inverse_multiplexed_conv2d_layer repack logic.

        Args:
            inputs: List of input ciphertext node lists
            input_n_channels: Number of channels for each input
            n_channel_per_ct: Channels packed per ciphertext
            shape: Spatial shape [H, W]
            skip: Skip values [skip_h, skip_w]
            mask_pts: Pre-created mask plaintext nodes, one per global channel

        Returns:
            Concatenated ciphertext node list (output level = input level - 1)
        """
        n_channel_per_block = skip[0] * skip[1]
        block_size = skip[0] * shape[0] * skip[1] * shape[1]

        # Compute channel offsets and total channels
        channel_offsets = []
        total_channels = 0
        for n_ch in input_n_channels:
            channel_offsets.append(total_channels)
            total_channels += n_ch
        import math

        n_out_ct = math.ceil(total_channels / n_channel_per_ct)

        # Step 1: For each global channel, mask the source CT
        masked_cts = []
        for global_ch in range(total_channels):
            # Find which input and local channel
            input_idx = 0
            local_ch = global_ch
            for i, n_ch in enumerate(input_n_channels):
                if local_ch < n_ch:
                    input_idx = i
                    break
                local_ch -= n_ch

            # Source CT and position
            src_ct_idx = local_ch // n_channel_per_ct
            masked = mult(inputs[input_idx][src_ct_idx], mask_pts[global_ch])
            masked_cts.append(masked)

        # Step 2: Rotate and accumulate into output CTs
        result = []
        for out_ct_idx in range(n_out_ct):
            packed = None
            for ch_in_ct in range(n_channel_per_ct):
                global_ch = out_ct_idx * n_channel_per_ct + ch_in_ct
                if global_ch >= total_channels:
                    break

                # Source position
                input_idx = 0
                local_ch = global_ch
                for i, n_ch in enumerate(input_n_channels):
                    if local_ch < n_ch:
                        input_idx = i
                        break
                    local_ch -= n_ch

                src_channel_in_ct = local_ch % n_channel_per_ct
                src_block = src_channel_in_ct // n_channel_per_block
                src_offset = src_channel_in_ct % n_channel_per_block
                src_cx = src_offset // skip[1]
                src_cy = src_offset % skip[1]
                src_slot_base = src_block * block_size + src_cx * (shape[1] * skip[1]) + src_cy

                # Target position in output CT
                dst_channel_in_ct = ch_in_ct
                dst_block = dst_channel_in_ct // n_channel_per_block
                dst_offset = dst_channel_in_ct % n_channel_per_block
                dst_cx = dst_offset // skip[1]
                dst_cy = dst_offset % skip[1]
                dst_slot_base = dst_block * block_size + dst_cx * (shape[1] * skip[1]) + dst_cy

                rot_step = -(dst_slot_base - src_slot_base)

                if rot_step == 0:
                    rotated = masked_cts[global_ch]
                else:
                    rotated = rotate_cols(masked_cts[global_ch], [rot_step])[0]

                if packed is None:
                    packed = rotated
                else:
                    packed = add(packed, rotated)
            result.append(rescale(packed))

        return result
