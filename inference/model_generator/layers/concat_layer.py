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
        Concatenate multiple input feature maps (along channel dimension)

        Args:
            inputs: List of multiple input ciphertext node lists

        Returns:
            Concatenated ciphertext node list

        Logic explanation:
            Corresponds to C++ run_multiple_inputs(inputs) method (concat_layer.cpp:45-96)
            - Concatenates all input ciphertexts in order
            - result.data = inputs[0].data + inputs[1].data + ... + inputs[n-1].data
        """
        if not inputs:
            raise ValueError('Empty input list in ConcatLayer.call_multiple_inputs')

        if len(inputs) == 1:
            # Only one input, return directly
            return inputs[0]

        # Concatenate all inputs in order
        result = []
        for input_nodes in inputs:
            for elem in input_nodes:
                x = elem
                result.append(x)

        return result
