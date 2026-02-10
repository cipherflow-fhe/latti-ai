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

import logging

# from typing import override
from typing_extensions import override
from . import ComputeNode, FeatureNode

log = logging.getLogger(__name__)


defaulf_scale = 0.01


class SquareComputeNode(ComputeNode):
    """Compute node for Square operation"""

    def __init__(
        self, layer_id: str, layer_type: str, feature_input: list[FeatureNode], feature_output: list[FeatureNode]
    ):
        super(SquareComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.layer_type = 'square2d'
        feature_output[0].scale = feature_input[0].scale
        feature_output[0].shape = feature_input[0].shape
        feature_output[0].skip = feature_input[0].skip
        feature_output[0].level = feature_input[0].level - 1

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating square code')
        init_str, forward_str = [], []
        params_str = {}
        params_str['num_features'] = self.feature_input[0].channel
        forward_str.append(f'{str(self.feature_output[0])} = self.Square({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}

    @staticmethod
    def gen_method():
        return """def Square(self, x):
        return x**2
        """
