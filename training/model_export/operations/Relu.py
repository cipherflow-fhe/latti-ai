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
from . import ComputeNode, FeatureNode, format_id
from onnx import NodeProto

log = logging.getLogger(__name__)


class ReluComputeNode(ComputeNode):
    """Compute node for Relu activation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        is_mpc=False,
    ):
        super(ReluComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        if feature_input[0].dim == 2:
            self.layer_type = 'relu2d'
        else:
            self.layer_type = 'relu0d'
        feature_output[0].scale = 1
        if is_mpc:
            feature_output[0].skip = [1, 1]
        else:
            feature_output[0].skip = feature_input[0].skip

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'ReluComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'relu2d'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]

        return ReluComputeNode(layer_id, layer_type, feature_input, feature_output)

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating relu code')
        init_str, forward_str = [], []
        params_str = {}
        forward_str.append(f'{str(self.feature_output[0])} = F.relu({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}
