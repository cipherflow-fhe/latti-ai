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


class ReshapeComputeNode(ComputeNode):
    """Compute node for Reshape operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        reshape_size: list = [-1, 1],
    ):
        super(ReshapeComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        feature_output[0].channel = feature_input[0].channel
        feature_output[0].level = feature_input[0].level
        self.reshape_size = reshape_size

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes, constant_nodes) -> 'ReshapeComputeNode':
        # TODO: support reshape op with user specified output shape
        layer_id = format_id(x.name)
        layer_type = 'reshape'
        feature_input = [features_nodes[format_id(x.input[0])]]
        try:
            reshape_size = constant_nodes[format_id(x.input[1])]
        except Exception:
            reshape_size = [-1, feature_input[0].channel]
        feature_output = [features_nodes[format_id(x.output[0])]]

        return ReshapeComputeNode(layer_id, layer_type, feature_input, feature_output, reshape_size)

    @override
    def to_json(self):
        info = dict()
        info['type'] = self.layer_type
        info['channel_input'] = self.feature_input[0].channel
        info['channel_output'] = self.feature_output[0].channel
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input[:1]]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        info['shape'] = [-1, 1]
        return info

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating reshape code')
        init_str, forward_str = [], []
        params_str = {}
        shape = self.reshape_size
        forward_str.append(
            f'{str(self.feature_output[0])} = {str(self.feature_input[0])}.view({str(self.feature_input[0])}.size(0), -1)',
        )
        return {'init': init_str, 'forward': forward_str}
