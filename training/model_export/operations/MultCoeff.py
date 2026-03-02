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

from typing_extensions import override
from . import ComputeNode, FeatureNode, format_id
from onnx import NodeProto

log = logging.getLogger(__name__)


class MultCoeffComputeNode(ComputeNode):
    """Compute node for MultCoeff operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        coeff: float,
    ):
        super().__init__(layer_id, layer_type, feature_input, feature_output)
        feature_output[0].shape = feature_input[0].shape
        feature_output[0].skip = feature_input[0].skip
        feature_output[0].level = feature_input[0].level
        self.coeff = coeff

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes, constant_nodes) -> 'MultCoeffComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'mult_coeff'
        log.debug('%s', x)
        if format_id(x.input[0]) in features_nodes:
            variable_input_name = format_id(x.input[0])
            const_input_name = format_id(x.input[1])
        elif format_id(x.input[1]) in features_nodes:
            variable_input_name = format_id(x.input[1])
            const_input_name = format_id(x.input[0])
        feature_input = [features_nodes[variable_input_name]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        coeff = round(float(constant_nodes[const_input_name][0]), 5)

        return MultCoeffComputeNode(layer_id, layer_type, feature_input, feature_output, coeff)

    @override
    def to_json(self):
        info = dict()
        self.feature_output[0].shape = self.feature_input[0].shape
        self.feature_output[0].skip = self.feature_input[0].skip
        info['type'] = self.layer_type
        info['coeff'] = self.coeff
        info['channel_input'] = int(self.feature_input[0].channel)
        info['channel_output'] = int(self.feature_output[0].channel)
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output[:1]]
        return info

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating mult code')
        init_str, forward_str = [], []
        params_str = dict()
        params_str['num_features'] = self.feature_input[0].channel
        forward_str.append(
            f'{str(self.feature_output[0])} = torch.mul({self.feature_output[0].scale}, {str(self.feature_input[0])})'
        )
        return {'init': init_str, 'forward': forward_str}
