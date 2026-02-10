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

# from typing import override
from typing_extensions import override
from . import ComputeNode, FeatureNode, format_id
from onnx import NodeProto


class RNNComputeNode(ComputeNode):
    """Compute node for RNN operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        is_mpc=False,
    ):
        super(RNNComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        feature_output[0].skip = [1, 1]

    @override
    def to_json(self):
        info = dict()
        info['type'] = self.layer_type
        info['channel_input'] = int(self.feature_input[0].channel)
        info['channel_output'] = int(self.feature_output[0].channel)
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'RNNComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'rnn'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)

        return RNNComputeNode(layer_id, layer_type, feature_input, feature_output)
