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


class MaxPoolComputeNode(ComputeNode):
    """Compute node for MaxPool operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        kernel_shape: list,
        stride: list,
    ):
        super(MaxPoolComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.stride = stride
        self.kernel_shape = kernel_shape
        feature_output[0].shape[0] = feature_input[0].shape[0] // stride[0]
        feature_output[0].shape[1] = feature_input[0].shape[1] // stride[1]
        feature_output[0].skip = [1, 1]

    @override
    def to_json(self):
        info = dict()
        # # info[self.layer_id] = self.layer_id
        # # self.feature_output[0].shape[0] = self.feature_input[0].shape[0] // self.stride
        # # self.feature_output[0].shape[1] = self.feature_input[0].shape[1] // self.stride
        # self.feature_output[0].skip[0] = self.feature_input[0].skip[0]
        # self.feature_output[0].skip[1] = self.feature_input[0].skip[1]
        info['type'] = self.layer_type
        info['channel_input'] = int(self.feature_input[0].channel)
        info['channel_output'] = int(self.feature_output[0].channel)
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['stride'] = self.stride
        info['kernel_shape'] = self.kernel_shape
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'MaxPoolComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'maxpool2d'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)
        stride = attrs['strides']
        kernel_shape = attrs['kernel_shape']

        return MaxPoolComputeNode(layer_id, layer_type, feature_input, feature_output, kernel_shape, stride)

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating maxpool code')
        init_str, forward_str = [], []
        params_str = {}
        params_str['padding'] = [1, 1]
        params_str['stride'] = self.stride
        params_str['kernel_size'] = self.kernel_shape
        params = self.gen_params_str(**params_str)
        init_str.append(f'self.{self.layer_id} = nn.MaxPool2d(**{{{params}}})')
        forward_str.append(f'{str(self.feature_output[0])} = self.{self.layer_id}({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}
