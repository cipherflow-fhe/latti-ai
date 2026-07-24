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
from . import ComputeNode, FeatureNode, format_id, dict_to_args
from onnx import NodeProto

log = logging.getLogger(__name__)


class AveragePoolComputeNode(ComputeNode):
    """Compute node for AveragePool operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        kernel_shape: list,
        stride: list = [1, 1],
        pads: list = [1, 1],
    ):
        super(AveragePoolComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = pads
        feature_output[0].level = feature_input[0].level
        feature_output[0].channel = feature_input[0].channel
        spatial_dims = len(stride)
        if all(stride[i] > 0 for i in range(spatial_dims)) and feature_input[0].shape[0] > 0:
            for i in range(spatial_dims):
                feature_output[0].skip[i] = feature_input[0].skip[i] * stride[i]
                feature_output[0].shape[i] = feature_input[0].shape[i] // stride[i]
        else:
            for i in range(spatial_dims):
                feature_output[0].shape[i] = 1
                feature_output[0].skip[i] = feature_input[0].skip[i]

    @override
    def to_json(self):
        info = dict()
        info['type'] = self.layer_type
        info['channel_input'] = self.feature_input[0].channel
        info['channel_output'] = self.feature_output[0].channel
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        info['stride'] = self.stride
        info['kernel_shape'] = self.kernel_shape
        info['padding'] = self.padding
        info['weight_path'] = self.layer_id + '.weight'
        info['bias_path'] = self.layer_id + '.bias'
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'AveragePoolComputeNode':
        layer_id = format_id(x.name)
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)
        stride = attrs['strides']
        kernel_shape = attrs['kernel_shape']
        pads_raw = attrs.get('pads', [0] * len(kernel_shape) * 2)
        pads = pads_raw[len(kernel_shape) : :]

        spatial_dims = len(kernel_shape)
        layer_type = f'avgpool{spatial_dims}d'

        return AveragePoolComputeNode(layer_id, layer_type, feature_input, feature_output, kernel_shape, stride, pads)

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating avgpool code')
        init_str, forward_str = [], []
        params_str = {}
        params_str['padding'] = self.padding
        params_str['stride'] = self.stride
        params_str['kernel_size'] = self.kernel_shape
        params = dict_to_args(params_str)
        spatial_dims = len(self.stride)
        pool_class = f'nn.AvgPool{spatial_dims}d'
        init_str.append(f'self.{self.layer_id} = {pool_class}({params})')
        forward_str.append(f'{str(self.feature_output[0])} = self.{self.layer_id}({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}
