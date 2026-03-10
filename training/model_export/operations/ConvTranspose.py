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


class ConvTransposeComputeNode(ComputeNode):
    """Compute node for ConvTranspose operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        groups: int,
        upsample_factor: list,
        kernel_shape: list,
        weight_path: str = None,
        bias_path: str = None,
    ):
        super().__init__(layer_id, layer_type, feature_input, feature_output)
        self.layer_id = layer_id
        self.groups = groups
        self.kernel_shape = kernel_shape
        self.upsample_factor = upsample_factor
        self.weight_path = weight_path
        self.bias_path = bias_path
        if feature_input[0].dim == 2:
            feature_output[0].skip[0] = feature_input[0].skip[0] // upsample_factor[0]
            feature_output[0].skip[1] = feature_input[0].skip[1] // upsample_factor[1]
            feature_output[0].shape[0] = feature_input[0].shape[0] * upsample_factor[0]
            feature_output[0].shape[1] = feature_input[0].shape[1] * upsample_factor[1]
        if feature_input[0].dim == 1:
            feature_output[0].skip[0] = feature_input[0].skip[0] // upsample_factor[0]
            feature_output[0].shape[0] = feature_input[0].shape[0] * upsample_factor[0]
        feature_output[0].level = feature_input[0].level - 1
        self.weight_path = self.weight_path
        self.bias_path = self.bias_path

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'ConvTransposeComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'conv2d'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)
        groups = attrs['group']
        kernel_shape = attrs['kernel_shape']
        upsample_factor_in = attrs['strides']
        weight_path = x.input[1]
        if len(x.input) >= 3:
            bias_path = x.input[2]
        else:
            bias_path = None

        return ConvTransposeComputeNode(
            layer_id,
            layer_type,
            feature_input,
            feature_output,
            groups,
            upsample_factor_in,
            kernel_shape,
            weight_path,
            bias_path,
        )

    @override
    def to_json(self) -> dict:
        info = dict()
        info['type'] = self.layer_type
        info['channel_input'] = self.feature_input[0].channel
        info['channel_output'] = self.feature_output[0].channel
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        info['groups'] = self.groups
        info['stride'] = [1, 1]
        info['upsample_factor'] = self.upsample_factor
        info['kernel_shape'] = self.kernel_shape
        info['weight_path'] = self.weight_path
        info['bias_path'] = self.bias_path if self.bias_path is not None else None
        return info

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating conv code')
        init_str, forward_str = [], []
        params_str = dict()
        params_str['groups'] = self.groups
        params_str['stride'] = [1, 1]
        params_str['upsample_factor'] = self.upsample_factor
        params_str['kernel_size'] = self.kernel_shape
        if self.feature_input[0].dim == 2:
            params_str['padding'] = [1, 1]
            if params_str['kernel_size'] == [1, 1]:
                params_str['padding'] = [0, 0]
        if self.feature_input[0].dim == 1:
            params_str['padding'] = [1]
            if params_str['kernel_size'] == [1]:
                params_str['padding'] = [0]
        params_str['in_channels'] = self.feature_input[0].channel
        params_str['out_channels'] = self.feature_output[0].channel
        params_str['bias'] = False
        params = dict_to_args(params_str)
        if self.feature_input[0].dim == 2:
            init_str.append(f'self.{self.layer_id} = nn.Conv2d({params})')
        if self.feature_input[0].dim == 1:
            init_str.append(f'self.{self.layer_id} = nn.Conv1d({params})')
        forward_str.append(f'{str(self.feature_output[0])} = self.{self.layer_id}({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}
