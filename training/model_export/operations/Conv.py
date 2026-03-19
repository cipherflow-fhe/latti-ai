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
from onnx import NodeProto, GraphProto

log = logging.getLogger(__name__)


class ConvComputeNode(ComputeNode):
    """Compute node for Conv operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        groups: int,
        stride: list,
        kernel_shape: list,
        weight_path: str = None,
        bias_path: str = None,
        style: str = 'ordinary',
    ):
        super().__init__(layer_id, layer_type, feature_input, feature_output)
        self.layer_id = layer_id
        self.groups = groups
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.weight_path = weight_path
        self.bias_path = bias_path
        if feature_input[0].dim == 2:
            feature_output[0].skip[0] = feature_input[0].skip[0] * stride[0]
            feature_output[0].skip[1] = feature_input[0].skip[1] * stride[1]
            feature_output[0].shape[0] = feature_input[0].shape[0] // stride[0]
            feature_output[0].shape[1] = feature_input[0].shape[1] // stride[1]
        if feature_input[0].dim == 1:
            feature_output[0].skip[0] = feature_input[0].skip[0] * stride[0]
            feature_output[0].shape[0] = feature_input[0].shape[0] // stride[0]
        feature_output[0].level = feature_input[0].level - 1
        self.weight_path = self.weight_path
        self.bias_path = self.weight_path.replace('.weight', '.bias')
        self.style = style
        self.is_big_size = False

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes, style, graph: GraphProto) -> 'ConvComputeNode':
        from onnx import numpy_helper

        weight_path = x.input[1]
        if len(x.input) >= 3:
            bias_path = x.input[2]
        else:
            bias_path = None

        # Find weight tensor from graph.initializer
        weight_tensor = None
        for init in graph.initializer:
            if init.name == weight_path:
                weight_tensor = init
                break
        else:
            raise ValueError('Could find weight tensor.')

        layer_id = format_id(x.name)
        log.debug('%s', x)
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]

        attrs = ComputeNode.get_attr_value_dict(x)
        groups = attrs['group']
        stride = attrs['strides']
        pads = attrs['pads']
        dilations = attrs['dilations']

        weight_shape = numpy_helper.to_array(weight_tensor).shape
        out_channels = weight_shape[0]
        in_channels = weight_shape[1] * groups
        spatial_dims = len(weight_shape) - 2  # 1 for Conv1d, 2 for Conv2d
        kernel_shape = list(weight_shape[2:])

        # Update channel for feature_input and feature_output
        feature_input[0].channel = in_channels
        feature_output[0].channel = out_channels

        log.debug('Dynamically inferred Conv channel: input=%s, output=%s', in_channels, out_channels)

        layer_type = f'conv{spatial_dims}d'
        if dilations != [1] * spatial_dims:
            raise ValueError('Unsupported dilation value: ' + str(dilations))
        # Make sure the relation "feature_output.shape = feature_input.shape // stride" holds
        # ONNX pads format: [pad_begin_0, ..., pad_begin_N, pad_end_0, ..., pad_end_N]
        for i in range(spatial_dims):
            if (
                not 0
                <= pads[i] + pads[i + spatial_dims] - dilations[i] * (kernel_shape[i] - 1) - 1 + stride[i]
                < stride[i]
            ):
                raise ValueError('Unsupported padding value: ' + str(pads))

        if (groups != 1) and (groups != in_channels):
            raise ValueError('Unsupported groups value: ' + str(groups))

        return ConvComputeNode(
            layer_id,
            layer_type,
            feature_input,
            feature_output,
            groups,
            stride,
            kernel_shape,
            weight_path,
            bias_path,
            style,
        )

    @override
    def to_json(self) -> dict:
        info = dict()
        info['type'] = self.layer_type
        info['style'] = self.style
        info['channel_input'] = self.feature_input[0].channel
        info['channel_output'] = self.feature_output[0].channel
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        info['groups'] = self.groups
        info['stride'] = self.stride
        info['is_big_size'] = self.is_big_size
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
        params_str['stride'] = self.stride
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
