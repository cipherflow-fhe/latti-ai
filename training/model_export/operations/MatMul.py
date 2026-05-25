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


class MatMulComputeNode(ComputeNode):
    """Compute node for MatMul operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        weight_path: str = '',
        weight_shape: list[int] | None = None,
        bias_path: str = '',
        is_mpc=False,
        to_expand: bool = False,
    ):
        super(MatMulComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.weight_path = weight_path
        self.weight_shape = weight_shape or []
        self.bias_path = bias_path
        self.to_expand = to_expand
        feature_output[0].skip = [1, 1]

    @override
    def to_json(self):
        info = dict()
        info['type'] = self.layer_type
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        if self.layer_type == 'parcpmm' and self.weight_path:
            info['weight_path'] = self.weight_path
        if self.layer_type == 'parcpmm' and self.weight_shape:
            info['weight_shape'] = self.weight_shape
        if self.layer_type == 'parcpmm' and self.bias_path:
            info['bias_path'] = self.bias_path
        if self.layer_type == 'parcpmm' and self.to_expand:
            info['to_expand'] = True
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes, weight_shapes: dict | None = None) -> 'MatMulComputeNode':
        layer_id = format_id(x.name)
        input1_id = format_id(x.input[1])
        weight_path = ''
        weight_shape = []
        bias_path = ''
        if input1_id in features_nodes:
            layer_type = 'parccmm'
            feature_input = [features_nodes[format_id(x.input[0])], features_nodes[input1_id]]
        else:
            layer_type = 'parcpmm'
            feature_input = [features_nodes[format_id(x.input[0])]]
            weight_path = x.input[1]
            if weight_shapes and x.input[1] in weight_shapes:
                weight_shape = list(weight_shapes[x.input[1]])
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)
        log.debug('%s', attrs)
        has_bias_input = len(x.input) > 2 and bool(x.input[2])
        has_fused_bias_input = has_bias_input and 'fused_bias' in x.input[2]
        to_expand = layer_type == 'parcpmm' and x.op_type == 'Linear' and has_fused_bias_input
        if layer_type == 'parcpmm' and x.op_type == 'Linear':
            if has_bias_input:
                bias_path = x.input[2]
                

        return MatMulComputeNode(
            layer_id,
            layer_type,
            feature_input,
            feature_output,
            weight_path,
            weight_shape,
            bias_path=bias_path,
            to_expand=to_expand,
        )
