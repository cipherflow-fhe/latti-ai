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

from typing_extensions import override
from . import ComputeNode, FeatureNode, format_id
from onnx import NodeProto


class LayerNormComputeNode(ComputeNode):
    """Compute node for LayerNorm.

    Produced from the single ``nn_tools::LayerNorm`` ONNX custom op emitted by
    ``FHELayerNorm`` during export.  The node carries the weight/bias initialiser
    names so that the H5 export pipeline can locate them.
    """

    def __init__(
        self,
        layer_id: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        epsilon: float = 1e-5,
        weight_name: str = '',
        bias_name: str = '',
        inv_std_scale: float = 1.0,
        inv_var_scale: float = 1.0,
        c0: float = 6.19067182,
        c1: float = -16.15885111,
        c2: float = 11.52830778,
        num_iters: int = 2,
    ):
        super().__init__(layer_id, 'layernorm', feature_input, feature_output)
        self.epsilon = epsilon
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.inv_std_scale = inv_std_scale
        self.inv_var_scale = inv_var_scale
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.num_iters = num_iters

    @override
    def to_json(self) -> dict:
        info: dict = {
            'type': 'layernorm',
            'feature_input': [i.node_id for i in self.feature_input],
            'feature_output': [i.node_id for i in self.feature_output],
            'epsilon': self.epsilon,
            'inv_std_scale': self.inv_std_scale,
            'inv_var_scale': self.inv_var_scale,
            'c0': self.c0,
            'c1': self.c1,
            'c2': self.c2,
            'num_iters': self.num_iters,
        }
        if self.weight_name:
            info['weight_path'] = self.weight_name
        if self.bias_name:
            info['bias_path'] = self.bias_name
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'LayerNormComputeNode':
        layer_id = format_id(x.name)
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]

        attrs = ComputeNode.get_attr_value_dict(x)
        epsilon = float(attrs.get('eps', attrs.get('epsilon', 1e-5)))
        inv_std_scale = float(attrs.get('inv_std_scale', 1.0))
        inv_var_scale = float(attrs.get('inv_var_scale', 1.0))
        c0 = float(attrs.get('c0', 6.19067182))
        c1 = float(attrs.get('c1', -16.15885111))
        c2 = float(attrs.get('c2', 11.52830778))
        num_iters = int(attrs.get('num_iters', 2))

        # input[1] = weight initialiser, input[2] = bias initialiser (raw ONNX names)
        weight_name = x.input[1] if len(x.input) > 1 else ''
        bias_name = x.input[2] if len(x.input) > 2 else ''

        return LayerNormComputeNode(
            layer_id=layer_id,
            feature_input=feature_input,
            feature_output=feature_output,
            epsilon=epsilon,
            weight_name=weight_name,
            bias_name=bias_name,
            inv_std_scale=inv_std_scale,
            inv_var_scale=inv_var_scale,
            c0=c0,
            c1=c1,
            c2=c2,
            num_iters=num_iters,
        )
