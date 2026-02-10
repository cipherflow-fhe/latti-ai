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
from . import ComputeNode, FeatureNode, format_id, dict_to_args
from onnx import NodeProto

log = logging.getLogger(__name__)


class BatchNormComputeNode(ComputeNode):
    """Compute node for BatchNorm operation"""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        weight_path: str = None,
        bias_path: str = None,
        running_mean_path: str = None,
        running_var_path: str = None,
    ):
        super().__init__(layer_id, layer_type, feature_input, feature_output)
        feature_output[0].shape = feature_input[0].shape
        feature_output[0].skip = feature_input[0].skip
        feature_output[0].level = feature_input[0].level
        self.weight_path = weight_path
        self.bias_path = bias_path
        self.running_mean_path = running_mean_path
        self.running_var_path = running_var_path

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'BatchNormComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'batchnorm2d'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        weight_path = x.input[1]
        bias_path = x.input[2]
        running_mean_path = x.input[3]
        running_var_path = x.input[4]

        return BatchNormComputeNode(
            layer_id,
            layer_type,
            feature_input,
            feature_output,
            weight_path,
            bias_path,
            running_mean_path,
            running_var_path,
        )

    @override
    def to_json(self):
        info = dict()
        self.feature_output[0].shape = self.feature_input[0].shape
        self.feature_output[0].skip = self.feature_input[0].skip
        info['type'] = self.layer_type
        info['channel_input'] = int(self.feature_input[0].channel)
        info['channel_output'] = int(self.feature_output[0].channel)
        info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
        info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output[:1]]
        info['weight_path'] = self.weight_path
        info['bias_path'] = self.bias_path
        info['running_mean_path'] = self.running_mean_path
        info['running_var_path'] = self.running_var_path
        return info

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating batchnorm code')
        init_str, forward_str = [], []
        params_str = dict()
        params_str['num_features'] = self.feature_input[0].channel
        params = dict_to_args(params_str)
        if self.feature_input[0].dim == 0:
            init_str.append(f'self.{self.layer_id} = nn.BatchNorm1d({params})')
        else:
            init_str.append(f'self.{self.layer_id} = nn.BatchNorm2d({params})')
        forward_str.append(f'{str(self.feature_output[0])} = self.{self.layer_id}({str(self.feature_input[0])})')
        return {'init': init_str, 'forward': forward_str}
