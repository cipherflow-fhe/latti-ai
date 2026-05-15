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


class TransposeComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        perm: list[int] | None = None,
    ):
        super(TransposeComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.perm = perm

    @override
    def to_json(self):
        info = dict()
        info['type'] = self.layer_type
        info['feature_input'] = [i.node_id for i in self.feature_input]
        info['feature_output'] = [i.node_id for i in self.feature_output]
        return info

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'TransposeComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'partranspose'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        attrs = ComputeNode.get_attr_value_dict(x)
        perm = list(attrs.get('perm', []))
        log.debug('%s', attrs)
        return TransposeComputeNode(layer_id, layer_type, feature_input, feature_output, perm=perm)
