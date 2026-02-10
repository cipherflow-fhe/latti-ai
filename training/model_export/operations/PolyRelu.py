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


class PolyReluComputeNode(ComputeNode):
    """Compute node for PolyRelu activation"""

    def __init__(
        self, layer_id: str, layer_type: str, feature_input: list[FeatureNode], feature_output: list[FeatureNode]
    ):
        super(PolyReluComputeNode, self).__init__(layer_id, layer_type, feature_input, feature_output)
        self.layer_type = 'poly_relu2d'
        feature_output[0].channel = feature_input[0].channel
        feature_output[0].level = feature_input[0].level - 2
        feature_output[0].shape = feature_input[0].shape
        feature_output[0].skip = feature_output[0].skip
        feature_output[0].scale = feature_input[0].scale
        feature_output[0].skip = feature_input[0].skip

    @staticmethod
    def from_onnx_node(x: NodeProto, features_nodes) -> 'PolyReluComputeNode':
        layer_id = format_id(x.name)
        layer_type = 'poly_relu2d'
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]

        return PolyReluComputeNode(layer_id, layer_type, feature_input, feature_output)

    @override
    def to_torch_code(self) -> dict[str, list[str]]:
        log.debug('Generating polyrelu code')
        init_str, forward_str = [], []
        params_str = {}
        params_str['num_features'] = self.feature_input[0].channel
        params_str['num_tensors'] = 2
        params = dict_to_args(params_str)
        init_str.append(f'self.{self.layer_id} = PolyReluListIndependent({params})')
        forward_str.append(
            f'{str(self.feature_output[0])} = self.{self.layer_id}({self.feature_input[0]}, [0.5, 1 / np.sqrt(4.0 * np.pi)])'
        )
        return {'init': init_str, 'forward': forward_str}

    @staticmethod
    def gen_method():
        return """
from typing import Union, Optional

class PolyReluListIndependentFunction(torch.autograd.Function):
    eps = 1e-5
    affine = True

    @staticmethod
    def forward(ctx, *inputs):
        x = inputs[0]
        a_tensor = inputs[1]
        weights = inputs[2]
        biases = inputs[3]

        dim = x.dim()
        spatial_dims = tuple(range(2, dim))
        a_tensor = a_tensor.view(-1, *([1] * dim))

        x_list = [x, x / np.sqrt(2)]

        normalized_list = []
        for i, x in enumerate(x_list):
            mean = x.mean(dim=(0, *spatial_dims), keepdim=True)
            var = x.var(dim=(0, *spatial_dims), keepdim=True, unbiased=False)
            x_norm = (x - mean) / np.sqrt(var + PolyReluListIndependentFunction.eps)

            if PolyReluListIndependentFunction.affine and weights is not None and biases is not None:
                weight = weights[i].view(1, -1, *([1] * (dim - 2)))
                bias = biases[i].view(1, -1, *([1] * (dim - 2)))
                x_norm = x_norm * weight + bias

            normalized_list.append(x_norm)

        ctx.save_for_backward(x, a_tensor, weights, biases)
        return (torch.stack(normalized_list) * a_tensor).sum(dim=0)

    @staticmethod
    def symbolic(g, *inputs):
        return g.op("custom::PolyReluListIndependent", *inputs)

class PolyReluListIndependent(nn.Module):
    def __init__(self, num_features, num_tensors, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.num_tensors = num_tensors
        self.eps = eps
        self.affine = affine

        if affine:
            self.weights = nn.Parameter(torch.ones(num_tensors, num_features))
            self.biases = nn.Parameter(torch.zeros(num_tensors, num_features))
        else:
            self.register_parameter("weights", None)
            self.register_parameter("biases", None)

        # Set static hyperparameters for Function
        PolyReluListIndependentFunction.eps = eps
        PolyReluListIndependentFunction.affine = affine

    def forward(self, x, a_list):
        a_tensor = torch.tensor(a_list, dtype=torch.float32, device=x.device)
        inputs = [x, a_tensor, self.weights, self.biases]
        return PolyReluListIndependentFunction.apply(*inputs)

        """
