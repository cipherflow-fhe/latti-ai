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
"""Convert an ONNX model to the JSON format used by the encrypted inference engine."""

import json
import logging

import onnx
from onnx import numpy_helper
from onnx import shape_inference

from .operations import FeatureNode, ComputeNode, format_id, get_type_id, get_op_code_generator
from .operations.Conv import ConvComputeNode
from .operations.BatchNorm import BatchNormComputeNode
from .operations.Dense import DenseComputeNode
from .operations.Relu import ReluComputeNode
from .operations.Reshape import ReshapeComputeNode
from .operations.Dropout import DropoutComputeNode
from .operations.MultCoeff import MultCoeffComputeNode
from .operations.AveragePool import AveragePoolComputeNode
from .operations.MaxPool import MaxPoolComputeNode
from .operations.Sigmoid import SigmoidComputeNode
from .operations.PolyRelu import PolyReluComputeNode
from .operations.ConvTranspose import ConvTransposeComputeNode
from .operations.PolyAct import PolyActComputeNode, PolyActRNPolyComputeNode
from .operations.MatMul import MatMulComputeNode
from .operations.Transpose import TransposeComputeNode
from .operations.LayerNorm import LayerNormComputeNode
from .onnx_model_manipulations import simplify_onnx_model

log = logging.getLogger(__name__)


def gen_data_nodes(value_infos, feature_mat: bool = False) -> dict[str, FeatureNode]:
    """Build FeatureNode dict from ONNX value_info entries."""
    data_nodes: dict[str, FeatureNode] = dict()
    for key, feature in value_infos.items():
        tensor_shape = []
        key = format_id(key)
        for s in feature.type.tensor_type.shape.dim:
            tensor_shape.append(s.dim_value)
        if len(tensor_shape) == 0:
            continue
        if feature_mat:
            if len(tensor_shape) == 3:
                shape = tensor_shape[1:]
            else:
                shape = list(tensor_shape)
            dim = 2
            channel = 1
        elif len(tensor_shape) == 1:
            shape = tensor_shape
            dim = 0
            channel = tensor_shape[0]
        elif len(tensor_shape) == 2:
            shape = tensor_shape[1::]
            dim = 0
            channel = tensor_shape[1]
        else:
            shape = tensor_shape[2::]
            channel = tensor_shape[1]
            dim = len(tensor_shape) - 2
        scale = 1
        skip = [1] * max(dim, 1)
        ckks_parameter_id = 'param0'
        node = FeatureNode(key, dim, channel, scale, skip, ckks_parameter_id, shape)
        if feature_mat:
            node.data_type = 'feature_mat'
        data_nodes[key] = node
    return data_nodes


def get_constant(const_node: onnx.NodeProto):
    """Extract constant value from an ONNX Constant node."""
    const_value = None
    for attr in const_node.attribute:
        if attr.name == 'value':
            const_value = onnx.helper.get_attribute_value(attr)
            if isinstance(const_value, onnx.TensorProto):
                return numpy_helper.to_array(const_value)
            elif hasattr(const_value, 'decode'):
                return const_value.decode('utf-8')
        else:
            raise ValueError(f"Unexpected attribute '{attr.name}' in Constant node, expected 'value'")
    return const_value


class CustomMultiHeadAttentionComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        feature_input: list[FeatureNode],
        feature_output: list[FeatureNode],
        q_weight_path: str,
        k_weight_path: str,
        v_weight_path: str,
        proj_weight_path: str,
        gamma_path: str,
        poly_weight_path: str,
        q_bias_path: str = '',
        k_bias_path: str = '',
        v_bias_path: str = '',
        proj_bias_path: str = '',
        poly_order: int = 4,
    ):
        super().__init__(layer_id, 'CustomMultiHeadAttention', feature_input, feature_output)
        self.q_weight_path = q_weight_path
        self.k_weight_path = k_weight_path
        self.v_weight_path = v_weight_path
        self.proj_weight_path = proj_weight_path
        self.gamma_path = gamma_path
        self.poly_weight_path = poly_weight_path
        self.q_bias_path = q_bias_path
        self.k_bias_path = k_bias_path
        self.v_bias_path = v_bias_path
        self.proj_bias_path = proj_bias_path
        self.poly_order = poly_order

    @staticmethod
    def _split_qkv_weight_paths(qkv_weight_path: str, layer_id: str) -> tuple[str, str, str]:
        if qkv_weight_path.endswith('.qkv.weight'):
            prefix = qkv_weight_path[: -len('.qkv.weight')]
            return f'{prefix}.q.weight', f'{prefix}.k.weight', f'{prefix}.v.weight'
        return f'{layer_id}.q.weight', f'{layer_id}.k.weight', f'{layer_id}.v.weight'

    @staticmethod
    def _split_qkv_bias_paths(qkv_bias_path: str, layer_id: str) -> tuple[str, str, str]:
        if not qkv_bias_path:
            return '', '', ''
        if qkv_bias_path.endswith('.qkv.bias'):
            prefix = qkv_bias_path[: -len('.qkv.bias')]
            return f'{prefix}.q.bias', f'{prefix}.k.bias', f'{prefix}.v.bias'
        return f'{layer_id}.q.bias', f'{layer_id}.k.bias', f'{layer_id}.v.bias'

    @staticmethod
    def _gamma_path(running_max_path: str, layer_id: str) -> str:
        if running_max_path.endswith('.running_max_concat'):
            return running_max_path[: -len('.running_max_concat')] + '.gamma'
        return f'{layer_id}.gamma'

    @staticmethod
    def _poly_weight_path(poly_coeff_paths: list[str], layer_id: str) -> str:
        if poly_coeff_paths:
            first_path = poly_coeff_paths[0]
            suffix = first_path.rsplit('.', 1)[-1]
            if suffix.startswith('a') and suffix[1:].isdigit():
                return first_path.rsplit('.', 1)[0] + '.weight'
        return f'{layer_id}.poly.weight'

    @staticmethod
    def _poly_order(poly_coeff_paths: list[str], default: int = 4) -> int:
        order = 0
        for path in poly_coeff_paths:
            suffix = path.rsplit('.', 1)[-1]
            if suffix.startswith('a') and suffix[1:].isdigit():
                order = max(order, int(suffix[1:]))
        return order or default

    @staticmethod
    def from_onnx_node(x: onnx.NodeProto, features_nodes) -> 'CustomMultiHeadAttentionComputeNode':
        layer_id = format_id(x.name)
        feature_input = [features_nodes[format_id(x.input[0])]]
        feature_output = [features_nodes[format_id(x.output[0])]]
        qkv_weight_path = x.input[1] if len(x.input) > 1 else ''
        qkv_bias_path = x.input[2] if len(x.input) > 2 else ''
        q_weight_path, k_weight_path, v_weight_path = CustomMultiHeadAttentionComputeNode._split_qkv_weight_paths(
            qkv_weight_path, layer_id
        )
        q_bias_path, k_bias_path, v_bias_path = CustomMultiHeadAttentionComputeNode._split_qkv_bias_paths(
            qkv_bias_path, layer_id
        )
        proj_weight_path = x.input[3] if len(x.input) > 3 else f'{layer_id}.proj.weight'
        proj_bias_path = x.input[4] if len(x.input) > 4 else ''
        running_max_path = x.input[5] if len(x.input) > 5 else ''
        poly_coeff_paths = list(x.input[6:])

        return CustomMultiHeadAttentionComputeNode(
            layer_id=layer_id,
            feature_input=feature_input,
            feature_output=feature_output,
            q_weight_path=q_weight_path,
            k_weight_path=k_weight_path,
            v_weight_path=v_weight_path,
            proj_weight_path=proj_weight_path,
            gamma_path=CustomMultiHeadAttentionComputeNode._gamma_path(running_max_path, layer_id),
            poly_weight_path=CustomMultiHeadAttentionComputeNode._poly_weight_path(poly_coeff_paths, layer_id),
            q_bias_path=q_bias_path,
            k_bias_path=k_bias_path,
            v_bias_path=v_bias_path,
            proj_bias_path=proj_bias_path,
            poly_order=CustomMultiHeadAttentionComputeNode._poly_order(poly_coeff_paths),
        )

    def to_json(self) -> dict:
        info = {
            'type': self.layer_type,
            'feature_input': [i.node_id for i in self.feature_input],
            'feature_output': [i.node_id for i in self.feature_output],
            'q_weight_path': self.q_weight_path,
            'k_weight_path': self.k_weight_path,
            'v_weight_path': self.v_weight_path,
            'proj_weight_path': self.proj_weight_path,
            'gamma_path': self.gamma_path,
            'poly_weight_path': self.poly_weight_path,
            'poly_order': self.poly_order,
        }
        if self.q_bias_path:
            info['q_bias_path'] = self.q_bias_path
        if self.k_bias_path:
            info['k_bias_path'] = self.k_bias_path
        if self.v_bias_path:
            info['v_bias_path'] = self.v_bias_path
        if self.proj_bias_path:
            info['proj_bias_path'] = self.proj_bias_path
        return info


def onnx_to_json(onnx_filename: str, output_filename: str, style: str, feature_mat: bool = False):
    """Convert an ONNX model file to the JSON format for encrypted inference.

    Args:
        onnx_filename:  Path to the input ``.onnx`` model.
        output_filename: Path to the output ``.json`` file.
        style:          Packing style (``'ordinary'`` or ``'multiplexed'``).
    """
    onnx_model = onnx.load(onnx_filename)
    simplify_onnx_model(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    graph = onnx_model.graph
    input_value_infos = {i.name: i for i in graph.input}
    output_value_infos = {i.name: i for i in graph.output}
    value_infos = {}
    value_infos.update(input_value_infos)
    value_infos.update(output_value_infos)
    value_infos.update({i.name: i for i in graph.value_info})
    features_nodes = gen_data_nodes(value_infos, feature_mat=feature_mat)
    compute_nodes: dict[str, ComputeNode] = {}

    constant_nodes = {
        format_id(init.name): [numpy_helper.to_array(init), numpy_helper.to_array(init)] for init in graph.initializer
    }
    weight_shapes = {init.name: list(init.dims) for init in graph.initializer}

    for n in graph.node:
        name = format_id(n.output[0])
        if n.op_type in ('Unsqueeze', 'Cast'):
            continue
        if n.op_type == 'Constant':
            data = get_constant(n)
            constant_nodes[name] = list([data, data])
            features_nodes.pop(name, None)
            continue
        inp = [format_id(i) for i in n.input]
        out = [format_id(i) for i in n.output]

        match n.op_type:
            case 'Conv':
                compute_node = ConvComputeNode.from_onnx_node(n, features_nodes, style=style, graph=graph)
            case 'BatchNormalization':
                compute_node = BatchNormComputeNode.from_onnx_node(n, features_nodes)
            case 'Gemm':
                compute_node = DenseComputeNode.from_onnx_node(n, features_nodes)
            case 'Relu':
                compute_node = ReluComputeNode.from_onnx_node(n, features_nodes)
            case 'Reshape':
                compute_node = ReshapeComputeNode.from_onnx_node(n, features_nodes, constant_nodes)
            case 'Dropout':
                compute_node = DropoutComputeNode.from_onnx_node(n, features_nodes)
            case 'Mul':
                compute_node = MultCoeffComputeNode.from_onnx_node(n, features_nodes, constant_nodes)
            case 'AveragePool':
                compute_node = AveragePoolComputeNode.from_onnx_node(n, features_nodes)
            case 'GlobalAveragePool':
                layer_id = format_id(n.name)
                feature_input = [features_nodes[format_id(n.input[0])]]
                feature_output = [features_nodes[format_id(n.output[0])]]
                # GlobalAveragePool outputs 1x1; kernel equals input spatial size
                input_shape = feature_input[0].shape
                if input_shape[0] > 0 and input_shape[1] > 0:
                    ks = list(input_shape)
                else:
                    # Fallback when shape inference failed
                    ks = list(feature_output[0].shape) if feature_output[0].shape[0] > 0 else [1, 1]
                compute_node = AveragePoolComputeNode(
                    layer_id, 'avgpool2d', feature_input, feature_output, kernel_shape=ks, stride=ks, pads=[0, 0]
                )
            case 'MaxPool':
                compute_node = MaxPoolComputeNode.from_onnx_node(n, features_nodes)
            case 'Dense':
                compute_node = DenseComputeNode.from_onnx_node(n, features_nodes)
            case 'ConvTranspose':
                compute_node = ConvTransposeComputeNode.from_onnx_node(n, features_nodes)
            case 'MatMul':
                compute_node = MatMulComputeNode.from_onnx_node(n, features_nodes, weight_shapes)
            case 'Transpose':
                compute_node = TransposeComputeNode.from_onnx_node(n, features_nodes)
            case 'LayerNorm':
                compute_node = LayerNormComputeNode.from_onnx_node(n, features_nodes)
            case 'RangeNormPoly2d':
                compute_node = PolyActComputeNode.from_onnx_node(n, features_nodes)
            case 'RangeNormPoly1d':
                compute_node = PolyActComputeNode.from_onnx_node(n, features_nodes)
            case 'PolyAct':
                compute_node = PolyActComputeNode.from_onnx_node(n, features_nodes)
            case 'PolyActRNPoly':
                compute_node = PolyActRNPolyComputeNode.from_onnx_node(n, features_nodes)
            case 'Linear':
                compute_node = MatMulComputeNode.from_onnx_node(n, features_nodes, weight_shapes)
            case 'CustomLayerNorm':
                compute_node = LayerNormComputeNode.from_onnx_node(n, features_nodes)
            case 'CustomMultiHeadAttention':
                compute_node = CustomMultiHeadAttentionComputeNode.from_onnx_node(n, features_nodes)
            case 'PolyActRN':
                compute_node = PolyActRNPolyComputeNode.from_onnx_node(n, features_nodes)
            case _:
                kwargs = {}
                if 'Add' in n.op_type:
                    inp = [format_id(i) for i in n.input]

                kwargs['layer_id'] = format_id(n.name)
                kwargs['layer_type'] = get_type_id(n.op_type)
                kwargs['feature_input'] = [features_nodes[i] for i in inp if i in features_nodes]
                kwargs['feature_output'] = [features_nodes[i] for i in out if i in features_nodes]
                compute_node = get_op_code_generator(n.op_type, **kwargs)

        compute_nodes[format_id(n.name)] = compute_node

    # Determine graph-level input/output feature IDs
    input_feature_ids = [format_id(i.name) for i in graph.input if format_id(i.name) in features_nodes]
    output_feature_ids = [format_id(o.name) for o in graph.output if format_id(o.name) in features_nodes]

    with open(output_filename, 'w') as f:
        json.dump(
            {
                'input_feature': input_feature_ids,
                'output_feature': output_feature_ids,
                'feature': {key: feature_node.to_json() for key, feature_node in features_nodes.items()},
                'layer': {key: compute_node.to_json() for key, compute_node in compute_nodes.items()},
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
