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
from .operations.Simple_Polyrelu import Simple_PolyreluComputeNode
from .onnx_model_manipulations import simplify_onnx_model

log = logging.getLogger(__name__)


def gen_data_nodes(value_infos) -> dict[str, FeatureNode]:
    """Build FeatureNode dict from ONNX value_info entries."""
    data_nodes: dict[str, FeatureNode] = dict()
    for key, feature in value_infos.items():
        tensor_shape = []
        key = format_id(key)
        for s in feature.type.tensor_type.shape.dim:
            tensor_shape.append(s.dim_value)
        if len(tensor_shape) == 0:
            continue
        if len(tensor_shape) == 1:
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


def onnx_to_json(onnx_filename: str, output_filename: str, style: str):
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
    features_nodes = gen_data_nodes(value_infos)
    compute_nodes: dict[str, ComputeNode] = {}

    constant_nodes = dict()

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
            case 'RangeNormPoly2d':
                compute_node = Simple_PolyreluComputeNode.from_onnx_node(n, features_nodes)
            case 'RangeNormPoly1d':
                compute_node = Simple_PolyreluComputeNode.from_onnx_node(n, features_nodes)
            case 'Simple_Polyrelu':
                compute_node = Simple_PolyreluComputeNode.from_onnx_node(n, features_nodes)
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
