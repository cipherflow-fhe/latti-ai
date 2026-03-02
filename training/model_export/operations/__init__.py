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

import glob
import logging
import os
import math
import re
import onnx

log = logging.getLogger(__name__)

N = 65536

modules = glob.glob(os.path.join(os.path.dirname(__file__), '*.py'))
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.endswith('__init__.py')] + [
    'get_op_code_generator',
    'get_type_id',
]


def format_id(onnx_id: str) -> str:
    return re.sub('[:/.]', '_', onnx_id)


class FeatureNode(object):
    """Node of data"""

    """
    key: node id
    dim: dimension of feature, can be 0, 1, or 2
    channel: number of channels
    scale: scaling factor of feature, a double type number
    skip: skip number in ciphertext; when dim=0 no shape; when dim=1 shape is a list; when dim=2 shape is a list
    ckks_parameter_id: parameter of current ciphertext
    shape: shape size of model; when dim=0 no shape; when dim=1 shape is a number; when dim=2 shape is a list
    """

    def __init__(self, key: str, dim: int, channel: int, scale: float, skip: list, ckks_parameter_id: str, shape: list):
        self.node_id = key
        self.dim = dim
        self.channel = channel
        self.scale = scale
        self.shape = shape
        self.skip = skip
        self.ckks_parameter_id = ckks_parameter_id
        self.pack_num = None
        self.isvisit = False
        self.level = 0
        self.computer_nodes: list[ComputeNode] = []

    def __repr__(self) -> str:
        return f'{self.node_id}'

    def to_json(self) -> dict:
        info = dict()

        info['dim'] = self.dim
        info['channel'] = self.channel
        info['scale'] = self.scale
        if self.dim == 2:
            info['shape'] = self.shape
            info['skip'] = self.skip
            try:
                info['pack_num'] = math.ceil(N / 2 / self.shape[0] / self.shape[1] / self.skip[0] / self.skip[1])
            except Exception:
                info['pack_num'] = 1
        if self.dim == 1:
            info['shape'] = self.shape
            info['skip'] = self.skip
            info['pack_num'] = math.ceil(N / 2 / self.shape[0] / self.skip[1])
        if self.dim == 0:
            info['virtual_shape'] = [8, 8]
            info['virtual_skip'] = [1, 1]
            info['skip'] = self.skip[0]
            info['pack_num'] = math.ceil(N / 2 / info['virtual_shape'][0] / info['virtual_shape'][1])

        info['ckks_parameter_id'] = self.ckks_parameter_id
        info['level'] = 5 + int(self.level)
        return info


class ComputeNode:
    def __init__(
        self, layer_id: str, layer_type: str, feature_input: list[FeatureNode], feature_output: list[FeatureNode]
    ):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.feature_input = feature_input
        self.feature_output = feature_output
        self.next = None
        self.level = -1
        if feature_input and feature_output:
            if feature_output[0].shape and all(s == 0 for s in feature_output[0].shape):
                if feature_input[0].shape and any(s > 0 for s in feature_input[0].shape):
                    feature_output[0].shape = list(feature_input[0].shape)
                    feature_output[0].skip = list(feature_input[0].skip)

    def to_json(self) -> dict:
        info = dict()
        try:
            self.feature_output[0].shape = self.feature_input[0].shape
            self.feature_output[0].skip = self.feature_output[0].skip
            info['type'] = self.layer_type
            info['channel_input'] = int(self.feature_input[0].channel)
            info['channel_output'] = int(self.feature_output[0].channel)
            info['ckks_parameter_id_input'] = self.feature_input[0].ckks_parameter_id
            info['ckks_parameter_id_output'] = self.feature_output[0].ckks_parameter_id
            info['feature_input'] = [i.node_id for i in self.feature_input]
            info['feature_output'] = [i.node_id for i in self.feature_output]
        except Exception:
            info['type'] = self.layer_type
            info['channel_input'] = 0
            info['channel_output'] = 0
            # info["ckks_parameter_id_input"] = self.feature_input[0].ckks_parameter_id
            # info["ckks_parameter_id_output"] = self.feature_output[0].ckks_parameter_id
            info['feature_input'] = [i.node_id for i in self.feature_input]
            info['feature_output'] = [i.node_id for i in self.feature_output]
        return info

    def to_torch_code(self) -> dict[str, list[str]]:
        raise NotImplementedError()

    @staticmethod
    def get_attr_value_dict(node):
        attr_value_dict = {}
        for a in node.attribute:
            attr_value_dict[a.name] = onnx.helper.get_attribute_value(a)
        return attr_value_dict

    def gen_params_str(self, **kwargs):
        params = []
        for k, v in kwargs.items():
            v_str = v if isinstance(v, str) else v.__repr__()
            params.append(f"'{k}': {v_str}")
        return ', '.join(params).__repr__()[1:-1]

    def __str__(self):
        return self.__class__.__name__


def dict_to_args(d: dict) -> str:
    result: list[str] = list()
    for k, v in d.items():
        result.append(f'{k}={v}')
    result = ', '.join(result)
    return result


def get_op_id(op: str):
    if op == 'add2d' or op == 'Add':
        output = 'Add'
    elif op == 'conv2d' or op == 'Conv':
        output = 'Conv'
    elif op == 'batchnorm' or op == 'BatchNormalization':
        output = 'BatchNorm'
    elif op == 'square2d' or op == 'Pow':
        output = 'Square'
    elif op == 'relu2d' or op == 'Relu':
        output = 'Relu'
    elif op == 'Clip':
        output = 'Relu6'
    elif op == 'reshape' or op == 'Reshape' or op == 'Flatten':
        output = 'Reshape'
    elif op == 'avgpool' or op == 'avgpool2d' or op == 'AveragePool' or op == 'GlobalAveragePool':
        output = 'AveragePool'
    elif op == 'maxpool' or op == 'MaxPool':
        output = 'MaxPool'
    elif op == 'fc0' or op == 'Gemm':
        output = 'Dense'
    elif op == 'Dropout' or op == 'dropout':
        output = 'Dropout'
    elif op == 'Mul' or op == 'constmul':
        output = 'ConstMul'
    elif op == 'Sigmoid' or op == 'sigmoid':
        output = 'Sigmoid'
    elif op == 'AddMorph' or op == 'morph':
        output = 'AddMorph'
    elif op == 'mult_scalar' or op == 'mult_scalar':
        output = 'mult_scalar'
    elif op == 'drop_level' or op == 'drop_level':
        output = 'drop_level'
    elif op == 'poly_relu2d' or op == 'PolyReluListIndependent':
        output = 'PolyRelu'
    elif op == 'bootstrapping':
        output = 'Identity'
    elif op == 'ConvTranspose':
        output = 'ConvTranspose'
    elif op == 'Concat':
        output = 'Concat'
    elif op == 'HermitePoly' or op == 'identity':
        output = 'Identity'
    elif op == 'Softmax' or op == 'softmax':
        output = 'Softmax'
    elif op == 'Sigmoid' or op == 'sigmoid':
        output = 'Sigmoid'
    elif op == 'RangeNorm2d' or op == 'range_norm2d':
        output = 'RangeNorm2d'
    elif op == 'RangeNorm' or op == 'range_norm':
        output = 'RangeNorm'
    elif op == 'Simple_Polyrelu' or op == 'simple_polyrelu' or op == 'SimplePolyrelu' or op == 'RangeNormPoly2d':
        output = 'Simple_Polyrelu'
    elif op == 'Split' or op == 'split':
        output = 'Split'
    elif op == 'Resize' or op == 'resize':
        output = 'Resize'
    elif op == 'MatMul' or op == 'matmul':
        output = 'MatMul'
    elif op == 'RNN' or op == 'rnn':
        output = 'RNN'
    else:
        raise TypeError(f'Current operator {op} is not supported')
    return output


def get_type_id(op: str):
    if op == 'add2d' or op == 'Add':
        output = 'add2d'
    elif op == 'conv2d' or op == 'Conv':
        output = 'conv2d'
    elif op == 'batchnorm' or op == 'BatchNormalization':
        output = 'batchnorm'
    elif op == 'square2d' or op == 'Pow':
        output = 'square2d'
    elif op == 'relu2d' or op == 'Relu':
        output = 'relu2d'
    elif op == 'Clip':
        output = 'relu6'
    elif op == 'reshape' or op == 'Reshape' or op == 'Flatten':
        output = 'reshape'
    elif op == 'avgpool2d' or op == 'AveragePool' or op == 'avgpool' or op == 'GlobalAveragePool':
        output = 'avgpool'
    elif op == 'maxpool' or op == 'MaxPool':
        output = 'maxpool'
    elif op == 'fc0' or op == 'Gemm':
        output = 'fc0'
    elif op == 'Dropout' or op == 'dropout':
        output = 'dropout'
    elif op == 'Mul' or op == 'mul':
        output = 'mult_coeff'
    elif op == 'Sigmoid' or op == 'sigmoid':
        output = 'sigmoid'
    elif op == 'Sub':
        output = 'sub'
    elif op == 'poly_relu2d' or op == 'PolyReluListIndependent':
        output = 'poly_relu2d'
    elif op == 'ConvTranspose':
        output = 'conv2d'
    elif op == 'Concat':
        output = 'concat2d'
    elif op == 'HermitePoly':
        output = 'identity'
    elif op == 'Softmax':
        output = 'softmax'
    elif op == 'RangeNorm2d':
        output = 'range_norm2d'
    elif op == 'RangeNorm':
        output = 'range_norm'
    elif op == 'Simple_Polyrelu':
        output = 'simple_polyrelu'
    elif op == 'Split':
        output = 'split'
    elif op == 'Resize':
        output = 'resize'
    elif op == 'MatMul':
        output = 'matmul'
    elif op == 'MyRNN' or op == 'rnn':
        output = 'rnn'
    else:
        raise TypeError(f'Current operator {op} is not supported')
    return output


__op_gen_dict = {}


def get_op_code_generator(op: str, **kwargs):
    log.debug('get_op=%s', op)
    if op == 'Div':
        return
    op_code_gen_name = '{}ComputeNode'.format(get_op_id(op))
    mod = globals().get(get_op_id(op), None)
    __op_gen_dict[op_code_gen_name] = getattr(mod, op_code_gen_name)(**kwargs)
    return __op_gen_dict[op_code_gen_name]


def clear_op_code_generator():
    __op_gen_dict.clear()
