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

import json
from typing import Optional
import math
from ordered_set import OrderedSet
import numpy as np
import networkx as nx
from enum import Enum
import os
import copy


class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r', encoding='utf8') as f:
                config_dict = json.load(f)
            cls._instance.poly_n = config_dict.get('POLY_N', 65536)
            cls._instance.block_shape = config_dict.get('block_shape', (1, 1))
            cls._instance.graph_type = config_dict.get('GRAPH_TYPE', 'btp')
            cls._instance.style = config_dict.get('STYLE', 'multiplexed')
            cls._instance.mpc_refresh = config_dict.get('MPC_REFRESH', False)
            cls._instance.approx_poly_type = config_dict.get('APPROX_POLY_TYPE', 'simple_polyrelu')
            cls._instance.set_max_level = config_dict.get('SET_LEVEL_MAX', True)
            if cls._instance.mpc_refresh:
                cls._instance.absorbable_layers = ['conv2d', 'fc0', 'fc1', 'mult_scalar', 'simple_polyrelu']
            else:
                cls._instance.absorbable_layers = ['conv2d', 'fc0', 'fc1', 'mult_scalar']


        return cls._instance


config = GlobalConfig()


f_name_index_dict = dict()
concat_dict = dict()
# MAX_LEVEL = None
# block_shape = None
IS_ABSORB_POLYRELU = False
# POLY_N = None
# STYLE = None
# MPC_REFRESH = None

YOLO_TYPE = True
IS_BALANCE = False
DEFAULT_SCALE = 1


def get_multithread_rate_for_btp(task_num: int):
    if single_thread:
        return 1
    if task_num == 1:
        return 1
    elif task_num == 2:
        return 1.5
    elif task_num > 2 and task_num < 16:
        return task_num * 0.8
    elif task_num >= 16:
        return 12


def get_multithread_rate(task_num: int):
    if single_thread:
        return 1
    if task_num == 1:
        return 1
    elif task_num == 2:
        return 1.6
    elif task_num <= 4:
        return 2.8
    elif task_num <= 8:
        return 5.2
    elif task_num <= 16:
        return 8
    else:
        return 8


def get_multithread_rate_for_block_rotation(task_num: int):
    if single_thread:
        return 1
    if task_num == 1:
        return 1
    elif task_num == 2:
        return 1.2
    elif task_num <= 4:
        return 1.8
    elif task_num <= 8:
        return 2.7
    elif task_num <= 16:
        return 5.9
    else:
        return 5.9


def get_multithread_rate_for_kernel_rotation(task_num: int):
    if single_thread:
        return 1
    if task_num == 1:
        return 1
    elif task_num == 2:
        return 1.8
    elif task_num <= 4:
        return 2.7
    elif task_num <= 8:
        return 4
    elif task_num <= 16:
        return 6.5
    else:
        return 6.5


single_thread = False


def get_multithread_rate_for_weight_ops(task_num: int):
    if single_thread:
        return 1
    if task_num == 1:
        return 1
    elif task_num == 2:
        return 1.7
    elif task_num <= 4:
        return 3.5
    elif task_num <= 8:
        return 4.8
    elif task_num <= 16:
        return 6.1
    else:
        return 6.1


mult_plain_time = {
    65536: {
        1: 0.00279,
        2: 0.00428,
        3: 0.00582,
        4: 0.00726,
        5: 0.00732,
        6: 0.00849,
        7: 0.00982,
        8: 0.0112,
        9: 0.0119,
    },
    16384: {
        1: 0.000686,
        2: 0.001037,
        3: 0.001333,
        4: 0.001670,
        5: 0.002007,
        6: 0.002378,
        7: 0.002809,
        8: 0.003,
        9: 0.003,
    },
    8192: {1: 0.000261, 2: 0.000362, 3: 0.000465, 4: 0.000596, 5: 0.000833},
}

mult_time = {
    65536: {
        1: 0.004064,
        2: 0.004677,
        3: 0.005629,
        4: 0.006466,
        5: 0.009199,
        6: 0.010248,
        7: 0.011975,
        8: 0.013173,
        9: 0.014370,
    },
    16384: {
        1: 0.003216,
        2: 0.004368,
        3: 0.005244,
        4: 0.007070,
        5: 0.008787,
        6: 0.011128,
        7: 0.012594,
        8: 0.013,
        9: 0.013,
    },
    8192: {1: 0.001429, 2: 0.002002, 3: 0.002831, 4: 0.003705, 5: 0.004831},
}

rotate_time = {
    65536: {
        0: 0.0186,
        1: 0.0214,
        2: 0.0235,
        3: 0.0257,
        4: 0.029,
        5: 0.0315,
        6: 0.0402,
        7: 0.0444,
        8: 0.0551,
        9: 0.0599,
    },
    16384: {
        0: 0.00283,
        1: 0.003980,
        2: 0.006282,
        3: 0.007841,
        4: 0.011171,
        5: 0.013181,
        6: 0.017956,
        7: 0.020501,
        8: 0.02,
        9: 0.02,
    },
    8192: {0: 0.000582, 1: 0.000981, 2: 0.001466, 3: 0.00222, 4: 0.00276, 5: 0.003626},
}

rescale_time = {
    8192: {1: 0.00027, 2: 0.0004, 3: 0.00055, 4: 0.00065, 5: 0.00082},
    16384: {
        1: 0.00056,
        2: 0.00085143,
        3: 0.00113714,
        4: 0.00144699,
        5: 0.00172215,
        6: 0.00202571,
        7: 0.00231143,
        8: 0.002,
        9: 0.002,
    },
    65536: {
        1: 0.00196,
        2: 0.00298,
        3: 0.00398,
        4: 0.00506446,
        5: 0.00602752,
        6: 0.00709,
        7: 0.00809,
        8: 0.00913,
        9: 0.0101,
    },
}

add_time = {
    65536: {
        0: 0.000086,
        1: 0.000183,
        2: 0.000276,
        3: 0.000367,
        4: 0.000471,
        5: 0.00106,
        6: 0.00184,
        7: 0.0019,
        8: 0.002,
        9: 0.0021,
    },
    16384: {
        0: 0.00002,
        1: 0.00004,
        2: 0.00007,
        3: 0.00009,
        4: 0.0001,
        5: 0.00025,
        6: 0.0004,
        7: 0.0005,
        8: 0.0002,
        9: 0.0002,
    },
    8192: {0: 0.00009, 1: 0.001021, 2: 0.001466, 3: 0.002185, 4: 0.003026, 5: 0.003026},
}

btp_time = {'8192': 7, '16384': 12, '65536': 24}


# def read_config(config_path):
#     with open(config_path, 'r', encoding='utf8') as fp:
#         config_ctx = json.load(fp)
#     return config_ctx


class EncryptParameterNode:
    def __init__(
        self,
        poly_modulus_degree: int,
        coeff_modulus_bit_length: int,
        special_prime_bit_length: int,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_modulus_bit_length = coeff_modulus_bit_length
        self.special_prime_bit_length = special_prime_bit_length

    def __repr__(self) -> str:
        return (
            f'poly_modulus_degree: {self.poly_modulus_degree}, '
            + f'mult_level: {self.mult_level}, '
            + f'coeff_modulus_bit_length: {self.coeff_modulus_bit_length}, '
            + f'special_prime_bit_length: {self.special_prime_bit_length}, '
            + f'pack_num: {self.pack_num}'
        )


class FeatureNode:
    def __init__(
        self,
        key: str,
        dim: int,
        channel: int,
        scale: float = 1.0,
        ckks_parameter_id: str = 'param0',
        ckks_scale=DEFAULT_SCALE,
        shape: list = [1, 1],
    ):
        self.node_id = key
        self.dim = dim
        self.channel = channel
        self.scale = scale
        self.ckks_scale = ckks_scale
        self.shape = shape
        self.ckks_parameter_id = ckks_parameter_id
        self.node_index = -1
        self.depth = -1
        self.is_total_graph_leading_node = False
        self.scale_up = 1
        self.scale_down = 1

    def __repr__(self) -> str:
        return f'{self.node_id}'

    def to_json(self) -> json:
        info = dict()
        info['dim'] = self.dim
        info['channel'] = self.channel
        info['scale'] = self.scale
        if self.dim == 2:
            info['shape'] = self.shape

        if self.dim == 0:
            virtual_shape = getattr(self, 'virtual_shape', [1, 1])
            virtual_skip = getattr(self, 'virtual_skip', [1, 1])
            info['skip'] = virtual_shape[0] * virtual_shape[1] * virtual_skip[0] * virtual_skip[1]
            info['pack_num'] = math.ceil(POLY_N / 2 / info['skip'])

        info['ckks_parameter_id'] = self.ckks_parameter_id
        info['level'] = int(self.level)
        info['ckks_scale'] = self.ckks_scale
        return info


class ComputeNode:
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
    ):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.channel_input = channel_input
        self.channel_output = channel_output
        self.ckks_parameter_id_input = ckks_parameter_id_input
        self.ckks_parameter_id_output = ckks_parameter_id_output
        self.level = -1
        self.depth = 100
        self.is_end = False
        self.up_scale_str = list()
        self.down_scale_str = list()
        self.vec_scale_str = list()
        self.upsample_factor_in = [1, 1]
        self.is_big_size = False
        self.order = 0
        self.scale_up = 1
        self.scale_down = 1
        self.is_resize = False
        self.change_skip_to = 0
        self.weight_scale = 1
        self.bias_scale = 1
        self.weight_scale_list = [1, 1, 1, 1, 1]
        self.path = ''

    def __repr__(self) -> str:
        return f'ComputeNode: {self.layer_id}'


class ConvComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
        groups: int = 0,
        stride: list = [0, 0],
        kernel_shape: list = [0, 0],
        parameter_paths: dict | None = None,
        upsample_factor_in: list = [1, 1],
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.groups = groups
        self.upsample_factor_in = upsample_factor_in
        self.bn_absorb_path = ''
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths

        self.scale_up = 1
        self.scale_down = 1
        self.vec_scale_path = ''
        self.weight_scale = 1
        self.bias_scale = 1
        self.is_conv_transpose = False


class DenseComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
        parameter_paths: dict | None = None,
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths
        self.bn_absorb_path = ''
        self.scale_up = 1
        self.scale_down = 1
        self.vec_scale_path = ''
        self.weight_scale = 1
        self.bias_scale = 1


class BatchNormComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
        parameter_paths: dict | None = None,
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths


class UpsampleComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )

        self.upsample_factor_in = [1, 1]


class PoolComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
        stride: list = [0, 0],
        kernel_shape: list = [0, 0],
        is_adaptive_avgpool=False,
        padding=[0, 0],
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.is_adaptive_avgpool = is_adaptive_avgpool
        self.padding = padding


class MultScalarComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        self.scale = 1
        self.scale_up = 1
        self.scale_down = 1
        self.vec_scale_path = ''
        self.weight_scale = 1
        self.bias_scale = 1


class ReshapeComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        shape: list[int],
        ckks_parameter_id_input: str = 'param0',
        ckks_parameter_id_output: str = 'param0',
    ):
        super().__init__(
            layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
        )
        self.shape = shape


class LayerAbstractGraph:
    def __init__(self, parent_graph: Optional['LayerAbstractGraph'] = None):
        self.parent_graph = parent_graph
        self.dag = nx.DiGraph()
        self.compute_nodes_in_topo_sort = list()
        self.graph_id = None
        self.parent_graph_id = None

        self.order_key: dict[str, int] = dict()
        self.list_layer_name: list[str] = list()
        self.layer_order_list: list[str] = list()
        self.is_mpc = False
        self.leading_nodes = None

    def __repr__(self):
        result: str = ''
        result += '<<<\n'
        result += 'Features:\n'
        for feature in self.dag.nodes:
            if not isinstance(feature, FeatureNode):
                continue
            result += f'- {feature.node_id}, scale={feature.scale}\n'
        result += 'Computes:\n'
        for compute in self.dag.nodes:
            if not isinstance(compute, ComputeNode):
                continue
            result += f'- {compute.layer_id}, {compute.layer_type}, {list(self.dag.predecessors(compute))}, {list(self.dag.successors(compute))}\n'
        result += '>>>\n'
        return result

    def get_leading_feature_nodes(self) -> list[FeatureNode]:
        leading_feature_nodes = []
        for node, in_degree in self.dag.in_degree():
            if in_degree == 0 and isinstance(node, FeatureNode):
                if any(isinstance(next_node, ComputeNode) for next_node in self.dag.successors(node)):
                    leading_feature_nodes.append(node)
        return leading_feature_nodes

    def get_output_feature_nodes(self) -> list[FeatureNode]:
        outputs = []
        for node, out_degree in self.dag.out_degree():
            if out_degree == 0 and isinstance(node, FeatureNode):
                outputs.append(node)
        return outputs

    @staticmethod
    def from_json(json_path, is_fpga=False) -> 'LayerAbstractGraph':
        with open(json_path, 'r', encoding='utf8') as f:
            graph_json = json.load(f)

        graph_info = LayerAbstractGraph()
        feature_dict = dict()
        f_index = 0
        for key, feature_json in graph_json['feature'].items():
            groups = 0
            dim = feature_json['dim']
            channel = feature_json['channel']
            scale = feature_json['scale']
            ckks_parameter_id = feature_json['ckks_parameter_id']
            if dim == 2:
                shape = feature_json['shape']
                skip = [1, 1]
                virtual_skip = [1, 1]
                virtual_shape = [1, 1]
                node = FeatureNode(key, dim, channel, scale, ckks_parameter_id, DEFAULT_SCALE, shape)
            if dim == 0:
                shape = [0, 0]
                skip = [1, 0]
                virtual_skip = feature_json['virtual_skip']
                virtual_shape = feature_json['virtual_shape']
                node = FeatureNode(key, dim, channel, scale, ckks_parameter_id, DEFAULT_SCALE, shape)
            node.node_index = f_index

            f_name_index_dict[node.node_id] = f_index
            graph_info.dag.add_node(node, name=key, skip=skip, virtual_shape=virtual_shape, virtual_skip=virtual_skip)
            feature_dict[key] = node
            f_index = f_index + 1

        for key, layer_json in graph_json['layer'].items():
            graph_info.layer_order_list.append(key)
            layer_type = layer_json['type']
            stride = [1, 1]
            kernel_shape = [1, 1]
            skip = [1, 1]
            upsample_factor_in = [1, 1]
            ckks_parameter_id_input = layer_json['ckks_parameter_id_input']
            ckks_parameter_id_output = layer_json['ckks_parameter_id_output']
            channel_input = layer_json['channel_input']
            channel_output = layer_json['channel_output']

            feature_input = [
                feature_dict[layer_json['feature_input'][i]] for i in range(len(layer_json['feature_input']))
            ]
            feature_output = [
                feature_dict[layer_json['feature_output'][i]] for i in range(len(layer_json['feature_output']))
            ]

            running_mean_path = None
            running_var_path = None

            if 'batchnorm' in layer_type:
                if 'weight_path' in layer_json:
                    weight_path = layer_json['weight_path']
                else:
                    weight_path = ''
                if 'bias_path' in layer_json:
                    bias_path = layer_json['bias_path']
                else:
                    bias_path = ''
                running_mean_path = ''
                running_var_path = ''
                if 'running_mean_path' in layer_json:
                    running_mean_path = layer_json['running_mean_path']
                if 'running_var_path' in layer_json:
                    running_var_path = layer_json['running_var_path']

                compute_node = BatchNormComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    ckks_parameter_id_input,
                    ckks_parameter_id_output,
                    parameter_paths={
                        'weight': weight_path,
                        'bias': bias_path,
                        'running_mean': running_mean_path,
                        'running_var': running_var_path,
                    },
                )
            elif 'conv' in layer_type:
                weight_path = layer_json['weight_path']
                bias_path = layer_json['bias_path']
                kernel_shape = layer_json['kernel_shape']
                stride = layer_json['stride']
                groups = layer_json['groups']
                is_conv_transpose = False
                if 'upsample_factor_in' in layer_json and layer_json['upsample_factor_in'][0] != 1:
                    upsample_factor_in = layer_json['upsample_factor_in']
                    is_conv_transpose = True

                compute_node = ConvComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    ckks_parameter_id_input,
                    ckks_parameter_id_output,
                    groups,
                    stride,
                    kernel_shape,
                    parameter_paths={
                        'weight': weight_path,
                        'bias': bias_path,
                        'running_mean': running_mean_path,
                        'running_var': running_var_path,
                    },
                    upsample_factor_in=upsample_factor_in,
                )
                compute_node.is_conv_transpose = is_conv_transpose
            elif layer_type == 'resize':
                if 'upsample_factor_in' in layer_json:
                    upsample_factor_in = layer_json['upsample_factor_in']
                compute_node = ComputeNode(
                    key, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
                )
                compute_node.is_resize = True
                compute_node.upsample_factor_in = upsample_factor_in
            elif 'fc' in layer_type:
                weight_path = layer_json['weight_path']
                bias_path = layer_json['bias_path']
                compute_node = DenseComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    ckks_parameter_id_input,
                    ckks_parameter_id_output,
                    parameter_paths={
                        'weight': weight_path,
                        'bias': bias_path,
                        'running_mean': running_mean_path,
                        'running_var': running_var_path,
                    },
                )
            elif 'pool' in layer_type:
                kernel_shape = layer_json['kernel_shape']
                stride = layer_json['stride']
                if 'padding' in layer_json:
                    padding = layer_json['padding']
                else:
                    padding = [1, 1]
                if layer_type == 'avgpool':
                    layer_type = 'avgpool2d'
                compute_node = PoolComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    ckks_parameter_id_input,
                    ckks_parameter_id_output,
                    stride,
                    kernel_shape,
                    padding=padding,
                )
            elif 'mult_scalar' in layer_type:
                compute_node = MultScalarComputeNode(
                    key, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
                )
            elif 'reshape' in layer_type:
                compute_node = ReshapeComputeNode(key, layer_type, channel_input, channel_output, layer_json['shape'])
            else:
                compute_node = ComputeNode(
                    key, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
                )
                if 'concat2d' == layer_type:
                    concat_input_index_list = list()
                    for name in feature_input:
                        concat_input_index_list.append(f_name_index_dict[name.node_id])

                    concat_dict.update({key: concat_input_index_list})
                if 'simple_polyrelu' == layer_type or 'relu2d' == layer_type:
                    compute_node.path = layer_json['weight_path']
                    if 'order' in layer_json.keys():
                        compute_node.order = layer_json['order']
                    else:
                        compute_node.order = 0

            level_cost = 0
            if 'conv' in layer_type:
                level_cost = 1
            elif 'fc' in layer_type:
                level_cost = 1
            elif 'mult_scalar' in layer_type:
                level_cost = 1
            elif 'relu2d' == layer_type:
                level_cost = math.ceil(math.log2(compute_node.order)) + 1
            elif 'resize' == layer_type:
                level_cost = 1
            elif 'batchnorm' in layer_type or 'pool' in layer_type:
                level_cost = 0

            graph_info.dag.add_node(compute_node, name=key, level_cost=level_cost)
            graph_info.dag.add_edges_from([(node, compute_node) for node in feature_input])
            graph_info.dag.add_edges_from([(compute_node, node) for node in feature_output])

        if is_fpga:
            pack_dict = dict()
            level_init_list = dict()
            for key, layer_json in graph_json['ckks_parameter'].items():
                pack = layer_json['pack_num']
                pack_dict.update({key: pack})
                level = layer_json['n_mult_level']
                level_init_list.update({key: level})
            return (graph_info, pack_dict, level_init_list)
        return graph_info

    def to_json(
        self,
        param: dict[str, EncryptParameterNode],
        output_path: str | None,
        is_last_mpc=False,
        score=0.0,
    ) -> None:
        param_dict = dict()
        poly_to_mod = {8192: 31, 16384: 34, 65536: 41}
        mod_bits = poly_to_mod.get(config.poly_n, 41)
        param_dict.update(
            {
                'param0': {
                    'poly_modulus_degree': config.poly_n,
                    'n_mult_level': config.max_level,
                    'coeff_modulus_bit_length': mod_bits,
                    'special_prime_bit_length': mod_bits,
                    'pack_num': 4,
                }
            }
        )
        layers = dict()

        compute_list: list[ComputeNode] = list()
        if len([node for node in self.dag.nodes if isinstance(node, ComputeNode)]) == 1:
            compute_list = [node for node in self.dag.nodes if isinstance(node, ComputeNode)]
        else:
            if not nx.is_directed_acyclic_graph(self.dag):
                raise ValueError('Cycle exists in graph, cannot perform topological sort!')

            all_nodes_in_topo_sort = list(nx.topological_sort(self.dag))
            compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]
            sorted_compute_nodes = compute_nodes_in_topo_sort
            for node in sorted_compute_nodes:
                compute_list.append(node)

        conv_num = 0
        conv_list = list()
        conv_name = ''
        last_relu_id = ''
        mpc_refresh_ids = []

        i = 0

        for layer in compute_list:
            preds = list(self.dag.predecessors(layer))
            succs = list(self.dag.successors(layer))
            layer_id = layer.layer_id
            layer_type = layer.layer_type

            channel_input = layer.channel_input
            channel_output = layer.channel_output
            input_feature_ids = [feature.node_id for feature in preds]
            output_feature_ids = [feature.node_id for feature in succs]

            if 'conv' in layer_type or 'pool' in layer_type:
                kernel_shape = layer.kernel_shape
                stride = layer.stride

            ckks_parameter_id_input = layer.ckks_parameter_id_input
            ckks_parameter_id_output = layer.ckks_parameter_id_output

            if (
                'square' in layer_type
                or 'relu2d' == layer_type
                or 'simple_polyrelu' == layer_type
                or 'reshape' in layer_type
                or 'add' in layer_type
                or 'constmul' in layer_type
                or 'concat2d' == layer_type
                or 'identity' == layer_type
            ):
                odrder_id = list()
                if 'concat2d' == layer_type:
                    for value in concat_dict[layer_id]:
                        for node in preds:
                            if isinstance(node, FeatureNode):
                                if node.node_index == value:
                                    odrder_id.append(node.node_id)
                    input_feature_ids = odrder_id
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                }
            if 'resize' == layer_type:
                layers[layer_id] = {
                    'type': 'upsample_nearest',
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'upsample_factor_in': layer.upsample_factor_in,
                }
            if 'upsample' in layer_type:
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'upsample_factor_in': layer.upsample_factor_in,
                }

            if 'conv' in layer_type:
                upsample_factor_in = layer.upsample_factor_in

                base_string = layer.parameter_paths['weight'].rsplit('.', 1)[0] + '.bias'
                conv_num = conv_num + 1
                conv_name = layer.layer_id

                absorb_type = list()
                absorb_path = list()

                if layer.bn_absorb_path:
                    absorb_type.append('batchnorm')
                    absorb_path.append(layer.bn_absorb_path)
                if layer.vec_scale_path:
                    if not YOLO_TYPE:
                        absorb_type.append('simple_polyrelu')
                        absorb_path.append(layer.vec_scale_path)
                if not IS_BALANCE:
                    layer.weight_scale = layer.scale_up * layer.scale_down
                    layer.bias_scale = layer.scale_up

                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'kernel_shape': kernel_shape,
                    'groups': layer.groups,
                    'stride': stride,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'weight_path': layer.parameter_paths['weight'],
                    'bias_path': layer.parameter_paths['bias'],
                    'is_normal': '1',
                    'upsample_factor_in': upsample_factor_in,
                    'weight_scale': layer.weight_scale,
                    'bias_scale': layer.bias_scale,
                    'absorb_type': absorb_type,
                    'absorb_path': absorb_path,
                    'is_big_size': layer.is_big_size,
                }
            if 'pool' in layer_type:
                if 'avgpool' in layer_type:
                    layer_type = 'avgpool2d'
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'kernel_shape': kernel_shape,
                    'stride': stride,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'is_big_size': layer.is_big_size,
                    'is_adaptive_avgpool': layer.is_adaptive_avgpool,
                    'padding': layer.padding,
                }
            if (
                'sigmoid' in layer_type
                or 'softmax' in layer_type
                or 'argmax' in layer_type
                or 'bootstrapping' in layer_type
                or 'mpc_refresh' in layer_type
            ):
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                }
                if 'mpc_refresh' in layer_type:
                    layers[layer_id]['is_end'] = False
                    mpc_refresh_ids.append(layer_id)
            if 'fc' in layer_type:
                absorb_type = list()
                absorb_path = list()

                if layer.bn_absorb_path:
                    absorb_type.append('batchnorm')
                    absorb_path.append(layer.bn_absorb_path)
                if layer.vec_scale_path:
                    if not YOLO_TYPE:
                        absorb_type.append('simple_polyrelu')
                        absorb_path.append(layer.vec_scale_path)
                base_string = layer.parameter_paths['weight'].rsplit('.', 1)[0] + '.bias'
                layer_type = 'fc0'
                if not IS_BALANCE:
                    layer.weight_scale = layer.scale_up * layer.scale_down
                    layer.bias_scale = layer.scale_up
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'weight_path': layer.parameter_paths['weight'],
                    'bias_path': base_string,
                    'weight_scale': layer.weight_scale,
                    'bias_scale': layer.bias_scale,
                    'absorb_type': absorb_type,
                    'absorb_path': absorb_path,
                }
            if 'batchnorm' in layer_type:
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'weight_path': layer.parameter_paths['weight'],
                    'bias_path': layer.parameter_paths['bias'],
                    'running_mean_path': layer.parameter_paths['running_mean'],
                    'running_var_path': layer.parameter_paths['running_var'],
                }
            if 'dropout' in layer_type or 'drop_level' in layer_type:
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                }
            if 'mult_scalar' == layer_type:
                absorb_type = list()
                absorb_path = list()
                if layer.vec_scale_path:
                    absorb_type.append('simple_polyrelu')
                    absorb_path.append(layer.vec_scale_path)
                if not IS_BALANCE:
                    layer.weight_scale = layer.scale_up * layer.scale_down
                    layer.bias_scale = layer.scale_up
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'weight_scale': layer.weight_scale,
                    'bias_scale': layer.bias_scale,
                    'absorb_type': absorb_type,
                    'absorb_path': absorb_path,
                    'weight_path': layer_id + '.weight',
                    'weight_scale_list': layer.weight_scale_list,
                }

            if 'maxpool' == layer_type:
                last_relu_id = layer_id
                layers[layer_id]['is_end'] = False

            if 'relu2d' == layer_type:
                last_relu_id = layer_id
                conv_list.append(conv_name)
                layers[layer_id] = {
                    'type': layer_type,
                    'is_end': False,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                }

            if 'reshape' in layer_type:
                layers[layer_id] = {
                    'type': layer_type,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                    'shape': layer.shape,
                }
            if is_last_mpc:
                layers[layer_id]['is_end'] = True
        for layer in compute_list:
            layer_id = layer.layer_id
            layer_type = layer.layer_type
            if 'simple_polyrelu' == layer_type:
                # layers[layer_id]['weight_path'] = layer_id + '.weight'
                layers[layer_id]['weight_path'] = layer.path
                layers[layer_id]['upsample_factor_in'] = layer.upsample_factor_in
                layers[layer_id]['is_big_size'] = layer.is_big_size
                layers[layer_id]['is_absorb_polyrelu'] = IS_ABSORB_POLYRELU
            if 'level_cost' in self.dag.nodes[layer]:
                layers[layer_id]['level_cost'] = self.dag.nodes[layer]['level_cost']
            if 'simple_polyrelu' == layer.layer_type:
                layers[layer_id]['order'] = layer.order
                if not IS_BALANCE:
                    layer.weight_scale = layer.scale_up * layer.scale_down
                    layer.bias_scale = layer.scale_up

                layers[layer_id]['weight_scale'] = layer.weight_scale
                layers[layer_id]['weight_scale_list'] = layer.weight_scale_list
            if 'conv' in layer_type or 'fc' in layer_type:
                if hasattr(layer, 'is_conv_transpose') and layer.is_conv_transpose:
                    conv_type = 'transpose'
                elif layer.is_big_size:
                    conv_type = 'big_size'
                else:
                    conv_type = config.style
                layers[layer_id]['style'] = conv_type

        if mpc_refresh_ids:
            last_mpc_refresh_id = mpc_refresh_ids[-1]
            if last_mpc_refresh_id in layers:
                layers[last_mpc_refresh_id]['is_end'] = True
                print(f'Set the is_end of the last mpc_refresh node {last_mpc_refresh_id} to True')

        features = dict()
        all_nodes_in_topo_sort = list(nx.topological_sort(self.dag))
        f_index = 0
        for feature in all_nodes_in_topo_sort:
            if isinstance(feature, FeatureNode):
                key = feature.node_id
                dim = feature.dim
                channel = feature.channel

                scale = feature.scale
                ckks_scale = feature.ckks_scale
                shape = [int(item) for item in feature.shape]
                skip = [int(item) for item in self.dag.nodes[feature]['skip']]
                try:
                    virtual_shape = [int(item) for item in self.dag.nodes[feature]['virtual_shape']]
                    virtual_skip = [int(item) for item in self.dag.nodes[feature]['virtual_skip']]
                except Exception as e:
                    print(
                        f'Failed to get node virtual_skip attribute! Node ID: {feature.node_id}, Error type: {type(e).__name__}, Details: {e}'
                    )
                    raise

                ckks_parameter_id = feature.ckks_parameter_id
                level = self.dag.nodes[feature]['level']
                depth = feature.depth
                pack_num = self.dag.nodes[feature]['pack_num']
                if dim == 0:
                    features[key] = {
                        'dim': dim,
                        'channel': channel,
                        'scale': scale,
                        'ckks_scale': ckks_scale,
                        'skip': skip[0],
                        'ckks_parameter_id': ckks_parameter_id,
                        'virtual_shape': virtual_shape,
                        'virtual_skip': virtual_skip,
                        'level': level,
                        'depth': depth,
                        'pack_num': pack_num,
                    }
                elif dim == 2:
                    features[key] = {
                        'dim': dim,
                        'channel': channel,
                        'scale': scale,
                        'ckks_scale': ckks_scale,
                        'shape': shape,
                        'skip': skip,
                        'ckks_parameter_id': ckks_parameter_id,
                        'level': level,
                        'depth': depth,
                        'pack_num': pack_num,
                    }
                else:
                    raise ValueError('Unsupported dim value.')

        input_feature = [node.node_id for node in list(self.dag.predecessors(compute_list[0]))]
        output_feature = [node.node_id for node in list(self.dag.successors(compute_list[-1]))]
        config_info = {
            'score': score,
            'ckks_parameter': param_dict,
            'feature': features,
            'layer': layers,
            'mpc_parameter': {'param0': {'scale_ord': 18, 'rangep': 128, 'ring_mod': 17592186044416}},
            'input_feature': input_feature,
            'output_feature': output_feature,
        }

        if output_path is not None:
            with open(output_path, 'w') as f:
                json.dump(config_info, f, indent=4, ensure_ascii=False)
        else:
            print(json.dumps(config_info))


class FheScoreParam:
    def __init__(
        self, dag: nx.DiGraph, compute_node: ComputeNode, param: dict[str, EncryptParameterNode], level
    ) -> None:
        preds: list[FeatureNode] = list(dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(dag.successors(compute_node))

        self.dag = dag
        self.acc_rate = 1
        self.compute_node = compute_node
        self.input_mult_level = dag.nodes[preds[0]]['level']
        self.output_mult_level = dag.nodes[succs[0]]['level']
        self.input_degree = param[compute_node.ckks_parameter_id_input].poly_modulus_degree
        self.output_degree = param[compute_node.ckks_parameter_id_output].poly_modulus_degree
        if compute_node.layer_type == 'conv2d':
            self.stride = compute_node.stride
            self.kernel_shape = compute_node.kernel_shape
        if compute_node.layer_type == 'avgpool2d':
            self.stride = compute_node.stride
        if preds[0].dim == 2:
            self.input_shape = preds[0].shape
            self.output_shape = succs[0].shape
            self.input_skip = dag.nodes[preds[0]]['skip']
            self.output_skip = dag.nodes[succs[0]]['skip']
        else:
            self.input_shape = dag.nodes[preds[0]]['virtual_shape']
            self.output_shape = dag.nodes[succs[0]]['virtual_shape']
            self.input_skip = dag.nodes[preds[0]]['virtual_skip']
            self.output_skip = dag.nodes[succs[0]]['virtual_skip']

        self.pack = dag.nodes[preds[0]]['pack_num']
        self.pack_out = dag.nodes[succs[0]]['pack_num']

        self.input_channel = compute_node.channel_input
        self.output_channel = compute_node.channel_output
        self.n_packed_in = math.ceil(self.input_channel / self.pack)
        self.n_packed_out = math.ceil(self.output_channel / self.pack_out)
        if level > 0:
            self.mult_score = mult_time[self.input_degree][level]
            self.mult_plain_score = mult_plain_time[self.input_degree][level]
            self.rescale_score = rescale_time[self.input_degree][level]
        self.rotate_score = rotate_time[self.input_degree][level]
        self.add_score = add_time[self.input_degree][level]

    def get_score(self) -> float:
        compute_score = 0.0
        if 'conv' in self.compute_node.layer_type:
            if config.style == 'ordinary':
                if self.compute_node.groups == 1:
                    n_mult_and_add_score = (
                        (self.n_packed_in * self.pack * self.n_packed_out * self.kernel_shape[0] * self.kernel_shape[1])
                        * (self.mult_plain_score + self.add_score)
                        / get_multithread_rate(self.n_packed_out)
                    ) + self.n_packed_out * self.rescale_score / get_multithread_rate(self.n_packed_out)
                else:
                    n_mult_and_add_score = (
                        (self.n_packed_out * self.kernel_shape[0] * self.kernel_shape[1])
                        * (self.mult_plain_score + self.add_score)
                        / get_multithread_rate(self.n_packed_out)
                    ) + self.n_packed_out * self.rescale_score / get_multithread_rate(self.n_packed_out)

                if self.compute_node.groups == 1:
                    n_rotate_step_1 = self.n_packed_in * (self.pack - 1)
                    n_rotate_step_2 = (
                        self.n_packed_in
                        * self.pack
                        * (self.kernel_shape[0] - 1 + self.kernel_shape[0] * (self.kernel_shape[1] - 1))
                    )
                    n_rotate_score = (
                        n_rotate_step_1 / get_multithread_rate(self.n_packed_in)
                        + n_rotate_step_2 / get_multithread_rate(self.n_packed_in * self.pack)
                    ) * self.rotate_score
                else:
                    n_rotate_step = self.n_packed_in * (
                        self.kernel_shape[0] - 1 + self.kernel_shape[0] * (self.kernel_shape[1] - 1)
                    )
                    n_rotate_score = n_rotate_step / get_multithread_rate(self.n_packed_in) * self.rotate_score
            else:
                x_size = (
                    math.ceil(self.input_channel / self.pack)
                    * math.ceil(self.input_shape[0] / config.block_shape[0])
                    * math.ceil(self.input_shape[1] / config.block_shape[1])
                )

                n_block_per_ct = math.ceil(self.pack / (self.input_skip[0] * self.input_skip[1]))
                rotate_num1 = x_size / get_multithread_rate_for_block_rotation(x_size) * (n_block_per_ct - 1)
                rotated_size = x_size * n_block_per_ct
                rotate_num2 = (
                    rotated_size
                    / get_multithread_rate_for_kernel_rotation(rotated_size)
                    * (self.kernel_shape[0] * self.kernel_shape[1] - 1)
                )
                weight_size = math.ceil(self.output_channel / n_block_per_ct)

                if self.stride[0] != 1 and self.input_skip[0] != 1:
                    rotate_num3 = (
                        weight_size
                        / get_multithread_rate_for_weight_ops(weight_size)
                        * (math.log2(self.input_skip[0]) * 2 + n_block_per_ct)
                    )
                else:
                    rotate_num3 = (
                        weight_size
                        / get_multithread_rate_for_weight_ops(weight_size)
                        * (math.log2(self.input_skip[0]))
                        * 2
                    )

                n_in_ct = math.ceil(self.input_channel / self.pack)
                n_out_ct = math.ceil(self.output_channel / self.pack_out)
                if self.stride[0] != 1:
                    mult_num = weight_size * (
                        n_in_ct * n_block_per_ct * self.kernel_shape[0] * self.kernel_shape[1] + n_block_per_ct
                    )
                else:
                    mult_num = weight_size * (n_in_ct * n_block_per_ct * self.kernel_shape[0] * self.kernel_shape[1])
                mult_num = mult_num / get_multithread_rate_for_weight_ops(weight_size)
                add_num = (
                    weight_size
                    / get_multithread_rate_for_weight_ops(weight_size)
                    * (
                        (math.log2(self.input_skip[0])) * 2
                        + n_in_ct * n_block_per_ct * self.kernel_shape[0] * self.kernel_shape[1]
                    )
                    + n_out_ct
                )

                n_rescale_score = (
                    weight_size / get_multithread_rate_for_weight_ops(weight_size) * n_block_per_ct * self.rescale_score
                )
                n_mult_and_add_score = mult_num * self.mult_plain_score + add_num * self.add_score + n_rescale_score
                n_rotate_score = (rotate_num1 + rotate_num2 + rotate_num3) * self.rotate_score
            compute_score = n_mult_and_add_score + n_rotate_score

            return compute_score * self.acc_rate
        elif 'fc' in self.compute_node.layer_type:
            if STYLE == 'ordinary':
                pred_node = list(self.dag.predecessors(self.compute_node))[0]
                num = math.log2(self.dag.nodes[pred_node]['skip'][0])
                n_mult_plain_score = (self.n_packed_in * self.pack * self.n_packed_out) * self.mult_plain_score
                n_add_score = (self.n_packed_in * (self.pack - 1) + self.n_packed_out * num) * self.add_score
                n_rotate_score = (
                    self.n_packed_in * (self.pack - 1) / get_multithread_rate(self.n_packed_in)
                    + self.n_packed_out * num
                ) * self.rotate_score
                compute_score = n_mult_plain_score + n_add_score + n_rotate_score
                return compute_score * self.acc_rate
            else:
                n_block_input = math.ceil(self.input_channel / (self.input_skip[0] * self.input_skip[1]))
                n_num_pre_ct = (
                    self.input_degree
                    / 2
                    / (self.input_skip[0] * self.input_skip[1] * self.input_shape[0] * self.input_shape[1])
                )

                n_packed_out_feature_for_mult_pack = math.ceil(self.output_channel / n_num_pre_ct)
                x_size = math.ceil(self.input_channel / self.pack)
                acc_rate = get_multithread_rate(n_packed_out_feature_for_mult_pack)
                rot_time = (n_block_input - 1) * x_size + n_packed_out_feature_for_mult_pack / acc_rate * (
                    math.log2(self.input_shape[0] * self.input_shape[1] * self.input_skip[0] * self.input_skip[1])
                )
                mult_time = n_packed_out_feature_for_mult_pack / acc_rate * (x_size * n_block_input)

                add_time = n_packed_out_feature_for_mult_pack / acc_rate * (x_size * n_block_input + 1)

                rescale_time = n_packed_out_feature_for_mult_pack / acc_rate
                return (
                    rot_time * self.rotate_score
                    + mult_time * self.mult_plain_score
                    + add_time * self.add_score
                    + rescale_time * self.rescale_score
                )
        elif 'simple_polyrelu' in self.compute_node.layer_type:
            compute_score = (self.n_packed_in * (self.mult_score + self.add_score)) * (
                math.ceil(math.log2(self.compute_node.order)) + 1
            )
            return compute_score * self.acc_rate
        elif 'avgpool2d' == self.compute_node.layer_type:
            num = self.n_packed_in * (self.stride[0] - 1 + math.log2(self.stride[0]))
            n_add_score = num * self.add_score
            n_rotate_score = num * self.rotate_score
            compute_score = n_add_score + n_rotate_score
            return compute_score * self.acc_rate
        elif 'add2d' == self.compute_node.layer_type:
            n_add_score = self.n_packed_in * self.add_score
            compute_score = n_add_score
            return compute_score * self.acc_rate


mpc_refresh_rate = 1 / 15
ct_trans_rate = 1 / 10


class MpcScoreParam:
    def __init__(
        self,
        dag: LayerAbstractGraph,
        compute_node: ComputeNode,
        param: dict[str, EncryptParameterNode],
        bit_len=44,
        mpc_scale=16,
    ) -> None:
        graph = LayerAbstractGraph()
        graph.dag = dag
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(graph.dag.successors(compute_node))
        compute_node.ckks_parameter_id_input = preds[0].ckks_parameter_id
        compute_node.ckks_parameter_id_output = succs[0].ckks_parameter_id
        self.preds = preds
        self.succs = succs
        self.compute_node = compute_node
        self.input_coeff_mod = param[compute_node.ckks_parameter_id_input].coeff_modulus_bit_length
        self.output_coeff_mod = param[compute_node.ckks_parameter_id_output].coeff_modulus_bit_length
        self.input_special_mod = param[compute_node.ckks_parameter_id_input].special_prime_bit_length
        self.output_special_mod = param[compute_node.ckks_parameter_id_output].special_prime_bit_length
        self.input_mult_level = graph.dag.nodes[preds[0]]['level']
        self.output_mult_level = graph.dag.nodes[succs[0]]['level']
        self.input_degree = param[compute_node.ckks_parameter_id_input].poly_modulus_degree
        self.output_degree = param[compute_node.ckks_parameter_id_output].poly_modulus_degree
        MB_scale = 2**23
        self.relu_score = bit_len * mpc_scale / MB_scale
        self.input_ct_score = (88 + self.input_coeff_mod * self.input_mult_level) * self.input_degree * 2 / MB_scale
        self.output_ct_score = (
            (88 + self.output_coeff_mod * self.output_mult_level) * self.output_degree * 1.5 / MB_scale
        )

        self.input_channel = compute_node.channel_input
        self.output_channel = compute_node.channel_output
        if (not preds[0].shape) and (not graph.dag.nodes[preds[0]]['skip']):
            input_shape = preds[0].shape
            output_shape = succs[0].shape
            input_skip = graph.dag.nodes[preds[0]]['skip']
            output_skip = graph.dag.nodes[succs[0]]['skip']
        else:
            input_shape = graph.dag.nodes[preds[0]]['virtual_shape']
            output_shape = graph.dag.nodes[succs[0]]['virtual_shape']
            input_skip = graph.dag.nodes[preds[0]]['virtual_skip']
            output_skip = graph.dag.nodes[succs[0]]['virtual_skip']

        self.input_channel = compute_node.channel_input
        self.output_channel = compute_node.channel_output
        temp_num_in = input_shape[0] * input_shape[1] * input_skip[0] * input_skip[1]
        temp_num_out = output_shape[0] * output_shape[1]

        self.n_packed_in = math.ceil(self.input_channel * temp_num_in / self.input_degree / 2)
        self.n_packed_out = math.ceil(self.output_channel * temp_num_out / self.input_degree / 2)

    def get_score(self) -> float:
        if 'relu2d' in self.compute_node.layer_type or 'pool' in self.compute_node.layer_type:
            if 'relu2d' == self.compute_node.layer_type or MPC_REFRESH:
                kernel_scale = 1
            elif 'pool' in self.compute_node.layer_type:
                kernel_scale = self.compute_node.kernel_shape[0] * self.compute_node.kernel_shape[1]
            shape = self.preds[0].shape
            n_relu_score = self.input_channel * shape[0] * shape[1] * self.relu_score / kernel_scale * mpc_refresh_rate
            n_ct_score = (
                self.n_packed_in * self.input_ct_score + self.n_packed_out * self.output_ct_score
            ) * ct_trans_rate
            return n_relu_score + n_ct_score
        if 'bootstrapping' in self.compute_node.layer_type and MPC_REFRESH:
            shape = self.preds[0].shape
            n_ct_score = (
                self.n_packed_in * self.input_ct_score + self.n_packed_out * self.output_ct_score
            ) * ct_trans_rate
            n_mpc_refresh = self.input_channel * shape[0] * shape[1] * self.relu_score * mpc_refresh_rate
            return n_ct_score + n_mpc_refresh


class BtpScoreParam:
    def __init__(self, dag: nx.DiGraph, compute_node: ComputeNode, param: dict[str, EncryptParameterNode]) -> None:
        graph = LayerAbstractGraph()
        graph.dag = dag
        pred = list(graph.dag.predecessors(compute_node))[0]
        self.n = param[pred.ckks_parameter_id].poly_modulus_degree
        btp_slot = int(self.n / 2)

        self.ct_num = math.ceil(pred.shape[0] * pred.shape[1] * compute_node.channel_input / btp_slot)

    def get_score(self):
        score = self.ct_num * btp_time[str(self.n)] / get_multithread_rate_for_btp(self.ct_num)
        return score


if __name__ == '__main__':
    print()
