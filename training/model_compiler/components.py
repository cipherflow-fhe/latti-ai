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


# Class hierarchy overview:
#
#   FheParameter                      – CKKS encryption parameters (degree, modulus, …); held by GlobalConfig
#
#   GlobalConfig                      – singleton for compiler configuration; exposes fhe_param,
#                                       poly_n, and block_shape (the latter two as properties)
#
#   FeatureNode                       – a ciphertext tensor (node between compute layers)
#
#   ComputeNode                       – base class for all compute/layer nodes
#   ├── SpatialComputeNode            – layers with stride / upsample geometry
#   │   ├── ConvComputeNode           – convolution (regular and transposed)
#   │   ├── UpsampleComputeNode       – nearest-neighbour upsample (split from ConvTranspose)
#   │   ├── UpsampleNearestComputeNode– upsample from a 'resize' op in the original graph
#   │   └── PoolComputeNode           – average / adaptive pooling
#   ├── DenseComputeNode              – fully-connected layer
#   ├── BatchNormComputeNode          – batch normalisation
#   ├── MultScalarComputeNode         – element-wise scalar multiply
#   ├── MultCoeffComputeNode          – multiply by a fixed scalar coefficient
#   ├── ReshapeComputeNode            – reshape / flatten
#   └── ActivationComputeNode         – activation functions (simple_polyrelu, relu2d,
#                                       square, sigmoid)
#
#   LayerAbstractGraph                – DAG of FeatureNode / ComputeNode; owns from_json
#                                       and to_json for the task config format

import json
import math
import networkx as nx
import os


class FheParameter:
    def __init__(
        self,
        poly_modulus_degree: int,
        n_mult_level: int,
        coeff_modulus_bit_length: int,
        block_shape: list | None = None,
    ):
        self.poly_modulus_degree = poly_modulus_degree
        self.max_level = n_mult_level
        self.coeff_modulus_bit_length = coeff_modulus_bit_length
        self.special_prime_bit_length = coeff_modulus_bit_length
        self.block_shape = block_shape

    def to_dict(self) -> dict:
        return {
            'poly_modulus_degree': self.poly_modulus_degree,
            'n_mult_level': self.max_level,
            'coeff_modulus_bit_length': self.coeff_modulus_bit_length,
            'special_prime_bit_length': self.special_prime_bit_length,
            'block_shape': self.block_shape,
        }

    def __repr__(self) -> str:
        return (
            f'FheParameter(poly_modulus_degree={self.poly_modulus_degree}, '
            f'n_mult_level={self.max_level}, '
            f'coeff_modulus_bit_length={self.coeff_modulus_bit_length}, '
            f'special_prime_bit_length={self.special_prime_bit_length}, '
            f'block_shape={self.block_shape})'
        )


class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r', encoding='utf8') as f:
                config_dict = json.load(f)
            # fhe_param will be overwritten by initialize_config() before first use
            cls._instance.fhe_param = FheParameter(
                poly_modulus_degree=65536,
                n_mult_level=0,
                coeff_modulus_bit_length=0,
                block_shape=(1, 1),
            )
            cls._instance.graph_type = config_dict.get('GRAPH_TYPE', 'btp')
            cls._instance.style = config_dict.get('STYLE', 'multiplexed')
            cls._instance.mpc_refresh = config_dict.get('MPC_REFRESH', False)
            cls._instance.approx_poly_type = config_dict.get('APPROX_POLY_TYPE', 'simple_polyrelu')
            cls._instance.set_max_level = config_dict.get('SET_LEVEL_MAX', True)
            cls._instance.absorbable_layers = ['conv2d', 'fc0', 'fc1', 'mult_scalar', 'simple_polyrelu']

        return cls._instance


config = GlobalConfig()


IS_ABSORB_POLYRELU = False
YOLO_TYPE = True
IS_BALANCE = False
DEFAULT_SCALE = 1
single_thread = False


class FeatureNode:
    def __init__(
        self,
        key: str,
        dim: int,
        channel: int,
        scale: float = 1.0,
        ckks_parameter_id: str = 'param0',
        ckks_scale=DEFAULT_SCALE,
        shape: list = None,
    ):
        if shape is None:
            shape = [1, 1]
        self.node_id = key
        self.dim = dim  # number of spatial dimensions
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
        if self.dim in (1, 2):
            info['shape'] = self.shape

        if self.dim == 0:
            virtual_shape = getattr(self, 'virtual_shape', [1, 1])
            virtual_skip = getattr(self, 'virtual_skip', [1, 1])
            info['skip'] = virtual_shape[0] * virtual_shape[1] * virtual_skip[0] * virtual_skip[1]
            info['pack_num'] = math.ceil(config.fhe_param.poly_modulus_degree / 2 / info['skip'])

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
    ):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.channel_input = channel_input
        self.channel_output = channel_output
        self.depth = 100
        self.is_end = False
        self.up_scale_str = list()
        self.down_scale_str = list()
        self.vec_scale_str = list()
        self.is_big_size = False
        self.order = 0
        self.scale_up = 1
        self.scale_down = 1
        self.change_skip_to = 0
        self.weight_scale = 1
        self.bias_scale = 1
        self.weight_scale_list = [1, 1, 1, 1, 1]
        self.path = ''

    def __repr__(self) -> str:
        return f'ComputeNode: {self.layer_id}'


class SpatialComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        dim: int = 2,
        stride: list = None,
        upsample_factor: list = None,
        upsample_factor_in: list = None,
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        self.dim = dim
        if stride is None:
            stride = [1] * dim
        if upsample_factor is None:
            upsample_factor = [1] * dim
        if upsample_factor_in is None:
            upsample_factor_in = [1] * dim
        if len(stride) != dim or len(upsample_factor) != dim or len(upsample_factor_in) != dim:
            raise ValueError(
                f'stride, upsample_factor, and upsample_factor_in must all have length dim={dim}, '
                f'got {len(stride)}, {len(upsample_factor)}, {len(upsample_factor_in)}'
            )
        self.stride = stride
        self.upsample_factor = upsample_factor  # the "stride" from ConvTranpose
        self.upsample_factor_in = upsample_factor_in  # absorbed from some downstream upsampling layer


class ConvComputeNode(SpatialComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        dim: int = 2,
        stride: list = None,
        upsample_factor: list = None,
        upsample_factor_in: list = None,
        groups: int = 0,
        kernel_shape: list = None,
        parameter_paths: dict | None = None,
    ):
        super().__init__(
            layer_id,
            layer_type,
            channel_input,
            channel_output,
            dim=dim,
            stride=stride,
            upsample_factor=upsample_factor,
            upsample_factor_in=upsample_factor_in,
        )
        if kernel_shape is None:
            kernel_shape = [1] * self.dim
        self.kernel_shape = kernel_shape
        self.groups = groups
        self.bn_absorb_path = ''
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths
        self.vec_scale_path = ''
        self.is_conv_transpose = False


class DenseComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        parameter_paths: dict | None = None,
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths
        self.bn_absorb_path = ''
        self.vec_scale_path = ''


class BatchNormComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        parameter_paths: dict | None = None,
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        if parameter_paths is None:
            self.parameter_paths = dict()
        else:
            self.parameter_paths = parameter_paths


class UpsampleComputeNode(SpatialComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        dim: int = 2,
        stride: list = None,
        upsample_factor: list = None,
        upsample_factor_in: list = None,
    ):
        super().__init__(
            layer_id,
            layer_type,
            channel_input,
            channel_output,
            dim=dim,
            stride=stride,
            upsample_factor=upsample_factor,
            upsample_factor_in=upsample_factor_in,
        )


class UpsampleNearestComputeNode(SpatialComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        dim: int = 2,
        stride: list = None,
        upsample_factor: list = None,
        upsample_factor_in: list = None,
    ):
        super().__init__(
            layer_id,
            layer_type,
            channel_input,
            channel_output,
            dim=dim,
            stride=stride,
            upsample_factor=upsample_factor,
            upsample_factor_in=upsample_factor_in,
        )


class PoolComputeNode(SpatialComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        dim: int = 2,
        stride: list = None,
        upsample_factor: list = None,
        upsample_factor_in: list = None,
        kernel_shape: list = None,
        is_adaptive_avgpool=False,
        padding=[0, 0],
    ):
        super().__init__(
            layer_id,
            layer_type,
            channel_input,
            channel_output,
            dim=dim,
            stride=stride,
            upsample_factor=upsample_factor,
            upsample_factor_in=upsample_factor_in,
        )
        if kernel_shape is None:
            kernel_shape = [1] * self.dim
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
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        self.scale = 1
        self.vec_scale_path = ''


class MultCoeffComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        coeff: float,
        channel_input: int,
        channel_output: int,
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        self.coeff = coeff


class ReshapeComputeNode(ComputeNode):
    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
        *,
        new_shape: list[int],
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        self.new_shape = new_shape


class ActivationComputeNode(ComputeNode):
    """Represents activation layers: simple_polyrelu, relu2d, square, sigmoid."""

    def __init__(
        self,
        layer_id: str,
        layer_type: str,
        channel_input: int,
        channel_output: int,
    ):
        super().__init__(layer_id, layer_type, channel_input, channel_output)
        self.zero_skip = [1, 1]


class LayerAbstractGraph:
    def __init__(self):
        self.dag = nx.DiGraph()
        self.is_mpc = False

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

    @staticmethod
    def from_json(json_path, is_fpga=False) -> 'LayerAbstractGraph':
        with open(json_path, 'r', encoding='utf8') as f:
            graph_json = json.load(f)

        graph_info = LayerAbstractGraph()
        feature_dict = dict()
        f_index = 0
        for key, feature_json in graph_json['feature'].items():
            dim = feature_json['dim']
            channel = feature_json['channel']
            scale = 1.0
            ckks_parameter_id = feature_json['ckks_parameter_id']
            if dim in (1, 2):
                shape = feature_json['shape']
                skip = [1] * dim
                virtual_skip = [1] * dim
                virtual_shape = [1] * dim
                node = FeatureNode(key, dim, channel, scale, ckks_parameter_id, DEFAULT_SCALE, shape)
            elif dim == 0:
                shape = [0, 0]
                skip = [1, 0]
                virtual_skip = feature_json['virtual_skip']
                virtual_shape = feature_json['virtual_shape']
                node = FeatureNode(key, dim, channel, scale, ckks_parameter_id, DEFAULT_SCALE, shape)
            else:
                raise ValueError(f'Unsupported feature dim: {dim}')
            node.node_index = f_index

            graph_info.dag.add_node(node, name=key, skip=skip, virtual_shape=virtual_shape, virtual_skip=virtual_skip)
            feature_dict[key] = node
            f_index = f_index + 1

        for key, layer_json in graph_json['layer'].items():
            layer_type = layer_json['type']
            channel_input = layer_json['channel_input']
            channel_output = layer_json['channel_output']

            feature_input = [feature_dict[fid] for fid in layer_json['feature_input']]
            feature_output = [feature_dict[fid] for fid in layer_json['feature_output']]

            running_mean_path = None
            running_var_path = None

            if 'batchnorm' in layer_type:
                weight_path = layer_json.get('weight_path', '')
                bias_path = layer_json.get('bias_path', '')
                running_mean_path = layer_json.get('running_mean_path', '')
                running_var_path = layer_json.get('running_var_path', '')

                compute_node = BatchNormComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
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
                dim = feature_input[0].dim
                upsample_factor = layer_json.get('upsample_factor', [1] * dim)
                is_conv_transpose = False
                if 'upsample_factor' in layer_json and layer_json['upsample_factor'][0] != 1:
                    upsample_factor = layer_json['upsample_factor']
                    is_conv_transpose = True

                compute_node = ConvComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    dim=dim,
                    groups=groups,
                    stride=stride,
                    kernel_shape=kernel_shape,
                    parameter_paths={
                        'weight': weight_path,
                        'bias': bias_path,
                        'running_mean': running_mean_path,
                        'running_var': running_var_path,
                    },
                    upsample_factor=upsample_factor,
                )
                compute_node.is_conv_transpose = is_conv_transpose

            elif layer_type == 'resize':
                if 'upsample_factor' in layer_json:
                    upsample_factor = layer_json['upsample_factor']
                compute_node = UpsampleNearestComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    upsample_factor=upsample_factor,
                )

            elif 'fc' in layer_type:
                weight_path = layer_json['weight_path']
                bias_path = layer_json['bias_path']
                compute_node = DenseComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    parameter_paths={
                        'weight': weight_path,
                        'bias': bias_path,
                    },
                )

            elif 'pool' in layer_type:
                kernel_shape = layer_json['kernel_shape']
                stride = layer_json['stride']
                padding = layer_json.get('padding', [1, 1])
                if layer_type == 'avgpool':
                    layer_type = 'avgpool2d'
                compute_node = PoolComputeNode(
                    key,
                    layer_type,
                    channel_input,
                    channel_output,
                    stride=stride,
                    kernel_shape=kernel_shape,
                    padding=padding,
                )

            elif 'mult_scalar' in layer_type:
                compute_node = MultScalarComputeNode(key, layer_type, channel_input, channel_output)

            elif 'reshape' in layer_type:
                compute_node = ReshapeComputeNode(
                    key, layer_type, channel_input, channel_output, new_shape=layer_json['shape'][1:]
                )

            elif layer_type == 'mult_coeff':
                compute_node = MultCoeffComputeNode(key, layer_type, layer_json['coeff'], channel_input, channel_output)

            elif layer_type in ('simple_polyrelu', 'relu2d', 'square', 'sigmoid'):
                if layer_type == 'relu2d' and not config.mpc_refresh:
                    raise ValueError('Relu2d is not supported in current mode')
                compute_node = ActivationComputeNode(key, layer_type, channel_input, channel_output)
                if layer_type in ('simple_polyrelu', 'relu2d'):
                    compute_node.path = layer_json.get('weight_path', '')
                    compute_node.order = layer_json.get('order', 0)

            else:
                compute_node = ComputeNode(key, layer_type, channel_input, channel_output)

            graph_info.dag.add_node(compute_node, name=key)
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
        param: dict[str, FheParameter],
        output_path: str | None,
        is_last_mpc=False,
        score=0.0,
    ) -> None:
        param_dict = dict()
        param_dict['param0'] = {**config.fhe_param.to_dict(), 'pack_num': 4}
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
        mpc_refresh_ids = []

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

            ckks_parameter_id_input = preds[0].ckks_parameter_id
            ckks_parameter_id_output = succs[0].ckks_parameter_id

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
                if 'concat2d' == layer_type:
                    # The ordering of concat2d inputs is recovered from node_index on the FeatureNodes, which is set at parse time.
                    input_feature_ids = [n.node_id for n in sorted(preds, key=lambda n: n.node_index)]
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
                    'upsample_factor': layer.upsample_factor,
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
                    'upsample_factor': layer.upsample_factor,
                }

            if 'conv' in layer_type:
                upsample_factor_in = layer.upsample_factor_in

                base_string = layer.parameter_paths['weight'].rsplit('.', 1)[0] + '.bias'
                conv_num = conv_num + 1

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
                layers[layer_id]['is_end'] = False

            if 'relu2d' == layer_type:
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
                    'shape': layer.new_shape,
                }
            if layer_type == 'mult_coeff':
                layers[layer_id] = {
                    'type': layer_type,
                    'coeff': layer.coeff,
                    'channel_input': channel_input,
                    'channel_output': channel_output,
                    'ckks_parameter_id_input': ckks_parameter_id_input,
                    'ckks_parameter_id_output': ckks_parameter_id_output,
                    'feature_input': input_feature_ids,
                    'feature_output': output_feature_ids,
                }
            if is_last_mpc:
                layers[layer_id]['is_end'] = True
        for layer in compute_list:
            layer_id = layer.layer_id
            layer_type = layer.layer_type
            if 'simple_polyrelu' == layer_type:
                # layers[layer_id]['weight_path'] = layer_id + '.weight'
                layers[layer_id]['weight_path'] = layer.path
                layers[layer_id]['zero_skip'] = layer.zero_skip
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
                elif dim in (1, 2):
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


if __name__ == '__main__':
    print()
