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

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.activation_layer import *
from inference.model_generator.layers.add_pack import *
from inference.model_generator.layers.avgpool2d_layer import *
from inference.model_generator.layers.concat_layer import *
from inference.model_generator.layers.conv1d_packed_layer import *
from inference.model_generator.layers.conv2d_depthwise import *
from inference.model_generator.layers.conv2d_packed_layer import *
from inference.model_generator.layers.dense_packed_layer import *
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import *
from inference.model_generator.layers.mult_scaler import *
from inference.model_generator.layers.multiplexed_conv1d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer import *
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import *
from inference.model_generator.layers.poly_relu2d import *
from inference.model_generator.layers.upsample_layer import *
from training.model_compiler.components import (
    N16QP1546H192H32,
    PN13QP218,
    PN14QP438,
    PN15QP880,
    PN16QP1761,
)


def read_config(config_path):
    with open(config_path, 'r', encoding='utf8') as fp:
        config_ctx = json.load(fp)
    return config_ctx


_FHE_PARAMS = {
    'PN13QP218': PN13QP218,
    'PN14QP438': PN14QP438,
    'PN15QP880': PN15QP880,
    'PN16QP1761': PN16QP1761,
    'N16QP1546H192H32': N16QP1546H192H32,
}


def set_param(param_name):
    if param_name not in _FHE_PARAMS:
        raise ValueError(f'Unsupported FHE parameter name: {param_name!r}')
    fhe = _FHE_PARAMS[param_name]
    if param_name == 'N16QP1546H192H32':
        param = CkksBtpParam.create_default_param()
    else:
        param = Param.create_ckks_custom_param(n=fhe.poly_modulus_degree, p=fhe.p, q=fhe.q)
    set_fhe_param(param)


def gen_custom_task(task_path, param_name='PN14QP438', use_gpu=True, style='ordinary'):
    n = _FHE_PARAMS[param_name].poly_modulus_degree
    set_param(param_name)
    task_config_info = read_config(os.path.join(task_path, 'task_config.json'))
    try:
        block_shape = task_config_info['block_shape']
    except Exception:
        block_shape = [64, 64]
    config_info = read_config(os.path.join(task_path, 'nn_layers_ct_0.json'))
    input_args = list()
    feature_id_to_nodes_map = {}
    task_output_feature_ids = config_info['output_feature']

    for layer_id, layer_config in config_info['layer'].items():
        if layer_config['type'] == 'relu2d':
            continue
        layer_input_feature_ids = layer_config['feature_input']
        layer_output_feature_ids = layer_config['feature_output']
        groups = 1
        n_in_channel = int(layer_config['channel_input'])
        n_out_channel = int(layer_config['channel_output'])

        skip = config_info['feature'][layer_input_feature_ids[0]]['skip']
        pack = int(config_info['feature'][layer_input_feature_ids[0]]['pack_num'])
        level = int(config_info['feature'][layer_input_feature_ids[0]]['level'])
        n_packed_in_channel = math.ceil(n_in_channel / pack)
        n_packed_out_channel = math.ceil(n_out_channel / pack)
        if layer_config['type'] == 'fc0':
            n_packed_in_channel = math.ceil(n_in_channel / 8192)
            n_packed_out_channel = math.ceil(n_out_channel / pack)

        # For big_conv/big_avgpool, input CT count differs from the default n_packed_in_channel
        if (layer_config['type'] == 'conv2d' or 'avgpool' in layer_config['type']) and layer_config.get(
            'is_big_size', False
        ):
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            block_expansion = (
                math.ceil(input_shape[0] / block_shape[0]),
                math.ceil(input_shape[1] / block_shape[1]),
            )
            n_packed_in_channel = n_in_channel * block_expansion[0] * block_expansion[1]

        for input_node in layer_input_feature_ids:
            if input_node not in feature_id_to_nodes_map.keys():
                x = [CkksCiphertextNode(input_node + f'input{k}', level=level) for k in range(n_packed_in_channel)]
                feature_id_to_nodes_map.update({input_node: x})
                input_args.append(Argument(input_node, x))

        if layer_config['type'] == 'reshape':
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'conv2d':
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            groups = layer_config['groups']
            kernel_shape = layer_config['kernel_shape']
            stride = layer_config['stride']
            index = int(kernel_shape[0] * kernel_shape[1])
            is_big_conv = layer_config['is_big_size']
            block_expansion = (math.ceil(input_shape[0] / block_shape[0]), math.ceil(input_shape[1] / block_shape[1]))
            next_stride = [block_expansion[0] // stride[0], block_expansion[1] // stride[1]]
            padding = [-1, -1]
            if is_big_conv:
                big_conv = InverseMultiplexedConv2DLayer(
                    n_out_channel,
                    n_in_channel,
                    input_shape,
                    padding,
                    kernel_shape,
                    stride,
                    next_stride,
                    skip,
                    block_shape,
                )

                weight_pt = [
                    [
                        [
                            CkksPlaintextRingtNode(f'convw_{layer_id}_{k}_{n}_{i}')
                            for i in range(int(index * next_stride[0] * next_stride[1]))
                        ]
                        for n in range(n_in_channel)
                    ]
                    for k in range(n_out_channel)
                ]

                bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_out_channel)]

                layer_output_nodes = big_conv.call(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                )
                feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                input_args.append(Argument(f'convb_{layer_id}', bias_pt))
            else:
                if style == 'ordinary':
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = Conv2DPackedDepthwiseLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{i}') for i in range(index)]
                            for n in range(n_packed_out_channel)
                        ]
                    else:
                        conv0_layer = Conv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{m}_{i}') for i in range(index)]
                                for m in range(int(n_packed_in_channel * pack))
                            ]
                            for n in range(n_packed_out_channel)
                        ]

                    bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out_channel)]
                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    layer_output_nodes = conv0_layer.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
                if style == 'multiplexed':
                    n_in_channel_per_ct = pack
                    n_block_per_ct = n_in_channel_per_ct // (skip[0] * skip[1])
                    n_out_channel_per_ct = n_in_channel_per_ct * stride[0] * stride[1]
                    n_pack_in_channel = math.ceil(n_in_channel / n_in_channel_per_ct)
                    n_pack_out_channel = math.ceil(n_out_channel / n_out_channel_per_ct)
                    kernel_size = kernel_shape[0] * kernel_shape[1]
                    if groups == n_out_channel and groups != 1:
                        conv0_layer = ParMultiplexedConv2DPackedLayerDepthwise(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_in_channel_per_ct,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        weight_pt = [
                            [CkksPlaintextRingtNode(f'convw_{layer_id}_{j}_{k}') for k in range(kernel_size)]
                            for j in range(n_pack_in_channel)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)]
                        if stride[0] != 1:
                            mask_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_out_channel)]
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))
                        else:
                            mask_pt = []
                    else:
                        conv0_layer = ParMultiplexedConv2DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            pack,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )

                        size_0 = math.ceil(n_out_channel / n_block_per_ct)
                        size_1 = n_pack_in_channel * n_block_per_ct

                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(kernel_size)]
                                for j in range(size_1)
                            ]
                            for i in range(size_0)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)]
                        if stride[0] == 1 and stride[1] == 1 and skip[0] == 1 and skip[1] == 1:
                            mask_pt = []
                        else:
                            mask_pt = [
                                [
                                    CkksPlaintextRingtNode(f'convm_{layer_id}_{i}_{j}')
                                    for j in range(min(n_block_per_ct, n_out_channel - i * n_block_per_ct))
                                ]
                                for i in range(size_0)
                            ]
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))

                    input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                    input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    layer_output_nodes = conv0_layer.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, mask_pt
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('batchnorm2d', 'dropout', 'constmul', 'identity'):
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'square2d':
            act_layer = SquareLayer(level)
            layer_output_nodes = act_layer.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('poly_relu2d', 'simple_polyrelu'):
            feat = config_info['feature'][layer_input_feature_ids[0]]
            level = int(feat['level'])
            order = layer_config['order']
            n_pack_in_channel = math.ceil(n_in_channel / pack)
            if feat['dim'] == 0:
                input_shape = [1, 1]
                skip = [1, 1]
            else:
                input_shape = feat['shape']
            weight_pt = [
                [CkksPlaintextRingtNode(f'poly_reluw_{layer_id}_{i}_{j}') for j in range(n_pack_in_channel)]
                for i in range(order + 1)
            ]
            polyrelu = PolyRelu(input_shape, order, skip, pack)
            feature_id_in_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            drop_level_n = feature_id_in_nodes[0].level - level

            if level < feature_id_in_nodes[0].level:
                feature_id_in_nodes = [drop_level(node, drop_level_n) for node in feature_id_in_nodes]
            if feat['dim'] == 0:
                layer_output_nodes = polyrelu.call_bsgs_feature0d(feature_id_in_nodes, weight_pt)
            else:
                layer_output_nodes = polyrelu.call_bsgs_feature2d(feature_id_in_nodes, weight_pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            for i in range(len(weight_pt)):
                input_args.append(Argument(f'poly_reluw_{layer_id}_{i}', weight_pt[i]))

        elif layer_config['type'] == 'conv1d':
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape'][0]
            kernel_shape = layer_config['kernel_shape'][0]
            stride = layer_config['stride'][0]
            skip_1d = skip[0] if isinstance(skip, list) else skip
            n_channel_per_ct = int(n // 2 // input_shape // skip_1d)
            n_pack_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
            n_packed_out_channel = math.ceil(n_out_channel / (n_channel_per_ct * stride))
            rot_n_channel_per_ct = min(n_in_channel, n_channel_per_ct)
            input_ct = [
                CkksCiphertextNode(f'{layer_input_feature_ids[0]}input{k}', level=level)
                for k in range(n_pack_in_channel)
            ]
            if layer_input_feature_ids[0] not in feature_id_to_nodes_map:
                feature_id_to_nodes_map[layer_input_feature_ids[0]] = input_ct
                input_args.append(Argument(layer_input_feature_ids[0], input_ct))
            weight_pt = [
                [
                    [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{k}_{j}') for k in range(kernel_shape)]
                    for j in range(rot_n_channel_per_ct)
                ]
                for i in range(n_packed_out_channel)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out_channel)]
            conv1d = Conv1DPackedLayer(
                n_out_channel,
                n_in_channel,
                input_shape,
                kernel_shape,
                stride,
                skip_1d,
                n_channel_per_ct,
                n_pack_in_channel,
                n_packed_out_channel,
            )
            layer_output_nodes = conv1d.call(feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'convw_{layer_id}', weight_pt))
            input_args.append(Argument(f'convb_{layer_id}', bias_pt))

        elif layer_config['type'] == 'mult_scalar':
            mult_scalar_layer = MultScalarLayer()
            input_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            pt = [CkksPlaintextRingtNode(f'mult_scalar_{layer_id}_{i}') for i in range(len(input_nodes))]
            layer_output_nodes = mult_scalar_layer.call(input_nodes, pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'mult_scalar_{layer_id}', pt))

        elif layer_config['type'] == 'mult_coeff':
            raise ValueError(
                f"Layer '{layer_id}' has type 'mult_coeff', which should have been absorbed "
                f"into adjacent layers or converted to 'mult_scalar' during compilation."
            )

        elif layer_config['type'] == 'drop_level':
            level_in = config_info['feature'][layer_input_feature_ids[0]]['level']
            level_out = config_info['feature'][layer_output_feature_ids[0]]['level']
            drop_level_n = level_in - level_out
            layer_output_nodes = [
                drop_level(node, drop_level_n) for node in feature_id_to_nodes_map[layer_input_feature_ids[0]]
            ]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'bootstrapping':
            layer_output_nodes = []
            for node in feature_id_to_nodes_map[layer_input_feature_ids[0]]:
                if node.level > 0:
                    node = drop_level(node, node.level)
                layer_output_nodes.append(bootstrap(node))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] in ('add', 'add2d'):
            layer_output_nodes = [
                add(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]][i],
                    feature_id_to_nodes_map[layer_input_feature_ids[1]][i],
                )
                for i in range(len(feature_id_to_nodes_map[layer_input_feature_ids[0]]))
            ]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'concat2d' in layer_config['type']:
            # concat is a runtime-only operation: just merge node lists from all inputs
            layer_output_nodes = []
            for input_fid in layer_input_feature_ids:
                layer_output_nodes.extend(feature_id_to_nodes_map[input_fid])
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'upsample_nearest' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            upsample_factor = layer_config['upsample_factor']
            n_channel_per_ct = pack
            upsample_layer = UpsampleNearestLayer(
                shape=input_shape,
                skip=skip,
                upsample_factor=upsample_factor,
                n_channel_per_ct=n_channel_per_ct,
                level=level,
            )
            out_channels_per_ct = n_channel_per_ct // (upsample_factor[0] * upsample_factor[1])
            n_select_pt = out_channels_per_ct
            select_tensor_pt = [
                CkksPlaintextRingtNode(f'upsample_select_pt_{layer_id}_{i}') for i in range(n_select_pt)
            ]
            layer_output_nodes = upsample_layer.call(
                feature_id_to_nodes_map[layer_input_feature_ids[0]],
                select_tensor_pt,
                n_in_channel,
            )
            input_args.append(Argument(f'upsample_select_pt_{layer_id}', select_tensor_pt))
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif 'avgpool' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            stride = layer_config['stride']
            is_adaptive = layer_config.get('is_adaptive_avgpool', True)
            is_big_size = layer_config.get('is_big_size', False)
            avgpool = Avgpool2DLayer(stride, input_shape, channel=n_in_channel, skip=skip)
            if is_big_size:
                block_expansion = [
                    math.ceil(input_shape[0] / block_shape[0]),
                    math.ceil(input_shape[1] / block_shape[1]),
                ]
                layer_output_nodes = avgpool.call_interleaved_avgpool(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], block_expansion
                )
            else:
                if is_adaptive:
                    # level_cost=0: only rotations + adds, normalization absorbed into adjacent layers
                    if style == 'ordinary':
                        layer_output_nodes = avgpool.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
                    else:
                        layer_output_nodes = avgpool.run_adaptive_avgpool(
                            feature_id_to_nodes_map[layer_input_feature_ids[0]], n=n
                        )
                else:
                    # level_cost=1: non-adaptive avgpool needs mult+rescale (select_tensor)
                    n_channel_per_ct = int(math.ceil(n / 2 / (input_shape[0] * input_shape[1])))
                    out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]
                    n_select_pt = min(n_in_channel, out_channels_per_ct)
                    select_tensor_pt = [CkksPlaintextRingtNode(f'select_pt_{layer_id}_{i}') for i in range(n_select_pt)]
                    layer_output_nodes = avgpool.call_multiplexed_avgpool(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]],
                        select_tensor_pt,
                        n_in_channel,
                        n_channel_per_ct,
                    )
                    input_args.append(Argument(f'select_tensor_pt_{layer_id}', select_tensor_pt))

            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        elif layer_config['type'] == 'fc0':
            n_packed_in_channel = math.ceil(n_in_channel / (n // 2))
            n_packed_out_channel = math.ceil(n_out_channel / pack)
            if 'special_info' not in config_info['feature'][layer_input_feature_ids[0]]:
                # call_skip_0d path — matching test_fc_fc_feature0d Layer 1
                skip_0d = config_info['feature'][layer_input_feature_ids[0]]['skip']
                n_channel_per_ct = int(n // 2 // skip_0d)
                pack_0d = n_channel_per_ct
                n_packed_in_feature = math.ceil(n_in_channel / n_channel_per_ct)
                n_packed_out_feature = math.ceil(n_out_channel / n_channel_per_ct)
                weight_pt_size = n_packed_in_feature * pack_0d
                weight_pt = [
                    [CkksPlaintextRingtNode(f'densew_{layer_id}_{m}_{i}') for i in range(weight_pt_size)]
                    for m in range(n_packed_out_feature)
                ]
                bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out_feature)]
                fc_layer = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    [1, 1],
                    [1, 1],
                    pack_0d,
                    n_packed_in_feature,
                    n_packed_out_feature,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = fc_layer.call_skip_0d(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, skip_0d
                )
            else:
                special_info = config_info['feature'][layer_input_feature_ids[0]]['special_info']
                special_shape = special_info['shape']
                special_skip = special_info['skip']
                invalid_fill = special_info.get('invalid_fill', [1, 1])
                input_shape_ct = [special_shape[0] * special_skip[0], special_shape[1] * special_skip[1]]
                n_num_per_ct = math.ceil(n // 2 / (input_shape_ct[0] * input_shape_ct[1]))
                n_packed_out_feature_for_mult_apck = math.ceil(n_out_channel / n_num_per_ct)
                valid_skip_0 = special_skip[0] // invalid_fill[0]
                valid_skip_1 = special_skip[1] // invalid_fill[1]
                n_channel_per_block = valid_skip_0 * valid_skip_1
                n_channel = n_in_channel // (special_shape[0] * special_shape[1])
                n_block_input = math.ceil(n_channel / (n_num_per_ct * n_channel_per_block)) * n_num_per_ct
                weight_pt = [
                    [CkksPlaintextRingtNode(f'densew_{layer_id}_{i}_{j}') for j in range(n_block_input)]
                    for i in range(n_packed_out_feature_for_mult_apck)
                ]
                bias_pt = [
                    CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out_feature_for_mult_apck)
                ]
                dense = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    special_shape,
                    special_skip,
                    n_num_per_ct,
                    n_in_channel,
                    n_out_channel,
                    invalid_fill=invalid_fill,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = dense.call_multiplexed(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        else:
            raise ValueError(f'Unsupported layer type: {layer_config["type"]}')

    output_args = [Argument(output_id, feature_id_to_nodes_map[output_id]) for output_id in task_output_feature_ids]

    process_custom_task(input_args=input_args, output_args=output_args, output_instruction_path=task_path)


if __name__ == '__main__':
    if hasattr(sys, 'frozen'):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='FPGA config generator.')
    parser.add_argument('task_path', type=str, help='Path of the server directory')
    args = parser.parse_args()

    task_path = args.task_path
    with open(os.path.join(task_path, 'task_config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    for _, is_fpga in config['server_task'].items():
        if is_fpga['enable_fpga']:
            gen_custom_task(os.path.join(task_path, 'server'), use_gpu=True)
