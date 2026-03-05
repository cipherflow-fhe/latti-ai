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
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model_generator.layers.add_pack import *
from model_generator.layers.avgpool import *
from model_generator.layers.conv_dw import *
from model_generator.layers.conv_pack import *
from model_generator.layers.dense_pack import *
from model_generator.layers.mult_conv import *
from model_generator.layers.mult_conv_dw import *
from model_generator.layers.mult_scalar import *
from model_generator.layers.poly_relu import *
from model_generator.layers.square_pack import *
from model_generator.layers.inverse_multiplexed_conv2d_layer import *
from model_generator.layers.upsample_layer import *
from model_generator.layers.concat_layer import *


def read_config(config_path):
    with open(config_path, 'r', encoding='utf8') as fp:
        config_ctx = json.load(fp)
    return config_ctx


def set_param(n=16384):
    if n == 16384:
        q = [
            0x200000008001,
            0x400018001,
            0x3FFFD0001,
            0x400060001,
            0x400068001,
            0x3FFF90001,
            0x400080001,
            0x4000A8001,
            0x400108001,
            0x3FFEB8001,
        ]
        p = [0x7FFFFFD8001, 0x7FFFFFC8001]
        param = Param.create_ckks_custom_param(n=16384, p=p, q=q)
    elif n == 65536:
        param = CkksBtpParam.create_default_param()
    set_fhe_param(param)


def gen_custom_task(task_path, n=16384, use_gpu=True, style='ordinary'):
    T_SCALE = 2**6
    set_param(n=n)
    task_config_info = read_config(os.path.join(task_path, 'task_config.json'))
    try:
        block_shape = task_config_info['block_shape']
    except Exception:
        block_shape = [64, 64]
    config_info = read_config(os.path.join(task_path, 'nn_layers_ct_0.json'))
    input_args = list()
    feature_id_to_nodes_map = {}
    task_input_feature_ids = config_info['input_feature']
    task_output_feature_ids = config_info['output_feature']

    for layer_id, layer_config in config_info['layer'].items():
        if 'relu2d' in layer_config['type']:
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
        if 'fc' in layer_config['type']:
            virtual_shape = config_info['feature'][layer_input_feature_ids[0]]['virtual_shape']
            virtual_skip = config_info['feature'][layer_input_feature_ids[0]]['virtual_skip']
            virtual_shape_out = config_info['feature'][layer_output_feature_ids[0]]['virtual_shape']
            virtual_skip_out = config_info['feature'][layer_output_feature_ids[0]]['virtual_skip']
            n_packed_in_channel = math.ceil(n_in_channel / 8192)
            n_packed_out_channel = math.ceil(n_out_channel / pack)

        for input_node in layer_input_feature_ids:
            if input_node not in feature_id_to_nodes_map.keys():
                x = [CkksCiphertextNode(input_node + f'input{k}', level=level) for k in range(n_packed_in_channel)]
                feature_id_to_nodes_map.update({input_node: x})
                input_args.append(Argument(input_node, x))

        if 'reshape' in layer_config['type']:
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'conv' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            groups = layer_config['groups']
            kernel_shape = layer_config['kernel_shape']
            stride = layer_config['stride']
            index = int(kernel_shape[0] * kernel_shape[1])
            is_big_conv = layer_config['is_big_size']
            block_expansion = [input_shape[0] / block_shape[0], input_shape[1] / block_shape[1]]
            next_stride = [int(block_expansion[0] / stride[0]), int(block_expansion[1] / stride[1])]
            padding = [-1, -1]
            # style = layer_config['style']
            if is_big_conv:
                input_args.clear()
                input_node = layer_input_feature_ids[0]
                level = config_info['feature'][layer_input_feature_ids[0]]['level']
                block_expansion = [input_shape[0] / block_shape[0], input_shape[1] / block_shape[1]]
                feature_id_to_nodes_map[layer_input_feature_ids[0]] = [
                    CkksCiphertextNode(input_node + f'input{k}', level=level)
                    for k in range(int(n_in_channel * block_expansion[0] * block_expansion[1]))
                ]
                input_args.append(Argument(input_node, feature_id_to_nodes_map[layer_input_feature_ids[0]]))
                big_conv = InverseMultiplexedConv2d(
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
                        conv0_layer = Conv2DepthwiseLayer(
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
                        if use_gpu:
                            weight_pt = [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{i}') for i in range(index)]
                                for n in range(n_packed_out_channel)
                            ]
                        else:
                            weight_pt = [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{i}') for i in range(index)]
                                for n in range(n_packed_out_channel)
                            ]

                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_packed_out_channel)]

                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
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
                        if use_gpu:
                            weight_pt = [
                                [
                                    [CkksPlaintextRingtNode(f'convw_{layer_id}_{n}_{m}_{i}') for i in range(index)]
                                    for m in range(int(n_packed_in_channel * pack))
                                ]
                                for n in range(n_packed_out_channel)
                            ]
                        else:
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
                    if groups == n_out_channel and groups != 1:
                        n_in_channel_per_ct = pack
                        n_block_per_ct = int(np.floor(n_in_channel_per_ct / (skip[0] * skip[1])))
                        n_out_channel_per_ct = int(n_in_channel_per_ct * stride[0] * stride[1])
                        n_pack_in_channel = int(np.ceil(n_in_channel / n_in_channel_per_ct))
                        n_pack_out_channel = int(np.ceil(n_out_channel / n_out_channel_per_ct))
                        conv0_layer = MultConv2DPackedDepthwiseLayer(
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
                        size_2 = kernel_shape[0] * kernel_shape[1]
                        weight_pt = [
                            [CkksPlaintextRingtNode(f'convw_{layer_id}_{j}_{k}') for k in range(size_2)]
                            for j in range(n_pack_in_channel)
                        ]
                        if stride[0] != 1:
                            bias_pt = [
                                CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)
                            ]
                            mask_pt = [CkksPlaintextRingtNode(f'convm_{layer_id}_{i}') for i in range(n_out_channel)]

                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))
                        else:
                            bias_pt = [
                                CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)
                            ]
                            mask_pt = []
                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    else:
                        n_in_channel_per_ct = pack
                        n_block_per_ct = int(np.floor(n_in_channel_per_ct / (skip[0] * skip[1])))
                        n_out_channel_per_ct = int(n_in_channel_per_ct * stride[0] * stride[1])
                        n_pack_in_channel = int(np.ceil(n_in_channel / n_in_channel_per_ct))
                        n_pack_out_channel = int(np.ceil(n_out_channel / n_out_channel_per_ct))
                        conv0_layer = MultConv2DPackedLayer(
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

                        size_0 = int(np.ceil(n_out_channel / n_block_per_ct))
                        size_1 = int(n_pack_in_channel * n_block_per_ct)
                        size_2 = int(kernel_shape[0] * kernel_shape[1])

                        # weight_pt、bias_pt、mask_pt
                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'convw_{layer_id}_{i}_{j}_{k}') for k in range(size_2)]
                                for j in range(size_1)
                            ]
                            for i in range(size_0)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)]
                        if stride[0] == 1 and stride[1] == 1 and skip[0] == 1 and skip[1] == 1:
                            bias_pt = [
                                CkksPlaintextRingtNode(f'convb_{layer_id}_{i}') for i in range(n_pack_out_channel)
                            ]
                            mask_pt = []
                        else:
                            mask_pt = [
                                [
                                    CkksPlaintextRingtNode(f'convm_{layer_id}_{i}_{j}')
                                    for j in range(min(n_block_per_ct, n_out_channel))
                                ]
                                for i in range(size_0)
                            ]
                            remove_mask = {}
                            for idx in range(len(mask_pt)):
                                for jdx in range(len(mask_pt[0])):
                                    m_idx = idx * len(mask_pt[0]) + jdx
                                    remove_mask[idx] = []
                                    if m_idx >= n_out_channel:
                                        remove_mask[idx].append(mask_pt[idx][jdx])
                            for k in remove_mask.keys():
                                v = remove_mask[k]
                                for vi in v:
                                    mask_pt[k].remove(vi)
                            input_args.append(Argument(f'convm_{layer_id}', mask_pt))

                        input_args.append(Argument(f'convw_{layer_id}', weight_pt))
                        input_args.append(Argument(f'convb_{layer_id}', bias_pt))
                    layer_output_nodes = conv0_layer.call(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, mask_pt
                    )
                    feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if (
            'batchnorm' in layer_config['type']
            or 'dropout' in layer_config['type']
            or 'constmul' in layer_config['type']
            or 'identity' in layer_config['type']
        ):
            layer_output_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'square' in layer_config['type']:
            if 'square2d' in layer_config['type']:
                act_layer = Square_layer(level)
                layer_output_nodes = act_layer.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
            else:
                continue
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'poly_relu' in layer_config['type'] or 'simple_polyrelu' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            level = config_info['feature'][layer_input_feature_ids[0]]['level']
            order = layer_config['order']
            n_pack_in_channel = int(np.ceil(n_in_channel / pack))
            if order == 4:
                level_order = [level - 3, level - 2, level - 2, level - 2, level - 1]
                weight_pt = [
                    [CkksPlaintextRingtNode(f'poly_reluw_{layer_id}_{i}_{j}') for j in range(n_pack_in_channel)]
                    for i in range(order + 1)
                ]
            else:
                weight_pt = [
                    [CkksPlaintextRingtNode(f'poly_reluw_{layer_id}_{i}_{j}') for j in range(n_pack_in_channel)]
                    for i in range(order + 1)
                ]
            polyrelu = PolyReluLayer(input_shape, order, skip, pack)
            feature_id_in_nodes = feature_id_to_nodes_map[layer_input_feature_ids[0]]
            drop_level_n = feature_id_in_nodes[0].level - level

            if level < feature_id_in_nodes[0].level:
                feature_id_in_nodes = [drop_level(node, drop_level_n) for node in feature_id_in_nodes]
            layer_output_nodes = polyrelu.call_bsgs(feature_id_in_nodes, weight_pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            for i in range(len(weight_pt)):
                input_args.append(Argument(f'poly_reluw_{layer_id}_{i}', weight_pt[i]))

        if 'mult_scalar' in layer_config['type']:
            mult_scalar_layer = MultScalarLayer()
            pt = CkksPlaintextRingtNode('')
            layer_output_nodes = mult_scalar_layer.call(feature_id_to_nodes_map[layer_input_feature_ids[0]], pt)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})
            input_args.append(Argument(f'mult_scalar_{layer_id}', pt))

        if 'drop_level' in layer_config['type']:
            level_in = config_info['feature'][layer_input_feature_ids[0]]['level']
            level_out = config_info['feature'][layer_output_feature_ids[0]]['level']
            drop_level_n = level_in - level_out
            layer_output_nodes = list()
            for i in range(len(feature_id_to_nodes_map[layer_input_feature_ids[0]])):
                layer_output_nodes.append(
                    drop_level(feature_id_to_nodes_map[layer_input_feature_ids[0]][i], drop_level_n)
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'bootstrap' in layer_config['type']:
            layer_output_nodes = list()

            for i in range(len(feature_id_to_nodes_map[layer_input_feature_ids[0]])):
                if feature_id_to_nodes_map[layer_input_feature_ids[0]][i].level > 0:
                    drop_level_n = feature_id_to_nodes_map[layer_input_feature_ids[0]][i].level
                    feature_id_to_nodes_map[layer_input_feature_ids[0]][i] = drop_level(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]][i], drop_level_n
                    )
                y = bootstrap(feature_id_to_nodes_map[layer_input_feature_ids[0]][i])
                layer_output_nodes.append(y)
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'add' in layer_config['type']:
            layer_output_nodes = list()
            for i in range(len(feature_id_to_nodes_map[layer_input_feature_ids[0]])):
                layer_output_nodes.append(
                    add(
                        feature_id_to_nodes_map[layer_input_feature_ids[0]][i],
                        feature_id_to_nodes_map[layer_input_feature_ids[1]][i],
                    )
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'avgpool' in layer_config['type']:
            input_shape = config_info['feature'][layer_input_feature_ids[0]]['shape']
            stride = layer_config['stride']
            avgpool = Avgpool_layer(stride, input_shape, channel=n_in_channel, skip=skip)
            if style == 'ordinary':
                layer_output_nodes = avgpool.call(feature_id_to_nodes_map[layer_input_feature_ids[0]])
            else:
                layer_output_nodes = avgpool.run_adaptive_avgpool(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], n=n
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

        if 'fc' in layer_config['type']:
            if style == 'ordinary':
                if use_gpu:
                    weight_pt = [
                        [
                            CkksPlaintextRingtNode(f'densew_{layer_id}_{m}_{i}')
                            for i in range(n_packed_in_channel * pack)
                        ]
                        for m in range(n_packed_out_channel)
                    ]
                else:
                    weight_pt = [
                        [
                            CkksPlaintextRingtNode(f'densew_{layer_id}_{m}_{i}')
                            for i in range(n_packed_in_channel * pack)
                        ]
                        for m in range(n_packed_out_channel)
                    ]

                bias_pt = [CkksPlaintextRingtNode(f'denseb_{layer_id}_{i}') for i in range(n_packed_out_channel)]
                fc_layer = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    virtual_shape,
                    virtual_skip,
                    pack,
                    n_packed_in_channel,
                    n_packed_out_channel,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = fc_layer.call(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt
                )
            elif style == 'multiplexed':
                input_shape_ct = [virtual_shape[0] * virtual_skip[0], virtual_shape[1] * virtual_skip[1]]
                n_num_per_ct = int(np.ceil(n / 2 / (input_shape_ct[0] * input_shape_ct[1])))
                n_packed_out_feature_for_mult_apck = int(np.ceil(n_out_channel / n_num_per_ct))
                n_block_input = (
                    int(np.ceil(n_in_channel * virtual_shape[0] * virtual_shape[1] / (n / 2))) * n_num_per_ct
                )
                weight_pt = [
                    [CkksPlaintextRingtNode(f'weight_pt_{i}_{j}') for j in range(n_block_input)]
                    for i in range(n_packed_out_feature_for_mult_apck)
                ]
                bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature_for_mult_apck)]
                dense = DensePackedLayer(
                    n_out_channel,
                    n_in_channel,
                    virtual_shape,
                    virtual_skip,
                    n_num_per_ct,
                    n_in_channel,
                    n_out_channel,
                )
                input_args.append(Argument(f'densew_{layer_id}', weight_pt))
                input_args.append(Argument(f'denseb_{layer_id}', bias_pt))
                layer_output_nodes = dense.call_mult_pack(
                    feature_id_to_nodes_map[layer_input_feature_ids[0]], weight_pt, bias_pt, n=n
                )
            feature_id_to_nodes_map.update({layer_output_feature_ids[0]: layer_output_nodes})

    output_args = []
    for output_id in task_output_feature_ids:
        output_args.append(Argument(output_id, feature_id_to_nodes_map[output_id]))

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
