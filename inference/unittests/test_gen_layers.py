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

import math
import unittest
import json
import os
import sys

import torch
from torch import nn

import sys
import os

# Add mega_ag_generator to path for importing frontend module
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find project root by walking up until we find the 'training' directory.
_dir = script_dir
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, 'training')):
        break
    _dir = os.path.dirname(_dir)
project_root = _dir

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'inference', 'lattisense'))

from frontend.custom_task import *

# examples directory is at project_root/examples
examples_root = os.path.join(project_root, 'examples')

from inference.model_generator.deploy_cmds import *  # noqa: E402
from training.model_export.onnx_to_json import *  # noqa: E402

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hetero')


def export_to_onnx(model, inputs, output_names, onnx_path, dynamic_axes=None, opset_version=11):
    """
    Export the PyTorch model to ONNX format.

    Parameters:

    model (torch.nn.Module): The PyTorch model to export.

    inputs (tuple or torch.Tensor): Inputs to the model, which can be a single tensor or a tuple (supporting multiple inputs).

    output_names (list): A list of names for the output tensors (supporting multiple outputs).

    onnx_path (str): The file path for the exported ONNX file.

    dynamic_axes (dict): Configuration for dynamic axes, used to support dynamic input/output shapes.

    opset_version (int): The ONNX operator set version, default is 11.
    """

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if dynamic_axes is None:
        dynamic_axes = {}

    torch.onnx.export(
        model=model,
        args=inputs,
        f=onnx_path,
        input_names=[f'input_{i}' for i in range(len(inputs))],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
    )


def gen_conv_mega_ag(
    model, n_in_channel, n_out_channel, input_shape, kernel_shape, stride, skip, groups, init_level, style='ordinary'
):
    set_param(n=16384)
    conv = nn.Conv2d(
        n_in_channel, n_out_channel, kernel_shape[0], stride[0], padding=int(kernel_shape[0] / 2), groups=groups
    )
    model.conv1 = conv

    if isinstance(input_shape, (list, set)):
        input_shape = tuple(input_shape)
    if isinstance(kernel_shape, (list, set)):
        kernel_shape = tuple(kernel_shape)
    if isinstance(stride, (list, set)):
        stride = tuple(stride)

    if style == 'multiplexed':
        if groups != 1:  # dw
            task_name = f'CKKS_multiplexed_dw_conv2d_{n_in_channel}_in_{n_out_channel}_out_channel_{stride[0]}_stride_{input_shape[0]}_{input_shape[1]}_{kernel_shape[0]}_{kernel_shape[1]}'
        else:
            task_name = f'CKKS_multiplexed_conv2d_{n_in_channel}_in_{n_out_channel}_out_channel_{stride[0]}_stride_{input_shape[0]}_{input_shape[1]}_{kernel_shape[0]}_{kernel_shape[1]}'
    else:
        if groups != 1:  # dw
            task_name = f'CKKS_dw_conv2d_{n_in_channel}_in_{n_out_channel}_out_channel_{stride[0]}_stride_{input_shape[0]}_{input_shape[1]}_{kernel_shape[0]}_{kernel_shape[1]}'
        else:
            task_name = f'CKKS_conv2d_{n_in_channel}_in_{n_out_channel}_out_channel_{stride[0]}_stride_{input_shape[0]}_{input_shape[1]}_{kernel_shape[0]}_{kernel_shape[1]}'
    task_path = os.path.join(base_path, task_name, f'level_{init_level}')
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    task_server_path = os.path.join(task_path, 'server')
    if not os.path.exists(task_server_path):
        os.makedirs(task_server_path)

    inputx = torch.randn(1, n_in_channel, input_shape[0], input_shape[1])

    export_to_onnx(model, inputx, ['output'], f'{task_server_path}/0.onnx')
    onnx_to_json(f'{task_server_path}/0.onnx', f'{task_server_path}/erg0.json', style)

    with open(f'{task_server_path}/erg0.json') as f:
        model_json = json.load(f)

    model_json['feature']['input_0']['level'] = init_level
    model_json['feature']['input_0']['pack_num'] = math.ceil(8192 / (input_shape[0] * input_shape[1]))
    model_json['feature']['output']['level'] = init_level - 1
    if style == 'multiplexed' and stride[0] != 1:
        model_json['feature']['output']['level'] = init_level - 2
    model_json['feature']['output']['pack_num'] = math.ceil(
        8192 / (input_shape[0] * input_shape[1]) * (stride[0] * stride[1])
    )
    with open(f'{task_server_path}/erg0.json', 'w') as f:
        json.dump(
            {
                'feature': model_json['feature'],
                'layer': model_json['layer'],
                'input_feature': ['input_0'],
                'output_feature': ['output'],
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    task_config = {
        'task_type': 'fhe',
        'task_num': 1,
        'server_start_id': 0,
        'server_end_id': 1,
        'server_task': {'0': {'enable_fpga': True}},
        'task_input_id': 'input_0',
        'task_output_id': 'output',
        'task_input_param': model_json['feature']['input_0'],
        'task_output_param': model_json['feature']['output'],
    }
    with open(f'{task_path}/task_config.json', 'w') as f:
        json.dump(task_config, f, indent=4, ensure_ascii=False)

    with open(os.path.join(task_path, 'task_config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    for _ in range(len(config['server_task'])):
        gen_custom_task(os.path.join(task_server_path), use_gpu=True, style=style)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class TestLayerExport(unittest.TestCase):
    def test_sq(self):
        N = 16384
        set_param(n=N)
        n_in_level = 2
        shapes = {16, 32, 64}
        for s in shapes:
            input_ct = [CkksCiphertextNode(f'input_ct_{i}', n_in_level) for i in range(int(np.ceil(s * s * 1 / 8192)))]
            square = Square_layer(level=n_in_level)
            output_ct = square.call(input_ct)
            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=os.path.join(
                    base_path, f'CKKS_square_{s}_{s}', f'level_{n_in_level}', 'server'
                ),
            )

    def test_conv_1ch_s1(self):
        n_in_channel = 1
        n_out_channel = 1
        stride = (1, 1)
        skip = (1, 1)
        groups = 1
        init_level = 2

        input_shapes = {(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)}
        kernel_shapes = {(1, 1), (3, 3), (5, 5)}

        model = SimpleCNN()
        for input_shape in input_shapes:
            for kernel_shape in kernel_shapes:
                gen_conv_mega_ag(
                    model,
                    n_in_channel,
                    n_out_channel,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    groups,
                    init_level,
                )

    def test_conv_1ch_s2(self):
        n_in_channel = 1
        n_out_channel = 1
        stride = [2, 2]
        skip = [1, 1]
        groups = 1
        init_level = 2

        input_shapes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
        kernel_shapes = [(1, 1), (3, 3), (5, 5)]

        model = SimpleCNN()
        for input_shape in input_shapes:
            for kernel_shape in kernel_shapes:
                gen_conv_mega_ag(
                    model,
                    n_in_channel,
                    n_out_channel,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    groups,
                    init_level,
                )

    def test_conv_mch_s1(self):
        n_in_channels = {1, 3, 4, 16, 17}
        n_out_channels = {1, 3, 4, 32, 33}
        stride = [1, 1]
        skip = [1, 1]
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        groups = 1
        init_level = 2

        model = SimpleCNN()
        for n_in_channel in n_in_channels:
            for n_out_channel in n_out_channels:
                gen_conv_mega_ag(
                    model,
                    n_in_channel,
                    n_out_channel,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    groups,
                    init_level,
                )

    def test_conv_mch_s2(self):
        n_in_channels = {1, 3, 4, 16, 17}
        n_out_channels = {1, 3, 4, 32, 33}
        stride = [2, 2]
        skip = [1, 1]
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        groups = 1
        init_level = 2

        model = SimpleCNN()
        for n_in_channel in n_in_channels:
            for n_out_channel in n_out_channels:
                gen_conv_mega_ag(
                    model,
                    n_in_channel,
                    n_out_channel,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    groups,
                    init_level,
                )

    def test_dw_32ch_s1_32x32_k3(self):
        n_in_channel = 32
        n_out_channel = 32
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        stride = [1, 1]
        skip = [1, 1]
        groups = 1
        init_level = 5

        model = SimpleCNN()

        gen_conv_mega_ag(
            model,
            n_in_channel,
            n_out_channel,
            input_shape,
            kernel_shape,
            stride,
            skip,
            groups,
            init_level,
        )

    def test_dw_4ch_s2_32x32_k3(self):
        n_in_channel = 4
        n_out_channel = 4
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        stride = [2, 2]
        skip = [1, 1]
        init_level = 5
        groups = n_in_channel

        model = SimpleCNN()

        gen_conv_mega_ag(
            model,
            n_in_channel,
            n_out_channel,
            input_shape,
            kernel_shape,
            stride,
            skip,
            groups,
            init_level,
        )

    def test_mux_conv_s1_32x32_k3(self):
        n_in_channels = {4, 8, 32}
        n_out_channels = {4, 8, 32}
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        stride = [1, 1]
        skip = [1, 1]
        init_level = 5

        for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
            groups = 1

            model = SimpleCNN()

            gen_conv_mega_ag(
                model,
                n_in_channel,
                n_out_channel,
                input_shape,
                kernel_shape,
                stride,
                skip,
                groups,
                init_level,
                'multiplexed',
            )

    def test_mux_conv_s2_32x32_k3(self):
        n_in_channels = {4, 8, 32}
        n_out_channels = {4, 8, 32}
        input_shape = [32, 32]
        kernel_shape = [3, 3]
        stride = [2, 2]
        skip = [1, 1]
        init_level = 5

        for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
            groups = 1

            model = SimpleCNN()

            gen_conv_mega_ag(
                model,
                n_in_channel,
                n_out_channel,
                input_shape,
                kernel_shape,
                stride,
                skip,
                groups,
                init_level,
                'multiplexed',
            )

    def test_mux_dw_s2_64x64_k3(self):
        n_in_channels = {4, 8, 32}
        n_out_channels = {4, 8, 32}
        input_shape = [64, 64]
        kernel_shape = [3, 3]
        stride = [2, 2]
        skip = [1, 1]
        init_level = 5

        for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
            groups = n_in_channel

            model = SimpleCNN()

            gen_conv_mega_ag(
                model,
                n_in_channel,
                n_out_channel,
                input_shape,
                kernel_shape,
                stride,
                skip,
                groups,
                init_level,
                'multiplexed',
            )

    def test_poly_relu_bsgs(self):
        N = 16384
        set_param(n=N)
        n_in_channel = 32
        input_shape = [32, 32]
        skip = [1, 1]
        n_in_channel_per_ct = int(np.floor(N / 2 / (input_shape[0] * input_shape[1])))
        n_pack_in_channel = int(np.ceil(n_in_channel / n_in_channel_per_ct))
        orders = [2, 4, 6, 8, 10, 12, 16, 32, 64]
        level = 8

        for order in orders:
            input_ct = [CkksCiphertextNode(f'input{k}', level) for k in range(n_pack_in_channel)]
            weight_pt = [
                [CkksPlaintextRingtNode(f'polyw_{1}_{i}_{j}') for j in range(n_pack_in_channel)]
                for i in range(order + 1)
            ]

            poly_layer = PolyReluLayer(input_shape, order, skip, n_in_channel_per_ct)
            output_ct = poly_layer.call_bsgs(input_ct, weight_pt)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            for i in range(order + 1):
                input_args.append(Argument(f'weight_pt{i}', weight_pt[i]))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=os.path.join(
                    base_path, f'CKKS_poly_relu_bsgs_{n_in_channel}_channel_order_{order}', f'level_{level}'
                ),
            )

    def test_fc_cyclic(self):
        N = 16384
        set_param(n=N)
        level = 2
        w_shapes = [128, 512]
        shapes = [1, 2, 4, 8, 16]
        virtual_skip = [1, 1]

        for s in shapes:
            n_in_channel = w_shapes[1]
            n_out_channel = w_shapes[0]
            virtual_shape = [s, s]
            n_channel_per_ct = int(np.ceil(8192 / (s * s)))
            n_packed_out_feature = int(np.ceil(w_shapes[0] / n_channel_per_ct))
            n_packed_in_feature = int(np.ceil(w_shapes[1] / n_channel_per_ct))
            per_channel_num = int((virtual_shape[0] / virtual_skip[0]) * (virtual_shape[1] / virtual_skip[1]))
            weight_pt_size = int(np.ceil(n_packed_in_feature / per_channel_num)) * n_channel_per_ct
            input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(int(np.ceil(w_shapes[1] / 8192)))]
            weight_pt = [
                [CkksPlaintextRingtNode(f'weight_pt_{i}_{j}') for j in range(weight_pt_size)]
                for i in range(n_packed_out_feature)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature)]

            dense = DensePackedLayer(
                n_out_channel,
                n_in_channel,
                virtual_shape,
                virtual_skip,
                n_channel_per_ct,
                n_packed_in_feature,
                n_packed_out_feature,
            )
            output_ct = dense.call(input_ct, weight_pt, bias_pt)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            input_args.append(Argument(f'weight_pt', weight_pt))
            input_args.append(Argument(f'bias_pt', bias_pt))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=os.path.join(
                    base_path, f'CKKS_fc_prepare_weight1_1D_pack_cyclic_{s}_{s}', f'level_{level}', 'server'
                ),
            )

    def test_fc_pack_skip(self):
        N = 16384
        set_param(n=N)
        w_shape = [10, 4096]
        virtual_shape = [1, 1]
        level = 2

        skips = [2, 4, 8, 16]
        for s in skips:
            n_in_channel = w_shape[1]
            n_out_channel = w_shape[0]
            virtual_skip = [s, s]
            n_channel_per_ct = int(np.ceil(8192 / (s * s)))
            n_packed_out_feature = int(np.ceil(n_out_channel / n_channel_per_ct))
            n_packed_in_feature = int(np.ceil(n_in_channel / n_channel_per_ct))
            dense = DensePackedLayer(
                n_out_channel,
                n_in_channel,
                virtual_shape,
                virtual_skip,
                n_channel_per_ct,
                n_packed_in_feature,
                n_packed_out_feature,
            )
            per_channel_num = 1
            weight_pt_size = min(int(np.ceil(n_packed_in_feature / per_channel_num)) * n_channel_per_ct, n_in_channel)
            input_ct = [
                CkksCiphertextNode(f'input_ct_{i}', level) for i in range(int(np.ceil(n_in_channel / n_channel_per_ct)))
            ]
            weight_pt = [
                [CkksPlaintextRingtNode(f'weight_pt_{i}_{j}') for j in range(weight_pt_size)]
                for i in range(n_packed_out_feature)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature)]

            output_ct = dense.call(input_ct, weight_pt, bias_pt)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            input_args.append(Argument(f'weight_pt', weight_pt))
            input_args.append(Argument(f'bias_pt', bias_pt))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=os.path.join(
                    base_path, f'CKKS_fc_prepare_weight1_1D_pack_skip_{s}_{s}', f'level_{level}', 'server'
                ),
            )

    def test_fc_fc(self):
        N = 16384
        set_param(n=N)
        n_slot = 8192
        init_level = 2
        input_channel = 1024
        output_channel = 1024
        dense_shape = [4, 4]
        skip = [1, 1]

        output_channel1 = 128
        dense_shape1 = [1, 1]
        skip1 = [dense_shape[0] * skip[0], dense_shape[1] * skip[1]]

        n_channel_per_ct = int(np.floor(n_slot / (dense_shape[0] * dense_shape[1])))
        n_packed_in_feature = int(np.floor(input_channel / n_channel_per_ct))
        n_packed_out_feature = int(np.floor(output_channel / n_channel_per_ct))
        per_channel_num = int((dense_shape[0] / skip[0]) * (dense_shape[1] / skip[1]))
        weight_pt_size = int(np.ceil(n_packed_in_feature / per_channel_num)) * n_channel_per_ct
        input_ct = [
            CkksCiphertextNode(f'input_ct0_{i}', init_level) for i in range(int(np.ceil(input_channel / n_slot)))
        ]
        weight_pt0 = [
            [CkksPlaintextRingtNode(f'weight_pt0_{i}_{j}') for j in range(weight_pt_size)]
            for i in range(n_packed_out_feature)
        ]
        bias_pt0 = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature)]
        dense0 = DensePackedLayer(
            input_channel,
            output_channel,
            dense_shape,
            skip,
            n_channel_per_ct,
            n_packed_in_feature,
            n_packed_out_feature,
        )
        res0 = dense0.call(input_ct, weight_pt0, bias_pt0)

        n_channel_per_ct = int(np.floor(n_slot / (skip1[0] * skip1[1])))
        n_packed_in_feature = int(np.ceil(output_channel / n_channel_per_ct))
        n_packed_out_feature = int(np.ceil(output_channel1 / n_channel_per_ct))
        weight_pt_size = n_packed_in_feature * n_channel_per_ct
        weight_pt1 = [
            [CkksPlaintextRingtNode(f'weight_pt1_{i}_{j}') for j in range(weight_pt_size)]
            for i in range(n_packed_out_feature)
        ]
        bias_pt1 = [CkksPlaintextRingtNode(f'bias_pt1_{i}') for i in range(n_packed_out_feature)]
        dense1 = DensePackedLayer(
            input_channel,
            output_channel,
            dense_shape1,
            skip1,
            n_channel_per_ct,
            n_packed_in_feature,
            n_packed_out_feature,
        )
        output_ct = dense1.call(res0, weight_pt1, bias_pt1)

        input_args = list()
        input_args.append(Argument('input_node', input_ct))
        input_args.append(Argument(f'weight_pt0', weight_pt0))
        input_args.append(Argument(f'bias_pt0', bias_pt0))
        input_args.append(Argument(f'weight_pt1', weight_pt1))
        input_args.append(Argument(f'bias_pt1', bias_pt1))

        process_custom_task(
            input_args=input_args,
            output_args=[Argument('output_ct', output_ct)],
            output_instruction_path=os.path.join(
                base_path,
                f'CKKS_fc_fc_{input_channel}_{output_channel}_{output_channel1}',
                f'level_{init_level}',
                'server',
            ),
        )

    def test_dense_mult_pack(self):
        N = 16384
        set_param(n=N)
        level = 6
        w_shapes = [10, 64]
        shapes = [8]
        virtual_skip = [4, 4]

        for s in shapes:
            n_in_channel = w_shapes[1]
            n_out_channel = w_shapes[0]
            virtual_shape = [s, s]
            input_shape_ct = [s * virtual_skip[0], s * virtual_skip[1]]
            n_num_per_ct = int(np.ceil(8192 / (input_shape_ct[0] * input_shape_ct[1])))
            n_packed_out_feature_for_mult_apck = int(np.ceil(n_out_channel / n_num_per_ct))
            n_block_input = int(np.ceil(n_in_channel * s * s / (N / 2))) * n_num_per_ct
            input_ct = [
                CkksCiphertextNode(f'input_ct_{i}', level) for i in range(int(np.ceil(w_shapes[1] * s * s / 8192)))
            ]
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
            output_ct = dense.call_mult_pack(input_ct, weight_pt, bias_pt, n=N)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            input_args.append(Argument(f'weight_pt', weight_pt))
            input_args.append(Argument(f'bias_pt', bias_pt))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=os.path.join(
                    base_path, f'CKKS_fc_prepare_weight_pack_cyclic_{s}_{s}_mult_apck', f'level_{level}', 'server'
                ),
            )


if __name__ == '__main__':
    unittest.main()
