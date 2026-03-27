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
from pathlib import Path
import torch
from torch import nn


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
base_path = project_root / 'build' / 'inference' / 'hetero'

from inference.lattisense.frontend.custom_task import *
from inference.model_generator.deploy_cmds import *  # noqa: E402
from inference.model_generator.layers.add_pack import AddLayer  # noqa: E402
from inference.model_generator.layers.avgpool2d_layer import Avgpool2DLayer  # noqa: E402
from inference.model_generator.layers.mult_scaler import MultScalarLayer  # noqa: E402
from training.model_export.onnx_to_json import *  # noqa: E402


def export_to_onnx(model, inputs, output_names, onnx_path, dynamic_axes=None, opset_version=18):
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
    n_in_channel, n_out_channel, input_shape, kernel_shape, stride, skip, groups, init_level, style='ordinary'
):
    set_param('PN14QP438')
    model = SimpleCNN()
    conv = nn.Conv2d(
        n_in_channel, n_out_channel, kernel_shape[0], stride[0], padding=int(kernel_shape[0] / 2), groups=groups
    )
    model.conv1 = conv
    model.eval()

    if isinstance(input_shape, (list, set)):
        input_shape = tuple(input_shape)
    if isinstance(kernel_shape, (list, set)):
        kernel_shape = tuple(kernel_shape)
    if isinstance(stride, (list, set)):
        stride = tuple(stride)

    if style == 'multiplexed':
        if groups != 1:
            task_name = 'CKKS_multiplexed_dw_conv2d'
        else:
            task_name = 'CKKS_multiplexed_conv2d'
    else:
        if groups != 1:
            task_name = 'CKKS_dw_conv2d'
        else:
            task_name = 'CKKS_conv2d'
    task_path = (
        base_path
        / task_name
        / f'stride_{stride[0]}_{stride[1]}'
        / f'kernel_shape_{kernel_shape[0]}_{kernel_shape[1]}'
        / f'cin_{n_in_channel}_cout_{n_out_channel}'
        / f'input_shape_{input_shape[0]}_{input_shape[1]}'
        / f'level_{init_level}'
    )
    task_path.mkdir(parents=True, exist_ok=True)
    task_server_path = task_path / 'server'
    task_server_path.mkdir(parents=True, exist_ok=True)

    inputx = torch.randn(1, n_in_channel, input_shape[0], input_shape[1])

    export_to_onnx(model, inputx, ['output'], f'{task_server_path}/0.onnx')
    onnx_to_json(f'{task_server_path}/0.onnx', f'{task_server_path}/nn_layers_ct_0.json', style)

    with open(f'{task_server_path}/nn_layers_ct_0.json') as f:
        model_json = json.load(f)

    model_json['feature']['input_0']['level'] = init_level
    model_json['feature']['input_0']['pack_num'] = math.ceil(8192 / (input_shape[0] * input_shape[1]))
    model_json['feature']['output']['level'] = init_level - 1
    if style == 'multiplexed' and stride[0] != 1:
        model_json['feature']['output']['level'] = init_level - 2
    model_json['feature']['output']['pack_num'] = math.ceil(
        8192 / (input_shape[0] * input_shape[1]) * (stride[0] * stride[1])
    )
    with open(f'{task_server_path}/nn_layers_ct_0.json', 'w') as f:
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
    with open(f'{task_server_path}/task_config.json', 'w') as f:
        json.dump(task_config, f, indent=4, ensure_ascii=False)

    with open(os.path.join(task_server_path, 'task_config.json'), 'r', encoding='utf-8') as file:
        config = json.load(file)

    for _ in range(len(config['server_task'])):
        gen_custom_task(str(task_server_path), use_gpu=True, style=style)


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
        set_param('PN14QP438')
        n_in_level = 2
        shapes = [16, 32, 64]
        for s in shapes:
            print(f'sub-test: s={s}')
            input_ct = [CkksCiphertextNode(f'input_ct_{i}', n_in_level) for i in range(int(np.ceil(s * s * 1 / 8192)))]
            square = SquareLayer(level=n_in_level)
            output_ct = square.call(input_ct)
            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=base_path / f'CKKS_square_{s}_{s}' / f'level_{n_in_level}' / 'server',
            )

    def test_conv2d_packed(self):
        groups = 1
        skip = (1, 1)
        init_level = 2

        # Single-channel tests (cin=1, cout=1)
        for stride in [(1, 1), (2, 2)]:
            input_shapes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)] if stride == (1, 1) else [(32, 32), (64, 64)]
            for input_shape in input_shapes:
                for kernel_shape in [(1, 1), (3, 3), (5, 5)]:
                    print(
                        f'sub-test: stride={stride}, input_shape={input_shape}, kernel_shape={kernel_shape}, '
                        f'cin=1, cout=1'
                    )
                    gen_conv_mega_ag(1, 1, input_shape, kernel_shape, stride, skip, groups, init_level)

        # Multi-channel tests (various cin x cout)
        for stride in [(1, 1), (2, 2)]:
            for n_in_channel in [1, 3, 4, 16, 17]:
                for n_out_channel in [1, 3, 4, 32, 33]:
                    print(f'sub-test: stride={stride}, n_in_channel={n_in_channel}, n_out_channel={n_out_channel}')
                    gen_conv_mega_ag(n_in_channel, n_out_channel, (32, 32), (3, 3), stride, skip, groups, init_level)

    def test_conv2d_depthwise(self):
        channels = [4, 8, 32]
        input_shapes = [(16, 16), (32, 32)]
        kernel_shapes = [(1, 1), (3, 3), (5, 5)]
        strides = [(1, 1), (2, 2)]
        skip = (1, 1)
        init_level = 5

        for n_channel in channels:
            for input_shape in input_shapes:
                for kernel_shape in kernel_shapes:
                    for stride in strides:
                        print(f'sub-test: ch={n_channel}, input={input_shape}, kernel={kernel_shape}, stride={stride}')
                        gen_conv_mega_ag(
                            n_channel,
                            n_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_channel,  # groups = n_channel for depthwise
                            init_level,
                        )

    def test_mux_conv2d_packed(self):
        skip = (1, 1)
        init_level = 5

        # Varied stride tests (cin=cout only)
        for stride in [(1, 1), (2, 2)]:
            for n_channel in [4, 8, 32]:
                print(f'sub-test: varied_stride stride={stride}, ch={n_channel}')
                gen_conv_mega_ag(n_channel, n_channel, (32, 32), (3, 3), stride, skip, 1, init_level, 'multiplexed')

        # Varied input shape tests
        for input_shape in [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]:
            print(f'sub-test: varied_input_shape input={input_shape}')
            gen_conv_mega_ag(32, 32, input_shape, (3, 3), (1, 1), skip, 1, init_level, 'multiplexed')

        # Varied kernel shape tests
        for kernel_shape in [(1, 1), (3, 3), (5, 5)]:
            print(f'sub-test: varied_kernel kernel={kernel_shape}')
            gen_conv_mega_ag(32, 32, (32, 32), kernel_shape, (1, 1), skip, 1, init_level, 'multiplexed')

        # Varied channels tests
        for n_in_channel in [1, 3, 4, 16, 17]:
            for n_out_channel in [1, 3, 4, 32, 33]:
                print(f'sub-test: varied_channels cin={n_in_channel}, cout={n_out_channel}')
                gen_conv_mega_ag(
                    n_in_channel, n_out_channel, (32, 32), (3, 3), (1, 1), skip, 1, init_level, 'multiplexed'
                )

    def test_mux_dw_s2_64x64_k3(self):
        n_in_channels = [4, 8, 32]
        n_out_channels = [4, 8, 32]
        input_shape = (64, 64)
        kernel_shape = (3, 3)
        stride = (2, 2)
        skip = (1, 1)
        init_level = 5

        for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
            groups = n_in_channel

            print(f'sub-test: n_in_channel={n_in_channel}, n_out_channel={n_out_channel}')
            gen_conv_mega_ag(
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

    def test_inverse_mux_conv(self):
        N = 16384

        skip = [1, 1]
        padding = [-1, -1]
        init_level = 1

        kernel_shapes = [[1, 1], [3, 3], [5, 5]]
        strides = [[1, 1], [2, 2]]
        block_shapes = [[64, 64], [64, 128]]
        multipliers = [2, 4]
        n_in_channels = [2, 3, 5]
        n_out_channels = [3, 4, 15]

        for kernel_shape in kernel_shapes:
            for stride in strides:
                for block_shape in block_shapes:
                    for mult in multipliers:
                        input_shape = [block_shape[0] * mult, block_shape[1] * mult]

                        for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
                            set_param('PN14QP438')

                            next_stride = [
                                input_shape[0] // (block_shape[0] * stride[0]),
                                input_shape[1] // (block_shape[1] * stride[1]),
                            ]

                            n_block_per_channel = next_stride[0] * next_stride[1] * stride[0] * stride[1]
                            n_ct_in = n_in_channel * n_block_per_channel
                            level = init_level

                            print(
                                f'sub-test: kernel_shape={kernel_shape}, stride={stride}, '
                                f'block_shape={block_shape}, input_shape={input_shape}, '
                                f'cin={n_in_channel}, cout={n_out_channel}'
                            )

                            input_ct = [CkksCiphertextNode(f'input_0input{k}', level=level) for k in range(n_ct_in)]

                            index = kernel_shape[0] * kernel_shape[1]
                            weight_pt = [
                                [
                                    [
                                        CkksPlaintextRingtNode(f'convw__conv1_{k}_{n}_{i}')
                                        for i in range(int(index * next_stride[0] * next_stride[1]))
                                    ]
                                    for n in range(n_in_channel)
                                ]
                                for k in range(n_out_channel)
                            ]
                            bias_pt = [CkksPlaintextRingtNode(f'convb__conv1_{i}') for i in range(n_out_channel)]

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

                            output_ct = big_conv.call(input_ct, weight_pt, bias_pt, N)

                            used_indices = big_conv.get_used_input_indices()
                            used_input_ct = [input_ct[i] for i in sorted(used_indices)]
                            input_args = list()
                            input_args.append(Argument('input_0', used_input_ct))
                            input_args.append(Argument('convw__conv1', weight_pt))
                            input_args.append(Argument('convb__conv1', bias_pt))

                            process_custom_task(
                                input_args=input_args,
                                output_args=[Argument('output', output_ct)],
                                output_instruction_path=base_path
                                / 'CKKS_inverse_multiplexed_conv2d'
                                / f'stride_{stride[0]}_{stride[1]}'
                                / f'kernel_shape_{kernel_shape[0]}_{kernel_shape[1]}'
                                / f'cin_{n_in_channel}_cout_{n_out_channel}'
                                / f'input_shape_{input_shape[0]}_{input_shape[1]}'
                                / f'level_{init_level}'
                                / 'server',
                            )

    def test_inverse_mux_conv_repack(self):
        N = 16384

        stride = [4, 4]
        skip = [1, 1]
        padding = [-1, -1]
        init_level = 2

        kernel_shapes = [[1, 1], [3, 3], [5, 5]]
        block_shapes = [[64, 64], [64, 128]]
        n_in_channels = [2, 3, 3]
        n_out_channels = [3, 4, 15]

        for kernel_shape in kernel_shapes:
            for block_shape in block_shapes:
                input_shape = [block_shape[0] * 2, block_shape[1] * 2]

                for n_in_channel, n_out_channel in zip(n_in_channels, n_out_channels):
                    set_param('PN14QP438')

                    next_stride = [
                        max(1, input_shape[0] // (block_shape[0] * stride[0])),
                        max(1, input_shape[1] // (block_shape[1] * stride[1])),
                    ]

                    print(
                        f'sub-test: kernel_shape={kernel_shape}, '
                        f'block_shape={block_shape}, input_shape={input_shape}, '
                        f'cin={n_in_channel}, cout={n_out_channel}'
                    )

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

                    effective_stride = big_conv.stride
                    effective_next_stride = big_conv.stride_next

                    n_block_per_channel = (
                        effective_next_stride[0] * effective_next_stride[1] * effective_stride[0] * effective_stride[1]
                    )
                    n_ct_in = n_in_channel * n_block_per_channel
                    level = init_level

                    input_ct = [CkksCiphertextNode(f'input_0input{k}', level=level) for k in range(n_ct_in)]

                    index = kernel_shape[0] * kernel_shape[1]
                    weight_pt = [
                        [
                            [
                                CkksPlaintextRingtNode(f'convw__conv1_{k}_{n}_{i}')
                                for i in range(int(index * effective_next_stride[0] * effective_next_stride[1]))
                            ]
                            for n in range(n_in_channel)
                        ]
                        for k in range(n_out_channel)
                    ]
                    bias_pt = [CkksPlaintextRingtNode(f'convb__conv1_{i}') for i in range(n_out_channel)]

                    repack_mask_node = CkksPlaintextRingtNode('repack_mask__conv1') if big_conv.need_repack else None
                    output_ct = big_conv.call(input_ct, weight_pt, bias_pt, N, repack_mask_pt=repack_mask_node)

                    used_indices = big_conv.get_used_input_indices()
                    used_input_ct = [input_ct[i] for i in sorted(used_indices)]
                    input_args = list()
                    input_args.append(Argument('input_0', used_input_ct))
                    input_args.append(Argument('convw__conv1', weight_pt))
                    input_args.append(Argument('convb__conv1', bias_pt))
                    if repack_mask_node is not None:
                        input_args.append(Argument('repack_mask__conv1', [repack_mask_node]))

                    process_custom_task(
                        input_args=input_args,
                        output_args=[Argument('output', output_ct)],
                        output_instruction_path=base_path
                        / 'CKKS_inverse_multiplexed_conv2d'
                        / f'stride_{stride[0]}_{stride[1]}'
                        / f'kernel_shape_{kernel_shape[0]}_{kernel_shape[1]}'
                        / f'cin_{n_in_channel}_cout_{n_out_channel}'
                        / f'input_shape_{input_shape[0]}_{input_shape[1]}'
                        / f'level_{init_level}'
                        / 'server',
                    )

    def test_poly_bsgs(self):
        N = 16384
        set_param('PN14QP438')
        n_in_channel = 32
        input_shape = (32, 32)
        skip = (1, 1)
        n_in_channel_per_ct = int(np.floor(N / 2 / (input_shape[0] * input_shape[1])))
        n_pack_in_channel = int(np.ceil(n_in_channel / n_in_channel_per_ct))
        orders = [2, 4, 6, 8, 10, 12, 16, 32, 64]
        level = 8

        for order in orders:
            print(f'sub-test: order={order}')
            input_ct = [CkksCiphertextNode(f'input{k}', level) for k in range(n_pack_in_channel)]
            weight_pt = [
                [CkksPlaintextRingtNode(f'polyw_{1}_{i}_{j}') for j in range(n_pack_in_channel)]
                for i in range(order + 1)
            ]

            poly_layer = PolyRelu(input_shape, order, skip, n_in_channel_per_ct)
            output_ct = poly_layer.call_bsgs_feature2d(input_ct, weight_pt)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            for i in range(order + 1):
                input_args.append(Argument(f'weight_pt{i}', weight_pt[i]))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=base_path
                / f'CKKS_poly_relu_bsgs_{n_in_channel}_channel_order_{order}'
                / f'level_{level}',
            )

    def test_fc_pack_skip_feature0d(self):
        N = 16384
        set_param('PN14QP438')
        w_shape = [10, 4096]
        virtual_shape = [1, 1]
        level = 2

        skips = [2, 4, 8, 16]
        for s in skips:
            print(f'sub-test: skip={s}')
            n_in_channel = w_shape[1]
            n_out_channel = w_shape[0]
            virtual_skip = [s, s]
            skip_0d = s * s  # ciphertext_skip = skip[0] * skip[1]
            n_channel_per_ct = int(N / 2 / skip_0d)
            pack = n_channel_per_ct
            n_packed_out_feature = int(np.ceil(n_out_channel / n_channel_per_ct))
            n_packed_in_feature = int(np.ceil(n_in_channel / n_channel_per_ct))
            dense = DensePackedLayer(
                n_out_channel,
                n_in_channel,
                virtual_shape,
                virtual_skip,
                pack,
                n_packed_in_feature,
                n_packed_out_feature,
            )
            bsgs_bs = int(np.ceil(np.sqrt(pack)))
            weight_pt_size = n_packed_in_feature * pack
            input_ct = [
                CkksCiphertextNode(f'input_ct_{i}', level) for i in range(int(np.ceil(n_in_channel / n_channel_per_ct)))
            ]
            weight_pt = [
                [CkksPlaintextRingtNode(f'weight_pt_{i}_{j}') for j in range(weight_pt_size)]
                for i in range(n_packed_out_feature)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature)]

            output_ct = dense.call_skip_0d(input_ct, weight_pt, bias_pt, skip_0d)

            input_args = list()
            input_args.append(Argument('input_node', input_ct))
            input_args.append(Argument(f'weight_pt', weight_pt))
            input_args.append(Argument(f'bias_pt', bias_pt))

            process_custom_task(
                input_args=input_args,
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=base_path
                / f'CKKS_fc_prepare_weight1_1D_pack_skip_{s}_{s}'
                / f'level_{level}'
                / 'server',
            )

    def test_fc_fc_feature0d(self):
        N = 16384
        set_param('PN14QP438')
        n_slot = 8192
        init_level = 2
        input_channel = 1024
        output_channel = 1024
        dense_shape = [4, 4]
        skip = (1, 1)

        output_channel1 = 128
        dense_shape1 = [1, 1]
        skip1 = [dense_shape[0] * skip[0], dense_shape[1] * skip[1]]

        # --- Layer 0: multiplexed (2D spatial layout) ---
        input_ct_shape0 = [dense_shape[0] * skip[0], dense_shape[1] * skip[1]]
        n_num_pre_ct0 = int(np.ceil(n_slot / (input_ct_shape0[0] * input_ct_shape0[1])))
        n_packed_out_feature0 = int(np.ceil(output_channel / n_num_pre_ct0))

        valid_skip_0 = skip[0]
        valid_skip_1 = skip[1]
        n_channel_per_block = valid_skip_0 * valid_skip_1
        n_channel0 = input_channel // (dense_shape[0] * dense_shape[1])
        n_block_input0 = int(np.ceil(n_channel0 / (n_num_pre_ct0 * n_channel_per_block))) * n_num_pre_ct0

        input_ct = [
            CkksCiphertextNode(f'input_ct0_{i}', init_level) for i in range(int(np.ceil(input_channel / n_slot)))
        ]
        weight_pt0 = [
            [CkksPlaintextRingtNode(f'weight_pt0_{i}_{j}') for j in range(n_block_input0)]
            for i in range(n_packed_out_feature0)
        ]
        bias_pt0 = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_feature0)]
        dense0 = DensePackedLayer(
            output_channel,
            input_channel,
            dense_shape,
            skip,
            n_num_pre_ct0,
            int(np.ceil(input_channel / n_num_pre_ct0)),
            n_packed_out_feature0,
        )
        res0 = dense0.call_multiplexed(input_ct, weight_pt0, bias_pt0, N)

        # --- Layer 1: skip_0d ---
        skip_0d1 = skip1[0] * skip1[1]
        n_channel_per_ct1 = int(N / 2 / skip_0d1)
        pack1 = n_channel_per_ct1
        n_packed_in_feature1 = int(np.ceil(output_channel / n_channel_per_ct1))
        n_packed_out_feature1 = int(np.ceil(output_channel1 / n_channel_per_ct1))
        weight_pt_size1 = n_packed_in_feature1 * pack1
        weight_pt1 = [
            [CkksPlaintextRingtNode(f'weight_pt1_{i}_{j}') for j in range(weight_pt_size1)]
            for i in range(n_packed_out_feature1)
        ]
        bias_pt1 = [CkksPlaintextRingtNode(f'bias_pt1_{i}') for i in range(n_packed_out_feature1)]
        dense1 = DensePackedLayer(
            output_channel1,
            output_channel,
            dense_shape1,
            skip1,
            pack1,
            n_packed_in_feature1,
            n_packed_out_feature1,
        )
        output_ct = dense1.call_skip_0d(res0, weight_pt1, bias_pt1, skip_0d1)

        input_args = list()
        input_args.append(Argument('input_node', input_ct))
        input_args.append(Argument(f'weight_pt0', weight_pt0))
        input_args.append(Argument(f'bias_pt0', bias_pt0))
        input_args.append(Argument(f'weight_pt1', weight_pt1))
        input_args.append(Argument(f'bias_pt1', bias_pt1))

        process_custom_task(
            input_args=input_args,
            output_args=[Argument('output_ct', output_ct)],
            output_instruction_path=base_path
            / f'CKKS_fc_fc_{input_channel}_{output_channel}_{output_channel1}'
            / f'level_{init_level}'
            / 'server',
        )

    def test_fc_multiplexed_feature2d(self):
        N = 16384
        set_param('PN14QP438')
        level = 3
        n_in_channel = 64
        n_out_channel = 10

        configs = [
            ([1, 1], [2, 2], [1, 1]),
            ([1, 1], [4, 4], [1, 1]),
            ([1, 1], [8, 8], [1, 1]),
            ([1, 1], [32, 32], [8, 8]),
            ([1, 1], [16, 16], [4, 4]),
            ([2, 2], [4, 4], [4, 4]),
        ]

        for shape, skip, invalid_fill in configs:
            input_ct_shape = [shape[0] * skip[0], shape[1] * skip[1]]
            n_num_per_ct = int(np.ceil(N / 2 / (input_ct_shape[0] * input_ct_shape[1])))
            n_packed_out = int(np.ceil(n_out_channel / n_num_per_ct))
            valid_skip_0 = skip[0] // invalid_fill[0]
            valid_skip_1 = skip[1] // invalid_fill[1]
            n_channel_per_block = valid_skip_0 * valid_skip_1
            n_channel = n_in_channel // (shape[0] * shape[1])
            n_channel_per_ct = int((N / 2) / (shape[0] * shape[1]) / (invalid_fill[0] * invalid_fill[1]))
            n_input_ct = max(1, int(np.ceil(n_in_channel / n_channel_per_ct)))
            n_block_input = n_input_ct * n_num_per_ct

            input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(n_input_ct)]
            weight_pt = [
                [CkksPlaintextRingtNode(f'weight_pt_{i}_{j}') for j in range(n_block_input)]
                for i in range(n_packed_out)
            ]
            bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out)]

            dense = DensePackedLayer(
                n_out_channel,
                n_in_channel,
                shape,
                skip,
                n_num_per_ct,
                n_in_channel,
                n_out_channel,
                invalid_fill=invalid_fill,
            )
            output_ct = dense.call_multiplexed(input_ct, weight_pt, bias_pt, N)

            path_name = (
                f'CKKS_fc_multiplexed'
                f'_shape{shape[0]}x{shape[1]}'
                f'_skip{skip[0]}x{skip[1]}'
                f'_inv{invalid_fill[0]}x{invalid_fill[1]}'
            )
            process_custom_task(
                input_args=[
                    Argument('input_node', input_ct),
                    Argument('weight_pt', weight_pt),
                    Argument('bias_pt', bias_pt),
                ],
                output_args=[Argument('output_ct', output_ct)],
                output_instruction_path=base_path / path_name / f'level_{level}' / 'server',
            )

    def test_poly_relu_bsgs_feature2d(self):
        N = 16384
        set_param('PN14QP438')
        n_in_channel = 32
        input_shape = (32, 32)
        skip = (1, 1)
        n_in_channel_per_ct = int(np.floor(N / 2 / (input_shape[0] * input_shape[1])))
        n_pack_in_channel = int(np.ceil(n_in_channel / n_in_channel_per_ct))
        order0 = 7
        order1 = 7
        level_cost0 = PolyRelu.compute_bsgs_level_cost(order0)
        level_cost1 = PolyRelu.compute_bsgs_level_cost(order1)
        level = level_cost0 + level_cost1 + 1  # +1 for sign(x)*x multiplication

        input_ct = [CkksCiphertextNode(f'input{k}', level) for k in range(n_pack_in_channel)]
        weight_pt0 = [
            [CkksPlaintextRingtNode(f'poly0w_{i}_{j}') for j in range(n_pack_in_channel)] for i in range(order0 + 1)
        ]
        poly_layer0 = PolyRelu(input_shape, order0, skip, n_in_channel_per_ct)
        output_ct0 = poly_layer0.call_bsgs_feature2d(input_ct, weight_pt0)

        weight_pt1 = [
            [CkksPlaintextRingtNode(f'poly1w_{i}_{j}') for j in range(n_pack_in_channel)] for i in range(order1 + 1)
        ]
        poly_layer1 = PolyRelu(input_shape, order1, skip, n_in_channel_per_ct)
        output_ct1 = poly_layer1.call_bsgs_feature2d(output_ct0, weight_pt1)

        output_ct = list()
        for idx in range(len(output_ct1)):
            tmp = input_ct[idx]
            if tmp.level > output_ct1[idx].level:
                tmp = drop_level(tmp, tmp.level - output_ct1[idx].level)
            tmp0 = rescale(relin(mult(output_ct1[idx], tmp)))
            tmp1 = input_ct[idx]
            if tmp1.level > tmp0.level:
                tmp1 = drop_level(tmp1, tmp1.level - tmp0.level)
            output_ct.append(add(tmp0, tmp1))

        input_args = list()
        input_args.append(Argument('input_node', input_ct))
        for i in range(order0 + 1):
            input_args.append(Argument(f'poly0_weight_pt{i}', weight_pt0[i]))
        for i in range(order1 + 1):
            input_args.append(Argument(f'poly1_weight_pt{i}', weight_pt1[i]))
        process_custom_task(
            input_args=input_args,
            output_args=[Argument('output_ct', output_ct)],
            output_instruction_path=base_path
            / f'CKKS_poly_relu_bsgs_{n_in_channel}_channel_order_{order0}_{order1}'
            / f'level_{level}',
        )

    def test_poly_bsgs_feature0d(self):
        N = 16384
        set_param('PN14QP438')
        n_in_channel = 32
        orders = [2, 4, 6, 8]
        skips = [1, 2, 128, 256]

        for skip_val in skips:
            n_channel_per_ct = N // 2 // skip_val
            n_pack_in_channel = int(np.ceil(n_in_channel / n_channel_per_ct))
            level = 8

            for order in orders:
                level_cost = PolyRelu.compute_bsgs_level_cost(order)
                if level < level_cost:
                    continue

                print(f'sub-test: skip={skip_val}, order={order}')
                input_ct = [CkksCiphertextNode(f'input{k}', level) for k in range(n_pack_in_channel)]
                weight_pt = [
                    [CkksPlaintextRingtNode(f'polyw_0d_{i}_{j}') for j in range(n_pack_in_channel)]
                    for i in range(order + 1)
                ]

                poly_layer = PolyRelu.create_for_feature0d(order, skip_val, n_channel_per_ct)
                output_ct = poly_layer.call_bsgs_feature0d(input_ct, weight_pt)

                input_args = list()
                input_args.append(Argument('input_node', input_ct))
                for i in range(order + 1):
                    input_args.append(Argument(f'weight_pt{i}', weight_pt[i]))

                process_custom_task(
                    input_args=input_args,
                    output_args=[Argument('output_ct', output_ct)],
                    output_instruction_path=base_path
                    / f'CKKS_poly_relu_bsgs_feature0d_{n_in_channel}_channel_order_{order}_skip_{skip_val}'
                    / f'level_{level}',
                )

    def test_conv1d_layer(self):
        N = 16384
        set_param('PN14QP438')
        n_in_channel = 4
        n_out_channel = 4
        init_level = 5

        input_shapes = [32, 64, 512]
        kernel_shapes = [1, 3, 5]
        skips = [2, 4]
        strides = [1, 2]

        for input_shape in input_shapes:
            for kernel_shape in kernel_shapes:
                for skip in skips:
                    for stride in strides:
                        print(
                            f'sub-test: input_shape={input_shape}, kernel_shape={kernel_shape}, skip={skip}, stride={stride}'
                        )
                        n_channel_per_ct = int(N / 2 / input_shape / skip)
                        n_pack_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
                        n_packed_out_channel = math.ceil(n_out_channel / (n_channel_per_ct * stride))
                        input_ct = [CkksCiphertextNode(f'input_{k}', init_level) for k in range(n_pack_in_channel)]
                        rot_n_channel_per_ct = n_in_channel if n_in_channel < n_channel_per_ct else n_channel_per_ct
                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'weight_{i}_{k}_{j}') for k in range(kernel_shape)]
                                for j in range(rot_n_channel_per_ct)
                            ]
                            for i in range(n_packed_out_channel)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_channel)]

                        conv1d = Conv1DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_channel_per_ct,
                            n_pack_in_channel,
                            n_packed_out_channel,
                        )
                        output_ct = conv1d.call(input_ct, weight_pt, bias_pt)

                        input_args = list()
                        input_args.append(Argument('input_node', input_ct))
                        input_args.append(Argument(f'weight_pt', weight_pt))
                        input_args.append(Argument(f'bias_pt', bias_pt))

                        process_custom_task(
                            input_args=input_args,
                            output_args=[Argument('output_ct', output_ct)],
                            output_instruction_path=base_path
                            / f'conv1d_input_shape_{input_shape}_kernel_shape_{kernel_shape}_skip_{skip}_stride_{stride}'
                            / f'level_{init_level}'
                            / 'server',
                        )

    def test_mux_conv1d_layer(self):
        N = 16384
        set_param('PN14QP438')

        n_in_channel = 16
        n_out_channel = 32
        init_level = 5

        input_shapes = [32, 64, 512]
        kernel_shapes = [1, 3, 5]
        skips = [2, 4]
        strides = [1, 2]

        for input_shape in input_shapes:
            for kernel_shape in kernel_shapes:
                for skip in skips:
                    for stride in strides:
                        print(
                            f'sub-test: input_shape={input_shape}, kernel_shape={kernel_shape}, skip={skip}, stride={stride}'
                        )
                        n_channel_per_ct = math.ceil(N / 2 / input_shape)
                        n_packed_in_channel = math.ceil(n_in_channel / n_channel_per_ct)
                        n_packed_out_channel = math.ceil(n_out_channel / n_channel_per_ct)
                        n_block_per_ct = math.ceil(n_channel_per_ct / skip)
                        n_weight_pt = math.ceil(n_out_channel / n_block_per_ct)
                        input_ct = [CkksCiphertextNode(f'input_{k}', init_level) for k in range(n_packed_in_channel)]
                        weight_pt = [
                            [
                                [CkksPlaintextRingtNode(f'weight_pt_{i}_{k}_{j}') for k in range(kernel_shape)]
                                for j in range(n_packed_in_channel * n_block_per_ct)
                            ]
                            for i in range(n_weight_pt)
                        ]
                        bias_pt = [CkksPlaintextRingtNode(f'bias_pt_{i}') for i in range(n_packed_out_channel)]
                        conv1d = ParMultiplexedConv1DPackedLayer(
                            n_out_channel,
                            n_in_channel,
                            input_shape,
                            kernel_shape,
                            stride,
                            skip,
                            n_channel_per_ct,
                            n_packed_in_channel,
                            n_packed_out_channel,
                        )
                        n_select_pt = min(n_block_per_ct, n_out_channel)
                        block_select_pt = [CkksPlaintextRingtNode(f'select_pt_{i}') for i in range(n_select_pt)]
                        output_ct = conv1d.call(input_ct, weight_pt, bias_pt, block_select_pt)

                        input_args = list()
                        input_args.append(Argument('input_node', input_ct))
                        input_args.append(Argument(f'weight_pt', weight_pt))
                        input_args.append(Argument(f'bias_pt', bias_pt))
                        if len(block_select_pt) != 0:
                            input_args.append(Argument(f'block_select_pt', block_select_pt))

                        process_custom_task(
                            input_args=input_args,
                            output_args=[Argument('output_ct', output_ct)],
                            output_instruction_path=base_path
                            / f'multiplexed_conv1d_input_shape_{input_shape}_kernel_shape_{kernel_shape}_skip_{skip}_stride_{stride}'
                            / f'level_{init_level}'
                            / 'server',
                        )

    def test_add_layer(self):
        N = 16384
        set_param('PN14QP438')
        level = 2
        skip = (1, 1)
        shapes = [16, 32]
        channels = [4, 32]
        for n_channel in channels:
            for s in shapes:
                print(f'sub-test: n_channel={n_channel}, s={s}')
                n_ct = int(np.ceil(n_channel / (N / 2 / (s * s))))
                input_ct1 = [CkksCiphertextNode(f'input_ct1_{i}', level) for i in range(n_ct)]
                input_ct2 = [CkksCiphertextNode(f'input_ct2_{i}', level) for i in range(n_ct)]
                add_layer = AddLayer()
                output_ct = add_layer.call(input_ct1, input_ct2)
                input_args = list()
                input_args.append(Argument('input_node1', input_ct1))
                input_args.append(Argument('input_node2', input_ct2))
                process_custom_task(
                    input_args=input_args,
                    output_args=[Argument('output_ct', output_ct)],
                    output_instruction_path=base_path
                    / f'CKKS_add_layer/ch_{n_channel}_shape_{s}_{s}'
                    / f'level_{level}'
                    / 'server',
                )

    def test_avgpool2d_layer(self):
        N = 16384
        set_param('PN14QP438')
        level = 3
        skip = (1, 1)
        shapes = [8, 16, 32, 64]
        channels = [4, 10, 15, 32, 37]
        strides = [(2, 2), (4, 4), (8, 8)]
        for stride in strides:
            for n_channel in channels:
                for s in shapes:
                    if s < stride[0]:
                        continue  # shape must be >= stride
                    print(f'sub-test: stride={stride[0]}, n_channel={n_channel}, s={s}')
                    n_channel_per_ct = int(np.ceil(N / 2 / (s * s)))
                    n_ct = int(np.ceil(n_channel / n_channel_per_ct))
                    input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(n_ct)]
                    out_channels_per_ct = n_channel_per_ct * stride[0] * stride[1]
                    n_select_pt = min(n_channel, out_channels_per_ct)
                    select_tensor_pt = [CkksPlaintextRingtNode(f'select_pt_{i}') for i in range(n_select_pt)]
                    avgpool = Avgpool2DLayer(stride=list(stride), shape=[s, s], channel=n_channel, skip=[1, 1])
                    output_ct = avgpool.call_multiplexed_avgpool(
                        input_ct, select_tensor_pt, n_channel, n_channel_per_ct
                    )
                    input_args = list()
                    input_args.append(Argument('input_node', input_ct))
                    input_args.append(Argument('select_tensor_pt', select_tensor_pt))
                    process_custom_task(
                        input_args=input_args,
                        output_args=[Argument('output_ct', output_ct)],
                        output_instruction_path=base_path
                        / f'CKKS_avgpool2d/stride_{stride[0]}_{stride[1]}'
                        / f'ch_{n_channel}_shape_{s}_{s}'
                        / f'level_{level}'
                        / 'server',
                    )

    def test_adaptive_avgpool2d_layer(self):
        N = 16384
        set_param('PN14QP438')
        level = 3
        skip = (1, 1)
        shapes = [8, 16, 32, 64]
        channels = [4, 10, 15, 32, 37]
        strides = [(2, 2), (4, 4), (8, 8)]
        for stride in strides:
            for n_channel in channels:
                for s in shapes:
                    if s < stride[0]:
                        continue
                    print(f'sub-test: stride={stride[0]}, n_channel={n_channel}, s={s}')
                    n_channel_per_ct = int(np.ceil(N / 2 / (s * s)))
                    n_ct = int(np.ceil(n_channel / n_channel_per_ct))
                    input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(n_ct)]
                    avgpool = Avgpool2DLayer(stride=list(stride), shape=[s, s], channel=n_channel, skip=[1, 1])
                    output_ct = avgpool.run_adaptive_avgpool(input_ct, n=N)
                    input_args = list()
                    input_args.append(Argument('input_node', input_ct))
                    process_custom_task(
                        input_args=input_args,
                        output_args=[Argument('output_ct', output_ct)],
                        output_instruction_path=base_path
                        / f'CKKS_adaptive_avgpool2d/stride_{stride[0]}_{stride[1]}'
                        / f'ch_{n_channel}_shape_{s}_{s}'
                        / f'level_{level}'
                        / 'server',
                    )

    def test_interleaved_avgpool2d_layer(self):
        N = 16384
        set_param('PN14QP438')
        level = 3
        channels = [2, 4, 8]
        strides = [(2, 2), (4, 4)]
        block_shapes = [(64, 64)]
        multipliers = [2, 4]

        for stride in strides:
            for n_channel in channels:
                for block_shape in block_shapes:
                    for mult in multipliers:
                        if mult < stride[0]:
                            continue  # block_expansion must be >= stride
                        input_shape = [block_shape[0] * mult, block_shape[1] * mult]
                        block_expansion = [input_shape[0] // block_shape[0], input_shape[1] // block_shape[1]]
                        n_ct_in = n_channel * block_expansion[0] * block_expansion[1]

                        print(
                            f'sub-test: stride={stride}, n_channel={n_channel}, '
                            f'block_shape={block_shape}, input_shape={input_shape}'
                        )

                        input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(n_ct_in)]
                        avgpool = Avgpool2DLayer(
                            stride=list(stride),
                            shape=[block_shape[0], block_shape[1]],
                            channel=n_channel,
                            skip=[1, 1],
                        )
                        output_ct = avgpool.call_interleaved_avgpool(input_ct, block_expansion)

                        input_args = [Argument('input_node', input_ct)]
                        process_custom_task(
                            input_args=input_args,
                            output_args=[Argument('output_ct', output_ct)],
                            output_instruction_path=base_path
                            / f'CKKS_interleaved_avgpool2d/stride_{stride[0]}_{stride[1]}'
                            / f'ch_{n_channel}'
                            / f'block_shape_{block_shape[0]}_{block_shape[1]}'
                            / f'input_shape_{input_shape[0]}_{input_shape[1]}'
                            / f'level_{level}'
                            / 'server',
                        )

    def test_mult_scalar_layer(self):
        N = 16384
        set_param('PN14QP438')
        level = 3
        skip = (1, 1)
        s = 32
        n_channel = 32
        print(f'sub-test: n_channel={n_channel}, s={s}')
        n_channel_per_ct = int(np.ceil(N / 2 / (s * s)))
        n_ct = int(np.ceil(n_channel / n_channel_per_ct))
        input_ct = [CkksCiphertextNode(f'input_ct_{i}', level) for i in range(n_ct)]
        weight_pt = [CkksPlaintextRingtNode(f'weight_pt_{i}') for i in range(n_ct)]
        mult_layer = MultScalarLayer()
        output_ct = mult_layer.call(input_ct, weight_pt)
        input_args = list()
        input_args.append(Argument('input_node', input_ct))
        input_args.append(Argument('weight_pt', weight_pt))
        process_custom_task(
            input_args=input_args,
            output_args=[Argument('output_ct', output_ct)],
            output_instruction_path=base_path
            / f'CKKS_mult_scalar/ch_{n_channel}_shape_{s}_{s}'
            / f'level_{level}'
            / 'server',
        )


if __name__ == '__main__':
    unittest.main()
