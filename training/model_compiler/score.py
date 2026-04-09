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
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from components import (
    ComputeNode,
    FheParameter,
    FeatureNode,
    LayerAbstractGraph,
    config,
)
from inference.model_generator.layers.activation_layer import SquareLayer
from inference.model_generator.layers.add_pack import AddLayer
from inference.model_generator.layers.avgpool1d_layer import Avgpool1DLayer
from inference.model_generator.layers.avgpool2d_layer import Avgpool2DLayer
from inference.model_generator.layers.conv1d_packed_layer import Conv1DPackedLayer
from inference.model_generator.layers.conv2d_depthwise import Conv2DPackedDepthwiseLayer
from inference.model_generator.layers.conv2d_packed_layer import Conv2DPackedLayer
from inference.model_generator.layers.dense_packed_layer import DensePackedLayer
from inference.model_generator.layers.inverse_multiplexed_conv2d_layer import InverseMultiplexedConv2DLayer
from inference.model_generator.layers.mult_scaler import MultScalarLayer
from inference.model_generator.layers.multiplexed_conv1d_pack_layer import MultiplexedConv1DPackedLayer
from inference.model_generator.layers.multiplexed_conv2d_pack_layer import MultiplexedConv2DPackedLayer
from inference.model_generator.layers.multiplexed_conv2d_pack_layer_depthwise import (
    MultiplexedConv2DPackedLayerDepthwise,
)
from inference.model_generator.layers.multiplexed_dw_conv1d_pack_layer import MultiplexedDWConv1DPackedLayer
from inference.model_generator.layers.poly_relu0d import PolyRelu0D
from inference.model_generator.layers.poly_relu1d import PolyRelu1D
from inference.model_generator.layers.poly_relu2d import PolyRelu2D
from inference.model_generator.layers.upsample_layer import UpsampleNearestLayer


def get_multithread_rate_for_btp(task_num: int):
    if config.single_thread:
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
    if config.single_thread:
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
    if config.single_thread:
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
    if config.single_thread:
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


def get_multithread_rate_for_weight_ops(task_num: int):
    if config.single_thread:
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

mpc_refresh_rate = 1 / 15
ct_trans_rate = 1 / 10


class FheScoreParam:
    def __init__(self, dag: nx.DiGraph, compute_node: ComputeNode, param: dict[str, FheParameter], level) -> None:
        preds: list[FeatureNode] = list(dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(dag.successors(compute_node))

        self.dag = dag
        self.acc_rate = 1
        self.compute_node = compute_node
        self.input_mult_level = dag.nodes[preds[0]]['level']
        self.output_mult_level = dag.nodes[succs[0]]['level']
        self.input_degree = param[preds[0].ckks_parameter_id].poly_modulus_degree
        self.output_degree = param[succs[0].ckks_parameter_id].poly_modulus_degree
        if 'conv' in compute_node.layer_type:
            self.stride = compute_node.stride
            self.kernel_shape = compute_node.kernel_shape
        if compute_node.layer_type in {'avgpool1d', 'avgpool2d'}:
            self.stride = compute_node.stride
        if preds[0].dim != 0:
            self.input_shape = preds[0].shape
            self.output_shape = succs[0].shape
            self.input_skip = dag.nodes[preds[0]]['skip']
            self.output_skip = dag.nodes[succs[0]]['skip']
        # else:
        #     self.input_shape = preds[0].sp_info['shape']
        #     self.output_shape = succs[0].sp_info['shape']
        #     self.input_skip = preds[0].sp_info['skip']
        #     self.output_skip = succs[0].sp_info['skip']

        self.pack = dag.nodes[preds[0]]['pack_num']
        self.pack_out = dag.nodes[succs[0]]['pack_num']

        self.input_channel = compute_node.channel_input
        self.output_channel = compute_node.channel_output
        self.n_packed_in = math.ceil(self.input_channel / self.pack)
        self.n_packed_out = math.ceil(self.output_channel / self.pack_out)
        # self.level = level
        if level > 0:
            self.mult_score = mult_time[self.input_degree][level]
            self.mult_plain_score = mult_plain_time[self.input_degree][level]
            self.rescale_score = rescale_time[self.input_degree][level]
        else:
            self.mult_score = 0
            self.mult_plain_score = 0
            self.rescale_score = 0
        self.rotate_score = rotate_time[self.input_degree][level]
        self.add_score = add_time[self.input_degree][level]

    def get_score(self) -> float:
        """Compute layer latency score by instantiating the exact inference layer class,
        calling get_fhe_op_count() to get primitive op counts, then multiplying by per-op
        timing constants.

        op_counts keys: 'rotate', 'mult_plain', 'mult', 'add', 'rescale'
        """
        preds: list[FeatureNode] = list(self.dag.predecessors(self.compute_node))
        n = self.input_degree  # poly_modulus_degree = 2*N_slots
        layer_type = self.compute_node.layer_type
        style = config.style

        op_counts = self._build_layer_and_get_op_count(preds, n, layer_type, style)
        # op_counts may be a flat dict {str: int} (single level) or a level-grouped
        # dict {int: {str: int}} when ops span multiple levels (e.g. MultiplexedConv1D).
        if op_counts and isinstance(next(iter(op_counts)), int):
            score = 0.0
            for lv, ops in op_counts.items():
                r_score = rotate_time[self.input_degree][lv]
                mp_score = mult_plain_time[self.input_degree][lv] if lv > 0 else 0
                m_score = mult_time[self.input_degree][lv] if lv > 0 else 0
                a_score = add_time[self.input_degree][lv]
                rs_score = rescale_time[self.input_degree][lv] if lv > 0 else 0
                score += (
                    ops['rotate'] * r_score
                    + ops['mult_plain'] * mp_score
                    + ops['mult'] * m_score
                    + ops['add'] * a_score
                    + ops['rescale'] * rs_score
                )
        else:
            score = (
                op_counts['rotate'] * self.rotate_score
                + op_counts['mult_plain'] * self.mult_plain_score
                + op_counts['mult'] * self.mult_score
                + op_counts['add'] * self.add_score
                + op_counts['rescale'] * self.rescale_score
            )
        return score * self.acc_rate

    def _build_layer_and_get_op_count(self, preds, n, layer_type, style):
        """Instantiate the matching inference layer and return its get_fhe_op_count() dict.
        Returns None if the layer type is not handled here.
        """
        node = self.compute_node
        n_in = self.input_channel
        n_out = self.output_channel
        pack = self.pack
        n_packed_in = self.n_packed_in  # ceil(n_in / pack)
        n_packed_out = self.n_packed_out  # ceil(n_out / pack_out)

        # ── conv2d ──────────────────────────────────────────────────────────
        if layer_type == 'conv2d' and node.dim == 2:
            input_shape = self.input_shape
            kernel_shape = node.kernel_shape
            stride = node.stride
            skip = self.input_skip
            groups = node.groups
            is_depthwise = groups == n_out and groups != 1
            is_big_size = getattr(node, 'is_big_size', False)

            if is_big_size:
                block_shape = config.block_shape
                padding = [-1, -1]
                next_stride = [
                    math.ceil(input_shape[0] / block_shape[0]) // stride[0],
                    math.ceil(input_shape[1] / block_shape[1]) // stride[1],
                ]
                layer = InverseMultiplexedConv2DLayer(
                    n_out,
                    n_in,
                    input_shape,
                    padding,
                    kernel_shape,
                    stride,
                    next_stride,
                    skip,
                    block_shape,
                )
                return layer.get_fhe_op_count(n)

            if style == 'ordinary':
                if is_depthwise:
                    layer = Conv2DPackedDepthwiseLayer(
                        n_out,
                        n_in,
                        input_shape,
                        kernel_shape,
                        stride,
                        skip,
                        pack,
                        n_packed_in,
                        n_packed_out,
                    )
                else:
                    layer = Conv2DPackedLayer(
                        n_out,
                        n_in,
                        input_shape,
                        kernel_shape,
                        stride,
                        skip,
                        pack,
                        n_packed_in,
                        n_packed_out,
                    )
                return layer.get_fhe_op_count(self.input_mult_level)

            # style == 'multiplexed'
            n_in_channel_per_ct = pack
            if is_depthwise:
                layer = MultiplexedConv2DPackedLayerDepthwise(
                    n_out,
                    n_in,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    n_in_channel_per_ct,
                    n_packed_in,
                    n_packed_out,
                )
            else:
                layer = MultiplexedConv2DPackedLayer(
                    n_out,
                    n_in,
                    input_shape,
                    kernel_shape,
                    stride,
                    skip,
                    pack,
                    n_packed_in,
                    n_packed_out,
                )
            return layer.get_fhe_op_count(self.input_mult_level)

        # ── conv1d ──────────────────────────────────────────────────────────
        elif layer_type == 'conv1d' and node.dim == 1:
            input_shape_1d = self.input_shape[0]
            kernel_shape_1d = node.kernel_shape[0]
            stride_1d = node.stride[0]
            skip_1d = self.input_skip[0] if isinstance(self.input_skip, list) else self.input_skip
            groups = node.groups
            is_depthwise = groups == n_out and groups != 1

            if style == 'multiplexed':
                n_channel_per_ct = math.ceil(n // 2 / input_shape_1d)
                n_packed_in_ch = math.ceil(n_in / n_channel_per_ct)
                n_packed_out_ch = math.ceil(n_out / n_channel_per_ct)
                if is_depthwise:
                    n_packed_ct = math.ceil(n_out / n_channel_per_ct)
                    layer = MultiplexedDWConv1DPackedLayer(
                        n_out,
                        input_shape_1d,
                        kernel_shape_1d,
                        stride_1d,
                        skip_1d,
                        n_channel_per_ct,
                        n_packed_ct,
                    )
                else:
                    layer = MultiplexedConv1DPackedLayer(
                        n_out,
                        n_in,
                        input_shape_1d,
                        kernel_shape_1d,
                        stride_1d,
                        skip_1d,
                        n_channel_per_ct,
                        n_packed_in_ch,
                        n_packed_out_ch,
                    )
                return layer.get_fhe_op_count(self.input_mult_level)

            # style == 'ordinary'
            n_channel_per_ct = int(n // 2 // input_shape_1d // skip_1d)
            n_pack_in = math.ceil(n_in / n_channel_per_ct)
            n_packed_out_ch = math.ceil(n_out / (n_channel_per_ct * stride_1d))
            layer = Conv1DPackedLayer(
                n_out,
                n_in,
                input_shape_1d,
                kernel_shape_1d,
                stride_1d,
                skip_1d,
                n_channel_per_ct,
                n_pack_in,
                n_packed_out_ch,
            )
            return layer.get_fhe_op_count(self.input_mult_level)

        # ── fc0 (dense) ──────────────────────────────────────────────────────
        elif 'fc' in layer_type:
            from components import ReshapeComputeNode

            pred = preds[0]
            # pred_node = next(self.dag.predecessors(self.compute_node), None)
            pred_com = next(self.dag.predecessors(pred), None)

            if pred.dim == 0:
                sp_info = pred.sp_info
                special_shape = sp_info.get('shape', [1, 1])
                if not (pred_com and isinstance(pred_com, ReshapeComputeNode)):
                    # call_skip_0d path
                    skip_0d = sp_info['skip'][0] if isinstance(sp_info.get('skip'), list) else 1
                    n_channel_per_ct = int(n // 2 // skip_0d)
                    pack_0d = n_channel_per_ct
                    n_packed_in_feat = math.ceil(n_in / n_channel_per_ct)
                    n_packed_out_feat = math.ceil(n_out / n_channel_per_ct)
                    layer = DensePackedLayer(
                        n_out,
                        n_in,
                        [1, 1],
                        [1, 1],
                        pack_0d,
                        n_packed_in_feat,
                        n_packed_out_feat,
                    )
                    return layer.get_fhe_op_count_skip_0d(n_packed_in_feat, skip_0d, self.input_mult_level)

                elif len(special_shape) == 1:
                    # 1D multiplexed
                    shape_1d = int(special_shape[0])
                    skip_list = sp_info.get('skip', [1])
                    skip_1d = int(skip_list[0]) if isinstance(skip_list, list) else int(skip_list)
                    invalid_fill = sp_info.get('invalid_fill', [1])
                    invalid_fill_1d = int(invalid_fill[0]) if isinstance(invalid_fill, list) else int(invalid_fill)
                    block_size = shape_1d * skip_1d
                    n_block_per_ct = int(n // 2) // block_size
                    valid_sub = skip_1d // invalid_fill_1d
                    n_channel_per_ct_1d = n_block_per_ct * valid_sub
                    layer = DensePackedLayer(
                        n_out,
                        n_in,
                        [shape_1d, 1],
                        [skip_1d, 1],
                        n_channel_per_ct_1d,
                        math.ceil(n_in / n_channel_per_ct_1d),
                        math.ceil(n_out / n_block_per_ct),
                        invalid_fill=[invalid_fill_1d, 1],
                    )
                    n_input_ct = math.ceil(n_in / n_channel_per_ct_1d)
                    return layer.get_fhe_op_count_1d_multiplexed(n_input_ct, n, self.input_mult_level)

                else:
                    # 2D multiplexed
                    special_skip = sp_info.get('skip', [1, 1])
                    invalid_fill = sp_info.get('invalid_fill', [1, 1])
                    n_ct_mult = math.ceil(
                        n // 2 / (special_shape[0] * special_skip[0] * special_shape[1] * special_skip[1])
                    )
                    layer = DensePackedLayer(
                        n_out,
                        n_in,
                        special_shape,
                        special_skip,
                        n_ct_mult,
                        n_in,
                        n_out,
                        invalid_fill=invalid_fill,
                    )
                    n_input_ct = math.ceil(n_in / pack)
                    return layer.get_fhe_op_count_multiplexed(n_input_ct, n, self.input_mult_level)
            return None

        # ── avgpool ──────────────────────────────────────────────────────────
        elif layer_type == 'avgpool2d':
            input_shape = self.input_shape
            stride = node.stride
            skip = self.input_skip
            n_input_ct = n_packed_in
            layer = Avgpool2DLayer(stride, input_shape, channel=n_in, skip=skip)
            is_adaptive = getattr(node, 'is_adaptive_avgpool', True)
            is_big_size = getattr(node, 'is_big_size', False)
            if is_big_size:
                block_shape = config.block_shape
                block_expansion = [
                    math.ceil(input_shape[0] / block_shape[0]),
                    math.ceil(input_shape[1] / block_shape[1]),
                ]
                return layer.get_fhe_op_count_interleaved(n_input_ct, n, self.input_mult_level)
            if is_adaptive:
                if style == 'ordinary':
                    return layer.get_fhe_op_count_adaptive(n_input_ct, n, self.input_mult_level)
                return layer.get_fhe_op_count_multiplexed(n_input_ct, n_in, pack, self.input_mult_level)
            # non-adaptive multiplexed
            return layer.get_fhe_op_count_multiplexed(n_input_ct, n_in, pack, self.input_mult_level)

        elif layer_type == 'avgpool1d':
            input_shape = self.input_shape
            stride = node.stride
            skip_1d = self.input_skip[0] if isinstance(self.input_skip, list) else self.input_skip
            layer = Avgpool1DLayer(stride[0], input_shape[0], channel=n_in, skip=skip_1d)
            is_adaptive = getattr(node, 'is_adaptive_avgpool', True)
            if is_adaptive:
                return layer.get_fhe_op_count_adaptive(n_input_ct, n, self.input_mult_level)
            return layer.get_fhe_op_count_multiplexed(n_input_ct, n_in, pack, self.input_mult_level)

        # ── polyact / activation ─────────────────────────────────────────────
        elif layer_type == 'polyact':
            pred = preds[0]
            order = getattr(node, 'order', 0)
            if pred.dim == 0:
                skip_0d = pred.sp_info['skip'][0] if isinstance(pred.sp_info.get('skip'), list) else 1
                n_channel_per_ct_0d = int(n // 2 // skip_0d)
                layer = PolyRelu0D(order, skip_0d, n_channel_per_ct_0d)
                n_input_ct = n_packed_in
                return layer.get_fhe_op_count_bsgs_feature0d(n_input_ct, self.input_mult_level)
            if pred.dim == 1:
                shape_1d = pred.shape[0]
                skip_1d = pred.sp_info['skip'][0] if isinstance(pred.sp_info.get('skip'), list) else 1
                if style == 'multiplexed':
                    n_channel_per_ct_1d = int(n // 2 // shape_1d)
                    layer = PolyRelu1D(shape_1d, order, skip_1d, n_channel_per_ct_1d)
                    return layer.get_fhe_op_count_bsgs_mux(n_packed_in, self.input_mult_level)
                n_channel_per_ct_1d = int(n // 2 // shape_1d // skip_1d)
                layer = PolyRelu1D(shape_1d, order, skip_1d, n_channel_per_ct_1d)
                return layer.get_fhe_op_count_bsgs_skip(n_packed_in, self.input_mult_level)
            # dim == 2
            input_shape = self.input_shape
            skip = self.input_skip
            layer = PolyRelu2D(input_shape, order, skip, pack)
            return layer.get_fhe_op_count_call(n_packed_in, self.input_mult_level)

        # ── mult_scalar ──────────────────────────────────────────────────────
        elif layer_type == 'mult_scalar':
            layer = MultScalarLayer()
            return layer.get_fhe_op_count(n_packed_in, self.input_mult_level)

        # ── add / add2d ──────────────────────────────────────────────────────
        elif layer_type in ('add', 'add2d'):
            layer = AddLayer()
            return layer.get_fhe_op_count(n_packed_in, self.input_mult_level)

        # ── upsample_nearest ────────────────────────────────────────────────
        elif layer_type in ('upsample_nearest', 'resize'):
            input_shape = self.input_shape
            skip = self.input_skip
            upsample_factor = getattr(node, 'upsample_factor', [1, 1])
            layer = UpsampleNearestLayer(
                shape=input_shape,
                skip=skip,
                upsample_factor=upsample_factor,
                n_channel_per_ct=pack,
                level=self.input_mult_level,
            )
            return layer.get_fhe_op_count(n_in, self.input_mult_level)
        else:
            raise NotImplementedError(f"Unsupported layer type: '{layer_type}'")


class MpcScoreParam:
    def __init__(
        self,
        dag: LayerAbstractGraph,
        compute_node: ComputeNode,
        param: dict[str, FheParameter],
        bit_len=44,
        mpc_scale=16,
    ) -> None:
        graph = LayerAbstractGraph()
        graph.dag = dag
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(graph.dag.successors(compute_node))
        self.preds = preds
        self.succs = succs
        self.compute_node = compute_node
        self.input_coeff_mod = param[preds[0].ckks_parameter_id].log_default_scale
        self.output_coeff_mod = param[succs[0].ckks_parameter_id].log_default_scale
        self.input_special_mod = param[preds[0].ckks_parameter_id].log_default_scale
        self.output_special_mod = param[succs[0].ckks_parameter_id].log_default_scale
        self.input_mult_level = graph.dag.nodes[preds[0]]['level']
        self.output_mult_level = graph.dag.nodes[succs[0]]['level']
        self.input_degree = param[preds[0].ckks_parameter_id].poly_modulus_degree
        self.output_degree = param[succs[0].ckks_parameter_id].poly_modulus_degree
        MB_scale = 2**23
        self.relu_score = bit_len * mpc_scale / MB_scale
        self.input_ct_score = (88 + self.input_coeff_mod * self.input_mult_level) * self.input_degree * 2 / MB_scale
        self.output_ct_score = (
            (88 + self.output_coeff_mod * self.output_mult_level) * self.output_degree * 1.5 / MB_scale
        )

        self.input_channel = compute_node.channel_input
        self.output_channel = compute_node.channel_output
        if preds[0].dim == 2:
            input_shape = preds[0].shape
            output_shape = succs[0].shape
            input_skip = graph.dag.nodes[preds[0]]['skip']
            temp_num_in = input_shape[0] * input_shape[1] * input_skip[0] * input_skip[1]
            temp_num_out = output_shape[0] * output_shape[1]
            self.n_packed_in = math.ceil(self.input_channel * temp_num_in / self.input_degree / 2)
            self.n_packed_out = math.ceil(self.output_channel * temp_num_out / self.input_degree / 2)
        elif preds[0].dim == 0:
            self.n_packed_in = math.ceil(self.input_channel / self.input_degree / 2)
            self.n_packed_out = self.n_packed_in

    def get_score(self) -> float:
        if 'relu2d' in self.compute_node.layer_type or 'pool' in self.compute_node.layer_type:
            if 'relu2d' == self.compute_node.layer_type or config.mpc_refresh:
                kernel_scale = 1
            elif 'pool' in self.compute_node.layer_type:
                kernel_scale = self.compute_node.kernel_shape[0] * self.compute_node.kernel_shape[1]
            shape = self.preds[0].shape
            n_relu_score = self.input_channel * shape[0] * shape[1] * self.relu_score / kernel_scale * mpc_refresh_rate
            n_ct_score = (
                self.n_packed_in * self.input_ct_score + self.n_packed_out * self.output_ct_score
            ) * ct_trans_rate
            return n_relu_score + n_ct_score
        if 'bootstrapping' in self.compute_node.layer_type and config.mpc_refresh:
            shape = self.preds[0].shape
            n_ct_score = (
                self.n_packed_in * self.input_ct_score + self.n_packed_out * self.output_ct_score
            ) * ct_trans_rate
            n_mpc_refresh = self.input_channel * shape[0] * shape[1] * self.relu_score * mpc_refresh_rate
            return n_ct_score + n_mpc_refresh


class BtpScoreParam:
    def __init__(self, dag: nx.DiGraph, compute_node: ComputeNode, param: dict[str, FheParameter]) -> None:
        graph = LayerAbstractGraph()
        graph.dag = dag
        pred = list(graph.dag.predecessors(compute_node))[0]
        self.n = param[pred.ckks_parameter_id].poly_modulus_degree
        pack_num = graph.dag.nodes[pred]['pack_num']
        self.ct_num = math.ceil(compute_node.channel_input / pack_num)

    def get_score(self):
        score = self.ct_num * btp_time[str(self.n)] / get_multithread_rate_for_btp(self.ct_num)
        return score
