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
from inference.model_generator.layers.inverse_multiplexed_depthwise_conv2d_layer import (
    InverseMultiplexedDepthwiseConv2DLayer,
)
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
from inference.model_generator.layers.par_block_col_major_ccmm import ParBlockColMajorCCMM
from inference.model_generator.layers.par_block_col_major_cpmm import ParBlockColMajorCPMM
from inference.model_generator.layers.par_block_col_major_transpose import ParBlockColMajorTranspose
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
        1: 0.000645,
        2: 0.000955,
        3: 0.001286,
        4: 0.001693,
        5: 0.002071,
        6: 0.002533,
        7: 0.002927,
        8: 0.003312,
        9: 0.003672,
        10: 0.004072,
        11: 0.004483,
        12: 0.004946,
        13: 0.005378,
        14: 0.005794,
        15: 0.006245,
        16: 0.006750,
        17: 0.008039,
        18: 0.011034,
        19: 0.011939,
        20: 0.009225,
        21: 0.009596,
        22: 0.009506,
        23: 0.010015,
        24: 0.010616,
        25: 0.011239,
        26: 0.011462,
        27: 0.011907,
        28: 0.012331,
        29: 0.012847,
        30: 0.013183,
        31: 0.013643,
        32: 0.014365,
        33: 0.014539,
    },
    32768: {
        1: 0.000319,
        2: 0.000480,
        3: 0.000627,
        4: 0.000797,
        5: 0.000978,
        6: 0.001150,
        7: 0.001307,
        8: 0.001460,
        9: 0.001656,
        10: 0.001841,
        11: 0.002021,
        12: 0.002223,
        13: 0.002441,
        14: 0.002637,
        15: 0.002836,
        16: 0.003077,
        17: 0.003261,
    },
    16384: {
        1: 0.000159,
        2: 0.000240,
        3: 0.000323,
        4: 0.000438,
        5: 0.000493,
        6: 0.000577,
        7: 0.000651,
        8: 0.000748,
        9: 0.000799,
    },
    8192: {1: 0.000079, 2: 0.000125, 3: 0.000164, 4: 0.000204, 5: 0.000240},
}

mult_time = {
    65536: {
        1: 0.001297,
        2: 0.002067,
        3: 0.002840,
        4: 0.003725,
        5: 0.004631,
        6: 0.005665,
        7: 0.006471,
        8: 0.007276,
        9: 0.008332,
        10: 0.009225,
        11: 0.010151,
        12: 0.011048,
        13: 0.011926,
        14: 0.012798,
        15: 0.013509,
        16: 0.014298,
        17: 0.015058,
        18: 0.015949,
        19: 0.016687,
        20: 0.017589,
        21: 0.018701,
        22: 0.019610,
        23: 0.020542,
        24: 0.021369,
        25: 0.022213,
        26: 0.023056,
        27: 0.023901,
        28: 0.024989,
        29: 0.025529,
        30: 0.026570,
        31: 0.027370,
        32: 0.028111,
        33: 0.029033,
    },
    32768: {
        1: 0.000650,
        2: 0.000964,
        3: 0.001296,
        4: 0.001625,
        5: 0.001986,
        6: 0.002336,
        7: 0.002718,
        8: 0.003106,
        9: 0.003599,
        10: 0.003988,
        11: 0.004415,
        12: 0.004836,
        13: 0.005362,
        14: 0.005774,
        15: 0.006279,
        16: 0.006776,
        17: 0.007212,
    },
    16384: {
        1: 0.000336,
        2: 0.000484,
        3: 0.000649,
        4: 0.000823,
        5: 0.000992,
        6: 0.001159,
        7: 0.001301,
        8: 0.001488,
        9: 0.001653,
    },
    8192: {1: 0.000165, 2: 0.000252, 3: 0.000335, 4: 0.000411, 5: 0.000506},
}

rotate_time = {
    65536: {
        0: 0.031820,
        1: 0.037018,
        2: 0.041868,
        3: 0.047295,
        4: 0.065352,
        5: 0.074927,
        6: 0.082575,
        7: 0.090344,
        8: 0.115303,
        9: 0.127408,
        10: 0.137189,
        11: 0.147268,
        12: 0.179777,
        13: 0.194373,
        14: 0.205742,
        15: 0.234848,
        16: 0.268079,
        17: 0.290697,
        18: 0.300496,
        19: 0.316836,
        20: 0.363635,
        21: 0.381708,
        22: 0.396155,
        23: 0.419812,
        24: 0.474806,
        25: 0.493998,
        26: 0.514810,
        27: 0.534185,
        28: 0.593950,
        29: 0.622960,
        30: 0.680312,
        31: 0.664419,
        32: 0.729720,
        33: 0.761393,
    },
    32768: {
        0: 0.010012,
        1: 0.012742,
        2: 0.015201,
        3: 0.021951,
        4: 0.025976,
        5: 0.029408,
        6: 0.039015,
        7: 0.044234,
        8: 0.048706,
        9: 0.061113,
        10: 0.067824,
        11: 0.102140,
        12: 0.123991,
        13: 0.096738,
        14: 0.102584,
        15: 0.120680,
        16: 0.130158,
        17: 0.138828,
    },
    16384: {
        0: 0.006910,
        1: 0.008187,
        2: 0.008420,
        3: 0.011797,
        4: 0.021814,
        5: 0.014901,
        6: 0.019477,
        7: 0.022309,
        8: 0.027749,
        9: 0.030840,
    },
    8192: {0: 0.002134, 1: 0.003607, 2: 0.005506, 3: 0.007264, 4: 0.007383, 5: 0.008279},
}

rescale_time = {
    8192: {1: 0.000501, 2: 0.001168, 3: 0.001894, 4: 0.002176, 5: 0.001463},
    16384: {
        1: 0.001013,
        2: 0.001526,
        3: 0.002033,
        4: 0.002663,
        5: 0.003210,
        6: 0.003679,
        7: 0.004089,
        8: 0.004560,
        9: 0.005095,
    },
    32768: {
        1: 0.002139,
        2: 0.003221,
        3: 0.007748,
        4: 0.005414,
        5: 0.006890,
        6: 0.007664,
        7: 0.008777,
        8: 0.010601,
        9: 0.010815,
        10: 0.012363,
        11: 0.013190,
        12: 0.014246,
        13: 0.015366,
        14: 0.016245,
        15: 0.017305,
        16: 0.018369,
        17: 0.019983,
    },
    65536: {
        1: 0.004522,
        2: 0.006801,
        3: 0.009051,
        4: 0.011359,
        5: 0.013644,
        6: 0.015940,
        7: 0.018223,
        8: 0.021073,
        9: 0.022900,
        10: 0.025219,
        11: 0.027528,
        12: 0.030182,
        13: 0.032181,
        14: 0.034526,
        15: 0.037818,
        16: 0.041075,
        17: 0.041716,
        18: 0.045078,
        19: 0.046434,
        20: 0.048902,
        21: 0.051305,
        22: 0.053533,
        23: 0.057005,
        24: 0.061400,
        25: 0.067001,
        26: 0.070472,
        27: 0.069801,
        28: 0.068217,
        29: 0.070632,
        30: 0.073058,
        31: 0.075488,
        32: 0.077856,
        33: 0.081414,
    },
}

add_time = {
    65536: {
        0: 0.000234,
        1: 0.000373,
        2: 0.000559,
        3: 0.000951,
        4: 0.002018,
        5: 0.001937,
        6: 0.001690,
        7: 0.002032,
        8: 0.002653,
        9: 0.002711,
        10: 0.003243,
        11: 0.003291,
        12: 0.003596,
        13: 0.004107,
        14: 0.004233,
        15: 0.004581,
        16: 0.005133,
        17: 0.005336,
        18: 0.005533,
        19: 0.005759,
        20: 0.006218,
        21: 0.006780,
        22: 0.007028,
        23: 0.006955,
        24: 0.007138,
        25: 0.007521,
        26: 0.007867,
        27: 0.008269,
        28: 0.008604,
        29: 0.008866,
        30: 0.009031,
        31: 0.009339,
        32: 0.009601,
        33: 0.009969,
    },
    32768: {
        0: 0.000162,
        1: 0.000191,
        2: 0.000274,
        3: 0.000696,
        4: 0.000496,
        5: 0.000636,
        6: 0.000646,
        7: 0.000778,
        8: 0.000959,
        9: 0.001006,
        10: 0.001156,
        11: 0.001341,
        12: 0.001464,
        13: 0.001620,
        14: 0.001820,
        15: 0.001969,
        16: 0.002224,
        17: 0.002303,
    },
    16384: {
        0: 0.000086,
        1: 0.000108,
        2: 0.000186,
        3: 0.000314,
        4: 0.000362,
        5: 0.000343,
        6: 0.000443,
        7: 0.000554,
        8: 0.000670,
        9: 0.000658,
    },
    8192: {0: 0.000081, 1: 0.000094, 2: 0.000176, 3: 0.000208, 4: 0.000224, 5: 0.000241},
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
        if op_counts is None:
            return 0.0
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
                if is_depthwise:
                    layer = InverseMultiplexedDepthwiseConv2DLayer(
                        n_out,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        block_shape,
                    )
                else:
                    layer = InverseMultiplexedConv2DLayer(
                        n_out,
                        n_in,
                        input_shape,
                        padding,
                        kernel_shape,
                        stride,
                        block_shape,
                    )
                return layer.get_fhe_op_count(self.input_mult_level, n)

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
                if not (pred_com and pred.has_sp_info):
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
                    return layer.get_fhe_op_count(n_input_ct, n, self.input_mult_level)
                else:
                    return layer.get_fhe_op_count_adaptive(n_input_ct, n_in, self.input_mult_level)
            else:
                # non-adaptive multiplexed
                return layer.get_fhe_op_count_multiplexed(n_input_ct, n_in, pack, self.input_mult_level)

        elif layer_type == 'avgpool1d':
            input_shape = self.input_shape
            stride = node.stride
            skip_1d = self.input_skip[0] if isinstance(self.input_skip, list) else self.input_skip
            layer = Avgpool1DLayer(stride[0], input_shape[0], channel=n_in, skip=skip_1d)
            is_adaptive = getattr(node, 'is_adaptive_avgpool', True)
            if is_big_size:
                raise ('unsuport avgpool1d in big_size')
            if is_adaptive:
                if style == 'ordinary':
                    raise ('unsuport avgpool1d in ordinary and is_adaptive')
                else:
                    return layer.get_fhe_op_count_adaptive(n_input_ct, n, self.input_mult_level)
            else:
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
        elif layer_type == 'qkvcpmm':
            m = preds[0].shape[0]
            n_total = preds[0].shape[1]
            n_per_head = n_total // config.n_heads
            W_rows = n_total
            W_cols = self.output_shape[1]
            n_slot = n // 2
            try:
                layer = ParBlockColMajorCPMM(
                    shape_A=(m, n_per_head),
                    W_shape=(W_rows, W_cols),
                    block_size=config.matmul_block_size,
                    n_heads=config.n_heads,
                    n_slot=n_slot,
                )
                return layer.get_fhe_op_count(self.input_mult_level)
            except (AssertionError, ValueError):
                return None
        elif layer_type == 'transpose':
            m = preds[0].shape[0]
            n_per_head = preds[0].shape[1] // config.n_heads
            n_slot = n // 2
            try:
                layer = ParBlockColMajorTranspose(
                    shape=(m, n_per_head),
                    block_size=config.matmul_block_size,
                    n_heads=config.n_heads,
                    n_slot=n_slot,
                )
                return layer.get_fhe_op_count(self.input_mult_level)
            except (AssertionError, ValueError):
                return None
        elif layer_type == 'ccmm':
            m = preds[0].shape[0]
            n_per_head = preds[0].shape[1] // config.n_heads
            p_per_head = preds[1].shape[1] // config.n_heads
            n_slot = n // 2
            try:
                layer = ParBlockColMajorCCMM(
                    shape_A=(m, n_per_head),
                    shape_B=(n_per_head, p_per_head),
                    block_size=config.matmul_block_size,
                    n_heads=config.n_heads,
                    n_slot=n_slot,
                )
                return layer.get_fhe_op_count(self.input_mult_level)
            except (AssertionError, ValueError):
                return None
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
