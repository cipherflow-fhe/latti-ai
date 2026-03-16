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

import networkx as nx

from components import (
    ComputeNode,
    FheParameter,
    FeatureNode,
    LayerAbstractGraph,
    config,
    single_thread,
)


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
                    * math.ceil(self.input_shape[0] / config.fhe_param.block_shape[0])
                    * math.ceil(self.input_shape[1] / config.fhe_param.block_shape[1])
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
            if config.style == 'ordinary':
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
        self.input_coeff_mod = param[preds[0].ckks_parameter_id].coeff_modulus_bit_length
        self.output_coeff_mod = param[succs[0].ckks_parameter_id].coeff_modulus_bit_length
        self.input_special_mod = param[preds[0].ckks_parameter_id].special_prime_bit_length
        self.output_special_mod = param[succs[0].ckks_parameter_id].special_prime_bit_length
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
        btp_slot = int(self.n / 2)

        self.ct_num = math.ceil(pred.shape[0] * pred.shape[1] * compute_node.channel_input / btp_slot)

    def get_score(self):
        score = self.ct_num * btp_time[str(self.n)] / get_multithread_rate_for_btp(self.ct_num)
        return score
