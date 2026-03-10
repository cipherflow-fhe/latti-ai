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

import sys
import os
from components import *
import copy
from itertools import product
import math
import time
import multiprocessing
import os
from datetime import datetime
import json

import transforms


order = 4


def process_levels(graph: LayerAbstractGraph):
    if config.set_max_level:
        for node in graph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                succ = next(graph.dag.successors(node))
                graph.dag.nodes[succ]['level'] = config.max_level

    transforms.insert_drop_level_layers(graph)


def get_leading_feature_nodes(dag: nx.DiGraph) -> list[FeatureNode]:
    leading_feature_nodes = []
    for node, in_degree in dag.in_degree():
        if in_degree == 0 and isinstance(node, FeatureNode):
            if any(isinstance(next_node, ComputeNode) for next_node in dag.successors(node)):
                leading_feature_nodes.append(node)
    return leading_feature_nodes


def _calc_pack_num(dag: nx.DiGraph, feature_node, slot_num: int, use_skip: bool = True) -> int:
    attrs = dag.nodes[feature_node]
    if feature_node.dim == 0:
        return math.ceil(
            slot_num
            / (
                attrs['virtual_shape'][0]
                * attrs['virtual_shape'][1]
                * attrs['virtual_skip'][0]
                * attrs['virtual_skip'][1]
            )
        )
    else:
        denom = feature_node.shape[0] * feature_node.shape[1]
        if use_skip:
            denom *= attrs['skip'][0] * attrs['skip'][1]
        return math.ceil(slot_num / denom)


def populate_pack_num(dag: nx.DiGraph, node, slot_num: int):
    preds = list(dag.predecessors(node))
    succs = list(dag.successors(node))
    if config.style == 'multiplexed':
        for f_node in preds + succs:
            dag.nodes[f_node]['pack_num'] = _calc_pack_num(dag, f_node, slot_num, use_skip=False)
    else:
        if node.layer_type == 'reshape':
            for f_node in preds + succs:
                dag.nodes[f_node]['pack_num'] = _calc_pack_num(dag, f_node, slot_num)
        else:
            for f_node in preds + succs:
                dag.nodes[f_node]['pack_num'] = _calc_pack_num(dag, f_node, slot_num)


def update_subgraph_node_param(dag, param_dict: dict[str, EncryptParameterNode], param_id, print_flag=False):
    all_nodes_in_topo_sort = list(nx.topological_sort(dag))
    compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]
    for node in dag.nodes:
        if isinstance(node, FeatureNode):
            node.ckks_parameter_id = param_id

    sub = LayerAbstractGraph()
    sub.dag = dag
    slot_num = param_dict[param_id].poly_modulus_degree / 2
    # print(f'slot_num={slot_num}')
    if config.mpc_refresh:
        update_skip_for_btp(sub, print_flag)
        update_level_cost_for_btp(sub)
    for compute_node in compute_nodes_in_topo_sort:
        populate_pack_num(dag, compute_node, slot_num)


def add_btp_layer(graph: LayerAbstractGraph, feature: FeatureNode):
    refreshed_feature = copy.deepcopy(feature)

    refreshed_feature.node_id = f'{feature.node_id}_refreshed'
    feature_attrs = graph.dag.nodes[feature].copy() if feature in graph.dag.nodes else {}
    graph.dag.add_node(refreshed_feature, **feature_attrs)
    for s in list(graph.dag.successors(feature)):
        graph.dag.remove_edge(feature, s)
        graph.dag.add_edge(refreshed_feature, s)

    btp_node = ComputeNode(
        layer_id=f'{feature.node_id}_bootstrap',
        layer_type='bootstrapping',
        channel_input=feature.channel,
        channel_output=refreshed_feature.channel,
    )
    graph.dag.add_node(btp_node)
    graph.dag.add_edge(feature, btp_node)
    graph.dag.add_edge(btp_node, refreshed_feature)


def sync_node_attributes(source_graph: LayerAbstractGraph, target_graph: LayerAbstractGraph):
    """
    Synchronize node attributes from source_graph to the same nodes in target_graph

    Args:
        source_graph: Source graph
        target_graph: Target graph
    """
    for node in source_graph.dag.nodes:
        if node in target_graph.dag.nodes:
            target_graph.dag.nodes[node].update(source_graph.dag.nodes[node])


def set_param(sub: LayerAbstractGraph, param_dict: dict[str, EncryptParameterNode], param_id: str, is_print=False):
    update_subgraph_node_param(sub, param_dict, param_id, is_print)


def check_c_node(graph: LayerAbstractGraph, c_node: ComputeNode):
    pre_node = list(graph.dag.predecessors(c_node))[0]
    pre_compute = list(graph.dag.predecessors(pre_node))[0]
    pre_compute_input_node = list(graph.dag.predecessors(pre_compute))[0]
    if pre_compute.layer_type == 'batchnorm':
        graph.dag.add_edge(pre_compute_input_node, c_node)
        graph.dag.remove_node(pre_compute)
        graph.dag.remove_node(pre_node)


def substitute_layers_for_btp(subgraph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(subgraph.dag))
    for compute in all_nodes_in_topo_sort:
        if not isinstance(compute, ComputeNode):
            continue
        if compute.layer_type == 'relu2d' or compute.layer_type == 'simple_polyrelu':
            compute.layer_type = config.approx_poly_type
            subgraph.dag.nodes[compute]['level_cost'] = math.ceil(math.log2(compute.order)) + 1


mpc_scale = 1


def graph_to_task_config(graph: LayerAbstractGraph, file_path, use_btp: bool = True):
    server_task = {}
    if graph.is_mpc:
        server_task['nn_layers_ct_0'] = {'enable_fpga': False}
    else:
        server_task['nn_layers_ct_0'] = {'enable_fpga': True}

    input_roots = graph.get_leading_feature_nodes()

    if not nx.is_directed_acyclic_graph(graph.dag):
        raise ValueError('Cycle exists in graph, cannot perform topological sort!')

    output_roots = [node for node, out_deg in graph.dag.out_degree() if out_deg == 0]

    param_dict = dict()
    for node in input_roots + output_roots:
        if node.dim == 0:
            param_dict[node.node_id] = {
                'dim': node.dim,
                'channel': node.channel,
                'scale': node.scale,
                'ckks_scale': node.ckks_scale,
                'skip': int(graph.dag.nodes[node]['skip'][0]),
                'ckks_parameter_id': node.ckks_parameter_id,
                'virtual_shape': [int(x) for x in graph.dag.nodes[node]['virtual_shape']],
                'virtual_skip': [int(x) for x in graph.dag.nodes[node]['virtual_skip']],
                'level': graph.dag.nodes[node]['level'],
                'depth': node.depth,
                'pack_num': graph.dag.nodes[node]['pack_num'],
            }
        elif node.dim == 2:
            param_dict[node.node_id] = {
                'dim': node.dim,
                'channel': node.channel,
                'scale': node.scale,
                'ckks_scale': node.ckks_scale,
                'shape': node.shape,
                'skip': graph.dag.nodes[node]['skip'],
                'ckks_parameter_id': node.ckks_parameter_id,
                'level': graph.dag.nodes[node]['level'],
                'depth': node.depth,
                'pack_num': graph.dag.nodes[node]['pack_num'],
            }

    task_config = {
        'task_type': 'fhe',
        'task_num': 1,
        'server_start_id': 0,
        'server_end_id': 0,
        'block_shape': config.block_shape,
        'is_absorb_polyrelu': False,
        'pack_style': config.style,
        'task_input_id': [str(n.node_id) for n in input_roots],
        'task_output_id': [str(n.node_id) for n in output_roots],
        'task_input_param': {str(n.node_id): param_dict[n.node_id] for n in input_roots},
        'task_output_param': {str(n.node_id): param_dict[n.node_id] for n in output_roots},
        'server_task': server_task,
        'use_btp': use_btp,
    }
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, 'task_config.json'), 'w') as f:
        json.dump(task_config, f, indent=4, ensure_ascii=False)
    return


def process_avgpool2d(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, PoolComputeNode):
            node.is_adaptive_avgpool = True


def change_conv_transpose_shape(graph: LayerAbstractGraph):
    name_pair = dict()
    for c_node in graph.dag.nodes:
        if isinstance(c_node, ComputeNode):
            if isinstance(c_node, ConvComputeNode):
                if c_node.is_conv_transpose:
                    target_c_node = find_layer_in_linear_graph(graph, c_node, 'conv2d', 'up')
                    if target_c_node:
                        f_in = list(graph.dag.predecessors(target_c_node))[0]
                        f_out = list(graph.dag.successors(target_c_node))[0]
                        target_c_node.upsample_factor_in[0] = c_node.upsample_factor_in[0]
                        target_c_node.upsample_factor_in[1] = c_node.upsample_factor_in[1]
                        name_pair[target_c_node.layer_id] = (
                            c_node,
                            c_node.upsample_factor_in[0],
                            c_node.upsample_factor_in[1],
                        )
                        c_node.upsample_factor_in = [1, 1]
                        shortest_path = nx.shortest_path(graph.dag, target_c_node, c_node)
                        for node_in in shortest_path:
                            if isinstance(node_in, ComputeNode):
                                if node_in.layer_type == config.approx_poly_type:
                                    node_in.upsample_factor_in = target_c_node.upsample_factor_in

    return name_pair


def check_conv_upsample_factor(graph: LayerAbstractGraph, c_node: ConvComputeNode):
    if c_node.upsample_factor_in[0] != 1:
        f_in = list(graph.dag.predecessors(c_node))[0]
        if f_in.shape[0] * c_node.upsample_factor_in[0] > config.block_shape[0] or (
            f_in.shape[1] * c_node.upsample_factor_in[1] > config.block_shape[1]
        ):
            c_node.upsample_factor_in[0] = 1
            c_node.upsample_factor_in[1] = 1
            return True
    return False


def update_shape_for_btp(graph: LayerAbstractGraph):
    name_pair = change_conv_transpose_shape(graph)

    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]
    for compute_node in compute_nodes_in_topo_sort:
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(graph.dag.successors(compute_node))
        if isinstance(compute_node, ConvComputeNode):
            if check_conv_upsample_factor(graph, compute_node):
                for i in range(2):
                    succs[0].shape[i] = preds[0].shape[i] / compute_node.stride[i]

                upsample_layer = transforms.add_layer(
                    graph,
                    name_pair[compute_node.layer_id][0],
                    0,
                    0,
                    'upsample',
                    list(graph.dag.predecessors(name_pair[compute_node.layer_id][0])),
                    None,
                )

                upsample_layer.upsample_factor_in[0] = name_pair[compute_node.layer_id][1]
                upsample_layer.upsample_factor_in[1] = name_pair[compute_node.layer_id][2]
                upsample_node_in = list(graph.dag.predecessors(upsample_layer))[0]
                upsample_node_out = list(graph.dag.successors(upsample_layer))[0]
                for i in range(2):
                    upsample_node_out.shape[i] = upsample_node_in.shape[i] * upsample_layer.upsample_factor_in[i]
                    graph.dag.nodes[upsample_node_out]['skip'][i] = 1
                    graph.dag.nodes[upsample_layer]['level_cost'] = 0
            else:
                for i in range(2):
                    succs[0].shape[i] = preds[0].shape[i] / compute_node.stride[i] * compute_node.upsample_factor_in[i]
        elif isinstance(compute_node, PoolComputeNode):
            for i in range(2):
                if not compute_node.is_adaptive_avgpool:
                    succs[0].shape[i] = preds[0].shape[i] / compute_node.stride[i]
                else:
                    succs[0].shape[i] = preds[0].shape[i]
        elif compute_node.layer_type == 'resize':
            for i in range(2):
                succs[0].shape[i] = preds[0].shape[i] * compute_node.upsample_factor_in[i]
        else:
            for i in range(2):
                succs[0].shape[i] = preds[0].shape[i]


def update_level_cost_for_btp(graph: LayerAbstractGraph):
    for compute_node in graph.dag.nodes:
        if isinstance(compute_node, FeatureNode):
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(graph.dag.successors(compute_node))
        if isinstance(compute_node, ConvComputeNode):
            if config.style == 'ordinary':
                graph.dag.nodes[compute_node]['level_cost'] = 1
                continue
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                compute_node.is_big_size = True
                graph.dag.nodes[compute_node]['level_cost'] = 1
            else:
                if compute_node.groups == 1:
                    if compute_node.stride[0] == 1 and graph.dag.nodes[preds[0]]['skip'][0] == 1:
                        graph.dag.nodes[compute_node]['level_cost'] = 1
                    else:
                        graph.dag.nodes[compute_node]['level_cost'] = 2
                else:
                    if compute_node.stride[0] == 1:
                        graph.dag.nodes[compute_node]['level_cost'] = 1
                    else:
                        graph.dag.nodes[compute_node]['level_cost'] = 2

        elif compute_node.layer_type == 'avgpool2d':
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                graph.dag.nodes[compute_node]['level_cost'] = 0
                compute_node.is_big_size = True
                compute_node.is_adaptive_avgpool = False
            else:
                compute_node.is_big_size = False
                succs_sub = list(graph.dag.successors(succs[0]))
                if succs_sub and succs_sub[0].layer_type == 'reshape':
                    graph.dag.nodes[compute_node]['level_cost'] = 0
                    compute_node.is_adaptive_avgpool = True
                else:
                    graph.dag.nodes[compute_node]['level_cost'] = 1
                    compute_node.is_adaptive_avgpool = False
        elif compute_node.layer_type == config.approx_poly_type:
            graph.dag.nodes[compute_node]['level_cost'] = math.ceil(math.log2(compute_node.order)) + 1
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                compute_node.is_big_size = True


def get_slot_num(ckks_parameter_id_input: str, param_dict: dict) -> int:
    """
    Get the slot number based on the parameter dictionary.
    """
    return param_dict[ckks_parameter_id_input].poly_modulus_degree // 2


def set_is_adaptive_avgpool(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, PoolComputeNode):
            succ_f = next(graph.dag.successors(node))
            succ_c = next(graph.dag.successors(succ_f), None)
            if (succ_c is not None) and (succ_c.layer_type == 'reshape'):
                node.is_adaptive_avgpool = True
            else:
                node.is_adaptive_avgpool = False


def update_skip_for_btp(graph: LayerAbstractGraph, print_flag=False):
    nodes = graph.get_leading_feature_nodes()
    for node in nodes:
        graph.dag.nodes[node]['skip'] = [1, 1]

    for compute_node in nx.topological_sort(graph.dag):
        if not isinstance(compute_node, ComputeNode):
            continue

        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succs: list[FeatureNode] = list(graph.dag.successors(compute_node))

        if 'reshape' == compute_node.layer_type:
            graph.dag.nodes[succs[0]]['virtual_shape'] = preds[0].shape
            graph.dag.nodes[succs[0]]['virtual_skip'] = graph.dag.nodes[preds[0]]['skip']
            skip = (
                preds[0].shape[0]
                * preds[0].shape[1]
                * graph.dag.nodes[preds[0]]['skip'][0]
                * graph.dag.nodes[preds[0]]['skip'][1]
            )
            graph.dag.nodes[succs[0]]['skip'] = [skip, skip]

        if 'fc0' == compute_node.layer_type:
            graph.dag.nodes[succs[0]]['virtual_skip'] = graph.dag.nodes[preds[0]]['virtual_skip']
            graph.dag.nodes[succs[0]]['virtual_shape'] = graph.dag.nodes[preds[0]]['virtual_shape']
            graph.dag.nodes[succs[0]]['skip'] = graph.dag.nodes[preds[0]]['skip']

        if 'conv' in compute_node.layer_type:
            graph.dag.nodes[succs[0]]['skip'][0] = (
                graph.dag.nodes[preds[0]]['skip'][0] * compute_node.stride[0] / compute_node.upsample_factor_in[0]
            )
            graph.dag.nodes[succs[0]]['skip'][1] = (
                graph.dag.nodes[preds[0]]['skip'][1] * compute_node.stride[1] / compute_node.upsample_factor_in[1]
            )
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]

        if 'upsample' == compute_node.layer_type:
            graph.dag.nodes[succs[0]]['skip'] = [1, 1]
        if (
            'batchnorm' in compute_node.layer_type
            or 'drop_level' in compute_node.layer_type
            or 'mult_scalar' in compute_node.layer_type
            or 'bootstrapping' in compute_node.layer_type
            or config.approx_poly_type == compute_node.layer_type
            or 'relu2d' == compute_node.layer_type
            or 'identity' == compute_node.layer_type
        ):
            graph.dag.nodes[succs[0]]['skip'] = graph.dag.nodes[preds[0]]['skip']
            if config.mpc_refresh and 'bootstrapping' in compute_node.layer_type:
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]
            if 'bootstrapping' in compute_node.layer_type and compute_node.change_skip_to != 0:
                graph.dag.nodes[succs[0]]['skip'] = [compute_node.change_skip_to, compute_node.change_skip_to]

        if 'add' in compute_node.layer_type or 'concat2d' == compute_node.layer_type:
            check_res = check_preds_skip(graph, preds)
            if check_res:
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]
                continue
            graph.dag.nodes[succs[0]]['skip'] = graph.dag.nodes[preds[0]]['skip']
        if 'avgpool' in compute_node.layer_type:
            if compute_node.is_adaptive_avgpool:
                graph.dag.nodes[succs[0]]['skip'] = graph.dag.nodes[preds[0]]['skip']
            else:
                graph.dag.nodes[succs[0]]['skip'][0] = graph.dag.nodes[preds[0]]['skip'][0] * compute_node.stride[0]
                graph.dag.nodes[succs[0]]['skip'][1] = graph.dag.nodes[preds[0]]['skip'][1] * compute_node.stride[1]
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]
        if 'resize' == compute_node.layer_type:
            if (
                graph.dag.nodes[preds[0]]['skip'][0] < compute_node.upsample_factor_in[0]
                or graph.dag.nodes[preds[0]]['skip'][1] < compute_node.upsample_factor_in[1]
            ):
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]
                continue

            graph.dag.nodes[succs[0]]['skip'][0] = (
                graph.dag.nodes[preds[0]]['skip'][0] / compute_node.upsample_factor_in[0]
            )
            graph.dag.nodes[succs[0]]['skip'][1] = (
                graph.dag.nodes[preds[0]]['skip'][1] / compute_node.upsample_factor_in[1]
            )


def add_mpc_refresh_mult_scalar(graph: LayerAbstractGraph, node: ComputeNode):
    preds = list(graph.dag.predecessors(node))
    input = preds[0]
    level = 0
    if (
        graph.dag.nodes[input]['skip'][0] < node.upsample_factor_in[0]
        or graph.dag.nodes[input]['skip'][1] < node.upsample_factor_in[1]
    ) or node.layer_type != 'resize':
        add_mult_scalar_layer = transforms.add_layer(graph, node, 0, 0, 'mult_scalar', preds, None)
        preds = list(graph.dag.predecessors(add_mult_scalar_layer))
        add_mpc_refresh = transforms.add_layer(graph, add_mult_scalar_layer, 0, 0, 'bootstrapping', preds, None)

        nodes_to_process = [add_mpc_refresh, add_mult_scalar_layer]
        upsample_factors = node.upsample_factor_in
        add_mpc_refresh.change_skip_to = upsample_factors[0]
        for node_id in nodes_to_process:
            succ_node = list(graph.dag.successors(node_id))[0]
            graph.dag.nodes[succ_node]['skip'][:2] = upsample_factors
            if node_id.layer_type == 'mpc_refresh' or node_id.layer_type == 'bootstrapping':
                graph.dag.nodes[succ_node]['level'] = level + 1
            else:
                graph.dag.nodes[succ_node]['level'] = level


def check_preds_skip(graph: LayerAbstractGraph, preds: list) -> list:
    """
    Check the skip of predecessor nodes:
    - If all nodes have the same skip -> return True
    - If not equal -> return all nodes with skip not equal to 1 (supports dual-value array: include if any element != 1)
    """
    if not preds:
        return []

    base_skip = graph.dag.nodes[preds[0]]['skip'][:2]
    all_equal = all(graph.dag.nodes[pred]['skip'][:2] == base_skip for pred in preds)

    if all_equal:
        return []
    else:
        return [
            pred for pred in preds if graph.dag.nodes[pred]['skip'][0] != 1 or graph.dag.nodes[pred]['skip'][1] != 1
        ]


def change_skip_for_graph(graph: LayerAbstractGraph):
    resize_layer_list: list[ComputeNode] = list()
    concat_add_layer_list: list[ComputeNode] = list()
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type == 'resize':
                resize_layer_list.append(node)
            if node.layer_type in ['add2d', 'concat2d']:
                concat_add_layer_list.append(node)
    for node in resize_layer_list:
        pre = list(graph.dag.predecessors(node))[0]
        if graph.dag.nodes[pre]['skip'][0] >= node.upsample_factor_in[0]:
            continue
        add_mpc_refresh_mult_scalar(graph, node)
        succ = list(graph.dag.successors(node))[0]
        graph.dag.nodes[succ]['skip'] = [1, 1]
    for node in concat_add_layer_list:
        preds = list(graph.dag.predecessors(node))
        check_res = check_preds_skip(graph, preds)
        if check_res:
            for f_node in check_res:
                c_node = list(graph.dag.predecessors(f_node))[0]
                add_mpc_refresh_mult_scalar(graph, c_node)


def check_level_cost(graph: LayerAbstractGraph) -> bool:
    """
    Check that for each compute node: output_level - input_level == level_cost.

    Returns True if all compute nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode) or node.layer_type in ['drop_level', 'bootstrapping']:
            continue
        level_cost = graph.dag.nodes[node].get('level_cost')
        if level_cost is None:
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        succs: list[FeatureNode] = list(graph.dag.successors(node))
        if not preds or not succs:
            continue
        input_level = max(graph.dag.nodes[p]['level'] for p in preds)
        output_level = graph.dag.nodes[succs[0]]['level']
        if input_level - output_level != level_cost:
            print(
                f'[check_level_cost] FAIL: {node.layer_id} ({node.layer_type}): '
                f'input_level({input_level}) - output_level({output_level}) = '
                f'{input_level - output_level}, expected level_cost={level_cost}'
            )
            result = False
    return result


def check_multi_input_level_skip_aligned(graph: LayerAbstractGraph) -> bool:
    """
    Check that for each compute node with multiple input FeatureNodes,
    all inputs have the same skip and level.

    Returns True if all such nodes satisfy the constraint, False otherwise.
    """
    result = True
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        if len(preds) < 2:
            continue
        base_level = graph.dag.nodes[preds[0]]['level']
        base_skip = graph.dag.nodes[preds[0]]['skip'][:2]
        for p in preds[1:]:
            p_level = graph.dag.nodes[p]['level']
            p_skip = graph.dag.nodes[p]['skip'][:2]
            if p_level != base_level:
                print(
                    f'[check_multi_input_consistency] FAIL level: {node.layer_id} ({node.layer_type}): '
                    f'{preds[0].node_id} level={base_level} vs {p.node_id} level={p_level}'
                )
                result = False
            if p_skip != base_skip:
                print(
                    f'[check_multi_input_consistency] FAIL skip: {node.layer_id} ({node.layer_type}): '
                    f'{preds[0].node_id} skip={base_skip} vs {p.node_id} skip={p_skip}'
                )
                result = False
    return result


def check_feature_scale(graph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    for node in all_nodes_in_topo_sort:
        if not isinstance(node, ComputeNode):
            continue
        preds = list(graph.dag.predecessors(node))
        succs = list(graph.dag.successors(node))
        if not preds or not succs:
            continue
        assert all(p.scale == preds[0].scale for p in preds), (
            f'[calculate_feture_scale_for_test] preds scale mismatch at {node.layer_id}: {[p.scale for p in preds]}'
        )
        f_node = preds[0]
        out_node = succs[0]
        if node.layer_type in config.absorbable_layers:
            out_node.scale = f_node.scale * node.weight_scale
        elif node.layer_type == 'mult_coeff':
            out_node.scale = f_node.scale * (1 / node.coeff)
        elif node.layer_type == 'avgpool2d':
            if (not node.is_adaptive_avgpool) and (not node.is_big_size):
                out_node.scale = f_node.scale
            else:
                out_node.scale = f_node.scale * (node.kernel_shape[0] * node.kernel_shape[1])
        else:
            out_node.scale = f_node.scale

    output_nodes = [node for node, out_deg in graph.dag.out_degree() if out_deg == 0 and isinstance(node, FeatureNode)]
    return all(math.isclose(node.scale, 1.0) for node in output_nodes)


def set_depth_for_graph(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        node.depth = 0
    for node in reversed(list(nx.topological_sort(graph.dag))):
        if list(graph.dag.successors(node)):
            max_successor_depth = max(s.depth for s in graph.dag.successors(node))
        else:
            max_successor_depth = 0
        if isinstance(node, ComputeNode):
            node.depth = max_successor_depth + 1
        else:
            node.depth = max_successor_depth
    return 0


if __name__ == '__main__':
    print()
