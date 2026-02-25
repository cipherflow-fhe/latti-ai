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

order = 4


def _init_config_vars():
    """Initialize global variables from config"""
    global MAX_LEVEL, GRAPH_TYPE, block_shape
    global STYLE, MPC_REFRESH, APPROX_POLY_TYPE, SET_LEVEL_MAX
    global POLY_N
    global absorbable_layers

    if config is None:
        raise RuntimeError('Config not initialized. Please call init_config() in graph_splitter first.')

    MAX_LEVEL = config['MAX_LEVEL']
    GRAPH_TYPE = config['GRAPH_TYPE']
    block_shape = config['block_shape']  # Set by graph_splitter based on POLY_N
    POLY_N = config['POLY_N']
    STYLE = config['STYLE']
    MPC_REFRESH = config['MPC_REFRESH']
    APPROX_POLY_TYPE = config['APPROX_POLY_TYPE']
    SET_LEVEL_MAX = config['SET_LEVEL_MAX']

    # Update absorbable_layers based on MPC_REFRESH
    if MPC_REFRESH:
        absorbable_layers = ['conv2d', 'fc0', 'fc1', 'mult_scalar', 'simple_polyrelu']
    else:
        absorbable_layers = ['conv2d', 'fc0', 'fc1', 'mult_scalar']


# Global variable initialization
config = None  # Set by external module
MAX_LEVEL = None
GRAPH_TYPE = None
block_shape = None
STYLE = None
DEFAULT_SCALE = 1
MPC_REFRESH = None
APPROX_POLY_TYPE = None
SET_LEVEL_MAX = None


def process_level_for_graph(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type == 'bootstrapping':
                succs: list[FeatureNode] = list(graph.dag.successors(node))
                preds: list[FeatureNode] = list(graph.dag.predecessors(node))
                graph.dag.nodes[succs[0]]['level'] = MAX_LEVEL
    add_drop_level_for_graph(graph, None)


def add_layer(
    graph: LayerAbstractGraph,
    compute_node: ComputeNode,
    depth_out,
    index: int,
    layer_type: str,
    preds: list[FeatureNode],
    other_graph: LayerAbstractGraph | None,
    insert_node: ComputeNode = None,
):
    channel_input = compute_node.channel_input
    channel_output = compute_node.channel_input
    ckks_parameter_id_input = compute_node.ckks_parameter_id_input
    ckks_parameter_id_output = compute_node.ckks_parameter_id_input
    feature_node_in = preds[index]

    dim = feature_node_in.dim
    ckks_scale = feature_node_in.ckks_scale

    skip = list(graph.dag.nodes[feature_node_in]['skip'])

    shape = list(feature_node_in.shape)
    virtue_shape = list(graph.dag.nodes[feature_node_in]['virtual_shape'])
    virtue_skip = list(graph.dag.nodes[feature_node_in]['virtual_skip'])
    level = graph.dag.nodes[feature_node_in]['level']
    pack_num = graph.dag.nodes[feature_node_in]['pack_num']

    scale = feature_node_in.scale
    timestamp = int(time.time() * 1000000)
    layer_id = f'{compute_node.layer_id}_{layer_type}_idx{index}_ts{timestamp}'

    feature_node_out = FeatureNode(
        feature_node_in.node_id + str(id(virtue_shape)) + f'_{layer_type}_output',
        dim,
        channel_output,
        scale,
        ckks_parameter_id_output,
        ckks_scale,
        shape,
    )

    if hasattr(feature_node_in, 'node_index'):
        feature_node_out.node_index = feature_node_in.node_index

    graph.dag.add_node(
        feature_node_out,
        name=feature_node_out.node_id,
        skip=skip,
        virtual_shape=virtue_shape,
        virtual_skip=virtue_skip,
        level=level,
        pack_num=pack_num,
    )

    if insert_node:
        new_compute_node = insert_node
    else:
        if layer_type == 'mult_scalar':
            new_compute_node = MultScalarComputeNode(
                layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
            )
        elif layer_type == 'upsample':
            new_compute_node = UpsampleComputeNode(
                layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
            )
        else:
            new_compute_node = ComputeNode(
                layer_id, layer_type, channel_input, channel_output, ckks_parameter_id_input, ckks_parameter_id_output
            )

    new_compute_node.depth = depth_out

    level_cost = 0
    if layer_type == 'mult_scalar':
        level_cost = 1
    elif layer_type == 'drop_level':
        level_cost = 0

    graph.dag.add_node(new_compute_node, name=layer_id, level_cost=level_cost)
    graph.dag.remove_edge(feature_node_in, compute_node)
    graph.dag.add_edge(feature_node_in, new_compute_node)
    graph.dag.add_edge(new_compute_node, feature_node_out)
    graph.dag.add_edge(feature_node_out, compute_node)

    if other_graph is not None:
        other_graph.dag.add_node(
            feature_node_out,
            name=feature_node_out.node_id,
            skip=skip,
            virtual_shape=virtue_shape,
            virtual_skip=virtue_skip,
            level=level,
            pack_num=pack_num,
        )
        other_graph.dag.add_node(new_compute_node, name=layer_id, level_cost=level_cost)
        other_graph.dag.remove_edge(feature_node_in, compute_node)
        other_graph.dag.add_edge(feature_node_in, new_compute_node)
        other_graph.dag.add_edge(new_compute_node, feature_node_out)
        other_graph.dag.add_edge(feature_node_out, compute_node)
    return new_compute_node


def get_leading_feature_nodes(dag: nx.DiGraph) -> list[FeatureNode]:
    leading_feature_nodes = []
    for node, in_degree in dag.in_degree():
        if in_degree == 0 and isinstance(node, FeatureNode):
            if any(isinstance(next_node, ComputeNode) for next_node in dag.successors(node)):
                leading_feature_nodes.append(node)
    return leading_feature_nodes


def populate_pack_num(dag: nx.DiGraph, node, slot_num: int):
    sub = LayerAbstractGraph()
    sub.dag = dag
    preds = list(sub.dag.predecessors(node))
    succs = list(sub.dag.successors(node))
    if STYLE == 'multiplexed':
        in_node = preds[0]
        out_node = succs[0]
        if in_node.dim == 0:
            sub.dag.nodes[in_node]['pack_num'] = int(
                math.ceil(
                    slot_num
                    / (
                        sub.dag.nodes[in_node]['virtual_shape'][0]
                        * sub.dag.nodes[in_node]['virtual_shape'][1]
                        * sub.dag.nodes[in_node]['virtual_skip'][0]
                        * sub.dag.nodes[in_node]['virtual_skip'][1]
                    )
                )
            )
        else:
            sub.dag.nodes[in_node]['pack_num'] = math.ceil(slot_num / (in_node.shape[0] * in_node.shape[1]))
        if out_node.dim == 0:
            sub.dag.nodes[out_node]['pack_num'] = int(
                math.ceil(
                    slot_num
                    / (
                        sub.dag.nodes[out_node]['virtual_shape'][0]
                        * sub.dag.nodes[out_node]['virtual_shape'][1]
                        * sub.dag.nodes[out_node]['virtual_skip'][0]
                        * sub.dag.nodes[out_node]['virtual_skip'][1]
                    )
                )
            )
        else:
            sub.dag.nodes[out_node]['pack_num'] = math.ceil(slot_num / (out_node.shape[0] * out_node.shape[1]))
    else:
        if 'reshape' == node.layer_type:
            sub.dag.nodes[succs[0]]['pack_num'] = math.ceil(
                slot_num
                / (
                    sub.dag.nodes[succs[0]]['virtual_shape'][0]
                    * sub.dag.nodes[succs[0]]['virtual_shape'][1]
                    * sub.dag.nodes[succs[0]]['virtual_skip'][0]
                    * sub.dag.nodes[succs[0]]['virtual_skip'][1]
                )
            )
        elif preds[0].dim == 0:
            in_node = preds[0]
            out_node = succs[0]

            sub.dag.nodes[succs[0]]['pack_num'] = math.ceil(
                slot_num
                / (
                    sub.dag.nodes[succs[0]]['virtual_shape'][0]
                    * sub.dag.nodes[succs[0]]['virtual_shape'][1]
                    * sub.dag.nodes[succs[0]]['virtual_skip'][0]
                    * sub.dag.nodes[succs[0]]['virtual_skip'][1]
                )
            )
            sub.dag.nodes[preds[0]]['pack_num'] = math.ceil(
                slot_num
                / (
                    sub.dag.nodes[preds[0]]['virtual_shape'][0]
                    * sub.dag.nodes[preds[0]]['virtual_shape'][1]
                    * sub.dag.nodes[preds[0]]['virtual_skip'][0]
                    * sub.dag.nodes[preds[0]]['virtual_skip'][1]
                )
            )
        else:
            sub.dag.nodes[succs[0]]['pack_num'] = math.ceil(
                slot_num
                / (
                    succs[0].shape[0]
                    * succs[0].shape[1]
                    * sub.dag.nodes[succs[0]]['skip'][0]
                    * sub.dag.nodes[succs[0]]['skip'][1]
                )
            )
            sub.dag.nodes[preds[0]]['pack_num'] = math.ceil(
                slot_num
                / (
                    preds[0].shape[0]
                    * preds[0].shape[1]
                    * sub.dag.nodes[preds[0]]['skip'][0]
                    * sub.dag.nodes[preds[0]]['skip'][1]
                )
            )


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
    if MPC_REFRESH:
        update_skip_for_btp(sub, print_flag)
        update_level_cost_for_btp(sub)
    for compute_node in compute_nodes_in_topo_sort:
        compute_node.ckks_parameter_id_input = param_id
        compute_node.ckks_parameter_id_output = param_id

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
        ckks_parameter_id_input=feature.ckks_parameter_id,
        ckks_parameter_id_output=refreshed_feature.ckks_parameter_id,
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
            compute.layer_type = APPROX_POLY_TYPE
            subgraph.dag.nodes[compute]['level_cost'] = math.ceil(math.log2(compute.order)) + 1


def set_scale_for_node(graph: LayerAbstractGraph, c_node: ComputeNode, scale: float):
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_id in c_node.up_scale_str:
                node.scale_up = node.scale_up * scale
                c_node.up_scale_str.remove(node.layer_id)
                return node

            elif node.layer_id in c_node.down_scale_str:
                node.scale_down = node.scale_down * scale
                c_node.down_scale_str.remove(node.layer_id)
                return node


mpc_scale = 1


def set_feature_scale_for_graph(graph: LayerAbstractGraph):
    for compute in graph.dag.nodes:
        scale = 1.0
        if not isinstance(compute, ComputeNode):
            continue

        if compute.layer_type == 'relu2d' or compute.layer_type == 'mpc_refresh':
            scale = mpc_scale

        elif compute.layer_type == 'avgpool2d':
            if GRAPH_TYPE == 'mpc':
                scale = 1.0 / (compute.kernel_shape[0] * compute.kernel_shape[1])
            elif compute.is_adaptive_avgpool or compute.is_big_size:
                scale = 1.0 / (compute.kernel_shape[0] * compute.kernel_shape[1])

        if compute.layer_type == 'simple_polyrelu':
            while compute.up_scale_str:
                node_out = set_scale_for_node(graph, compute, 1)
                node_out.vec_scale_path = compute.layer_id
            continue
        while compute.up_scale_str or compute.down_scale_str:
            node_out = set_scale_for_node(graph, compute, scale)


def add_drop_level_for_graph(sub: LayerAbstractGraph, graph: LayerAbstractGraph | None):
    for compute in list(sub.dag.nodes):
        if not isinstance(compute, ComputeNode):
            continue
        if compute.layer_type == 'drop_level':
            continue
        preds: list[FeatureNode] = list(sub.dag.predecessors(compute))
        succs: list[FeatureNode] = list(sub.dag.successors(compute))
        if len(succs) == 0:
            continue
        for i in range(len(preds)):
            if 'level' not in sub.dag.nodes[preds[i]]:
                print(f"Warning: node {preds[i].node_id} missing 'level' attribute")
                continue
            if 'level' not in sub.dag.nodes[succs[0]]:
                print(f"Warning: node {succs[0].node_id} missing 'level' attribute")
                continue

            if (sub.dag.nodes[preds[i]]['level'] - sub.dag.nodes[succs[0]]['level']) > sub.dag.nodes[compute][
                'level_cost'
            ]:
                drop_level_layer = add_layer(sub, compute, compute.depth, i, 'drop_level', preds, graph)
                sub.dag.nodes[drop_level_layer]['level_cost'] = (
                    sub.dag.nodes[preds[i]]['level']
                    - sub.dag.nodes[succs[0]]['level']
                    - sub.dag.nodes[compute]['level_cost']
                )
                succ_sub = list(sub.dag.successors(drop_level_layer))[0]
                sub.dag.nodes[succ_sub]['level'] = (
                    sub.dag.nodes[preds[i]]['level'] - sub.dag.nodes[drop_level_layer]['level_cost']
                )
                if graph:
                    graph.dag.nodes[succ_sub]['level'] = sub.dag.nodes[succ_sub]['level']


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'


def find_linear_fhe_layer(
    compute_node: ComputeNode, graph: LayerAbstractGraph, direction: Direction
) -> tuple[bool, ComputeNode]:
    node = compute_node
    while True:
        if direction == Direction.UP:
            node_list = list(graph.dag.predecessors(node))
        else:
            node_list = list(graph.dag.successors(node))
        if not node_list:
            return [False, None]

        if isinstance(node_list[0], FeatureNode):
            node = node_list[0]
            continue

        if node_list[0].layer_type in absorbable_layers:
            return [True, node_list[0]]

        node = node_list[0]


def add_mult_scalar_behind_node(
    graph: LayerAbstractGraph, compute_node: ComputeNode, other_graph: LayerAbstractGraph
) -> ComputeNode:
    f_node = list(graph.dag.successors(compute_node))[0]

    skip = list(graph.dag.nodes[f_node]['skip'])
    virtual_shape = list(graph.dag.nodes[f_node]['virtual_shape'])
    virtual_skip = list(graph.dag.nodes[f_node]['virtual_skip'])
    level = graph.dag.nodes[f_node]['level']
    pack_num = graph.dag.nodes[f_node]['pack_num']

    added_f_node = copy.deepcopy(f_node)
    f_node.node_id = f_node.node_id + '_mult_scalar_output'
    f_node.scale = 1.0

    added_c_node = MultScalarComputeNode(
        compute_node.layer_id + '_mult_scalar_', 'mult_scalar', compute_node.channel_input, compute_node.channel_output
    )

    graph.dag.remove_edge(compute_node, f_node)

    graph.dag.add_node(
        added_f_node,
        name=added_f_node.node_id,
        skip=skip,
        virtual_shape=virtual_shape,
        virtual_skip=virtual_skip,
        level=level,
        pack_num=pack_num,
    )

    graph.dag.add_node(added_c_node, name=added_c_node.layer_id, level_cost=1)

    graph.dag.add_edge(compute_node, added_f_node)
    graph.dag.add_edge(added_f_node, added_c_node)
    graph.dag.add_edge(added_c_node, f_node)

    if other_graph:
        other_graph.dag.remove_edge(compute_node, f_node)

        other_graph.dag.add_node(
            added_f_node,
            name=added_f_node.node_id,
            skip=skip,
            virtual_shape=virtual_shape,
            virtual_skip=virtual_skip,
            level=level,
            pack_num=pack_num,
        )
        other_graph.dag.add_node(added_c_node, name=added_c_node.layer_id, level_cost=1)

        other_graph.dag.add_edge(compute_node, added_f_node)
        other_graph.dag.add_edge(added_f_node, added_c_node)
        other_graph.dag.add_edge(added_c_node, f_node)

    return added_c_node


def recovery_graph_from_subgraph(graph: LayerAbstractGraph, subgraphs: LayerAbstractGraph):
    for sub in subgraphs:
        sub_nodes = set(sub.dag.nodes)

        edges_to_remove = []
        for u, v in graph.dag.edges:
            if u in sub_nodes and v in sub_nodes:
                if not sub.dag.has_edge(u, v):
                    edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            graph.dag.remove_edge(u, v)

        graph.dag.add_nodes_from(sub.dag.nodes(data=True))
        graph.dag.add_edges_from(sub.dag.edges)


def sort_subgraphs(subgraphs: list[LayerAbstractGraph]):
    sorted_subgraphs = list()
    index_graph = nx.DiGraph()
    for i in range(len(subgraphs)):
        index_graph = nx.compose(compress_graph(subgraphs[i], i), index_graph)
    if not nx.is_directed_acyclic_graph(index_graph):
        print('Cycle exists')
        print(list(nx.simple_cycles(index_graph)))
    all_nodes_in_topo_sort = list(nx.topological_sort(index_graph))
    int_res = [value for value in all_nodes_in_topo_sort if isinstance(value, int)]
    index = 0

    pre_next_dict = dict()
    for value in int_res:
        sorted_subgraphs.append(subgraphs[value])
        pre_next_dict[value] = index
        index = index + 1
    next_sub_dict = dict()
    for node in index_graph.nodes:
        if not isinstance(node, int):
            preds = list(index_graph.predecessors(node))
            succs = list(index_graph.successors(node))
            if len(succs) >= 1:
                if preds:
                    next_sub_dict[pre_next_dict[preds[0]]] = [pre_next_dict[value] for value in succs]
    prev_sub_dict = dict()
    for u, vs in next_sub_dict.items():
        for v in vs:
            if v not in prev_sub_dict:
                prev_sub_dict[v] = []
            if u not in prev_sub_dict[v]:
                prev_sub_dict[v].append(u)
    return sorted_subgraphs, next_sub_dict, prev_sub_dict


def find_input_and_output(sub: LayerAbstractGraph):
    input_nodes = sub.get_leading_feature_nodes()
    output_nodes = sub.get_output_feature_nodes()
    return input_nodes, output_nodes


def graph_to_task_config(subgraphs: list[LayerAbstractGraph], file_path, use_btp: bool = True):
    server_task = {}
    for i in range(len(subgraphs)):
        sub = subgraphs[i]
        if sub.is_mpc:
            server_task['nn_layers_ct_' + f'{i}'] = {'enable_fpga': False}
        else:
            server_task['nn_layers_ct_' + f'{i}'] = {'enable_fpga': True}
    input_root = subgraphs[0].get_leading_feature_nodes()[0]

    if not nx.is_directed_acyclic_graph(subgraphs[-1].dag):
        raise ValueError('Cycle exists in graph, cannot perform topological sort!')

    all_nodes_in_topo_sort = list(nx.topological_sort(subgraphs[-1].dag))
    compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]

    succs = list(subgraphs[-1].dag.successors(compute_nodes_in_topo_sort[-1]))
    output_root = succs[0]

    param_dict = dict()
    for idx, node in enumerate([input_root, output_root]):
        graph_to_use = subgraphs[0] if idx == 0 else subgraphs[-1]
        if node.dim == 0:
            param_dict[node.node_id] = {
                'dim': node.dim,
                'channel': node.channel,
                'scale': node.scale,
                'ckks_scale': node.ckks_scale,
                'skip': int(graph_to_use.dag.nodes[node]['skip'][0]),
                'ckks_parameter_id': node.ckks_parameter_id,
                'virtual_shape': [int(x) for x in graph_to_use.dag.nodes[node]['virtual_shape']],
                'virtual_skip': [int(x) for x in graph_to_use.dag.nodes[node]['virtual_skip']],
                'level': graph_to_use.dag.nodes[node]['level'],
                'depth': node.depth,
                'pack_num': graph_to_use.dag.nodes[node]['pack_num'],
            }
        elif node.dim == 2:
            param_dict[node.node_id] = {
                'dim': node.dim,
                'channel': node.channel,
                'scale': node.scale,
                'ckks_scale': node.ckks_scale,
                'shape': node.shape,
                'skip': graph_to_use.dag.nodes[node]['skip'],
                'ckks_parameter_id': node.ckks_parameter_id,
                'level': graph_to_use.dag.nodes[node]['level'],
                'depth': node.depth,
                'pack_num': graph_to_use.dag.nodes[node]['pack_num'],
            }

    config = {
        'task_type': 'fhe' if len(subgraphs) == 1 else 'hybrid',
        'task_num': len(subgraphs),
        'server_start_id': 0,
        'server_end_id': len(subgraphs) - 1,
        'block_shape': block_shape,
        'is_absorb_polyrelu': False,
        'pack_style': STYLE,
        'task_input_id': str(input_root.node_id),
        'task_output_id': str(output_root.node_id),
        'task_input_param': {'input': param_dict[input_root.node_id]},
        'task_output_param': {'output': param_dict[output_root.node_id]},
        'server_task': server_task,
        'use_btp': use_btp,
    }
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, 'task_config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return


def process_avgpool2d(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, PoolComputeNode):
            node.is_adaptive_avgpool = True


def find_layer_in_linear_graph(graph: LayerAbstractGraph, c_node: ComputeNode, target_layer_type: str, direction: str):
    if direction == 'up':
        preds = list(graph.dag.predecessors(c_node))
        if len(preds) == 0 or len(preds) > 1:
            return False
        start_node = preds[0]
        while True:
            if isinstance(start_node, ComputeNode) and start_node.layer_type == target_layer_type:
                return start_node
            else:
                start_preds = list(graph.dag.predecessors(start_node))
                if len(start_preds) == 0 or len(start_preds) > 1:
                    return False
                start_node = start_preds[0]
                continue
    else:
        succs = list(graph.dag.successors(c_node))
        if len(succs) == 0 or len(succs) > 1:
            return False
        start_node = succs[0]
        while True:
            if isinstance(start_node, ComputeNode) and start_node.layer_type == target_layer_type:
                return start_node
            else:
                start_succs = list(graph.dag.successors(start_node))
                if len(start_succs) == 0 or len(start_succs) > 1:
                    return False
                start_node = start_succs[0]
                continue


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
                                if node_in.layer_type == APPROX_POLY_TYPE:
                                    node_in.upsample_factor_in = target_c_node.upsample_factor_in

    return name_pair


def check_conv_upsample_factor(graph: LayerAbstractGraph, c_node: ConvComputeNode):
    if c_node.upsample_factor_in[0] != 1:
        f_in = list(graph.dag.predecessors(c_node))[0]
        if f_in.shape[0] * c_node.upsample_factor_in[0] > block_shape[0] or (
            f_in.shape[1] * c_node.upsample_factor_in[1] > block_shape[1]
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

                upsample_layer = add_layer(
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
            if STYLE == 'ordinary':
                graph.dag.nodes[compute_node]['level_cost'] = 1
                continue
            if preds[0].shape[0] > block_shape[0] or preds[0].shape[1] > block_shape[1]:
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
            if preds[0].shape[0] > block_shape[0] or preds[0].shape[1] > block_shape[1]:
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
        elif compute_node.layer_type == APPROX_POLY_TYPE:
            graph.dag.nodes[compute_node]['level_cost'] = math.ceil(math.log2(compute_node.order)) + 1
            if preds[0].shape[0] > block_shape[0] or preds[0].shape[1] > block_shape[1]:
                compute_node.is_big_size = True


def get_slot_num(ckks_parameter_id_input: str, param_dict: dict) -> int:
    """
    Get the slot number based on the parameter dictionary.
    """
    return param_dict[ckks_parameter_id_input].poly_modulus_degree // 2


def add_drop_level(dag: nx.DiGraph, feature: FeatureNode, drop_level: int):
    level_decreased_feature = copy.deepcopy(feature)
    level_decreased_feature.node_id = f'{feature.node_id}_drop_level_output'
    dag.add_node(
        level_decreased_feature,
        level=dag.nodes[feature]['level'] - drop_level,
        skip=dag.nodes[feature]['skip'],
        virtual_shape=dag.nodes[feature]['virtual_shape'],
        virtual_skip=dag.nodes[feature]['virtual_skip'],
        pack_num=dag.nodes[feature]['pack_num'],
    )

    drop_node = ComputeNode(
        layer_id=f'{feature.node_id}_drop_level',
        layer_type='drop_level',
        channel_input=feature.channel,
        channel_output=level_decreased_feature.channel,
        ckks_parameter_id_input=feature.ckks_parameter_id,
        ckks_parameter_id_output=level_decreased_feature.ckks_parameter_id,
    )
    dag.add_node(drop_node, name=drop_node.layer_id, level_cost=drop_level)
    dag.add_edge(feature, drop_node)
    dag.add_edge(drop_node, level_decreased_feature)

    return level_decreased_feature


def set_is_adaptive_avgpool(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, PoolComputeNode):
            succ_f = list(graph.dag.successors(node))[0]
            succ_c = list(graph.dag.successors(succ_f))[0]
            if succ_c.layer_type == 'reshape':
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
            if preds[0].shape[0] > block_shape[0] or preds[0].shape[1] > block_shape[1]:
                graph.dag.nodes[succs[0]]['skip'] = [1, 1]

        if 'upsample' == compute_node.layer_type:
            graph.dag.nodes[succs[0]]['skip'][0] = 1
            graph.dag.nodes[succs[0]]['skip'] = [1, 1]
        if (
            'batchnorm' in compute_node.layer_type
            or 'drop_level' in compute_node.layer_type
            or 'mult_scalar' in compute_node.layer_type
            or 'bootstrapping' in compute_node.layer_type
            or APPROX_POLY_TYPE == compute_node.layer_type
            or 'relu2d' == compute_node.layer_type
            or 'identity' == compute_node.layer_type
        ):
            graph.dag.nodes[succs[0]]['skip'] = graph.dag.nodes[preds[0]]['skip']
            if MPC_REFRESH and 'bootstrapping' in compute_node.layer_type:
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
            if preds[0].shape[0] > block_shape[0] or preds[0].shape[1] > block_shape[1]:
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
        add_mult_scalar_layer = add_layer(graph, node, 0, 0, 'mult_scalar', preds, None)
        preds = list(graph.dag.predecessors(add_mult_scalar_layer))
        add_mpc_refresh = add_layer(graph, add_mult_scalar_layer, 0, 0, 'bootstrapping', preds, None)

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


def check_degree(graph: LayerAbstractGraph, node):
    if graph.dag.in_degree(node) > 1 or graph.dag.out_degree(node) > 1:
        return False
    else:
        return True


def add_identity_layer(graph: LayerAbstractGraph, node1, node2):
    identity_layer = ComputeNode(
        str(id(node1)) + '_identity',
        'identity',
        node1.channel,
        node1.channel,
        node1.ckks_parameter_id,
        node1.ckks_parameter_id,
    )
    node_out = FeatureNode(
        identity_layer.layer_id + '_output',
        node1.dim,
        node1.channel,
        1,
        node1.ckks_parameter_id,
        DEFAULT_SCALE,
        node1.shape,
    )
    node1_attrs = graph.dag.nodes[node1].copy()

    graph.dag.add_node(identity_layer)
    graph.dag.add_node(node_out, **node1_attrs)
    graph.dag.add_edge(node1, identity_layer)
    graph.dag.add_edge(identity_layer, node_out)
    graph.dag.add_edge(node_out, node2)
    graph.dag.remove_edge(node1, node2)


def del_identity_layer(graph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]
    for compute_node in compute_nodes_in_topo_sort:
        if compute_node.layer_type == 'identity':
            pre_n_node = list(graph.dag.predecessors(compute_node))[0]
            succ_n_node = list(graph.dag.successors(compute_node))[0]
            succ_c_node = list(graph.dag.successors(succ_n_node))[0]
            graph.dag.remove_edge(succ_n_node, succ_c_node)
            graph.dag.remove_edge(pre_n_node, compute_node)
            graph.dag.remove_node(compute_node)
            graph.dag.remove_node(succ_n_node)
            graph.dag.add_edge(pre_n_node, succ_c_node)


def pre_process(graph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    for node in all_nodes_in_topo_sort:
        if not check_degree(graph, node) and isinstance(node, FeatureNode):
            for next_node in list(graph.dag.successors(node)):
                if not check_degree(graph, next_node):
                    add_identity_layer(graph, node, next_node)


def build_edge_mappings(removed_edges):
    forward_map = {}
    backward_map = {}

    for node_in, node in removed_edges:
        if node_in not in forward_map:
            forward_map[node_in] = []
        forward_map[node_in].append(node)

        if node not in backward_map:
            backward_map[node] = []
        backward_map[node].append(node_in)

    return forward_map, backward_map


def add_node_for_graph(
    sub: LayerAbstractGraph, forward_map: dict, backward_map: dict, original_graph: LayerAbstractGraph
):
    all_nodes_in_topo_sort = list(nx.topological_sort(sub.dag))
    compute_nodes_in_topo_sort = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)]
    for node in compute_nodes_in_topo_sort:
        if sub.dag.in_degree(node) == 0:
            add_nodes = backward_map.get(node, [])
            for add_node in add_nodes:
                if add_node in original_graph.dag.nodes:
                    attrs = original_graph.dag.nodes[add_node]
                    sub.dag.add_node(add_node, **attrs)
                else:
                    sub.dag.add_node(add_node)
                sub.dag.add_edge(add_node, node)

    for node in reversed(compute_nodes_in_topo_sort):
        if sub.dag.out_degree(node) == 0:
            add_nodes = forward_map.get(node, [])
            for add_node in add_nodes:
                if add_node in original_graph.dag.nodes:
                    attrs = original_graph.dag.nodes[add_node]
                    sub.dag.add_node(add_node, **attrs)
                else:
                    sub.dag.add_node(add_node)
                sub.dag.add_edge(node, add_node)


def split_graph_to_linear_subgraph(graph: LayerAbstractGraph):
    original_graph = graph
    removed_edges = []
    for node in graph.dag.nodes:
        if graph.dag.in_degree(node) > 1 or graph.dag.out_degree(node) > 1:
            in_nodes_list = list(graph.dag.predecessors(node))
            out_nodes_list = list(graph.dag.successors(node))
            if len(in_nodes_list) > 1:
                for node_in in in_nodes_list:
                    graph.dag.remove_edge(node_in, node)
                    removed_edges.append((node_in, node))
            if len(out_nodes_list) > 1:
                for node_out in out_nodes_list:
                    graph.dag.remove_edge(node, node_out)
                    removed_edges.append((node, node_out))

    weak_components = list(nx.weakly_connected_components(graph.dag))

    subgraphs = list()
    forward_map, backward_map = build_edge_mappings(removed_edges)
    for component in weak_components:
        if len(component) > 1:
            sub = LayerAbstractGraph()
            sub.dag = graph.dag.subgraph(component).copy()
            subgraphs.append(sub)
            add_node_for_graph(sub, forward_map, backward_map, original_graph)

    return subgraphs, removed_edges


def set_graph_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    pre_process(graph)
    subgraphs, removed_edges = split_graph_to_linear_subgraph(graph)
    for sub in subgraphs:
        handle_valid_poly_subgraph(sub, use_mpc_refresh)

    recovery_graph_from_subgraph(graph, subgraphs)
    del_identity_layer(graph)
    set_feature_scale_for_graph(graph)


def absorb_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    pre_process(graph)
    subgraphs, removed_edges = split_graph_to_linear_subgraph(graph)
    subs_odered, next_dict, pre_dict = sort_subgraphs(subgraphs)

    index = 0
    invalid_index = []
    processed_index = []

    subgraph_invalid_poly_dict = dict()

    added_mult_scalar_ids = []

    for sub_in in subs_odered:
        invalid_poly_nodes = []
        if not check_approx_poly_subgraph(sub_in, invalid_poly_nodes, use_mpc_refresh):
            invalid_index.append(index)
        subgraph_invalid_poly_dict[index] = invalid_poly_nodes
        index = index + 1

    for i in range(len(subs_odered)):
        if i in invalid_index:
            added_id = handle_invalid_poly_subgraph(
                i, subs_odered, next_dict, pre_dict, subgraph_invalid_poly_dict, use_mpc_refresh
            )
            if added_id:
                added_mult_scalar_ids.append(added_id)

    recovery_graph_from_subgraph(graph, subgraphs)
    del_identity_layer(graph)

    return graph


def check_approx_poly_subgraph(subgraph: LayerAbstractGraph, invalid_list: list = None, use_mpc_refresh: bool = False):
    """Check if the approx poly nodes in the linear subgraph can be absorbed"""

    if use_mpc_refresh:
        approx_poly_layer = ['bootstrapping']
    else:
        approx_poly_layer = ['avgpool2d']
    valid_flag = True

    for node in subgraph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type in approx_poly_layer:
                if isinstance(node, PoolComputeNode) and (not node.is_adaptive_avgpool) and (not node.is_big_size):
                    continue
                is_find_dwon, target_node_down = find_linear_fhe_layer(node, subgraph, Direction.DOWN)
                is_find_up, target_node_up = find_linear_fhe_layer(node, subgraph, Direction.UP)
                if (not is_find_dwon) and (not is_find_up):
                    valid_flag = False
                    return valid_flag
                elif (not is_find_up) and is_find_dwon and target_node_down.layer_type != 'simple_polyrelu':
                    valid_flag = False
                    return valid_flag

    return valid_flag


def handle_invalid_poly_subgraph(
    subgraph_index, subs_odered, next_dict, pre_dict, subgraph_invalid_poly_dict, use_mpc_refresh: bool = False
):
    """Handle poly nodes that cannot be absorbed in the current subgraph, return the layer_id of the added mult_scalar"""
    current_sub = subs_odered[subgraph_index]
    all_nodes_in_topo_sort = list(nx.topological_sort(current_sub.dag))
    first_node = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)][0]
    mult_scalar_layer = add_mult_scalar_behind_node(current_sub, first_node, None)

    return mult_scalar_layer.layer_id


def handle_valid_poly_subgraph(subgraph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    """Handle poly nodes that can be absorbed in the current subgraph"""

    if not use_mpc_refresh:
        for node in subgraph.dag.nodes:
            if isinstance(node, ComputeNode):
                if node.layer_type == 'simple_polyrelu' or node.layer_type == 'relu2d':
                    find, res_node = find_linear_fhe_layer(node, subgraph, Direction.UP)
                    if find:
                        node.up_scale_str.append(res_node.layer_id)
                elif node.layer_type in {'avgpool2d'}:
                    find_down, res_node_down = find_linear_fhe_layer(node, subgraph, Direction.DOWN)
                    if find_down and res_node_down.layer_type != 'simple_polyrelu':
                        node.down_scale_str.append(res_node_down.layer_id)

                        continue
                    find_up, res_node_up = find_linear_fhe_layer(node, subgraph, Direction.UP)
                    if find_up:
                        node.up_scale_str.append(res_node_up.layer_id)
    else:
        candidates = {}

        for node in subgraph.dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                find_down, res_node_down = find_linear_fhe_layer(node, subgraph, Direction.DOWN)
                find_up, res_node_up = find_linear_fhe_layer(node, subgraph, Direction.UP)

                candidates[node] = {
                    'down': res_node_down if (find_down and res_node_down.layer_type != 'simple_polyrelu') else None,
                    'up': res_node_up if find_up else None,
                }

        initial_assignment = {}

        for btp_node, cands in candidates.items():
            if cands['down']:
                initial_assignment[btp_node] = ('down', cands['down'])
            elif cands['up']:
                initial_assignment[btp_node] = ('up', cands['up'])

        c_node_count = {}

        for btp_node, (direction, target) in initial_assignment.items():
            if target not in c_node_count:
                c_node_count[target] = []
            c_node_count[target].append(btp_node)

        for c_node, btp_list in list(c_node_count.items()):
            if len(btp_list) > 1:
                for btp_node in btp_list:
                    current_direction, current_target = initial_assignment[btp_node]
                    cands = candidates[btp_node]

                    alternative_direction = 'up' if current_direction == 'down' else 'down'
                    alternative_target = cands[alternative_direction]

                    if alternative_target and alternative_target != current_target:
                        if alternative_target not in c_node_count or len(c_node_count[alternative_target]) == 1:
                            initial_assignment[btp_node] = (alternative_direction, alternative_target)

                            c_node_count[current_target].remove(btp_node)
                            if alternative_target not in c_node_count:
                                c_node_count[alternative_target] = []
                            c_node_count[alternative_target].append(btp_node)

                            if len(c_node_count[c_node]) <= 1:
                                break

        for btp_node, (direction, target) in initial_assignment.items():
            if direction == 'down':
                btp_node.down_scale_str.append(target.layer_id)
            else:
                btp_node.up_scale_str.append(target.layer_id)


def compress_graph(graph: LayerAbstractGraph, graph_out_index: int):
    graph_out = nx.DiGraph()

    graph_out.add_node(graph_out_index)

    inputs, outputs = find_input_and_output(graph)
    for node in inputs:
        graph_out.add_node(node)
        graph_out.add_edge(node, graph_out_index)
    for node in outputs:
        graph_out.add_node(node)
        graph_out.add_edge(graph_out_index, node)
    return graph_out


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
