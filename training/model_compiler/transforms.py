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


import copy
import time
from enum import Enum
import networkx as nx

from components import *


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'


def _insert_layer_between_feature_and_compute(
    dag: nx.DiGraph,
    old_feature: FeatureNode,
    old_compute: ComputeNode,
    new_compute: ComputeNode,
    new_feature: FeatureNode,
    *,
    new_compute_args: dict | None = None,
    new_feature_args: dict | None = None,
):
    if new_compute_args is None:
        new_compute_args = dict()
    if new_feature_args is None:
        new_feature_args = dict()
    dag.add_node(new_compute, **new_compute_args)
    dag.add_node(new_feature, **new_feature_args)

    dag.remove_edge(old_feature, old_compute)
    dag.add_edge(old_feature, new_compute)
    dag.add_edge(new_compute, new_feature)
    dag.add_edge(new_feature, old_compute)


def _insert_layer_after_feature(
    dag: nx.DiGraph,
    old_feature: FeatureNode,
    new_compute: ComputeNode,
    new_feature: FeatureNode,
    *,
    new_compute_args: dict | None = None,
    new_feature_args: dict | None = None,
):
    if new_compute_args is None:
        new_compute_args = dict()
    if new_feature_args is None:
        new_feature_args = dict()
    dag.add_node(new_compute, **new_compute_args)
    dag.add_node(new_feature, **new_feature_args)

    old_computes = list(dag.successors(old_feature))
    for oc in old_computes:
        dag.remove_edge(old_feature, oc)
        dag.add_edge(new_feature, oc)
    dag.add_edge(old_feature, new_compute)
    dag.add_edge(new_compute, new_feature)


def init_levels(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, FeatureNode):
            graph.dag.nodes[node]['level'] = 0
            graph.dag.nodes[node]['pack_num'] = 1


def add_layer(
    graph: LayerAbstractGraph,
    compute_node: ComputeNode,
    depth_out,
    index: int,
    layer_type: str,
    preds: list[FeatureNode],
    insert_node: ComputeNode = None,
):
    channel_input = compute_node.channel_input
    channel_output = compute_node.channel_input
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
        feature_node_in.ckks_parameter_id,
        ckks_scale,
        shape,
    )

    if hasattr(feature_node_in, 'node_index'):
        feature_node_out.node_index = feature_node_in.node_index

    if insert_node:
        new_compute_node = insert_node
    else:
        if layer_type == 'mult_scalar':
            new_compute_node = MultScalarComputeNode(layer_id, layer_type, channel_input, channel_output)
        elif layer_type == 'upsample':
            new_compute_node = UpsampleComputeNode(layer_id, layer_type, channel_input, channel_output)
        else:
            new_compute_node = ComputeNode(layer_id, layer_type, channel_input, channel_output)

    new_compute_node.depth = depth_out

    level_cost = 0
    if layer_type == 'mult_scalar':
        level_cost = 1

    _insert_layer_between_feature_and_compute(
        graph.dag,
        feature_node_in,
        compute_node,
        new_compute_node,
        feature_node_out,
        new_compute_args={'name': layer_id, 'level_cost': level_cost},
        new_feature_args={
            'name': feature_node_out.node_id,
            'skip': skip,
            'virtual_shape': virtue_shape,
            'virtual_skip': virtue_skip,
            'level': level,
            'pack_num': pack_num,
        },
    )

    return new_compute_node


def add_btp_layer(dag: nx.DiGraph, upstream_feature: FeatureNode, param_dict: dict, restore_lv: int):
    refreshed_feature = copy.deepcopy(upstream_feature)
    base_id = upstream_feature.node_id
    counter = 0
    new_id = f'{base_id}_refreshed'

    while any(isinstance(n, FeatureNode) and n.node_id == new_id for n in dag.nodes):
        counter += 1
        new_id = f'{base_id}_refreshed_{counter}'

    if counter > 100:
        raise ValueError(f'refreshed nodes with same node id {new_id}. Something is wrong!')

    refreshed_feature.node_id = new_id
    if config.mpc_refresh:
        skip = [1, 1]
    else:
        skip = dag.nodes[upstream_feature]['skip']

    btp_node = ComputeNode(
        layer_id=f'{upstream_feature.node_id}_bootstrap',
        layer_type='bootstrapping',
        channel_input=upstream_feature.channel,
        channel_output=refreshed_feature.channel,
    )

    _insert_layer_after_feature(
        dag,
        upstream_feature,
        btp_node,
        refreshed_feature,
        new_compute_args={'name': btp_node.layer_id, 'level_cost': -restore_lv},
        new_feature_args={
            'level': dag.nodes[upstream_feature]['level'] + restore_lv,
            'skip': skip,
            'virtual_shape': [1, 1],
            'virtual_skip': [1, 1],
        },
    )

    slot_num = param_dict[upstream_feature.ckks_parameter_id].poly_modulus_degree // 2
    dag.nodes[refreshed_feature]['pack_num'] = dag.nodes[upstream_feature]['pack_num']

    return btp_node


def add_mult_scalar_behind_node(graph: LayerAbstractGraph, compute_node: ComputeNode) -> ComputeNode:
    f_node = list(graph.dag.successors(compute_node))[0]

    skip = list(graph.dag.nodes[f_node]['skip'])
    virtual_shape = list(graph.dag.nodes[f_node]['virtual_shape'])
    virtual_skip = list(graph.dag.nodes[f_node]['virtual_skip'])
    # level = graph.dag.nodes[f_node]['level']
    # pack_num = graph.dag.nodes[f_node]['pack_num']

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
        # level=level,
        # pack_num=pack_num,
    )
    graph.dag.add_node(added_c_node, name=added_c_node.layer_id)
    graph.dag.nodes[added_c_node]['level_cost'] = 1
    graph.dag.add_edge(compute_node, added_f_node)
    graph.dag.add_edge(added_f_node, added_c_node)
    graph.dag.add_edge(added_c_node, f_node)

    return added_c_node


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


def split_upsampling_layers(graph: LayerAbstractGraph):
    for conv_node in list(graph.dag.nodes):
        if not isinstance(conv_node, ConvComputeNode):
            continue
        if any(x > 1 for x in conv_node.upsample_factor):
            feature_in = next(graph.dag.predecessors(conv_node))
            upsample_layer = UpsampleComputeNode(
                layer_id=f'{conv_node.layer_id}_upsample',
                layer_type='upsample',
                channel_input=conv_node.channel_input,
                channel_output=conv_node.channel_output,
                upsample_factor=conv_node.upsample_factor,
            )
            upsample_layer.level_cost = 1
            upsampled_feature = FeatureNode(
                key=f'{upsample_layer.layer_id}_output',
                dim=2,
                channel=upsample_layer.channel_output,
                scale=feature_in.scale,
                ckks_parameter_id=feature_in.ckks_parameter_id,
            )
            _insert_layer_between_feature_and_compute(
                graph.dag,
                feature_in,
                conv_node,
                upsample_layer,
                upsampled_feature,
                new_feature_args={
                    'virtual_shape': list(graph.dag.nodes[feature_in]['virtual_shape']),
                    'virtual_skip': list(graph.dag.nodes[feature_in]['virtual_skip']),
                },
            )
            conv_node.upsample_factor = [1, 1]


def infer_shapes_and_skips(graph: LayerAbstractGraph):
    sorted_nodes = list(nx.topological_sort(graph.dag))
    sorted_compute_nodes = [node for node in sorted_nodes if isinstance(node, ComputeNode)]

    for compute_node in sorted_compute_nodes:
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succ: FeatureNode = next(graph.dag.successors(compute_node))
        graph.dag.nodes[succ]['skip'] = [1, 1]

        if 'reshape' == compute_node.layer_type:
            graph.dag.nodes[succ]['virtual_shape'] = preds[0].shape
            graph.dag.nodes[succ]['virtual_skip'] = graph.dag.nodes[preds[0]]['skip']
            skip = (
                preds[0].shape[0]
                * preds[0].shape[1]
                * graph.dag.nodes[preds[0]]['skip'][0]
                * graph.dag.nodes[preds[0]]['skip'][1]
            )
            graph.dag.nodes[succ]['skip'] = [skip]
            continue
        if isinstance(compute_node, PoolComputeNode):
            for i in range(2):
                if not compute_node.is_adaptive_avgpool:
                    succ.shape[i] = preds[0].shape[i] / compute_node.stride[i]
                    graph.dag.nodes[succ]['skip'][0] = graph.dag.nodes[preds[0]]['skip'][0] * compute_node.stride[0]
                    graph.dag.nodes[succ]['skip'][1] = graph.dag.nodes[preds[0]]['skip'][1] * compute_node.stride[1]
                else:
                    succ.shape[i] = preds[0].shape[i]
                    graph.dag.nodes[succ]['skip'] = graph.dag.nodes[preds[0]]['skip']
            continue
        if preds[0].dim == 0 and succ.dim == 0:
            graph.dag.nodes[succ]['virtual_skip'] = graph.dag.nodes[preds[0]]['virtual_skip']
            graph.dag.nodes[succ]['virtual_shape'] = graph.dag.nodes[preds[0]]['virtual_shape']
            graph.dag.nodes[succ]['skip'] = graph.dag.nodes[preds[0]]['skip']
            continue
        if isinstance(compute_node, SpatialComputeNode):
            for i in range(compute_node.dim):
                succ.shape[i] = (
                    preds[0].shape[i]
                    // compute_node.stride[i]
                    * compute_node.upsample_factor_in[i]
                    * compute_node.upsample_factor[i]
                )
                graph.dag.nodes[succ]['skip'][i] = (
                    graph.dag.nodes[preds[0]]['skip'][i]
                    * compute_node.stride[i]
                    // compute_node.upsample_factor_in[i]
                    // compute_node.upsample_factor[i]
                )
        else:
            for i in range(preds[0].dim):
                succ.shape[i] = preds[0].shape[i]
                graph.dag.nodes[succ]['skip'][i] = graph.dag.nodes[preds[0]]['skip'][i]
        if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
            graph.dag.nodes[succ]['skip'] = [1, 1]


def combine_convs_with_upsamples(graph: LayerAbstractGraph):
    for upsample_node in list(graph.dag.nodes):
        if not isinstance(upsample_node, UpsampleComputeNode):
            continue
        conv_node = find_layer_in_linear_graph(graph, upsample_node, 'conv2d', 'up')
        if conv_node is False:
            raise ValueError('Cannot find a conv node above the upsampling node.')
        conv_out = next(graph.dag.successors(conv_node))

        if (
            conv_out.shape[0] * upsample_node.upsample_factor[0] > config.block_shape[0]
            or conv_out.shape[1] * upsample_node.upsample_factor[1] > config.block_shape[1]
        ):
            continue

        for i in range(conv_node.dim):
            conv_node.upsample_factor_in[i] *= upsample_node.upsample_factor[i]

        cur_compute_node = conv_node
        while True:
            cur_feature_node = next(graph.dag.successors(cur_compute_node))
            for i in range(cur_feature_node.dim):
                cur_feature_node.shape[i] *= upsample_node.upsample_factor[i]
                graph.dag.nodes[cur_feature_node]['skip'][i] //= upsample_node.upsample_factor[i]

            cur_compute_node = next(graph.dag.successors(cur_feature_node))
            if cur_compute_node == upsample_node:
                break
            if cur_compute_node.layer_type in ('relu2d', 'simple_polyrelu'):
                for i in range(cur_feature_node.dim):
                    cur_compute_node.zero_skip[i] *= upsample_node.upsample_factor[i]

        upsample_node.upsample_factor = [1, 1]


def set_level_costs(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        compute_node: ComputeNode = node
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succ: FeatureNode = next(graph.dag.successors(compute_node))

        if isinstance(compute_node, ConvComputeNode):
            if config.style == 'ordinary':
                graph.dag.nodes[compute_node]['level_cost'] = 1
            elif config.style == 'multiplexed':
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
            else:
                raise ValueError('Unsupported config.style')

        elif compute_node.layer_type == 'avgpool2d':
            if preds[0].shape[0] > config.block_shape[0] or preds[0].shape[1] > config.block_shape[1]:
                graph.dag.nodes[compute_node]['level_cost'] = 0
                compute_node.is_big_size = True
                compute_node.is_adaptive_avgpool = False
            else:
                compute_node.is_big_size = False
                succs_sub = list(graph.dag.successors(succ))
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
        elif isinstance(compute_node, UpsampleComputeNode):
            if compute_node.upsample_factor[0] == 1 and compute_node.upsample_factor[1] == 1:
                graph.dag.nodes[compute_node]['level_cost'] = 0
            else:
                graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type.startswith('fc'):
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif 'mult_scalar' in compute_node.layer_type:
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif 'resize' in compute_node.layer_type:
            graph.dag.nodes[compute_node]['level_cost'] = 1
        else:
            graph.dag.nodes[compute_node]['level_cost'] = 0


def insert_drop_level_layers(graph: LayerAbstractGraph):
    for compute in list(graph.dag.nodes):
        if not isinstance(compute, ComputeNode):
            continue
        if compute.layer_type == 'drop_level':
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute))
        succ = next(graph.dag.successors(compute))
        for i in range(len(preds)):
            if 'level' not in graph.dag.nodes[preds[i]]:
                print(f"Warning: node {preds[i].node_id} missing 'level' attribute")
                continue
            pred_level = graph.dag.nodes[preds[i]]['level']
            if 'level' not in graph.dag.nodes[succ]:
                print(f"Warning: node {succ.node_id} missing 'level' attribute")
                continue
            succ_level = graph.dag.nodes[succ]['level']
            level_cost = graph.dag.nodes[compute]['level_cost']

            if (pred_level - succ_level) > level_cost:
                drop_level_layer = add_layer(graph, compute, compute.depth, i, 'drop_level', preds)
                graph.dag.nodes[drop_level_layer]['level_cost'] = pred_level - succ_level - level_cost
                succ_sub = next(graph.dag.successors(drop_level_layer))
                graph.dag.nodes[succ_sub]['level'] = pred_level - graph.dag.nodes[drop_level_layer]['level_cost']


def split_graph_to_linear_subgraph(graph: LayerAbstractGraph) -> list[LayerAbstractGraph]:
    dag_of_linear_subgraphs = graph.dag.copy()
    for node in graph.dag.nodes:
        if graph.dag.in_degree(node) > 1:
            for node_in in graph.dag.predecessors(node):
                if dag_of_linear_subgraphs.has_edge(node_in, node):
                    dag_of_linear_subgraphs.remove_edge(node_in, node)
        if graph.dag.out_degree(node) > 1:
            for node_out in graph.dag.successors(node):
                if dag_of_linear_subgraphs.has_edge(node, node_out):
                    dag_of_linear_subgraphs.remove_edge(node, node_out)

    components = list(nx.weakly_connected_components(dag_of_linear_subgraphs))
    subgraphs = list()
    for component in components:
        # A single feature_node does not constitute a subgraph
        if len(component) <= 1:
            continue
        sub = LayerAbstractGraph()
        sub.dag = dag_of_linear_subgraphs.subgraph(component).copy()
        subgraphs.append(sub)

    return subgraphs


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

        if node_list[0].layer_type in config.absorbable_layers:
            return [True, node_list[0]]

        node = node_list[0]


def handle_valid_poly_subgraph(subgraph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    """Handle poly nodes that can be absorbed in the current subgraph"""

    if not use_mpc_refresh:
        for node in subgraph.dag.nodes:
            if isinstance(node, ComputeNode):
                if node.layer_type == 'simple_polyrelu' or node.layer_type == 'relu2d':
                    find, res_node = find_linear_fhe_layer(node, subgraph, Direction.UP)
                    if find:
                        node.up_scale_str.append(res_node.layer_id)
                elif node.layer_type in {'avgpool2d', 'mult_coeff'}:
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


def set_graph_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    subgraphs = split_graph_to_linear_subgraph(graph)
    for sub in subgraphs:
        handle_valid_poly_subgraph(sub, use_mpc_refresh)

    set_feature_scales(graph)


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


def set_feature_scales(graph: LayerAbstractGraph):
    mpc_scale = 1.0
    for compute in graph.dag.nodes:
        scale = 1.0
        if not isinstance(compute, ComputeNode):
            continue

        if compute.layer_type == 'relu2d' or compute.layer_type == 'mpc_refresh':
            scale = mpc_scale

        elif compute.layer_type == 'avgpool2d':
            if config.graph_type == 'mpc':
                scale = 1.0 / (compute.kernel_shape[0] * compute.kernel_shape[1])
            elif compute.is_adaptive_avgpool or compute.is_big_size:
                scale = 1.0 / (compute.kernel_shape[0] * compute.kernel_shape[1])
        elif compute.layer_type == 'mult_coeff':
            scale = compute.coeff

        if compute.layer_type == 'simple_polyrelu':
            while compute.up_scale_str:
                node_out = set_scale_for_node(graph, compute, 1)
                node_out.vec_scale_path = compute.layer_id
            continue
        while compute.up_scale_str or compute.down_scale_str:
            node_out = set_scale_for_node(graph, compute, scale)


def check_subgraph_validity(subgraph: LayerAbstractGraph, invalid_list: list = None, use_mpc_refresh: bool = False):
    """Check if nodes in the linear subgraph can be absorbed"""

    if use_mpc_refresh:
        layers_to_absorb = ['bootstrapping']
    else:
        layers_to_absorb = ['avgpool2d', 'mult_coeff']
    valid_flag = True

    for node in subgraph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type in layers_to_absorb:
                if isinstance(node, PoolComputeNode) and (not node.is_adaptive_avgpool) and (not node.is_big_size):
                    continue
                is_find_dwon, target_node_down = find_linear_fhe_layer(node, subgraph, Direction.DOWN)
                is_find_up, target_node_up = find_linear_fhe_layer(node, subgraph, Direction.UP)
                if (not is_find_dwon) and (not is_find_up):
                    return False
                elif (not is_find_up) and is_find_dwon and target_node_down.layer_type == 'simple_polyrelu':
                    return False

    return valid_flag


def handle_invalid_poly_subgraph(
    graph, subgraph_index, subs_ordered, subgraph_invalid_poly_dict, use_mpc_refresh: bool = False
):
    """Handle poly nodes that cannot be absorbed in the current subgraph, return the layer_id of the added mult_scalar"""
    current_sub = subs_ordered[subgraph_index]
    all_nodes_in_topo_sort = list(nx.topological_sort(current_sub.dag))
    first_node = [node for node in all_nodes_in_topo_sort if isinstance(node, ComputeNode)][0]
    mult_scalar_layer = add_mult_scalar_behind_node(graph, first_node)

    return mult_scalar_layer.layer_id


def absorb_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    subgraphs = split_graph_to_linear_subgraph(graph)

    index = 0
    invalid_index = []
    subgraph_invalid_poly_dict = dict()
    added_mult_scalar_ids = []

    for sub_in in subgraphs:
        invalid_poly_nodes = []
        if not check_subgraph_validity(sub_in, invalid_poly_nodes, use_mpc_refresh):
            invalid_index.append(index)
        subgraph_invalid_poly_dict[index] = invalid_poly_nodes
        index = index + 1

    for i in range(len(subgraphs)):
        if i in invalid_index:
            added_id = handle_invalid_poly_subgraph(graph, i, subgraphs, subgraph_invalid_poly_dict, use_mpc_refresh)
            if added_id:
                added_mult_scalar_ids.append(added_id)

    return graph
