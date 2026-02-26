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

from components import LayerAbstractGraph, FeatureNode, ComputeNode, config, DEFAULT_SCALE, PoolComputeNode


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'


def _insert_two_nodes_between(dag: nx.DiGraph, old_node_0, old_node_1, new_node_0, new_node_1):
    dag.remove_edge(old_node_0, old_node_1)
    dag.add_edge(old_node_0, new_node_0)
    dag.add_edge(new_node_0, new_node_1)
    dag.add_edge(new_node_1, old_node_1)


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
    *,
    other_graph: LayerAbstractGraph | None = None,
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
    _insert_two_nodes_between(graph.dag, feature_node_in, compute_node, new_compute_node, feature_node_out)

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
        _insert_two_nodes_between(other_graph.dag, feature_node_in, compute_node, new_compute_node, feature_node_out)
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
    dag.add_node(
        refreshed_feature,
        level=dag.nodes[upstream_feature]['level'] + restore_lv,
        skip=skip,
        virtual_shape=[1, 1],
        virtual_skip=[1, 1],
    )
    for s in list(dag.successors(upstream_feature)):
        dag.remove_edge(upstream_feature, s)
        dag.add_edge(refreshed_feature, s)

    btp_node = ComputeNode(
        layer_id=f'{upstream_feature.node_id}_bootstrap',
        layer_type='bootstrapping',
        channel_input=upstream_feature.channel,
        channel_output=refreshed_feature.channel,
        ckks_parameter_id_input=upstream_feature.ckks_parameter_id,
        ckks_parameter_id_output=refreshed_feature.ckks_parameter_id,
    )
    dag.add_node(btp_node, name=btp_node.layer_id, level_cost=-restore_lv)
    dag.add_edge(upstream_feature, btp_node)
    dag.add_edge(btp_node, refreshed_feature)

    slot_num = param_dict[btp_node.ckks_parameter_id_input].poly_modulus_degree // 2
    dag.nodes[refreshed_feature]['pack_num'] = dag.nodes[upstream_feature]['pack_num']

    return btp_node


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


def insert_drop_level_layers(graph: LayerAbstractGraph, *, other_graph: LayerAbstractGraph | None = None):
    for compute in list(graph.dag.nodes):
        if not isinstance(compute, ComputeNode):
            continue
        if compute.layer_type == 'drop_level':
            continue
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute))
        try:
            succ = next(graph.dag.successors(compute))
        except StopIteration:
            continue
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
                drop_level_layer = add_layer(
                    graph, compute, compute.depth, i, 'drop_level', preds, other_graph=other_graph
                )
                graph.dag.nodes[drop_level_layer]['level_cost'] = pred_level - succ_level - level_cost
                succ_sub = next(graph.dag.successors(drop_level_layer))
                graph.dag.nodes[succ_sub]['level'] = pred_level - graph.dag.nodes[drop_level_layer]['level_cost']
                if other_graph:
                    other_graph.dag.nodes[succ_sub]['level'] = graph.dag.nodes[succ_sub]['level']


def split_graph_to_linear_subgraph(graph: LayerAbstractGraph) -> tuple[list[LayerAbstractGraph], list[tuple]]:
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


def set_graph_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    pre_process(graph)
    subgraphs, removed_edges = split_graph_to_linear_subgraph(graph)
    for sub in subgraphs:
        handle_valid_poly_subgraph(sub, use_mpc_refresh)

    recovery_graph_from_subgraph(graph, subgraphs)
    del_identity_layer(graph)
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

        if compute.layer_type == 'simple_polyrelu':
            while compute.up_scale_str:
                node_out = set_scale_for_node(graph, compute, 1)
                node_out.vec_scale_path = compute.layer_id
            continue
        while compute.up_scale_str or compute.down_scale_str:
            node_out = set_scale_for_node(graph, compute, scale)


def check_degree(graph: LayerAbstractGraph, node):
    if graph.dag.in_degree(node) > 1 or graph.dag.out_degree(node) > 1:
        return False
    else:
        return True


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


def pre_process(graph: LayerAbstractGraph):
    all_nodes_in_topo_sort = list(nx.topological_sort(graph.dag))
    for node in all_nodes_in_topo_sort:
        if not check_degree(graph, node) and isinstance(node, FeatureNode):
            for next_node in list(graph.dag.successors(node)):
                if not check_degree(graph, next_node):
                    add_identity_layer(graph, node, next_node)


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
            from processor import handle_invalid_poly_subgraph

            added_id = handle_invalid_poly_subgraph(
                i, subs_odered, next_dict, pre_dict, subgraph_invalid_poly_dict, use_mpc_refresh
            )
            if added_id:
                added_mult_scalar_ids.append(added_id)

    recovery_graph_from_subgraph(graph, subgraphs)
    del_identity_layer(graph)

    return graph
