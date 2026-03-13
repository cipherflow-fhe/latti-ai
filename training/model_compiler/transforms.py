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
import math
import time
from enum import Enum
import networkx as nx

from components import *


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'


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
    use_skip = config.style != 'multiplexed'
    for f_node in preds + succs:
        dag.nodes[f_node]['pack_num'] = _calc_pack_num(dag, f_node, slot_num, use_skip=use_skip)


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


def _insert_layer_after_compute(
    dag: nx.DiGraph,
    old_compute: ComputeNode,
    new_feature: FeatureNode,
    new_compute: ComputeNode,
    *,
    new_feature_args: dict | None = None,
    new_compute_args: dict | None = None,
):
    if new_compute_args is None:
        new_compute_args = dict()
    if new_feature_args is None:
        new_feature_args = dict()
    dag.add_node(new_feature, **new_feature_args)
    dag.add_node(new_compute, **new_compute_args)

    old_feature_list = list(dag.successors(old_compute))
    if len(old_feature_list) != 1:
        raise ValueError(
            f'Expected exactly one output feature for compute node {old_compute.layer_id}, got {len(old_feature_list)}'
        )
    old_feature = old_feature_list[0]

    dag.remove_edge(old_compute, old_feature)
    dag.add_edge(old_compute, new_feature)
    dag.add_edge(new_feature, new_compute)
    dag.add_edge(new_compute, old_feature)


def _delete_layer(
    dag: nx.DiGraph,
    compute: ComputeNode,
):
    """Remove *compute* and its output FeatureNode, rewiring the predecessor
    feature directly to all downstream compute nodes.

    Before: feature_in -> compute -> feature_out -> downstream_compute(s)
    After:  feature_in -> downstream_compute(s)
    """
    pred_list = list(dag.predecessors(compute))
    if len(pred_list) != 1:
        raise ValueError(f'Expected exactly one predecessor for compute node {compute.layer_id}, got {len(pred_list)}')
    feature_in = pred_list[0]

    feature_out_list = list(dag.successors(compute))
    if len(feature_out_list) != 1:
        raise ValueError(
            f'Expected exactly one output feature for compute node {compute.layer_id}, got {len(feature_out_list)}'
        )
    feature_out = feature_out_list[0]

    downstream_computes = list(dag.successors(feature_out))

    dag.remove_node(feature_out)
    dag.remove_node(compute)
    for dc in downstream_computes:
        dag.add_edge(feature_in, dc)


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
    old_output_feature = next(graph.dag.successors(compute_node))

    skip = list(graph.dag.nodes[old_output_feature]['skip'])
    virtual_shape = list(graph.dag.nodes[old_output_feature]['virtual_shape'])
    virtual_skip = list(graph.dag.nodes[old_output_feature]['virtual_skip'])

    mult_scalar_output = copy.deepcopy(old_output_feature)
    old_output_feature.node_id = old_output_feature.node_id + '_mult_scalar_output'
    old_output_feature.scale = 1.0

    mult_scalar_node = MultScalarComputeNode(
        compute_node.layer_id + '_mult_scalar_', 'mult_scalar', compute_node.channel_input, compute_node.channel_output
    )

    _insert_layer_after_compute(
        graph.dag,
        compute_node,
        mult_scalar_output,
        mult_scalar_node,
        new_feature_args={
            'name': mult_scalar_output.node_id,
            'skip': skip,
            'virtual_shape': virtual_shape,
            'virtual_skip': virtual_skip,
        },
        new_compute_args={'name': mult_scalar_node.layer_id, 'level_cost': 1},
    )


def find_layer_in_linear_graph(
    graph: LayerAbstractGraph, c_node: ComputeNode, target_layer_type: str, direction: str
) -> ComputeNode | None:
    node = c_node
    while True:
        if direction == 'up':
            if graph.dag.in_degree(node) != 1:
                return None
            node = next(graph.dag.predecessors(node))
        else:
            if graph.dag.out_degree(node) != 1:
                return None
            node = next(graph.dag.successors(node))

        if isinstance(node, ComputeNode) and node.layer_type == target_layer_type:
            return node


def find_absorbable_layer_in_linear_subgraph(
    subgraph: nx.DiGraph, c_node: ComputeNode, direction: Direction
) -> ComputeNode | None:
    node = c_node
    while True:
        if direction == Direction.UP:
            if subgraph.in_degree(node) != 1:
                return None
            node = next(subgraph.predecessors(node))
        else:
            if subgraph.out_degree(node) != 1:
                return None
            node = next(subgraph.successors(node))

        if isinstance(node, ComputeNode) and node.layer_type in config.absorbable_layers:
            return node


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


def infer_shapes_skips_and_pack_num(graph: LayerAbstractGraph):
    sorted_nodes = list(nx.topological_sort(graph.dag))
    sorted_compute_nodes = [node for node in sorted_nodes if isinstance(node, ComputeNode)]

    for compute_node in sorted_compute_nodes:
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succ: FeatureNode = next(graph.dag.successors(compute_node))
        graph.dag.nodes[succ]['skip'] = [1] * succ.dim
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
        populate_pack_num(graph.dag, compute_node, config.poly_n / 2)


def combine_convs_with_upsamples(graph: LayerAbstractGraph):
    for upsample_node in list(graph.dag.nodes):
        if not isinstance(upsample_node, UpsampleComputeNode):
            continue
        conv_node = find_layer_in_linear_graph(graph, upsample_node, 'conv2d', 'up')
        if conv_node is None:
            raise ValueError('Cannot find a conv node above the upsampling node.')
        conv_out = next(graph.dag.successors(conv_node))
        dim = upsample_node.dim

        if any(conv_out.shape[i] * upsample_node.upsample_factor[i] > config.block_shape[i] for i in range(dim)):
            continue

        for i in range(dim):
            conv_node.upsample_factor_in[i] *= upsample_node.upsample_factor[i]

        cur_compute_node = conv_node
        while True:
            cur_feature_node = next(graph.dag.successors(cur_compute_node))
            for i in range(dim):
                cur_feature_node.shape[i] *= upsample_node.upsample_factor[i]
                graph.dag.nodes[cur_feature_node]['skip'][i] //= upsample_node.upsample_factor[i]

            cur_compute_node = next(graph.dag.successors(cur_feature_node))
            if cur_compute_node == upsample_node:
                break
            if cur_compute_node.layer_type in ('relu2d', 'simple_polyrelu'):
                for i in range(dim):
                    cur_compute_node.zero_skip[i] *= upsample_node.upsample_factor[i]

        _delete_layer(graph.dag, upsample_node)


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


def split_graph_to_linear_subgraph(dag: nx.DiGraph) -> list[nx.DiGraph]:
    dag_of_linear_subgraphs = dag.copy()
    for node in dag.nodes:
        if dag.in_degree(node) > 1:
            for node_in in dag.predecessors(node):
                if dag_of_linear_subgraphs.has_edge(node_in, node):
                    dag_of_linear_subgraphs.remove_edge(node_in, node)
        if dag.out_degree(node) > 1:
            for node_out in dag.successors(node):
                if dag_of_linear_subgraphs.has_edge(node, node_out):
                    dag_of_linear_subgraphs.remove_edge(node, node_out)

    components = list(nx.weakly_connected_components(dag_of_linear_subgraphs))
    return [dag_of_linear_subgraphs.subgraph(component).copy() for component in components if len(component) > 1]


def handle_valid_poly_subgraph(subgraph: nx.DiGraph, use_mpc_refresh: bool = False):
    """Handle poly nodes that can be absorbed in the current subgraph"""

    if not use_mpc_refresh:
        for node in subgraph.nodes:
            if isinstance(node, ComputeNode):
                if node.layer_type == 'simple_polyrelu' or node.layer_type == 'relu2d':
                    res_node = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.UP)
                    if res_node is not None:
                        node.up_scale_str.append(res_node.layer_id)
                elif node.layer_type in {'avgpool2d', 'mult_coeff'}:
                    res_node_down = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.DOWN)
                    if res_node_down is not None and res_node_down.layer_type != 'simple_polyrelu':
                        node.down_scale_str.append(res_node_down.layer_id)

                        continue
                    res_node_up = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.UP)
                    if res_node_up is not None:
                        node.up_scale_str.append(res_node_up.layer_id)
    else:
        candidates = {}

        for node in subgraph.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                res_node_down = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.DOWN)
                res_node_up = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.UP)

                candidates[node] = {
                    'down': res_node_down
                    if (res_node_down is not None and res_node_down.layer_type != 'simple_polyrelu')
                    else None,
                    'up': res_node_up,
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
    subgraphs = split_graph_to_linear_subgraph(graph.dag)
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


def linear_subgraph_can_absorb_scale(subgraph: nx.DiGraph, use_mpc_refresh: bool = False):
    """Check if nodes in the linear subgraph can be absorbed"""
    if use_mpc_refresh:
        layers_to_absorb = ['bootstrapping']
    else:
        layers_to_absorb = ['avgpool2d', 'mult_coeff']

    for node in subgraph.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type in layers_to_absorb:
                if isinstance(node, PoolComputeNode) and (not node.is_adaptive_avgpool) and (not node.is_big_size):
                    continue
                target_node_down = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.DOWN)
                target_node_up = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.UP)
                if target_node_down is None and target_node_up is None:
                    return False
                elif (
                    target_node_up is None
                    and target_node_down is not None
                    and target_node_down.layer_type == 'simple_polyrelu'
                ):
                    return False
                else:
                    continue

    return True


def insert_mult_scalar_in_linear_subgraph(graph, subgraph: nx.DiGraph):
    first_compute_node = next(node for node in nx.topological_sort(subgraph) if isinstance(node, ComputeNode))
    add_mult_scalar_behind_node(graph, first_compute_node)


def absorb_scale(graph: LayerAbstractGraph, use_mpc_refresh: bool = False):
    subgraphs = split_graph_to_linear_subgraph(graph.dag)

    unchangable_subgraphs = list()
    for subgraph in subgraphs:
        if not linear_subgraph_can_absorb_scale(subgraph, use_mpc_refresh):
            unchangable_subgraphs.append(subgraph)

    for subgraph in unchangable_subgraphs:
        insert_mult_scalar_in_linear_subgraph(graph, subgraph)

    return graph
