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
        return math.ceil(slot_num / (attrs['skip'][0]))
    else:
        denom = math.prod(feature_node.shape) * math.prod(feature_node.invalid_fill)
        return math.ceil(slot_num / denom)


def populate_pack_num(dag: nx.DiGraph, node, slot_num: int):
    preds = list(dag.predecessors(node))
    succs = list(dag.successors(node))
    for f_node in preds + succs:
        dag.nodes[f_node]['pack_num'] = _calc_pack_num(dag, f_node, slot_num)


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

    old_edge_attrs = dict(dag.edges[old_feature, old_compute])
    dag.remove_edge(old_feature, old_compute)
    dag.add_edge(old_feature, new_compute, **old_edge_attrs)
    dag.add_edge(new_compute, new_feature)
    dag.add_edge(new_feature, old_compute, **old_edge_attrs)


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
        old_edge_attrs = dict(dag.edges[old_feature, oc])
        dag.remove_edge(old_feature, oc)
        dag.add_edge(new_feature, oc, **old_edge_attrs)
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
    downstream_edge_attrs = {dc: dict(dag.edges[feature_out, dc]) for dc in downstream_computes}

    dag.remove_node(feature_out)
    dag.remove_node(compute)
    for dc in downstream_computes:
        dag.add_edge(feature_in, dc, **downstream_edge_attrs[dc])


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
    level = graph.dag.nodes[feature_node_in]['level']
    pack_num = graph.dag.nodes[feature_node_in]['pack_num']

    scale = feature_node_in.scale
    timestamp = int(time.time() * 1000000)
    layer_id = f'{compute_node.layer_id}_{layer_type}_idx{index}_ts{timestamp}'

    feature_node_out = FeatureNode(
        feature_node_in.node_id + str(id(shape)) + f'_{layer_type}_output',
        dim,
        channel_output,
        scale,
        feature_node_in.ckks_parameter_id,
        ckks_scale,
        shape,
    )
    feature_node_out.sp_info = feature_node_in.sp_info.copy()

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
            'level': level,
            'pack_num': pack_num,
        },
    )

    return new_compute_node


def add_mult_scalar_between_feature_and_layer(
    graph: LayerAbstractGraph,
    f_node: FeatureNode,
    c_node: ComputeNode,
) -> MultScalarComputeNode:
    """Insert a mult_scalar compute node on the edge f_node -> c_node.

    Before: f_node -> c_node
    After:  f_node -> mult_scalar -> mult_scalar_f_node -> c_node

    Args:
        graph: the computation graph
        f_node: the upstream feature node
        c_node: the downstream compute node

    Returns:
        The newly created MultScalarComputeNode.
    """
    timestamp = int(time.time() * 1000000)
    layer_id = f'{c_node.layer_id}_mult_scalar_ts{timestamp}'

    mult_scalar = MultScalarComputeNode(layer_id, 'mult_scalar', f_node.channel, f_node.channel)

    mult_scalar_f_node = FeatureNode(
        f_node.node_id + f'_mult_scalar_out_ts{timestamp}',
        f_node.dim,
        f_node.channel,
        f_node.scale,
        f_node.ckks_parameter_id,
        f_node.ckks_scale,
        list(f_node.shape),
    )
    mult_scalar_f_node.sp_info = f_node.sp_info.copy()

    skip = list(graph.dag.nodes[f_node]['skip'])
    # level = graph.dag.nodes[f_node]['level']
    pack_num = graph.dag.nodes[f_node]['pack_num']

    _insert_layer_between_feature_and_compute(
        graph.dag,
        f_node,
        c_node,
        mult_scalar,
        mult_scalar_f_node,
        new_compute_args={'name': layer_id, 'level_cost': 1},
        new_feature_args={
            'name': mult_scalar_f_node.node_id,
            'skip': skip,
            # 'level': level,
            'pack_num': pack_num,
        },
    )

    return mult_scalar


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
        skip = [1] * upstream_feature.dim
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
        },
    )

    slot_num = param_dict[upstream_feature.ckks_parameter_id].poly_modulus_degree // 2
    dag.nodes[refreshed_feature]['pack_num'] = dag.nodes[upstream_feature]['pack_num']
    refreshed_feature.sp_info = upstream_feature.sp_info.copy()

    return btp_node


def add_mult_scalar_behind_node(graph: LayerAbstractGraph, compute_node: ComputeNode) -> ComputeNode:
    old_output_feature = next(graph.dag.successors(compute_node))

    skip = list(graph.dag.nodes[old_output_feature]['skip'])

    mult_scalar_output = copy.deepcopy(old_output_feature)
    old_output_feature.node_id = old_output_feature.node_id + '_mult_scalar_output'
    old_output_feature.scale = 1.0

    mult_scalar_node = MultScalarComputeNode(
        compute_node.layer_id + '_mult_scalar_', 'mult_scalar', compute_node.channel_input, compute_node.channel_output
    )

    # Inherit is_big_size from predecessor's shape vs block_shape
    if any(old_output_feature.shape[i] > config.block_shape[i] for i in range(old_output_feature.dim)):
        mult_scalar_node.is_big_size = True

    _insert_layer_after_compute(
        graph.dag,
        compute_node,
        mult_scalar_output,
        mult_scalar_node,
        new_feature_args={
            'name': mult_scalar_output.node_id,
            'skip': skip,
        },
        new_compute_args={'name': mult_scalar_node.layer_id, 'level_cost': 1},
    )

    return mult_scalar_node


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
                new_feature_args={},
            )
            conv_node.upsample_factor = [1] * conv_node.dim


def process_special_info(
    graph: LayerAbstractGraph, compute_node: ComputeNode, preds: list[FeatureNode], succ: FeatureNode
):
    """Process sp_info for dim=0 and reshape nodes. Returns True if caller should continue."""

    # 2d->2d, 1d->1d
    if preds[0].dim == succ.dim and succ.dim in (1, 2):
        invalid_fill_default = [1] * succ.dim
        if config.style == 'ordinary':
            succ.invalid_fill = graph.dag.nodes[succ]['skip'].copy()
        else:
            if isinstance(compute_node, PoolComputeNode):
                if compute_node.is_adaptive_avgpool:
                    succ.invalid_fill = compute_node.stride.copy()
                else:
                    succ.invalid_fill = invalid_fill_default
            elif isinstance(compute_node, ConvComputeNode):
                succ.invalid_fill = invalid_fill_default
            else:
                succ.invalid_fill = preds[0].invalid_fill
        return False
    # 2d -> 0d, 1d->0d: reshape
    if (preds[0].dim == 2 or preds[0].dim == 1) and succ.dim == 0:
        succ.has_sp_info = True
        succ.sp_info['skip'] = graph.dag.nodes[preds[0]]['skip'].copy()
        succ.sp_info['shape'] = preds[0].shape
        succ.sp_info['invalid_fill'] = preds[0].invalid_fill
        graph.dag.nodes[succ]['skip'] = [math.prod(succ.sp_info['skip']) * math.prod(succ.sp_info['shape'])]
        return True

    # 0d -> 0d
    if preds[0].dim == 0 and succ.dim == 0:
        graph.dag.nodes[succ]['skip'] = graph.dag.nodes[preds[0]]['skip'].copy()
        if len(preds) > 1 and not all(p.has_sp_info == preds[0].has_sp_info for p in preds[1:]):
            raise ValueError(
                f'Multi-input 0d->0d: all inputs must share the same has_sp_info, got {[p.has_sp_info for p in preds]}'
            )
        if preds[0].has_sp_info and 'fc' not in compute_node.layer_type:
            succ.has_sp_info = True
            succ.sp_info = preds[0].sp_info.copy()
        else:
            succ.has_sp_info = False
        return True


def infer_shapes_skips_and_pack_num(graph: LayerAbstractGraph):
    sorted_nodes = list(nx.topological_sort(graph.dag))
    sorted_compute_nodes = [node for node in sorted_nodes if isinstance(node, ComputeNode)]
    c_node_num = len(sorted_compute_nodes)

    N = config.fhe_param.poly_modulus_degree
    leading_skip = 2 ** math.floor(math.log2(N) / 2)
    for node in sorted_nodes:
        if isinstance(node, FeatureNode) and graph.dag.in_degree(node) == 0 and node.dim == 0:
            graph.dag.nodes[node]['skip'] = [leading_skip]

    for compute_node in sorted_compute_nodes:
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        succ: FeatureNode = next(graph.dag.successors(compute_node))
        # init skip,
        if succ.dim != 0:
            graph.dag.nodes[succ]['skip'] = [1] * succ.dim

        if process_special_info(graph, compute_node, preds, succ):
            populate_pack_num(graph.dag, compute_node, config.fhe_param.poly_modulus_degree / 2)
            continue
        if succ.dim > 0:
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
        if preds[0].dim >= 1 and any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
            graph.dag.nodes[succ]['skip'] = [1] * preds[0].dim
            if any(succ.shape[i] < config.block_shape[i] for i in range(succ.dim)):
                for i in range(preds[0].dim):
                    graph.dag.nodes[succ]['skip'][i] = config.block_shape[i] / succ.shape[i]

        process_special_info(graph, compute_node, preds, succ)
        populate_pack_num(graph.dag, compute_node, config.fhe_param.poly_modulus_degree / 2)


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
            if cur_compute_node.layer_type in ('relu2d', 'polyact'):
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
                if any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
                    compute_node.is_big_size = True
                    graph.dag.nodes[compute_node]['level_cost'] = 1
                    if any(succ.shape[i] < config.block_shape[i] for i in range(succ.dim)):
                        graph.dag.nodes[compute_node]['level_cost'] = 2
                else:
                    if compute_node.groups == 1:
                        if all(compute_node.stride[i] == 1 for i in range(compute_node.dim)) and all(
                            graph.dag.nodes[preds[0]]['skip'][i] == 1 for i in range(preds[0].dim)
                        ):
                            graph.dag.nodes[compute_node]['level_cost'] = 1
                        else:
                            graph.dag.nodes[compute_node]['level_cost'] = 2
                    else:
                        if all(compute_node.stride[i] == 1 for i in range(compute_node.dim)):
                            graph.dag.nodes[compute_node]['level_cost'] = 1
                        else:
                            graph.dag.nodes[compute_node]['level_cost'] = 2
            else:
                raise ValueError('Unsupported config.style')

        elif compute_node.layer_type in {'avgpool1d', 'avgpool2d'}:
            if any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
                compute_node.is_big_size = True
                compute_node.is_adaptive_avgpool = False
                if any(succ.shape[i] < config.block_shape[i] for i in range(succ.dim)):
                    graph.dag.nodes[compute_node]['level_cost'] = 1  # repack needed
                else:
                    graph.dag.nodes[compute_node]['level_cost'] = 0
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
            if any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
                compute_node.is_big_size = True
        elif isinstance(compute_node, UpsampleComputeNode):
            if all(compute_node.upsample_factor[i] == 1 for i in range(compute_node.dim)):
                graph.dag.nodes[compute_node]['level_cost'] = 0
            else:
                graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type.startswith('fc'):
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif 'mult_scalar' in compute_node.layer_type:
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif 'resize' in compute_node.layer_type:
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type == 'concat2d':
            has_uneven = any(p.channel % graph.dag.nodes[p]['pack_num'] != 0 for p in preds)
            graph.dag.nodes[compute_node]['level_cost'] = 1 if has_uneven else 0
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
                if node.layer_type == 'polyact' or node.layer_type == 'relu2d':
                    res_node = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.UP)
                    if res_node is not None:
                        node.up_scale_str.append(res_node.layer_id)
                elif node.layer_type in {'avgpool1d', 'avgpool2d', 'mult_coeff'}:
                    res_node_down = find_absorbable_layer_in_linear_subgraph(subgraph, node, Direction.DOWN)
                    if res_node_down is not None and res_node_down.layer_type != 'polyact':
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
                    if (res_node_down is not None and res_node_down.layer_type != 'polyact')
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

        elif compute.layer_type in {'avgpool1d', 'avgpool2d'}:
            kernel_prod = math.prod(compute.kernel_shape)
            if config.graph_type == 'mpc':
                scale = 1.0 / kernel_prod
            elif compute.is_adaptive_avgpool or compute.is_big_size:
                scale = 1.0 / kernel_prod
        elif compute.layer_type == 'mult_coeff':
            scale = compute.coeff

        if compute.layer_type == 'polyact':
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
        layers_to_absorb = ['avgpool1d', 'avgpool2d', 'mult_coeff']

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
                    target_node_up is None and target_node_down is not None and target_node_down.layer_type == 'polyact'
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


def _backward_level_dict(dag: nx.DiGraph) -> tuple[dict, dict]:
    """Compute backward FHE levels for all FeatureNodes in *dag*.

    Uses the same logic as inspect_level_backward: graph-output FeatureNodes
    start at level 0; going toward inputs, each ComputeNode adds its
    level_cost.

    Returns:
        level     : FeatureNode → int  (backward level of each FeatureNode)
        arm_level : (FeatureNode, ComputeNode) → int
                    level contribution of each output edge, i.e.
                    level[feat_out_of_compute] + level_cost[compute].
                    Used by fork decisions in _propagate_scale to avoid
                    recomputing per-arm contributions during recursion.

    Prerequisite: dag.nodes[compute]['level_cost'] must already be set
    (i.e. set_level_costs has been called before absorb_scale_new).
    """
    level: dict = {}
    arm_level: dict = {}
    for node in reversed(list(nx.topological_sort(dag))):
        if isinstance(node, ComputeNode):
            continue
        succs = list(dag.successors(node))
        if not succs:
            level[node] = 0
        else:
            best = 0
            for c in succs:
                contrib = level[next(dag.successors(c))] + dag.nodes[c]['level_cost']
                arm_level[(node, c)] = contrib
                if contrib > best:
                    best = contrib
            level[node] = best
    return level, arm_level


# ---------------------------------------------------------------------------
# Absorber layer types (used by graph_static_analysis and helpers)
# ---------------------------------------------------------------------------
_ABSORBER_TYPES = frozenset({'conv2d', 'fc0', 'fc1', 'mult_scalar', 'polyact'})
# polyact excluded: p(x/s) ≠ p(x)/s, so polyact cannot absorb in DOWN direction
_DOWN_ABSORBER_TYPES = frozenset({'conv2d', 'fc0', 'fc1', 'mult_scalar'})


def graph_static_analysis(dag: nx.DiGraph) -> tuple[dict, dict, dict, dict]:
    """Pass 1: static analysis of the DAG for scale absorption decisions.

    Returns
    -------
    feat_info : dict  keyed by FeatureNode
        'absorber_up'           : bool — every UP path from F reaches an absorber
        'bottleneck_succs'      : frozenset[ComputeNode] — output arms that set F's level
        'level_free_up'         : bool — at least one non-bottleneck output arm exists
        'nearest_convergence_up': FeatureNode | None — nearest shared upstream for join siblings
        'node_insert_cost'      : int — cost of terminating scale at this FeatureNode

    min_cover_above : dict  keyed by ComputeNode
        Minimum cost (in FREE/NONFREE units) to handle a scale that propagates
        ABOVE ComputeNode C.  Replaces the old per-FeatureNode min_ms_up.

        Key properties:
        - Absorber C: 0 (scale absorbed for free).
        - Single-input C: min of insert-at-f_above vs go-further-above.
        - JOIN C: sum over groups (convergent arm-groups pay once per shared ancestor).

    plan_above : dict  keyed by ComputeNode
        Ordered list of structural actions to execute when scale arrives ABOVE C.
        Each action is a tuple; see :func:`absorb_scale_new_dp` docstring for
        the supported action kinds.

    node_insert_cost : dict  keyed by FeatureNode
        Cost of terminating scale at this FeatureNode (Σ arm costs over all
        non-DOWN-absorber output arms, with non-bottleneck arms billed FREE and
        bottleneck arms billed NONFREE).
    """
    _FREE_COST = 1
    _NONFREE_COST = 10_000

    _, arm_level = _backward_level_dict(dag)
    topo = list(nx.topological_sort(dag))
    topo_rank = {n: i for i, n in enumerate(topo)}

    # ------------------------------------------------------------------ #
    # Phase 1: bottleneck_succs + level_free_up (no ordering constraint)
    # ------------------------------------------------------------------ #
    bottleneck_succs: dict[object, frozenset] = {}
    level_free_up: dict[object, bool] = {}
    for node in dag.nodes:
        if isinstance(node, ComputeNode):
            continue
        succs = list(dag.successors(node))
        if not succs:
            bottleneck_succs[node] = frozenset()
            level_free_up[node] = False
            continue
        arm_levels = {c: arm_level.get((node, c), 0) for c in succs}
        max_lv = max(arm_levels.values())
        bottleneck_succs[node] = frozenset(c for c, lv in arm_levels.items() if lv == max_lv)
        level_free_up[node] = len(bottleneck_succs[node]) < len(succs)

    # ------------------------------------------------------------------ #
    # Phase 2: absorber_up (topological, graph-inputs first)
    # ------------------------------------------------------------------ #
    absorber_up: dict[object, bool] = {}
    for node in topo:
        if isinstance(node, ComputeNode):
            continue
        preds = list(dag.predecessors(node))
        if not preds:
            absorber_up[node] = False
            continue
        all_covered = True
        for c in preds:
            if c.layer_type in _ABSORBER_TYPES:
                continue
            for f_above in dag.predecessors(c):
                if not absorber_up.get(f_above, False):
                    all_covered = False
                    break
            if not all_covered:
                break
        absorber_up[node] = all_covered

    # ------------------------------------------------------------------ #
    # Phase 3: node_insert_cost[F]
    # Cost of "stopping scale at F" = inserting ms on every non-DOWN-absorber
    # output arm.  Arm cost is FREE if arm is non-bottleneck, NONFREE otherwise.
    # ------------------------------------------------------------------ #
    node_insert_cost: dict[object, int] = {}
    for node in dag.nodes:
        if isinstance(node, ComputeNode):
            continue
        succs = list(dag.successors(node))
        if not succs:
            node_insert_cost[node] = 0
            continue
        cost = 0
        for s in succs:
            if s.layer_type in _DOWN_ABSORBER_TYPES:
                continue
            cost += _NONFREE_COST if s in bottleneck_succs[node] else _FREE_COST
        node_insert_cost[node] = cost

    # ------------------------------------------------------------------ #
    # Phase 4: nearest_convergence_up (topological)
    # ------------------------------------------------------------------ #
    id_to_feat: dict[int, object] = {id(n): n for n in dag.nodes if isinstance(n, FeatureNode)}

    def _upstream_feat_ids(start_feat) -> set:
        visited: set = set()
        stack = [start_feat]
        while stack:
            f = stack.pop()
            fid = id(f)
            if fid in visited:
                continue
            visited.add(fid)
            for c in dag.predecessors(f):
                if c.layer_type in _ABSORBER_TYPES:
                    continue
                for f_above in dag.predecessors(c):
                    stack.append(f_above)
        return visited

    nearest_convergence_up: dict[object, object | None] = {n: None for n in dag.nodes if isinstance(n, FeatureNode)}
    for node in topo:
        if not isinstance(node, ComputeNode):
            continue
        preds = list(dag.predecessors(node))
        if len(preds) < 2:
            continue
        reach = [(f, _upstream_feat_ids(f)) for f in preds]
        for i in range(len(reach)):
            f_i, set_i = reach[i]
            for j in range(i + 1, len(reach)):
                f_j, set_j = reach[j]
                shared = set_i & set_j
                if not shared:
                    continue
                nearest_fid = max(
                    (fid for fid in shared if fid in id_to_feat),
                    key=lambda fid: topo_rank.get(id_to_feat[fid], -1),
                    default=None,
                )
                if nearest_fid is None:
                    continue
                nearest_f = id_to_feat[nearest_fid]
                for arm_f in (f_i, f_j):
                    existing = nearest_convergence_up[arm_f]
                    if existing is None or topo_rank[nearest_f] > topo_rank[existing]:
                        nearest_convergence_up[arm_f] = nearest_f

    # ------------------------------------------------------------------ #
    # Phase 5: min_cover_above[C] per ComputeNode (topological, inputs first)
    #
    # Replaces the old per-FeatureNode min_ms_up.  Fixes two structural bugs:
    #   1. cost_insert was edge-level (missed compensation arms).
    #   2. shared-ancestor paths were double-counted in multi-input JOIN nodes.
    #
    # Key formulas:
    #   comp(F, excluded_cs) = sum of arm costs for non-excluded, non-DOWN-absorber succs of F
    #   insert_all_cost(F)   = node_insert_cost[F]  (stops scale at F on all arms)
    #   go_above_cost(F, ci_set) = comp(F, ci_set) + min_cover_above[pred_C_of_F]
    #   min_cover(F, ci_set) = min(insert_all_cost, go_above_cost)  [theoretical: only these two are optimal]
    #
    # For single-input C (pred = f_above):
    #   min_cover_above[C] = min_cover(f_above, {C})
    #
    # For JOIN C (preds = {f1, f2, ...}):
    #   Group preds by nearest_convergence_up:
    #     - Absorber preds: cost 0
    #     - Individual preds (ncu=None): min_cover_singleton(fi, C)
    #     - Convergence group {fi, fj, ...} with shared G:
    #         pay comp(fi, {C}) + comp(fj, {C}) + ... + min_cover(G, ci_set)
    # ------------------------------------------------------------------ #
    def _comp(f_node, excluded_cs) -> int:
        """Compensation cost: ms on non-excluded, non-DOWN-absorber succs of f_node."""
        cost = 0
        for s in dag.successors(f_node):
            if s in excluded_cs:
                continue
            if s.layer_type in _DOWN_ABSORBER_TYPES:
                continue
            cost += _NONFREE_COST if s in bottleneck_succs[f_node] else _FREE_COST
        return cost

    def _insert_plan_at(f_node, ci_set):
        """Actions when terminating scale at f_node (option-A: F unchanged,
        scale applied only on ci_set arms)."""
        actions = []
        for c in dag.successors(f_node):
            if c not in ci_set:
                continue
            if c.layer_type in _DOWN_ABSORBER_TYPES:
                actions.append(('absorb_up', c))
            else:
                actions.append(('insert_ms_up', f_node, c))
        return actions

    def _compensate_plan_at(f_node, ci_set):
        """Actions when passing UP through f_node: 1/s compensation on every
        non-ci_set successor (absorbers eat it for free, else propagate down)."""
        actions = []
        for c in dag.successors(f_node):
            if c in ci_set:
                continue
            if c.layer_type in _DOWN_ABSORBER_TYPES:
                actions.append(('absorb_down', c))
            else:
                actions.append(('compensate_down', f_node, c))
        return actions

    def _min_cover_at(f_node, ci_set):
        """min(insert_all, go_above) at f_node, where ci_set are the excluded arms.
        Returns (cost, plan)."""
        insert_cost = node_insert_cost[f_node]
        pred_cs = list(dag.predecessors(f_node))
        if not pred_cs:
            # graph input: cannot go further above — must terminate here
            return insert_cost, _insert_plan_at(f_node, ci_set)

        pred_c = pred_cs[0]  # FeatureNode has exactly one predecessor ComputeNode
        go_cost = _comp(f_node, ci_set) + min_cover_above.get(pred_c, _NONFREE_COST)

        if insert_cost <= go_cost:
            return insert_cost, _insert_plan_at(f_node, ci_set)
        plan = _compensate_plan_at(f_node, ci_set)
        plan.extend(plan_above.get(pred_c, []))
        return go_cost, plan

    def _find_ci_toward_fi(G, f_i):
        """Find the direct successor ComputeNode of G on the path toward f_i."""
        # Walk up from f_i until we find a FeatureNode whose pred ComputeNode has G as a pred.
        current_f = f_i
        while current_f is not G:
            pred_cs = list(dag.predecessors(current_f))
            if not pred_cs:
                return None
            pred_c = pred_cs[0]
            pred_fs = list(dag.predecessors(pred_c))
            if G in pred_fs:
                return pred_c  # direct succ of G toward f_i
            if len(pred_fs) != 1:
                return None  # multi-input node, cannot trace single path
            current_f = pred_fs[0]
        return None  # f_i is G itself (shouldn't happen)

    min_cover_above: dict[object, int] = {}
    plan_above: dict[object, list] = {}

    for C in topo:
        if not isinstance(C, ComputeNode):
            continue

        if C.layer_type in _ABSORBER_TYPES:
            min_cover_above[C] = 0
            plan_above[C] = [('absorb_up', C)]
            continue

        preds_f = list(dag.predecessors(C))

        if len(preds_f) == 1:
            # Single-input: straightforward
            f_above = preds_f[0]
            cost_C, plan_C = _min_cover_at(f_above, {C})
            min_cover_above[C] = cost_C
            plan_above[C] = plan_C
        else:
            # JOIN (add/cat/etc.): group preds by nearest_convergence_up
            total = 0
            plan_C: list = []
            grouped: dict[int, dict] = {}  # id(G) -> {G, f_list, comp_sum, ci_set, comp_plan}

            for f_i in preds_f:
                pred_cs_fi = list(dag.predecessors(f_i))
                if not pred_cs_fi:
                    # f_i is graph input — must insert ms at f_i boundary
                    total += node_insert_cost[f_i]
                    plan_C.extend(_insert_plan_at(f_i, {C}))
                    continue
                c_i = pred_cs_fi[0]
                if c_i.layer_type in _ABSORBER_TYPES:
                    # absorber handles this arm for free
                    plan_C.append(('absorb_up', c_i))
                    continue

                ncu = nearest_convergence_up.get(f_i)
                if ncu is None:
                    # Individual singleton: no convergence with siblings
                    cost_i, plan_i = _min_cover_at(f_i, {C})
                    total += cost_i
                    plan_C.extend(plan_i)
                else:
                    gid = id(ncu)
                    if gid not in grouped:
                        grouped[gid] = {'G': ncu, 'f_list': [], 'comp_sum': 0, 'ci_set': set(), 'comp_plan': []}
                    grouped[gid]['f_list'].append(f_i)
                    if f_i is ncu:
                        # f_i IS the convergence point — scale arrives at f_i via C
                        # directly (identity arm of the join). Do not add comp_sum;
                        # C goes straight into ci_set.
                        grouped[gid]['ci_set'].add(C)
                    else:
                        # Each f_i contributes its own compensation cost (non-C succs)
                        grouped[gid]['comp_sum'] += _comp(f_i, {C})
                        grouped[gid]['comp_plan'].extend(_compensate_plan_at(f_i, {C}))

            # Resolve ci_set for each convergence group
            for gid, grp in grouped.items():
                G = grp['G']
                for f_i in grp['f_list']:
                    if f_i is G:
                        continue  # already added C to ci_set above
                    ci = _find_ci_toward_fi(G, f_i)
                    if ci is not None:
                        grp['ci_set'].add(ci)

            # Each group pays: sum(comp per fi) + min_cover_at(G, ci_set)
            for grp in grouped.values():
                G = grp['G']
                ci_set = grp['ci_set']
                cost_g, plan_g = _min_cover_at(G, ci_set)
                total += grp['comp_sum'] + cost_g
                plan_C.extend(grp['comp_plan'])
                plan_C.extend(plan_g)

            min_cover_above[C] = total
            plan_above[C] = plan_C

    # ------------------------------------------------------------------ #
    # Assemble feat_info
    # ------------------------------------------------------------------ #
    feat_info: dict = {}
    for node in dag.nodes:
        if not isinstance(node, FeatureNode):
            continue
        feat_info[node] = {
            'absorber_up': absorber_up[node],
            'bottleneck_succs': bottleneck_succs[node],
            'level_free_up': level_free_up[node],
            'nearest_convergence_up': nearest_convergence_up[node],
            'node_insert_cost': node_insert_cost[node],
        }
    return feat_info, min_cover_above, plan_above, node_insert_cost


def _sum_level(dag: nx.DiGraph) -> int:
    """Sum of backward FHE levels across all FeatureNodes (lower is better)."""
    level, _ = _backward_level_dict(dag)
    return sum(level.values())


def absorb_scale_new(graph: LayerAbstractGraph):
    """Propagate scale factors from avgpool/mult_coeff nodes into absorbable layers.

    Trigger conditions (aligned with absorb_scale / set_feature_scales):
      - mult_coeff   : always; scale = node.coeff
      - avgpool1d/2d : only when is_adaptive_avgpool or is_big_size (fixed-size
                       regular pooling keeps its scale handled elsewhere);
                       scale = 1 / prod(kernel_shape)

    For each trigger node both an UP probe and a DOWN probe are run on deep
    copies of the current dag.  The probe whose resulting dag has the smaller
    total level sum (_sum_level) is adopted.  Ties go to UP.
    """
    layers_to_absorb = ['avgpool1d', 'avgpool2d', 'mult_coeff']

    # Use a while loop (not for-loop over list) so that after graph.dag is
    # replaced by a probe copy, subsequent iterations always operate on the
    # current graph's node objects.
    processed_layer_ids: set[str] = set()
    while True:
        node = next(
            (
                n
                for n in graph.dag.nodes
                if isinstance(n, ComputeNode)
                and n.layer_type in layers_to_absorb
                and n.layer_id not in processed_layer_ids
            ),
            None,
        )
        if node is None:
            break
        processed_layer_ids.add(node.layer_id)

        if node.layer_type in ('avgpool1d', 'avgpool2d'):
            if not (node.is_adaptive_avgpool or node.is_big_size):
                continue
            scale = 1.0 / math.prod(node.kernel_shape)
        else:
            scale = node.coeff

        pre_f_node = list(graph.dag.predecessors(node))[0]
        out_f_node = next(graph.dag.successors(node))
        preds_of_pre_f = list(graph.dag.predecessors(pre_f_node))

        if not preds_of_pre_f:
            # pre_f_node is a graph input — no upstream to explore, go DOWN directly
            _propagate_scale(graph.dag, node, out_f_node, Direction.DOWN, scale)
            continue

        pre_c_node = preds_of_pre_f[0]

        # --- UP probe ---
        dag_up = copy.deepcopy(graph.dag)
        pre_f_up = next(n for n in dag_up.nodes if isinstance(n, FeatureNode) and n.node_id == pre_f_node.node_id)
        pre_c_up = next(n for n in dag_up.nodes if isinstance(n, ComputeNode) and n.layer_id == pre_c_node.layer_id)
        _, arm_level_up = _backward_level_dict(dag_up)
        _propagate_scale(dag_up, pre_f_up, pre_c_up, Direction.UP, scale, arm_level=arm_level_up)

        # --- DOWN probe ---
        dag_down = copy.deepcopy(graph.dag)
        out_f_down = next(n for n in dag_down.nodes if isinstance(n, FeatureNode) and n.node_id == out_f_node.node_id)
        node_down = next(n for n in dag_down.nodes if isinstance(n, ComputeNode) and n.layer_id == node.layer_id)
        _propagate_scale(dag_down, node_down, out_f_down, Direction.DOWN, scale)

        # Adopt the probe with lower total level cost; ties go to UP.
        graph.dag = dag_up if _sum_level(dag_up) <= _sum_level(dag_down) else dag_down

    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type == 'mult_scalar':
                print('layer=', node.layer_id, 'scale_up=', node.scale_up, node.scale_down)
    _remove_identity_mult_scalars(graph)


def _propagate_scale_pass2(
    dag: nx.DiGraph,
    start_feat: 'FeatureNode',
    scale: float,
    feat_info: dict,
    min_cover_above: dict,
    arm_level: dict,
    source_node: 'ComputeNode | None' = None,
):
    """Pass 2: propagate *scale* upward from *start_feat* using Pass-1 info.

    Decision at each FeatureNode F (arriving from downstream ComputeNode C_from):

      absorber_up[F] is True
        → continue upward; absorb into the first conv/fc/ms/polyact found.

      absorber_up[F] is False
        → compare arm cost of inserting ms here vs min_cover_above of pred:
            arm_cost = FREE(1) if C_from ∉ bottleneck_succs[F] else NONFREE
            above_cost = min_cover_above[pred_C_of_F]
          if arm_cost <= above_cost → insert ms on (F → C_from), stop this arm.
          else → continue upward.

    Convergence (nearest_convergence_up):
      When two sibling arms of a join (add/cat) share upstream FeatureNodes,
      both carry nearest_convergence_up = G (the nearest shared node).
      Processing is deferred until ALL convergent arms have arrived at G;
      only then is G processed once, avoiding duplicate ms insertion.

    Compensation:
      When passing through a multi-input ComputeNode (add/cat) upward,
      all other input arms receive a 1/scale DOWN compensation via
      _propagate_scale (identical to existing behaviour).
    """
    _FREE_COST = 1
    _NONFREE_COST = 10_000

    graph_wrapper = LayerAbstractGraph()
    graph_wrapper.dag = dag

    def _insert_ms_up(f_node, c_node, s):
        """Insert ms on (f_node → c_node); ms absorbs scale in UP direction."""
        ms = add_mult_scalar_between_feature_and_layer(graph_wrapper, f_node, c_node)
        ms.scale_down *= s
        _, new_arm = _backward_level_dict(dag)
        arm_level.clear()
        arm_level.update(new_arm)

    def _absorb_into_up(compute_node, s):
        compute_node.scale_down *= s

    convergence_pending: dict = {}
    convergence_expected: dict = {}
    convergence_c_froms: dict = {}
    visited_convergence: set = set()

    # work_stack items: (feat_node, c_from, scale)
    work_stack: list = [(start_feat, None, scale)]

    while work_stack:
        feat_node, c_from, sc = work_stack.pop()

        # ------------------------------------------------------------------ #
        # Convergence deferral
        # ------------------------------------------------------------------ #
        ncu = feat_info.get(feat_node, {}).get('nearest_convergence_up')
        if ncu is not None and id(ncu) not in visited_convergence:
            pending = convergence_pending.setdefault(id(ncu), [])
            pending.append((feat_node, c_from, sc))

            if id(ncu) not in convergence_expected:
                join_node = next(
                    (s for s in dag.successors(feat_node) if isinstance(s, ComputeNode) and dag.in_degree(s) > 1),
                    None,
                )
                if join_node is not None:
                    count = sum(
                        1
                        for p in dag.predecessors(join_node)
                        if isinstance(p, FeatureNode) and feat_info.get(p, {}).get('nearest_convergence_up') is ncu
                    )
                    convergence_expected[id(ncu)] = max(count, 1)
                else:
                    convergence_expected[id(ncu)] = 1

            if len(pending) < convergence_expected[id(ncu)]:
                continue  # wait for remaining sibling arms

            # All sibling arms arrived — compute convergent c_froms
            visited_convergence.add(id(ncu))
            c_froms_set: set = set()
            for arm_feat, _arm_c_from, _arm_sc in pending:
                if arm_feat is ncu:
                    # arm_feat IS the convergence point — scale arrived via the
                    # original c_from (the join itself), not via any pred path.
                    if _arm_c_from is not None:
                        c_froms_set.add(_arm_c_from)
                    continue
                for c in dag.predecessors(arm_feat):
                    if ncu in dag.predecessors(c):
                        c_froms_set.add(c)
                        break
            convergence_c_froms[id(ncu)] = c_froms_set
            work_stack.append((ncu, None, sc))
            continue

        # ------------------------------------------------------------------ #
        # Dead-end: graph input
        # ------------------------------------------------------------------ #
        preds_of_feat = list(dag.predecessors(feat_node))
        if not preds_of_feat:
            if c_from is not None:
                _insert_ms_up(feat_node, c_from, sc)
            continue

        # ------------------------------------------------------------------ #
        # Decision: insert ms here or continue upward
        # New formula: arm_cost(F, c_from) <= min_cover_above[pred_C_of_F]
        # ------------------------------------------------------------------ #
        absorber_up_f = feat_info.get(feat_node, {}).get('absorber_up', False)
        bsuccs = feat_info.get(feat_node, {}).get('bottleneck_succs', frozenset())

        arm_cost = _FREE_COST if (c_from is not None and c_from not in bsuccs) else _NONFREE_COST

        # Determine the predecessor ComputeNode(s) of feat_node (those feeding it from above)
        pred_c_of_feat = preds_of_feat[0]  # FeatureNode has exactly one predecessor ComputeNode
        above_cost = min_cover_above.get(pred_c_of_feat, _NONFREE_COST)

        insert_here = not absorber_up_f and c_from is not None and arm_cost <= above_cost

        if insert_here:
            _insert_ms_up(feat_node, c_from, sc)
            continue

        # Continue upward: compensate all non-excluded output arms
        if c_from is not None:
            excluded = {c_from}
        else:
            excluded = set(convergence_c_froms.get(id(feat_node), set()))
        # At start_feat the scale originated from source_node; don't compensate
        # through it (the caller turns the source into identity afterwards).
        if feat_node is start_feat and source_node is not None:
            excluded.add(source_node)
        for c_other in list(dag.successors(feat_node)):
            if c_other in excluded:
                continue
            _propagate_scale(
                dag,
                feat_node,
                c_other,
                Direction.DOWN,
                1.0 / sc,
                arm_level=arm_level,
            )

        # Continue upward through each predecessor ComputeNode
        for c_up in preds_of_feat:
            if c_up.layer_type in _ABSORBER_TYPES:
                _absorb_into_up(c_up, sc)
                continue
            for f_above in dag.predecessors(c_up):
                work_stack.append((f_above, c_up, sc))


def absorb_scale_new_dp(graph: LayerAbstractGraph):
    """Two-pass scale absorption using Pass-1 static analysis + Pass-2 DP.

    Pass 1 (graph_static_analysis): read-only topological scan.
    Pass 2 (_propagate_scale_pass2): for each scale source, uses Pass-1 info
      to pick the globally cheapest insertion point (min_ms_up), then directly
      modifies the dag (inserts mult_scalar or absorbs into existing layers).

    Trigger conditions identical to absorb_scale_new:
      - mult_coeff   : always
      - avgpool1d/2d : only when is_adaptive_avgpool or is_big_size
    """
    layers_to_absorb = ['avgpool1d', 'avgpool2d', 'mult_coeff']

    processed_layer_ids: set[str] = set()
    while True:
        feat_info, min_cover_above, plan_above, node_insert_cost = graph_static_analysis(graph.dag)
        _, arm_level = _backward_level_dict(graph.dag)

        node = next(
            (
                n
                for n in graph.dag.nodes
                if isinstance(n, ComputeNode)
                and n.layer_type in layers_to_absorb
                and n.layer_id not in processed_layer_ids
            ),
            None,
        )
        if node is None:
            break
        processed_layer_ids.add(node.layer_id)

        if node.layer_type in ('avgpool1d', 'avgpool2d'):
            if not (node.is_adaptive_avgpool or node.is_big_size):
                continue
            scale = 1.0 / math.prod(node.kernel_shape)
        else:
            scale = node.coeff

        pre_f_node = list(graph.dag.predecessors(node))[0]
        out_f_node = next(graph.dag.successors(node))
        preds_of_pre_f = list(graph.dag.predecessors(pre_f_node))

        if not preds_of_pre_f:
            # Graph-input: no upstream, propagate DOWN
            _propagate_scale(graph.dag, node, out_f_node, Direction.DOWN, scale, arm_level=arm_level)
            continue

        # Build the source plan (Pass-1 driven): at pre_f_node we go UP unconditionally,
        # excluding the source from compensation. Then append plan_above of the pred
        # ComputeNode of pre_f_node. mc additionally clears its coeff after execution.
        src_plan = []
        for c in graph.dag.successors(pre_f_node):
            if c is node:
                continue
            if c.layer_type in _DOWN_ABSORBER_TYPES:
                src_plan.append(('absorb_down', c))
            else:
                src_plan.append(('compensate_down', pre_f_node, c))
        src_plan.extend(plan_above.get(preds_of_pre_f[0], []))
        if node.layer_type == 'mult_coeff':
            src_plan.append(('clear_source_mc', node))

        _NONFREE = 10_000

        def _fmt_cost(v):
            return 'NONFREE' if v >= _NONFREE else str(v)

        print(f'\n=== Pass-1 metrics for source {node.layer_id} (scale={scale}) ===')
        print('  node_insert_cost (per FeatureNode, topological):')
        for n in nx.topological_sort(graph.dag):
            if isinstance(n, FeatureNode):
                ident = getattr(n, 'node_id', repr(n))
                print(f'    {ident:55s}  {_fmt_cost(node_insert_cost[n])}')
        print('  min_cover_above (per ComputeNode, topological):')
        for n in nx.topological_sort(graph.dag):
            if isinstance(n, ComputeNode):
                ident = getattr(n, 'layer_id', repr(n))
                v = min_cover_above.get(n, None)
                print(f'    {ident:55s}  {("-" if v is None else _fmt_cost(v))}')

        print(f'\n=== Pass-1 plan for source {node.layer_id} (scale={scale}) ===')
        for i, action in enumerate(src_plan, 1):
            kind = action[0]
            tail = ', '.join(getattr(a, 'layer_id', getattr(a, 'node_id', repr(a))) for a in action[1:])
            print(f'  {i:2d}. {kind:18s}  {tail}')
        print()

        # For mult_coeff the source layer is destructively absorbed: its scale
        # is lifted upstream and the source becomes an identity mc afterwards.
        # Tell Pass 2 to skip DOWN compensation through the source so it does
        # not insert a redundant ms behind it (avgpool keeps its averaging and
        # still needs the DOWN compensation, so it doesn't pass source_node).
        if node.layer_type == 'mult_coeff':
            _propagate_scale_pass2(
                graph.dag, pre_f_node, scale, feat_info, min_cover_above, arm_level, source_node=node
            )
            node.coeff = 1.0
        else:
            _propagate_scale_pass2(graph.dag, pre_f_node, scale, feat_info, min_cover_above, arm_level)

    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode) and node.layer_type == 'mult_scalar':
            print('layer=', node.layer_id, 'scale_up=', node.scale_up, node.scale_down)
    _remove_identity_mult_scalars(graph)


def _remove_identity_mult_scalars(graph: LayerAbstractGraph):
    """Remove mult_scalar nodes whose scale == 1.0 (identity, no-op).

    For each such node the topology is:
        feat_in → mult_scalar → feat_out → next_compute
    After removal:
        feat_in → next_compute  (with the original edge attributes from feat_out→next_compute)
    """
    for node in list(graph.dag.nodes):
        if not isinstance(node, ComputeNode):
            continue
        if node.layer_type != 'mult_scalar':
            continue
        if node.scale_down * node.scale_up != 1.0 or node.poly_path:
            continue
        if node not in graph.dag.nodes:
            continue

        feat_in = list(graph.dag.predecessors(node))[0]
        feat_out = list(graph.dag.successors(node))[0]
        next_nodes = list(graph.dag.successors(feat_out))

        for next_node in next_nodes:
            edge_attrs = dict(graph.dag.edges[feat_out, next_node])
            graph.dag.add_edge(feat_in, next_node, **edge_attrs)

        graph.dag.remove_node(node)
        graph.dag.remove_node(feat_out)


def miniprocess(graph: LayerAbstractGraph, p: ComputeNode, res_list: list, polyact_id, approve_len: bool = False):
    value_list: list[FeatureNode] = list(graph.dag.predecessors(p))
    for value in value_list:
        # c_list = list(graph.dag.predecessors(value))
        if graph.dag.out_degree(value) > 1 or graph.dag.in_degree(value) == 0:
            mult_scalar_node = add_mult_scalar_between_feature_and_layer(graph, value, p)
            # res_list.append((polyact_id, mult_scalar_node))
            mult_scalar_node.poly_path = polyact_id
            continue
        else:
            c_value: ComputeNode = list(graph.dag.predecessors(value))[0]

        if 'conv' in c_value.layer_type or 'fc' in c_value.layer_type:
            # Direct conv/fc predecessor — record it
            # res_list.append((polyact_id, c_value))
            c_value.poly_path = polyact_id
        elif c_value.layer_type == 'polyact' or (not approve_len and graph.dag.in_degree(c_value) > 1):
            mult_scalar_node = add_mult_scalar_between_feature_and_layer(graph, value, p)
            # res_list.append((polyact_id, mult_scalar_node))
            mult_scalar_node.poly_path = polyact_id
        else:
            # Single non-conv/fc predecessor — recurse
            miniprocess(graph, c_value, res_list, polyact_id)


def _absorb_at_loop(dag: nx.DiGraph, cur_node, next_node, scale: float, direction: Direction, arm_level: dict = None):
    """Called when next_node is already on the active set (loop detected).
    Inserts a mult_scalar between cur_node and next_node to absorb the scale in place.
    """
    if isinstance(next_node, ComputeNode):
        feature_node, compute_node = cur_node, next_node
    else:
        feature_node, compute_node = next_node, cur_node
    graph_wrapper = LayerAbstractGraph()
    graph_wrapper.dag = dag
    mult_scalar = add_mult_scalar_between_feature_and_layer(graph_wrapper, feature_node, compute_node)
    if direction == Direction.UP:
        mult_scalar.scale_down *= scale
    else:
        mult_scalar.scale_up *= scale
    if arm_level is not None:
        _, new_arm = _backward_level_dict(dag)
        arm_level.clear()
        arm_level.update(new_arm)


def _can_absorb_up(dag: nx.DiGraph, node, visited: set = None) -> bool:
    """Read-only probe: from *node* upward, can every path find an absorber?

    An absorber is a ComputeNode whose layer_type is one of conv2d / fc0 / fc1 /
    mult_scalar / polyact.  Graph-input FeatureNodes (no predecessors) are
    dead-ends and cause the function to return False.

    Cycles are handled conservatively: a revisited node is assumed absorbable
    (True) so that the loop-detection path in _propagate_scale handles it.
    """
    visited = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if id(cur) in visited:
            continue  # cycle — assume absorbable
        visited.add(id(cur))
        if isinstance(cur, ComputeNode):
            if cur.layer_type in {'conv2d', 'fc0', 'fc1', 'mult_scalar', 'polyact'}:
                continue  # absorber found on this path
            preds = list(dag.predecessors(cur))
            stack.extend(preds)
        else:  # FeatureNode
            preds = list(dag.predecessors(cur))
            if not preds:
                return False  # graph input dead-end — no absorber on this path
            stack.extend(preds)
    return True


def _can_absorb_down(dag: nx.DiGraph, node, visited: set = None) -> bool:
    """Read-only probe: from *node* downward, can every path find an absorber?

    An absorber is a ComputeNode whose layer_type is one of conv2d / fc0 / fc1 /
    mult_scalar / polyact.  Graph-output FeatureNodes (no successors) are
    dead-ends and cause the function to return False.

    Cycles are handled conservatively: a revisited node is assumed absorbable
    (True) so that the loop-detection path in _propagate_scale handles it.
    """
    visited = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if id(cur) in visited:
            continue  # cycle — assume absorbable
        visited.add(id(cur))
        if isinstance(cur, ComputeNode):
            if cur.layer_type in {'conv2d', 'fc0', 'fc1', 'mult_scalar'}:
                continue  # absorber found on this path
            succs = list(dag.successors(cur))
            stack.extend(succs)
        else:  # FeatureNode
            succs = list(dag.successors(cur))
            if not succs:
                return False  # graph output dead-end — no absorber on this path
            stack.extend(succs)
    return True


def _propagate_scale_at_multi_input_no_absorber(dag, old_node, node, scale, arm_level):
    """Handle scale propagation when arriving DOWN at a multi-input ComputeNode
    (e.g. add) whose downstream path has no absorber.

    Strategy:
      1. Trial-insert ms on old_node → node (the arm we came from).
      2. Measure change in sum(level):
         - sum+1: ms is "free" (old_node arm was not the bottleneck) — keep it.
         - sum+more: ms raises add's level and propagates downstream — undo
    """
    graph_wrapper = LayerAbstractGraph()
    graph_wrapper.dag = dag
    flag = False
    sum_before = _sum_level(dag)

    # Trial: insert ms on old_node → node
    ms_trial = add_mult_scalar_between_feature_and_layer(graph_wrapper, old_node, node)
    ms_trial_out = next(dag.successors(ms_trial))

    sum_after = _sum_level(dag)

    if sum_after - sum_before <= 1:
        # old_node arm was not the bottleneck — ms here is free, keep it
        ms_trial.scale_up *= scale
        flag = True
    else:
        # ms raises add's level — undo trial and insert ms after add's output instead
        original_edge_data = dict(dag.edges[ms_trial_out, node])
        dag.remove_node(ms_trial_out)
        dag.remove_node(ms_trial)
        dag.add_edge(old_node, node, **original_edge_data)

        # ms_final = add_mult_scalar_behind_node(graph_wrapper, node)
        # ms_final.scale_up *= scale

    if arm_level is not None:
        _, new_arm = _backward_level_dict(dag)
        arm_level.clear()
        arm_level.update(new_arm)
    return flag


def _propagate_scale(
    dag: nx.DiGraph,
    old_node,
    node,
    direction: Direction,
    scale: float,
    arm_level: dict = None,
):
    """Iteratively propagate a scale factor through the DAG.

    Args:
        dag: the computation graph
        old_node: the node we came from (used to exclude it when fanning out)
        node: the current node being visited
        direction: Direction.DOWN means we arrived from above, Direction.UP from below
        scale: the scale factor to propagate
        arm_level: (FeatureNode, ComputeNode) → int level contribution table;
                   updated in-place whenever the graph is modified.

    Loop detection uses an explicit active set instead of the call stack.
    Each work item is (old_node, node, direction, scale).  A sentinel tuple
    (None, node, None, None) is pushed immediately after a node is activated
    so that active.discard(id(node)) fires when all work spawned by that node
    has been processed (LIFO order mirrors the original recursion).
    """
    _SENTINEL = None  # direction=None      → pop event: active.discard(node)
    _DEFERRED_UP = object()  # direction=_DEFERRED_UP   → deferred pred lookup (ComputeNode UP)
    _DEFERRED_DOWN = object()  # direction=_DEFERRED_DOWN → deferred succ lookup (FeatureNode DOWN)

    active: set = set()
    # work_stack entry types:
    #   (old_node,    node,    direction,      scale)  — normal work item
    #   (None,        node,    None,           None)   — sentinel: deactivate node
    #   (c_node,      arm_idx, _DEFERRED_UP,   scale)  — deferred: re-query pred via input_index
    #   (f_node,      arm_idx, _DEFERRED_DOWN, scale)  — deferred: re-query succ via output_index
    work_stack: list = [(old_node, node, direction, scale)]

    while work_stack:
        item = work_stack.pop()
        old_node, node, direction, scale = item

        # --- sentinel: deactivate node ---
        if direction is None:
            active.discard(id(node))
            continue

        # --- deferred pred lookup (ComputeNode UP, multi-arm) ---
        # old_node = compute_node, node = arm_idx (int), scale = scale value
        if direction is _DEFERRED_UP:
            compute_node = old_node
            arm_idx = node
            current_pred = next(
                (p for p in dag.predecessors(compute_node) if dag.edges[p, compute_node].get('input_index') == arm_idx),
                None,
            )
            if current_pred is not None:
                work_stack.append((compute_node, current_pred, Direction.UP, scale))
            continue

        # --- deferred succ lookup (FeatureNode DOWN, multi-arm) ---
        # old_node = feat_node, node = arm_idx (int), scale = scale value
        if direction is _DEFERRED_DOWN:
            feat_node = old_node
            arm_idx = node
            current_succ = next(
                (s for s in dag.successors(feat_node) if dag.edges[feat_node, s].get('output_index') == arm_idx),
                None,
            )
            if current_succ is not None:
                work_stack.append((feat_node, current_succ, Direction.DOWN, scale))
            continue

        # --- loop detection (replaces _call_or_absorb guard) ---
        if id(node) in active:
            _absorb_at_loop(dag, old_node, node, scale, direction, arm_level)
            continue

        active.add(id(node))
        # Sentinel is pushed below, at the point where all child work items
        # for this node have been determined.  For most paths the sentinel is
        # pushed immediately (so it pops after all children).  For the
        # FeatureNode-UP path it is pushed *before* the sibling compensation
        # items so that active.discard fires only after compensation is done.

        # ------------------------------------------------------------------ #
        # FeatureNode
        # ------------------------------------------------------------------ #
        if isinstance(node, FeatureNode):
            if direction == Direction.DOWN:
                succs = list(dag.successors(node))
                if not succs:
                    # Graph output dead-end
                    if isinstance(old_node, ComputeNode):
                        graph_wrapper = LayerAbstractGraph()
                        graph_wrapper.dag = dag
                        ms_node = add_mult_scalar_behind_node(graph_wrapper, old_node)
                        ms_node.scale_up *= scale
                    work_stack.append((_SENTINEL, node, None, None))
                    continue

                if len(succs) > 1:
                    # Multi-arm fan-out identified by output_index.
                    # Push in reverse order so lowest index is processed first.
                    arm_indices = sorted({dag.edges[node, s]['output_index'] for s in succs}, reverse=True)
                    seen_arms: set = set()
                    work_stack.append((_SENTINEL, node, None, None))
                    for idx in arm_indices:
                        if idx in seen_arms:
                            continue
                        seen_arms.add(idx)
                        # Deferred: re-query succ at pop time so arm N+1 sees
                        # graph mutations made by arm N (e.g. ms inserted on edge).
                        work_stack.append((node, idx, _DEFERRED_DOWN, scale))
                else:
                    work_stack.append((_SENTINEL, node, None, None))
                    work_stack.append((node, succs[0], Direction.DOWN, scale))

            else:  # Direction.UP — f_node，向上方向
                preds = list(dag.predecessors(node))
                if not preds:
                    # Graph input dead-end
                    if isinstance(old_node, ComputeNode):
                        graph_wrapper = LayerAbstractGraph()
                        graph_wrapper.dag = dag
                        mult_scalar = add_mult_scalar_between_feature_and_layer(graph_wrapper, node, old_node)
                        mult_scalar.scale_down *= scale
                        if arm_level is not None:
                            _, new_arm = _backward_level_dict(dag)
                            arm_level.clear()
                            arm_level.update(new_arm)
                    work_stack.append((_SENTINEL, node, None, None))
                    continue

                # Fork detection: if this FeatureNode fans out to multiple
                # successors (out_degree > 1), decide whether to insert a
                # mult_scalar on the current edge (node → old_node) or
                # continue propagating upward.
                #
                # Decision rule (arm-level lookup):
                #   current   = arm_level[(node, old_node)]
                #   other_max = max arm_level[(node, c)] for all other arms
                #
                # If current < other_max: old_node arm is not the bottleneck
                # → insert ms here (free), stop.
                # Otherwise: continue upward; real absorber is better.
                if (
                    dag.out_degree(node) > 1
                    and isinstance(old_node, ComputeNode)
                    and arm_level is not None
                    and not _can_absorb_up(dag, node)
                ):
                    current = arm_level.get((node, old_node), 0)
                    other_max = max(
                        (arm_level.get((node, c), 0) for c in dag.successors(node) if c is not old_node),
                        default=0,
                    )
                    if current < other_max:
                        graph_wrapper = LayerAbstractGraph()
                        graph_wrapper.dag = dag
                        mult_scalar = add_mult_scalar_between_feature_and_layer(graph_wrapper, node, old_node)
                        mult_scalar.scale_up *= scale
                        _, new_arm = _backward_level_dict(dag)
                        arm_level.clear()
                        arm_level.update(new_arm)
                        work_stack.append((_SENTINEL, node, None, None))
                        continue

                succs = list(dag.successors(node))
                # For FeatureNode UP: sentinel must be pushed BEFORE the
                # sibling compensation items so that active.discard(node)
                # fires only after ALL compensation work is done (LIFO order:
                # sentinel pops last, UP item pops first).
                work_stack.append((_SENTINEL, node, None, None))
                # Push sibling compensation items (processed after UP subtree)
                for node_in in succs:
                    if node_in is old_node:
                        continue
                    work_stack.append((node, node_in, Direction.DOWN, 1.0 / scale))
                # Push main upward item last (processed next, i.e. first)
                work_stack.append((node, preds[0], Direction.UP, scale))

        # ------------------------------------------------------------------ #
        # ComputeNode
        # ------------------------------------------------------------------ #
        else:
            # conv2d/fc/mult_scalar: absorbable in any direction
            if node.layer_type in {'conv2d', 'fc0', 'fc1', 'mult_scalar'}:
                if direction == Direction.UP:
                    node.scale_down = node.scale_down * scale
                else:
                    node.scale_up = node.scale_up * scale
                work_stack.append((_SENTINEL, node, None, None))
                continue

            # polyact: direction-dependent
            #   UP   → absorb into polynomial coefficients
            #   DOWN → f(s·x) ≠ s·f(x), treat as barrier
            if node.layer_type == 'polyact':
                if direction == Direction.UP:
                    node.scale_down = node.scale_down * scale
                else:
                    if isinstance(old_node, FeatureNode):
                        graph_wrapper = LayerAbstractGraph()
                        graph_wrapper.dag = dag
                        mult_scalar = add_mult_scalar_between_feature_and_layer(graph_wrapper, old_node, node)
                        mult_scalar.scale_up *= scale
                        if arm_level is not None:
                            _, new_arm = _backward_level_dict(dag)
                            arm_level.clear()
                            arm_level.update(new_arm)
                work_stack.append((_SENTINEL, node, None, None))
                continue

            if direction == Direction.DOWN:
                # Arrived from above — propagate downward (single successor) and
                # send 1/scale compensation to all sibling predecessors
                preds = list(dag.predecessors(node))
                succs = list(dag.successors(node))
                if len(preds) > 1 and not _can_absorb_down(dag, node):
                    Flag = _propagate_scale_at_multi_input_no_absorber(dag, old_node, node, scale, arm_level)
                    print('flag=', Flag)
                    if Flag:
                        work_stack.append((_SENTINEL, node, None, None))
                        continue
                work_stack.append((_SENTINEL, node, None, None))
                # Push sibling compensation items first (processed later)
                for node_in in preds:
                    if node_in is old_node:
                        continue
                    work_stack.append((node, node_in, Direction.UP, 1.0 / scale))
                # Push main downward item last (processed next)
                work_stack.append((node, succs[0], Direction.DOWN, scale))

            else:  # Direction.UP — compute，向上传
                # Arrived from below — fan upward to all predecessors.
                # Arms must be explored strictly in order (like recursion): arm N+1
                # starts only after arm N's full subtree is done, so that arm N+1's
                # re-query of the current predecessor reflects graph mutations from arm N.
                # Use _DEFERRED items: each item re-queries its predecessor at pop time.
                preds = list(dag.predecessors(node))
                work_stack.append((_SENTINEL, node, None, None))
                if len(preds) > 1:
                    arm_indices = sorted(
                        {dag.edges[p, node].get('input_index') for p in preds},
                        reverse=True,  # reverse so lowest index pops first (LIFO)
                    )
                    seen_arms: set = set()
                    for idx in arm_indices:
                        if idx in seen_arms:
                            continue
                        seen_arms.add(idx)
                        # Push a deferred item: predecessor re-queried at pop time,
                        # after all earlier arms have finished modifying the graph.
                        work_stack.append((node, idx, _DEFERRED_UP, scale))
                else:
                    for node_in in preds:
                        work_stack.append((node, node_in, Direction.UP, scale))


def propagate_scale_from_node(
    dag: nx.DiGraph,
    start_node,
    scale: float,
    direction: Direction = Direction.DOWN,
):
    """Propagate a scale factor starting from *start_node*."""
    _propagate_scale(dag, None, start_node, direction, scale)


def process_polyact(graph: LayerAbstractGraph) -> list:
    """
    Traverse the graph in reverse topological order and call miniprocess for
    every polyact node.

    Returns:
        res_list: list of (polyact_id, target_node) pairs collected by miniprocess
    """
    res_list = []
    all_nodes_reversed = list(reversed(list(nx.topological_sort(graph.dag))))
    for p in all_nodes_reversed:
        if isinstance(p, ComputeNode) and p.layer_type == 'polyact':
            miniprocess(graph, p, res_list, p.layer_id, True)
    return res_list
