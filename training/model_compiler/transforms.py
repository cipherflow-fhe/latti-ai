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
from inference.model_generator.layers.poly_relu_base import PolyReluBase


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

    old_edge_attrs = dag.edges[old_feature, old_compute]
    dag.remove_edge(old_feature, old_compute)
    dag.add_edge(old_feature, new_compute)
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
    feature_node_out.data_type = feature_node_in.data_type

    if feature_node_in.head_shape is not None:
        feature_node_out.head_shape = list(feature_node_in.head_shape)

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


def expand_parcpmm_add_pt(graph: LayerAbstractGraph):
    used_ids = set()
    for node in graph.dag.nodes:
        if hasattr(node, 'node_id'):
            used_ids.add(node.node_id)
        if hasattr(node, 'layer_id'):
            used_ids.add(node.layer_id)

    def make_unique_id(base_id: str) -> str:
        if base_id not in used_ids:
            used_ids.add(base_id)
            return base_id
        idx = 1
        while f'{base_id}_{idx}' in used_ids:
            idx += 1
        unique_id = f'{base_id}_{idx}'
        used_ids.add(unique_id)
        return unique_id

    for parcpmm_node in list(graph.dag.nodes):
        if not isinstance(parcpmm_node, ComputeNode):
            continue
        if parcpmm_node.layer_type != 'parcpmm' or not getattr(parcpmm_node, 'to_expand', False):
            continue

        bias_path = getattr(parcpmm_node, 'bias_path', '')
        if not bias_path:
            raise ValueError(f'parcpmm layer {parcpmm_node.layer_id} has to_expand=True but no bias_path')

        old_feature_list = list(graph.dag.successors(parcpmm_node))
        if len(old_feature_list) != 1:
            raise ValueError(
                f'Expected exactly one output feature for parcpmm layer {parcpmm_node.layer_id}, '
                f'got {len(old_feature_list)}'
            )
        old_feature = old_feature_list[0]

        new_feature = FeatureNode(
            key=make_unique_id(f'{parcpmm_node.layer_id}_add_pt_input'),
            dim=old_feature.dim,
            channel=old_feature.channel,
            scale=old_feature.scale,
            ckks_parameter_id=old_feature.ckks_parameter_id,
            ckks_scale=old_feature.ckks_scale,
            shape=list(old_feature.shape),
        )
        new_feature.invalid_fill = list(old_feature.invalid_fill)
        new_feature.sp_info = copy.deepcopy(old_feature.sp_info)
        new_feature.has_sp_info = old_feature.has_sp_info
        new_feature.data_type = old_feature.data_type
        if old_feature.head_shape is not None:
            new_feature.head_shape = list(old_feature.head_shape)

        add_pt_id = make_unique_id(f'{parcpmm_node.layer_id}_add_pt')
        add_pt_node = ComputeNode(add_pt_id, 'pcm_add_pt', parcpmm_node.channel_output, parcpmm_node.channel_output)
        add_pt_node.path = bias_path

        new_feature_args = copy.deepcopy(graph.dag.nodes[old_feature])
        new_feature_args['name'] = new_feature.node_id
        _insert_layer_after_compute(
            graph.dag,
            parcpmm_node,
            new_feature,
            add_pt_node,
            new_feature_args=new_feature_args,
            new_compute_args={'name': add_pt_id, 'level_cost': 0},
        )
        parcpmm_node.bias_path = ''
        parcpmm_node.to_expand = False


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

    # Initialize head_shape for feature_mat input nodes.
    n_heads = getattr(config, 'n_heads', 1)
    head_dim = getattr(config, 'head_dim', 0)
    for node in sorted_nodes:
        if isinstance(node, FeatureNode) and graph.dag.in_degree(node) == 0:
            if node.data_type == 'feature_mat' and node.head_shape is None and n_heads > 1:
                if head_dim <= 0:
                    raise ValueError('HEAD_DIM must be set when N_HEADS > 1 for feature_mat inputs')
                node.head_shape = [node.shape[0], head_dim]

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
                if compute_node.layer_type == 'parcpmm':
                    succ.shape[0] = preds[0].shape[0]
                    weight_shape = getattr(compute_node, 'weight_shape', [])
                    if len(weight_shape) >= 2:
                        succ.shape[1] = weight_shape[-1]
                    if preds[0].head_shape is not None:
                        succ.head_shape = list(preds[0].head_shape)
                elif compute_node.layer_type == 'partranspose':
                    if preds[0].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[1], preds[0].head_shape[0]]
                        succ.shape[0] = succ.head_shape[0]
                        succ.shape[1] = succ.head_shape[1] * n_heads
                    else:
                        succ.shape[0] = preds[0].shape[1]
                        succ.shape[1] = preds[0].shape[0]
                elif compute_node.layer_type == 'parccmm':
                    succ.shape[0] = preds[0].shape[0]
                    if preds[0].head_shape is not None and len(preds) > 1 and preds[1].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[0], preds[1].head_shape[1]]
                        succ.shape[1] = succ.head_shape[1] * n_heads
                    elif len(preds) > 1 and len(preds[1].shape) > 1:
                        succ.shape[1] = preds[1].shape[1]
                else:
                    for i in range(preds[0].dim):
                        succ.shape[i] = preds[0].shape[i]
                        graph.dag.nodes[succ]['skip'][i] = graph.dag.nodes[preds[0]]['skip'][i]
                # Propagate head_shape for feature_mat pass-through layers
                if succ.head_shape is None and preds[0].head_shape is not None:
                    succ.head_shape = list(preds[0].head_shape)
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
            graph.dag.nodes[compute_node]['level_cost'] = PolyReluBase.compute_bsgs_level_cost(compute_node.order)
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
        elif compute_node.layer_type == 'parcpmm':
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in {'add_pt', 'pcm_add_pt'}:
            graph.dag.nodes[compute_node]['level_cost'] = 0
        elif compute_node.layer_type == 'partranspose':
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type == 'parccmm':
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type == 'pcmgamma':
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type == 'pcmpoly':
            graph.dag.nodes[compute_node]['level_cost'] = 2 if compute_node.order == 2 else 3
        elif compute_node.layer_type == 'pcmstats':
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type == 'pcmcenter':
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type == 'pcminit':
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type == 'pcmgs':
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type == 'pcmaffine':
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type == 'pcmmul':
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


def expand_multi_head_attention(graph: LayerAbstractGraph):
    for vit_node in list(graph.dag.nodes):
        if not (isinstance(vit_node, ComputeNode) and vit_node.layer_type == 'CustomMultiHeadAttention'):
            continue

        preds = list(graph.dag.predecessors(vit_node))
        succs = list(graph.dag.successors(vit_node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(
                f'ViT node {vit_node.layer_id} must have exactly 1 input and 1 output, '
                f'got {len(preds)} inputs and {len(succs)} outputs'
            )
        x_in: FeatureNode = preds[0]
        out: FeatureNode = succs[0]

        base_id = vit_node.layer_id
        x_in_attrs = graph.dag.nodes[x_in]
        skip = list(x_in_attrs.get('skip', [1] * x_in.dim))
        level = x_in_attrs.get('level', 0)
        pack_num = x_in_attrs.get('pack_num', 1)
        m = x_in.shape[0]
        n = x_in.shape[1]
        n_heads = max(1, config.n_heads)

        def make_feature(name: str, shape: list[int] | None = None) -> FeatureNode:
            f = FeatureNode(
                key=name,
                dim=x_in.dim,
                channel=x_in.channel,
                scale=x_in.scale,
                ckks_parameter_id=x_in.ckks_parameter_id,
                ckks_scale=x_in.ckks_scale,
                shape=list(shape if shape is not None else x_in.shape),
            )
            f.data_type = x_in.data_type
            return f

        def f_attrs(f: FeatureNode) -> dict:
            return {'name': f.node_id, 'skip': list(skip), 'level': level, 'pack_num': pack_num}

        def c_attrs(level_cost: int, name: str) -> dict:
            return {'name': name, 'level_cost': level_cost}

        q = make_feature(f'{base_id}_q')
        k = make_feature(f'{base_id}_k')
        v = make_feature(f'{base_id}_v')
        kt = make_feature(f'{base_id}_kt', [n // n_heads, m * n_heads])
        qkt = make_feature(f'{base_id}_qkt', [m, m * n_heads])
        qkt_poly = make_feature(f'{base_id}_qkt_polyact', [m, m * n_heads])
        qktv = make_feature(f'{base_id}_qktv', list(out.shape))

        q_node = ComputeNode(f'{base_id}_q_layer', 'parcpmm', 1, 1)
        q_node.path = getattr(vit_node, 'q_weight_path', f'{base_id}.q.weight')
        q_node.bias_path = getattr(vit_node, 'q_bias_path', '')
        q_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        k_node = ComputeNode(f'{base_id}_k_layer', 'parcpmm', 1, 1)
        k_node.path = getattr(vit_node, 'k_weight_path', f'{base_id}.k.weight')
        k_node.bias_path = getattr(vit_node, 'k_bias_path', '')
        k_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        v_node = ComputeNode(f'{base_id}_v_layer', 'parcpmm', 1, 1)
        v_node.path = getattr(vit_node, 'v_weight_path', f'{base_id}.v.weight')
        v_node.bias_path = getattr(vit_node, 'v_bias_path', '')
        v_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        kt_node = ComputeNode(f'{base_id}_kt_layer', 'partranspose', 1, 1)
        qkt_node = ComputeNode(f'{base_id}_qkt_layer', 'parccmm', 1, 1)
        poly_node = ComputeNode(f'{base_id}_poly', 'pcmpoly', 1, 1)
        poly_node.path = getattr(vit_node, 'poly_weight_path', f'{base_id}.poly.weight')
        poly_node.coeffs_path = poly_node.path
        poly_node.gamma_path = getattr(vit_node, 'gamma_path', f'{base_id}.gamma')
        poly_node.order = getattr(vit_node, 'poly_order', 4)
        qktv_node = ComputeNode(f'{base_id}_qktv_layer', 'parccmm', 1, 1)
        out_node = ComputeNode(f'{base_id}_out', 'parcpmm', 1, 1)
        out_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        out_node.path = getattr(vit_node, 'proj_weight_path', f'{base_id}.proj.weight')
        out_node.bias_path = getattr(vit_node, 'proj_bias_path', '')

        graph.dag.remove_node(vit_node)

        for node in (q_node, k_node, v_node):
            graph.dag.add_node(node, **c_attrs(2, node.layer_id))
        graph.dag.add_node(kt_node, **c_attrs(1, kt_node.layer_id))
        graph.dag.add_node(qkt_node, **c_attrs(3, qkt_node.layer_id))
        graph.dag.add_node(poly_node, **c_attrs(3, poly_node.layer_id))
        graph.dag.add_node(qktv_node, **c_attrs(3, qktv_node.layer_id))
        graph.dag.add_node(out_node, **c_attrs(2, out_node.layer_id))

        for f in (q, k, v, kt, qkt, qkt_poly, qktv):
            graph.dag.add_node(f, **f_attrs(f))

        graph.dag.add_edge(x_in, q_node)
        graph.dag.add_edge(q_node, q)
        graph.dag.add_edge(x_in, k_node)
        graph.dag.add_edge(k_node, k)
        graph.dag.add_edge(x_in, v_node)
        graph.dag.add_edge(v_node, v)

        graph.dag.add_edge(k, kt_node)
        graph.dag.add_edge(kt_node, kt)

        graph.dag.add_edge(q, qkt_node, input_index=0)
        graph.dag.add_edge(kt, qkt_node, input_index=1)
        graph.dag.add_edge(qkt_node, qkt)

        graph.dag.add_edge(qkt, poly_node)
        graph.dag.add_edge(poly_node, qkt_poly)

        graph.dag.add_edge(qkt_poly, qktv_node, input_index=0)
        graph.dag.add_edge(v, qktv_node, input_index=1)
        graph.dag.add_edge(qktv_node, qktv)

        graph.dag.add_edge(qktv, out_node)
        graph.dag.add_edge(out_node, out)


def expand_poly_act_rn(graph: LayerAbstractGraph):
    for node in list(graph.dag.nodes):
        if not (isinstance(node, ComputeNode) and node.layer_type == 'PolyActRN'):
            continue

        preds = list(graph.dag.predecessors(node))
        succs = list(graph.dag.successors(node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(
                f'PolyActRN node {node.layer_id} must have exactly 1 input and 1 output, '
                f'got {len(preds)} inputs and {len(succs)} outputs'
            )
        x_in: FeatureNode = preds[0]
        out: FeatureNode = succs[0]

        base_id = node.layer_id
        running_max_path = getattr(node, 'running_max_path', '') or node.path
        gamma_path = getattr(node, 'gamma_path', '')
        coeffs_path = getattr(node, 'coeffs_path', '')
        rn_suffix = '.rangenorm.running_max'
        if running_max_path.endswith(rn_suffix):
            prefix = running_max_path[: -len(rn_suffix)]
            gamma_path = gamma_path or f'{prefix}.gamma'
            coeffs_path = coeffs_path or f'{prefix}.weight'
        order = node.order

        input_edge_attrs = copy.deepcopy(graph.dag.edges[x_in, node])
        output_edge_attrs = copy.deepcopy(graph.dag.edges[node, out])

        poly_node = ComputeNode(base_id, 'pcmpoly', node.channel_input, node.channel_output)
        poly_node.depth = node.depth
        poly_node.path = coeffs_path or running_max_path
        poly_node.running_max_path = running_max_path
        poly_node.gamma_path = gamma_path
        poly_node.coeffs_path = coeffs_path
        poly_node.order = order

        graph.dag.remove_node(node)

        graph.dag.add_node(poly_node, name=poly_node.layer_id)
        graph.dag.add_edge(x_in, poly_node, **input_edge_attrs)
        graph.dag.add_edge(poly_node, out, **output_edge_attrs)


def expand_layer_norm(graph: LayerAbstractGraph, n_iter: int = 2):
    for ln_node in list(graph.dag.nodes):
        if not isinstance(ln_node, LayerNormComputeNode):
            continue

        preds = list(graph.dag.predecessors(ln_node))
        succs = list(graph.dag.successors(ln_node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(
                f'LayerNorm node {ln_node.layer_id} must have exactly 1 input and 1 output, '
                f'got {len(preds)} inputs and {len(succs)} outputs'
            )
        x_in: FeatureNode = preds[0]
        out: FeatureNode = succs[0]

        base_id = ln_node.layer_id
        epsilon = ln_node.epsilon
        weight_path = ln_node.weight_path
        bias_path = ln_node.bias_path

        x_in_attrs = graph.dag.nodes[x_in]
        skip = list(x_in_attrs.get('skip', [1] * x_in.dim))
        level = x_in_attrs.get('level', 0)
        pack_num = x_in_attrs.get('pack_num', 1)

        def make_feature(name: str) -> FeatureNode:
            f = FeatureNode(
                key=name,
                dim=x_in.dim,
                channel=x_in.channel,
                scale=x_in.scale,
                ckks_parameter_id=x_in.ckks_parameter_id,
                ckks_scale=x_in.ckks_scale,
                shape=list(x_in.shape),
            )
            f.data_type = x_in.data_type
            return f

        def f_attrs(f: FeatureNode) -> dict:
            return {'name': f.node_id, 'skip': list(skip), 'level': level, 'pack_num': pack_num}

        def c_attrs(level_cost: int, name: str) -> dict:
            return {'name': name, 'level_cost': level_cost}

        # Intermediate feature nodes
        a = make_feature(f'{base_id}_a')
        x_c = make_feature(f'{base_id}_x_c')
        y0 = make_feature(f'{base_id}_y0')
        y_nodes = [y0] + [make_feature(f'{base_id}_y{i + 1}') for i in range(n_iter)]

        # Sub-compute nodes
        pcmstats = ComputeNode(f'{base_id}_pcmstats', 'pcmstats', 1, 1)
        pcmstats.epsilon = epsilon
        pcmcenter = ComputeNode(f'{base_id}_pcmcenter', 'pcmcenter', 1, 1)
        pcminit = ComputeNode(f'{base_id}_pcminit', 'pcminit', 1, 1)
        pcmgs_nodes = [ComputeNode(f'{base_id}_pcmgs_{i}', 'pcmgs', 1, 1) for i in range(n_iter)]
        pcmaffine = ComputeNode(f'{base_id}_pcmaffine', 'pcmaffine', 1, 1)
        pcmaffine.weight_path = weight_path
        pcmaffine.bias_path = bias_path

        # Remove the original layernorm node (keeps x_in and out in the graph)
        graph.dag.remove_node(ln_node)

        # 1a. x_in → pcmstats → a  (computes mean/variance stats, costs 3 levels)
        graph.dag.add_node(pcmstats, **c_attrs(3, pcmstats.layer_id))
        graph.dag.add_node(a, **f_attrs(a))
        graph.dag.add_edge(x_in, pcmstats)
        graph.dag.add_edge(pcmstats, a)

        # 1b. x_in → pcmcenter → x_c  (centers x, costs 1 level)
        graph.dag.add_node(pcmcenter, **c_attrs(1, pcmcenter.layer_id))
        graph.dag.add_node(x_c, **f_attrs(x_c))
        graph.dag.add_edge(x_in, pcmcenter)
        graph.dag.add_edge(pcmcenter, x_c)

        # 2. a → pcminit → y0
        graph.dag.add_node(pcminit, **c_attrs(2, pcminit.layer_id))
        graph.dag.add_node(y0, **f_attrs(y0))
        graph.dag.add_edge(a, pcminit)
        graph.dag.add_edge(pcminit, y0)

        # 3. [y_prev, a] → pcmgs_i → y_next  (repeated n_iter times)
        for i, (pcmgs_i, y_next) in enumerate(zip(pcmgs_nodes, y_nodes[1:])):
            y_prev = y_nodes[i]
            graph.dag.add_node(pcmgs_i, **c_attrs(3, pcmgs_i.layer_id))
            graph.dag.add_node(y_next, **f_attrs(y_next))
            graph.dag.add_edge(y_prev, pcmgs_i, input_index=0)
            graph.dag.add_edge(a, pcmgs_i, input_index=1)
            graph.dag.add_edge(pcmgs_i, y_next)

        # 4. [x_c, y_final] → pcmaffine → out
        y_final = y_nodes[-1]
        graph.dag.add_node(pcmaffine, **c_attrs(2, pcmaffine.layer_id))
        graph.dag.add_edge(x_c, pcmaffine, input_index=0)
        graph.dag.add_edge(y_final, pcmaffine, input_index=1)
        graph.dag.add_edge(pcmaffine, out)


def set_pcm_K(graph: LayerAbstractGraph):
    """Set K attribute on pcmgamma and pcmpoly nodes."""
    base_feat_dim = config.n_heads * config.matmul_block_size
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        if node.layer_type not in ('pcmgamma', 'pcmpoly'):
            continue
        preds = list(graph.dag.predecessors(node))
        if not preds:
            continue
        in_shape = preds[0].shape
        if base_feat_dim > 0 and len(in_shape) >= 2:
            node.K = math.ceil(in_shape[1] / base_feat_dim)
        else:
            node.K = 1

    def get_prev_compute(feature_node: FeatureNode) -> ComputeNode | None:
        prev_compute_nodes = [p for p in graph.dag.predecessors(feature_node) if isinstance(p, ComputeNode)]
        if not prev_compute_nodes:
            return None
        return prev_compute_nodes[0]

    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode) or node.layer_type not in ('pcmgamma', 'pcmpoly'):
            continue
        preds = list(graph.dag.predecessors(node))
        if not preds:
            continue
        prev_compute = get_prev_compute(preds[0])
        if node.layer_type == 'pcmgamma' and prev_compute is not None and prev_compute.layer_type == 'parccmm':
            node.K = 1
        elif node.layer_type == 'pcmpoly' and prev_compute is not None:
            if prev_compute.layer_type == 'parccmm':
                node.K = 1
            elif prev_compute.layer_type == 'pcmgamma':
                prev_preds = list(graph.dag.predecessors(prev_compute))
                if prev_preds:
                    prev_prev_compute = get_prev_compute(prev_preds[0])
                    if prev_prev_compute is not None and prev_prev_compute.layer_type == 'parccmm':
                        node.K = 1
