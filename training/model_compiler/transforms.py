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
from maxpool_plan import build_maxpool_plan, build_maxpool_stage_plan
from inference.model_generator.parameter_sources import public_source, secret_source
from inference.model_generator.layers.poly_relu_base import PolyReluBase
from inference.model_generator.layers.poly_base import PolyBase
from inference.model_generator.layers.mat_poly_base import MatPolyBase


class Direction(Enum):
    UP = 'up'
    DOWN = 'down'


PCM_TO_PDM_TYPES = {
    'pcmgeneralpoly': 'pdmupperpoly',
    'pcmmulsquare': 'pdmmulsquare',
    'pcmheadcolsum': 'pdmheadcolsum',
    'pcminvinit': 'pdminvinit',
    'pcminviter': 'pdminviter',
    'pcmctmul': 'pdmctmul',
    'pcmpolymultct': 'pdmupperpolymultct',

    'pcm_add_pt': 'pdm_add_pt',
    'pcmgamma': 'pdmgamma',
    'pcmpoly': 'pdmpoly',
    'pcmstats': 'pdmstats',
    'pcmcenter': 'pdmcenter',
    'pcminit': 'pdminit',
    'pcmgs': 'pdmgs',
    'pcmaffine': 'pdmaffine',
}


def _is_diagonal_mat_pack() -> bool:
    return getattr(config, 'mat_pack_style', '') == 'par_diagonal_pack'


def _positive_int_attr(value, attr_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _positive_int_attr(value[0], attr_name)

    result = int(value)
    if result <= 0:
        raise ValueError(f'{attr_name} must be positive, got {value}')
    return result


def get_bert_l_prepad_seq_len(graph: LayerAbstractGraph) -> int | None:
    seq_lens = []
    for node in graph.dag.nodes:
        if not (isinstance(node, ComputeNode) and node.layer_type == 'CustomMultiHeadAttention'):
            continue
        if getattr(config, 'model_type', '') != 'bert' and getattr(node, 'model_type', '') != 'bert':
            continue
        seq_len = _positive_int_attr(getattr(node, 'L_prepad', None), f'{node.layer_id}.L_prepad')
        if seq_len is not None:
            seq_lens.append(seq_len)

    if not seq_lens:
        return None
    seq_len = seq_lens[0]
    if any(item != seq_len for item in seq_lens[1:]):
        raise ValueError(f'Conflicting BERT L_prepad values: {seq_lens}')
    return seq_len


def apply_bert_l_prepad_to_feature_mat_shapes(graph: LayerAbstractGraph) -> int | None:
    seq_len = get_bert_l_prepad_seq_len(graph)
    if seq_len is None:
        return None

    for node in graph.dag.nodes:
        if not (
            isinstance(node, FeatureNode) and node.data_type == 'feature_mat' and node.dim == 2 and len(node.shape) >= 2
        ):
            continue
        node.shape[0] = seq_len
    return seq_len


def _pack_pcm_type(layer_type: str) -> str:
    if _is_diagonal_mat_pack():
        return PCM_TO_PDM_TYPES.get(layer_type, layer_type)
    return layer_type


def sort_preds_by_input_index(dag: nx.DiGraph, node, preds: list[FeatureNode]) -> list[FeatureNode]:
    if len(preds) <= 1:
        return preds
    edge_indices = {pred: dag.edges[pred, node].get('input_index') for pred in preds}
    if all(idx is not None for idx in edge_indices.values()):
        return sorted(preds, key=lambda pred: edge_indices[pred])
    return preds


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
    if hasattr(feature_node_in, 'maxpool_state'):
        # Level/scale adapters preserve the full group-major live set.
        feature_node_out.maxpool_state = copy.deepcopy(feature_node_in.maxpool_state)
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

    slot_num = param_dict[upstream_feature.ckks_parameter_id].poly_modulus_degree
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
        if parcpmm_node.layer_type not in ('parcpmm', 'pdmpcmm') or not getattr(parcpmm_node, 'to_expand', False):
            continue

        bias_path = getattr(parcpmm_node, 'bias_path', '')
        if not bias_path:
            raise ValueError(
                f'{parcpmm_node.layer_type} layer {parcpmm_node.layer_id} has to_expand=True but no bias_path'
            )

        old_feature_list = list(graph.dag.successors(parcpmm_node))
        if len(old_feature_list) != 1:
            raise ValueError(
                f'Expected exactly one output feature for {parcpmm_node.layer_type} layer {parcpmm_node.layer_id}, '
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
        add_pt_node = ComputeNode(
            add_pt_id, _pack_pcm_type('pcm_add_pt'), parcpmm_node.channel_output, parcpmm_node.channel_output
        )
        add_pt_node.path = bias_path
        add_pt_node.parameter_sources = {'add_pt': secret_source('onnx_initializer', bias_path)}

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
                if _is_diagonal_mat_pack():
                    if len(node.shape) < 2:
                        raise ValueError(f'feature_mat input {node.node_id} must be rank-2 for par_diagonal_pack')
                    seq_len, total_dim = node.shape[0], node.shape[1]
                    node.shape[0] = total_dim
                    node.shape[1] = seq_len
                    node.head_shape = [head_dim, seq_len]
                    node.is_transposed = True
                else:
                    node.head_shape = [node.shape[0], head_dim]

    for compute_node in sorted_compute_nodes:
        preds: list[FeatureNode] = list(graph.dag.predecessors(compute_node))
        preds = sort_preds_by_input_index(graph.dag, compute_node, preds)
        succ: FeatureNode = next(graph.dag.successors(compute_node))
        # init skip,
        if succ.dim != 0:
            graph.dag.nodes[succ]['skip'] = [1] * succ.dim

        if compute_node.layer_type.startswith('maxpool2d_'):
            # Tap normalization already writes dense output planes; generic
            # pass-through inference would overwrite them with input shape.
            succ.shape = list(compute_node.maxpool_output_shape)
            populate_pack_num(graph.dag, compute_node, config.fhe_param.poly_modulus_degree / 2)
            continue

        if process_special_info(graph, compute_node, preds, succ):
            populate_pack_num(graph.dag, compute_node, config.fhe_param.poly_modulus_degree)
            continue
        if succ.dim > 0:
            if isinstance(compute_node, SpatialComputeNode):
                maxpool_output_shape = None
                if compute_node.layer_type == 'maxpool2d':
                    plan = build_maxpool_plan(
                        {
                            'kernel_shape': compute_node.kernel_shape,
                            'stride': compute_node.stride,
                            'padding': compute_node.padding,
                        },
                        preds[0].shape,
                    )
                    maxpool_output_shape = plan['output_shape']

                for i in range(compute_node.dim):
                    if maxpool_output_shape is not None:
                        succ.shape[i] = maxpool_output_shape[i]
                    else:
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
                if compute_node.layer_type == 'pcmstats':
                    succ.shape = [preds[0].shape[0], 1]
                    succ.head_shape = [preds[0].shape[0], 1]
                elif compute_node.layer_type == 'parcpmm':
                    succ.shape[0] = preds[0].shape[0]
                    weight_shape = getattr(compute_node, 'weight_shape', [])
                    if len(weight_shape) >= 2:
                        succ.shape[1] = weight_shape[-1]
                    if preds[0].head_shape is not None:
                        succ.head_shape = list(preds[0].head_shape)
                elif compute_node.layer_type == 'pdmpcmm':
                    weight_shape = getattr(compute_node, 'weight_shape', [])
                    if preds[0].head_shape is not None:
                        succ.head_shape = list(preds[0].head_shape)
                    if len(weight_shape) >= 1:
                        succ.shape[0] = weight_shape[0]
                    else:
                        succ.shape[0] = preds[0].shape[0]
                    if len(preds[0].shape) > 1:
                        succ.shape[1] = preds[0].shape[1]
                elif compute_node.layer_type == 'partranspose':
                    if preds[0].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[1], preds[0].head_shape[0]]
                        succ.shape[0] = succ.head_shape[0]
                        succ.shape[1] = succ.head_shape[1] * n_heads
                    else:
                        succ.shape[0] = preds[0].shape[1]
                        succ.shape[1] = preds[0].shape[0]
                elif compute_node.layer_type == 'pdmtranspose':
                    if preds[0].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[1], preds[0].head_shape[0]]
                        succ.shape[0] = succ.head_shape[0]
                        succ.shape[1] = succ.head_shape[1] * n_heads
                        succ.is_transposed = False
                    else:
                        succ.shape[0] = preds[0].shape[1]
                        succ.shape[1] = preds[0].shape[0]
                        succ.is_transposed = False
                elif compute_node.layer_type == 'parccmm':
                    succ.shape[0] = preds[0].shape[0]
                    if preds[0].head_shape is not None and len(preds) > 1 and preds[1].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[0], preds[1].head_shape[1]]
                        succ.shape[1] = succ.head_shape[1] * n_heads
                    elif len(preds) > 1 and len(preds[1].shape) > 1:
                        succ.shape[1] = preds[1].shape[1]
                elif compute_node.layer_type == 'pdmccmm':
                    if preds[0].head_shape is not None and len(preds) > 1 and preds[1].head_shape is not None:
                        succ.head_shape = [preds[0].head_shape[0], preds[1].head_shape[1]]
                        succ.shape[0] = succ.head_shape[0] * n_heads
                        succ.shape[1] = succ.head_shape[1]
                    else:
                        succ.shape[0] = preds[0].shape[0]
                        if len(preds) > 1 and len(preds[1].shape) > 1:
                            succ.shape[1] = preds[1].shape[1]
                else:
                    for i in range(preds[0].dim):
                        succ.shape[i] = preds[0].shape[i]
                        graph.dag.nodes[succ]['skip'][i] = graph.dag.nodes[preds[0]]['skip'][i]
                if not _is_diagonal_mat_pack() and compute_node.layer_type not in (
                    'pcmstats', 'parcpmm', 'parccmm', 'partranspose'
                ) and preds[0].head_shape is not None:
                    succ.head_shape = list(preds[0].head_shape)
                # Propagate head_shape for feature_mat pass-through layers
                if succ.head_shape is None and preds[0].head_shape is not None:
                    succ.head_shape = list(preds[0].head_shape)
        if preds[0].dim >= 1 and any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
            graph.dag.nodes[succ]['skip'] = [1] * preds[0].dim
            if any(succ.shape[i] < config.block_shape[i] for i in range(succ.dim)):
                for i in range(preds[0].dim):
                    graph.dag.nodes[succ]['skip'][i] = max(1, config.block_shape[i] // succ.shape[i])

        process_special_info(graph, compute_node, preds, succ)
        populate_pack_num(graph.dag, compute_node, config.fhe_param.poly_modulus_degree)


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


def replace_avgpool1d_with_depthwise_conv(graph: LayerAbstractGraph):
    """Replace non-adaptive AvgPool1d nodes with synthetic depthwise Conv1d nodes.

    The runtime multiplexed avgpool1d path preserves sparse skip layout, while the
    downstream multiplexed conv1d path expects conv-style packed output. Replacing
    avgpool1d with an equivalent depthwise conv1d during compilation keeps the
    packing semantics consistent for subsequent conv1d layers.

    The synthetic conv uses an odd kernel so the existing conv1d implementation's
    implicit same-padding can reproduce the original left-aligned avgpool window:
      avgpool window size = k
      synthetic conv kernel size = 2 * k - 1
      active taps are the last k positions, each weighted by 1 / k.
    """
    for node in list(graph.dag.nodes):
        if not isinstance(node, PoolComputeNode):
            continue
        if node.layer_type != 'avgpool1d' or node.is_adaptive_avgpool:
            continue

        preds: list[FeatureNode] = list(graph.dag.predecessors(node))
        succs: list[FeatureNode] = list(graph.dag.successors(node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(f'avgpool1d replacement expects single input/output: {node.layer_id}')

        pool_kernel = int(node.kernel_shape[0])
        synthetic_kernel = 2 * pool_kernel - 1
        conv_node = ConvComputeNode(
            node.layer_id,
            'conv1d',
            node.channel_input,
            node.channel_output,
            dim=1,
            stride=list(node.stride),
            groups=node.channel_input,
            kernel_shape=[synthetic_kernel],
            parameter_paths={
                'weight': f'{node.layer_id}.weight',
                'bias': f'{node.layer_id}.bias',
            },
        )
        conv_node.synthetic_source = 'avgpool1d'
        conv_node.source_pool_kernel_shape = list(node.kernel_shape)
        conv_node.source_pool_padding = list(node.padding)
        conv_node.is_big_size = getattr(node, 'is_big_size', False)

        node_attrs = dict(graph.dag.nodes[node])
        graph.dag.add_node(conv_node, **node_attrs)
        for pred in preds:
            graph.dag.add_edge(pred, conv_node, **dict(graph.dag.edges[pred, node]))
        for succ in succs:
            graph.dag.add_edge(conv_node, succ, **dict(graph.dag.edges[node, succ]))
        graph.dag.remove_node(node)


def expand_maxpool_for_btp(graph: LayerAbstractGraph):
    """Lower the Retina 3x3 MaxPool into compiler-visible stages.

    Preserve configuration on executable stages and derive round costs from
    the selected fixed profile. DP chooses refresh boundaries between stages.
    """
    for pool in list(graph.dag.nodes):
        if not isinstance(pool, ComputeNode) or pool.layer_type != 'maxpool2d':
            continue
        preds = list(graph.dag.predecessors(pool))
        succs = list(graph.dag.successors(pool))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(f'MaxPool node {pool.layer_id} must have one input and output')
        if list(pool.kernel_shape) != [3, 3] or list(pool.stride) != [2, 2] or list(pool.padding) != [1, 1]:
            continue

        input_feature = preds[0]
        output_feature = succs[0]
        from inference.model_generator.layers.poly_maxpool_cost import get_maxpool_compare_cost
        alpha = getattr(pool, 'alpha', 7)
        compare_cost = get_maxpool_compare_cost(alpha)
        polynomial_costs = None
        if compare_cost > config.fhe_param.max_level:
            from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile
            from inference.model_generator.layers.poly_base import PolyBase
            polynomial_costs = [PolyBase.compute_stockmeyer_level_cost(d)
                                for d in get_maxpool_profile(alpha)['degrees']]
            if max(polynomial_costs) > config.fhe_param.max_level:
                raise ValueError('MaxPool polynomial stage exceeds CKKS level budget')
        for name in ('upper_bound', 'B'):
            value = getattr(pool, name, 1.0)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'MaxPool {name} must be positive and finite')
        plan = build_maxpool_plan(pool.__dict__, input_feature.shape)
        stage_plan = build_maxpool_stage_plan(plan, compare_cost=compare_cost,
                                              polynomial_costs=polynomial_costs)
        prev = input_feature
        new_nodes = []
        for index, stage in enumerate(stage_plan['stages']):
            stage_id = f'{pool.layer_id}_{stage["name"]}'
            node = ComputeNode(stage_id, f'maxpool2d_{stage["name"]}', pool.channel_input, pool.channel_output)
            node.kernel_shape = list(pool.kernel_shape)
            node.stride = list(pool.stride)
            node.padding = list(pool.padding)
            node.maxpool_stage = stage['name']
            node.maxpool_plan = plan
            node.maxpool_output_shape = list(plan['output_shape'])
            node.is_adaptive_avgpool = False
            node.level_cost = stage['level_cost']
            node.running_max = getattr(pool, 'running_max', None)
            node.alpha = alpha
            node.upper_bound = getattr(pool, 'upper_bound', 1.0)
            node.B = getattr(pool, 'B', 1.0)
            is_last = index == len(stage_plan['stages']) - 1
            stage_shape = list(plan['output_shape'])
            stage_channel = output_feature.channel if stage['name'] in {'repack', 'restore'} else input_feature.channel
            feature = output_feature if is_last else FeatureNode(
                f'{stage_id}_out', input_feature.dim, stage_channel,
                scale=input_feature.scale, ckks_parameter_id=input_feature.ckks_parameter_id,
                ckks_scale=input_feature.ckks_scale, shape=stage_shape,
            )
            graph.dag.add_node(node, name=node.layer_id, level_cost=stage['level_cost'])
            if feature is not output_feature:
                # Explicit live-set metadata; do not pretend reduction groups
                # are additional packed CNN channels. Generator stage entries
                # use group-major/channel-minor order, one CT per plane.
                if stage['name'] == 'normalize_taps' or stage['name'].startswith('reduce_round_'):
                    feature.maxpool_state = {
                        'pool_id': pool.layer_id,
                        'groups': stage['live_tensors'],
                        'channels': input_feature.channel,
                        'order': 'group_channel',
                    }
                    from training.model_compiler.maxpool_plan import parse_maxpool_split_stage
                    split = parse_maxpool_split_stage(stage['name'])
                    if split is not None and split[1] != 'finish':
                        feature.maxpool_state['layout'] = 'pair_sum_difference_sign_then_doubled_carry'
                        feature.maxpool_state['round'] = split[0]
                        feature.maxpool_state['polynomial'] = split[1]
                graph.dag.add_node(feature, name=feature.node_id, skip=list(graph.dag.nodes[input_feature].get('skip', [1] * input_feature.dim)), level=graph.dag.nodes[input_feature].get('level', 0), pack_num=graph.dag.nodes[input_feature].get('pack_num', 1))
            graph.dag.add_edge(prev, node)
            graph.dag.add_edge(node, feature)
            prev = feature
            new_nodes.append(node)

        for successor in list(graph.dag.successors(pool)):
            edge_attrs = dict(graph.dag.edges[pool, successor])
            graph.dag.add_edge(output_feature, successor, **edge_attrs) if successor is not output_feature else None
        graph.dag.remove_node(pool)


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

        elif compute_node.layer_type.startswith('maxpool2d_'):
            # Lowered stages carry their own cost from maxpool_plan.  Do not
            # collapse them back to the fused 31-level node before DP runs.
            level_cost = int(
                getattr(compute_node, 'level_cost', graph.dag.nodes[compute_node].get('level_cost', 0))
            )
            from inference.model_generator.layers.poly_maxpool_cost import get_maxpool_compare_cost
            stage = compute_node.maxpool_stage
            from training.model_compiler.maxpool_plan import parse_maxpool_split_stage
            split = parse_maxpool_split_stage(stage)
            if split is not None:
                from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile
                from inference.model_generator.layers.poly_base import PolyBase
                degrees = get_maxpool_profile(getattr(compute_node, 'alpha', 7))['degrees']
                expected_cost = 1 if split[1] == 'finish' else PolyBase.compute_stockmeyer_level_cost(degrees[split[1]])
            elif stage.startswith('reduce_round_'):
                expected_cost = get_maxpool_compare_cost(getattr(compute_node, 'alpha', 7))
            elif stage in {'normalize_taps', 'repack', 'restore'}:
                expected_cost = 1
            else:
                raise ValueError(f'Unknown MaxPool stage: {stage}')
            if level_cost != expected_cost:
                raise ValueError(f'Invalid MaxPool stage cost: {level_cost}; expected {expected_cost}')
            graph.dag.nodes[compute_node]['level_cost'] = level_cost

        elif compute_node.layer_type == 'maxpool2d':
            kernel = list(compute_node.kernel_shape)
            stride = list(compute_node.stride)
            padding = list(compute_node.padding)

            if kernel == [2, 2] and stride == [2, 2] and padding == [0, 0]:
                level_cost = 19

            elif kernel == [3, 3] and stride == [2, 2] and padding == [1, 1]:
                level_cost = 31

            else:
                raise NotImplementedError(
                    f"Unsupported PolyMaxPool2D: "
                    f"kernel={kernel}, stride={stride}, padding={padding}"
                )

            graph.dag.nodes[compute_node]['level_cost'] = level_cost


        elif compute_node.layer_type in {'avgpool1d', 'avgpool2d'}:
            if any(preds[0].shape[i] > config.block_shape[i] for i in range(preds[0].dim)):
                compute_node.is_big_size = True
                compute_node.is_adaptive_avgpool = False
                if any(succ.shape[i] < config.block_shape[i] for i in range(succ.dim)):
                    graph.dag.nodes[compute_node]['level_cost'] = 1  # repack needed
                else:
                    graph.dag.nodes[compute_node]['level_cost'] = 1  # pool normalization
            else:
                compute_node.is_big_size = False
                succs_sub = list(graph.dag.successors(succ))
                if succs_sub and succs_sub[0].layer_type == 'reshape':
                    graph.dag.nodes[compute_node]['level_cost'] = 1
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
        elif compute_node.layer_type in ('parcpmm', 'pdmpcmm'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in {'add_pt', 'pcm_add_pt', 'pdm_add_pt'}:
            graph.dag.nodes[compute_node]['level_cost'] = 0
        elif compute_node.layer_type in ('partranspose', 'pdmtranspose'):
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type in ('parccmm', 'pdmccmm'):
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type in ('pcmgamma', 'pdmgamma'):
            graph.dag.nodes[compute_node]['level_cost'] = 1
        elif compute_node.layer_type == 'pdmupperaddpt':
            graph.dag.nodes[compute_node]['level_cost'] = 0
        elif compute_node.layer_type in ('pcmgeneralpoly', 'pdmupperpoly'):
            graph.dag.nodes[compute_node]['level_cost'] = MatPolyBase.compute_stockmeyer_level_cost(compute_node.order)
        elif compute_node.layer_type in ('pcmmulsquare', 'pdmmulsquare'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in ('pcmheadcolsum', 'pdmheadcolsum'):
            graph.dag.nodes[compute_node]['level_cost'] = 2 if compute_node.layer_type == 'pcmheadcolsum' else 1
        elif compute_node.layer_type in ('pcminvinit', 'pdminvinit'):
            graph.dag.nodes[compute_node]['level_cost'] = 0
        elif compute_node.layer_type in ('pcminviter', 'pdminviter'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in ('pcmctmul', 'pcmpolymultct', 'pdmctmul', 'pdmupperpolymultct'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in ('pcmpoly', 'pdmpoly'):
            graph.dag.nodes[compute_node]['level_cost'] = 2 if compute_node.order == 2 else 3
        elif compute_node.layer_type == 'pcmstats':
            graph.dag.nodes[compute_node]['level_cost'] = 4
        elif compute_node.layer_type == 'pdmstats':
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type in ('pcmcenter', 'pdmcenter'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in ('pcminit', 'pdminit'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
        elif compute_node.layer_type in ('pcmgs', 'pdmgs'):
            graph.dag.nodes[compute_node]['level_cost'] = 3
        elif compute_node.layer_type in ('pcmaffine', 'pdmaffine'):
            graph.dag.nodes[compute_node]['level_cost'] = 2
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
                elif node.layer_type == 'mult_coeff':
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
        layers_to_absorb = ['mult_coeff']

    for node in subgraph.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type in layers_to_absorb:
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


PDM_UPPER_PASS_THROUGH_TYPES = {
    'pdmupperaddpt',
    'pcmgeneralpoly',
    'pdmupperpoly',
    'pcmmulsquare',
    'pdmmulsquare',
    'pcmheadcolsum',
    'pdmheadcolsum',
    'pcminvinit',
    'pdminvinit',
    'pcminviter',
    'pdminviter',
    'pcmctmul',
    'pdmctmul',
    'pcmpolymultct',
    'pdmupperpolymultct',
}


def _parse_float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(',') if item.strip()]
    return [float(item) for item in value]


def _float_attr(node: ComputeNode, name: str, default: float) -> float:
    value = getattr(node, name, default)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return float(value)


def _int_attr(node: ComputeNode, name: str, default: int) -> int:
    value = getattr(node, name, default)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return int(value)


def _node_str_attr(node: ComputeNode, name: str, default: str = '') -> str:
    value = getattr(node, name, default)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


def _bert_softmax_denominator_scale(attn_node: ComputeNode, attr_name: str, config_name: str, default: float) -> float:
    if hasattr(attn_node, attr_name):
        scale = _float_attr(attn_node, attr_name, default)
    else:
        scale = float(getattr(config, config_name, default))
        if attr_name == 'initial_denominator_scale':
            if _node_str_attr(attn_node, 'softmax_profile').lower() == 'thor-wide':
                wide_scale = float(getattr(config, 'bert_softmax_wide_initial_denominator_scale', 65536.0))
                scale = max(scale, wide_scale)
    if scale == 0:
        raise ValueError(f'BERT softmax denominator scale {attr_name} cannot be 0')
    return scale


def _layernorm_iterations(node: ComputeNode, default: int) -> int:
    return _int_attr(
        node,
        'max_inverse_sqrt_iterations',
        _int_attr(node, 'max_inverse_sqrt_iteration', _int_attr(node, 'num_iters', default)),
    )


def _bert_layernorm_runtime_attrs(ln_node: ComputeNode, default_n_iter: int) -> dict:
    eps = _float_attr(ln_node, 'epsilon', 1.0e-5)
    min_var = _float_attr(ln_node, 'min_var', 0.75)
    max_var = _float_attr(ln_node, 'max_var', 2500.0)
    w_buffer = _float_attr(ln_node, 'w_buffer', 1.05)
    input_scale = _float_attr(ln_node, 'input_scale', 1.0)
    max_denominator = (max_var * w_buffer + eps) * input_scale * input_scale
    if max_denominator <= 0:
        raise ValueError(f'BERT LayerNorm node {ln_node.layer_id} has non-positive max_denominator={max_denominator}')
    inv_var = 1.0 / max_denominator
    inv_std = math.sqrt(inv_var)
    normalized_epsilon = (min_var + eps) / max_denominator
    use_asor = _int_attr(ln_node, 'use_asor', 0)
    if use_asor != 0:
        raise ValueError(
            f'BERT LayerNorm node {ln_node.layer_id} requests use_asor={use_asor}, but only use_asor=0 is supported'
        )

    return {
        'epsilon': eps,
        'min_var': min_var,
        'max_var': max_var,
        'w_buffer': w_buffer,
        'input_scale': input_scale,
        'max_denominator': max_denominator,
        'normalized_epsilon': normalized_epsilon,
        'inv_var': inv_var,
        'inv_std': inv_std,
        'c0': _float_attr(ln_node, 'c0', config.layernorm_c0),
        'c1': _float_attr(ln_node, 'c1', config.layernorm_c1),
        'c2': _float_attr(ln_node, 'c2', config.layernorm_c2),
        'num_iters': _layernorm_iterations(ln_node, default_n_iter),
        'profile': getattr(ln_node, 'profile', ''),
        'use_asor': use_asor,
    }


def _new_feature_like(source: FeatureNode, name: str, shape: list[int] | None = None) -> FeatureNode:
    feature = FeatureNode(
        key=name,
        dim=source.dim,
        channel=source.channel,
        scale=source.scale,
        ckks_parameter_id=source.ckks_parameter_id,
        ckks_scale=source.ckks_scale,
        shape=list(shape if shape is not None else source.shape),
    )
    feature.data_type = source.data_type
    feature.invalid_fill = list(source.invalid_fill)
    feature.sp_info = copy.deepcopy(source.sp_info)
    feature.has_sp_info = source.has_sp_info
    if source.head_shape is not None:
        feature.head_shape = list(source.head_shape)
    return feature


def _feature_attrs_like(graph: LayerAbstractGraph, source: FeatureNode, feature: FeatureNode) -> dict:
    source_attrs = graph.dag.nodes[source]
    return {
        'name': feature.node_id,
        'skip': list(source_attrs.get('skip', [1] * source.dim)),
        'level': source_attrs.get('level', 0),
        'pack_num': source_attrs.get('pack_num', 1),
    }


def _compute_attrs(level_cost: int, name: str) -> dict:
    return {'name': name, 'level_cost': level_cost}


def _append_unary_compute(
    graph: LayerAbstractGraph,
    input_feature: FeatureNode,
    compute_node: ComputeNode,
    output_id: str,
    level_cost: int,
) -> FeatureNode:
    output_feature = _new_feature_like(input_feature, output_id)
    graph.dag.add_node(compute_node, **_compute_attrs(level_cost, compute_node.layer_id))
    graph.dag.add_node(output_feature, **_feature_attrs_like(graph, input_feature, output_feature))
    graph.dag.add_edge(input_feature, compute_node)
    graph.dag.add_edge(compute_node, output_feature)
    return output_feature


def _append_binary_compute(
    graph: LayerAbstractGraph,
    lhs: FeatureNode,
    rhs: FeatureNode,
    compute_node: ComputeNode,
    output_id: str,
    level_cost: int,
) -> FeatureNode:
    output_feature = _new_feature_like(lhs, output_id)
    graph.dag.add_node(compute_node, **_compute_attrs(level_cost, compute_node.layer_id))
    graph.dag.add_node(output_feature, **_feature_attrs_like(graph, lhs, output_feature))
    graph.dag.add_edge(lhs, compute_node, input_index=0)
    graph.dag.add_edge(rhs, compute_node, input_index=1)
    graph.dag.add_edge(compute_node, output_feature)
    return output_feature


def _make_scalar_gamma(layer_id: str, scalar: float, source=None) -> ComputeNode:
    node = ComputeNode(layer_id, _pack_pcm_type('pcmgamma'), 1, 1)
    node.path = f'{layer_id}.weight'
    node.scalar_value = float(scalar)
    node.parameter_sources = {'gamma_pt': source or public_source(
        'approximation_config', {'scalar_value': float(scalar)})}
    return node


def _make_upper_poly(layer_id: str, coefficients: list[float]) -> ComputeNode:
    node = ComputeNode(layer_id, _pack_pcm_type('pcmgeneralpoly'), 1, 1)
    node.path = f'{layer_id}.weight'
    node.coeffs_path = node.path
    node.coefficients = list(coefficients)
    node.order = len(coefficients) - 1
    node.parameter_sources = {'stockmeyer_coeff_pt': public_source(
        'approximation_config', {'coefficients': list(coefficients)})}
    return node


def _append_softmax_normalize(
    graph: LayerAbstractGraph,
    base_id: str,
    values: FeatureNode,
    denominator_scale: float,
    max_inverse_iterations: int,
    values_scale: float = 1.0,
) -> FeatureNode:
    denom = _append_unary_compute(
        graph,
        values,
        ComputeNode(f'{base_id}_head_col_sum', _pack_pcm_type('pcmheadcolsum'), 1, 1),
        f'{base_id}_denominator',
        1 if _is_diagonal_mat_pack() else 2,
    )
    scale_ratio = denominator_scale / values_scale
    if scale_ratio != 1.0:
        denom = _append_unary_compute(
            graph,
            denom,
            _make_scalar_gamma(f'{base_id}_denominator_scale', scale_ratio),
            f'{base_id}_denominator_scaled',
            1,
        )

    # If b is scaled by s, carry a/s throughout Newton's iteration. Scaling
    # b alone changes the result of the exporter's fixed iteration count.
    inverse_init = (
        ComputeNode(f'{base_id}_inverse_init', _pack_pcm_type('pcminvinit'), 1, 1)
        if denominator_scale == 1.0 else
        _make_upper_poly(f'{base_id}_inverse_init', [2.0 / denominator_scale, -1.0 / denominator_scale**2])
    )
    inv = _append_unary_compute(
        graph,
        denom,
        inverse_init,
        f'{base_id}_inverse_init_out',
        0 if denominator_scale == 1.0 else 1,
    )
    for idx in range(1, max_inverse_iterations):
        inv = _append_binary_compute(
            graph,
            inv,
            denom,
            ComputeNode(f'{base_id}_inverse_iter_{idx}', _pack_pcm_type('pcminviter'), 1, 1),
            f'{base_id}_inverse_iter_{idx}_out',
            2,
        )

    probabilities_output_id = (
        f'{base_id}_probabilities_unscaled' if scale_ratio != 1.0 else f'{base_id}_probabilities'
    )
    probabilities = _append_binary_compute(
        graph,
        values,
        inv,
        ComputeNode(f'{base_id}_multiply', _pack_pcm_type('pcmctmul'), 1, 1),
        probabilities_output_id,
        2,
    )
    if scale_ratio != 1.0:
        probabilities = _append_unary_compute(
            graph,
            probabilities,
            _make_scalar_gamma(f'{base_id}_probabilities_scale', scale_ratio),
            f'{base_id}_probabilities',
            1,
        )
    return probabilities


def _append_bert_softmax(graph: LayerAbstractGraph, base_id: str, scores: FeatureNode, attn_node: ComputeNode):
    range_min = _float_attr(attn_node, 'range_min', -3.0)
    range_max = _float_attr(attn_node, 'range_max', 3.0)
    mid = (range_min + range_max) / 2.0
    exp_divisor = _float_attr(attn_node, 'exp_divisor', 1.0)
    coefficients = _parse_float_list(getattr(attn_node, 'exp_coefficients', []))
    if not coefficients:
        raise ValueError(f'BERT attention node {attn_node.layer_id} is missing exp_coefficients')
    max_inverse_iterations = _int_attr(
        attn_node,
        'max_inverse_iterations',
        _int_attr(attn_node, 'max_inverse_iteration', 15),
    )
    delta_1 = _int_attr(attn_node, 'delta_1', 2)
    delta_2 = _int_attr(attn_node, 'delta_2', 1)
    initial_denominator_scale = _bert_softmax_denominator_scale(
        attn_node,
        'initial_denominator_scale',
        'bert_softmax_initial_denominator_scale',
        16.0,
    )
    first_refinement_denominator_scale = _bert_softmax_denominator_scale(
        attn_node,
        'first_refinement_denominator_scale',
        'bert_softmax_first_refinement_denominator_scale',
        2.0,
    )
    later_refinement_denominator_scale = _bert_softmax_denominator_scale(
        attn_node,
        'later_refinement_denominator_scale',
        'bert_softmax_later_refinement_denominator_scale',
        1.0,
    )

    add_mid_node = ComputeNode(f'{base_id}_softmax_add_mid', ('pdmupperaddpt' if _is_diagonal_mat_pack() else 'pcm_add_pt'), 1, 1)
    add_mid_node.value = -mid
    add_mid_node.path = f'{add_mid_node.layer_id}.weight'
    add_mid_node.parameter_sources = {'add_pt': secret_source(
        'calibration', f'{attn_node.layer_id}.range_min', f'{attn_node.layer_id}.range_max')}
    centered = _append_unary_compute(
        graph,
        scores,
        add_mid_node,
        f'{base_id}_softmax_centered',
        0,
    )

    exp_arg = _append_unary_compute(
        graph,
        centered,
        _make_scalar_gamma(f'{base_id}_softmax_exp_gamma', 1.0 / exp_divisor),
        f'{base_id}_softmax_exp_arg',
        1,
    )
    exp_poly = _append_unary_compute(
        graph,
        exp_arg,
        # Fold the signal scale into the polynomial before squaring. A large
        # scalar plaintext (65536 for thor-wide) overflows native ring-T encoding.
        _make_upper_poly(f'{base_id}_softmax_exp_poly', [
            c * initial_denominator_scale ** (1.0 / max(1, delta_1)) for c in coefficients
        ]),
        f'{base_id}_softmax_exp_poly_out',
        PolyBase.compute_stockmeyer_level_cost(len(coefficients) - 1),
    )

    values = exp_poly
    delta = max(1, delta_1)
    square_idx = 0
    while delta > 1:
        values = _append_unary_compute(
            graph,
            values,
            ComputeNode(f'{base_id}_softmax_delta1_square_{square_idx}', _pack_pcm_type('pcmmulsquare'), 1, 1),
            f'{base_id}_softmax_delta1_square_{square_idx}_out',
            2,
        )
        delta >>= 1
        square_idx += 1

    probabilities = _append_softmax_normalize(
        graph,
        f'{base_id}_softmax_normalize_initial',
        values,
        initial_denominator_scale,
        max_inverse_iterations,
        values_scale=initial_denominator_scale,
    )

    refinement_idx = 0
    delta = max(1, delta_2)
    while delta > 1:
        denominator_scale = (
            first_refinement_denominator_scale if refinement_idx == 0 else later_refinement_denominator_scale
        )
        # Scale before squaring: amplifying the small squared ciphertext later
        # also amplifies its rescaling error by the full denominator scale.
        square_input = probabilities
        if denominator_scale != 1.0:
            square_input = _append_unary_compute(
                graph,
                probabilities,
                _make_scalar_gamma(f'{base_id}_softmax_refine_scale_{refinement_idx}', math.sqrt(denominator_scale)),
                f'{base_id}_softmax_refine_scaled_{refinement_idx}',
                1,
            )
        squared = _append_unary_compute(
            graph,
            square_input,
            ComputeNode(f'{base_id}_softmax_refine_square_{refinement_idx}', _pack_pcm_type('pcmmulsquare'), 1, 1),
            f'{base_id}_softmax_refine_square_{refinement_idx}_out',
            2,
        )
        probabilities = _append_softmax_normalize(
            graph,
            f'{base_id}_softmax_normalize_refine_{refinement_idx}',
            squared,
            denominator_scale,
            max_inverse_iterations,
            values_scale=denominator_scale,
        )
        delta >>= 1
        refinement_idx += 1

    return probabilities


def _add_ccmm_inputs(graph: LayerAbstractGraph, lhs: FeatureNode, rhs: FeatureNode, node: ComputeNode):
    if node.layer_type == 'pdmccmm':
        graph.dag.add_edge(lhs, node, input_index=1)
        graph.dag.add_edge(rhs, node, input_index=0)
    else:
        graph.dag.add_edge(lhs, node, input_index=0)
        graph.dag.add_edge(rhs, node, input_index=1)


def _expand_bert_multi_head_attention(graph: LayerAbstractGraph, attn_node: ComputeNode):
    if getattr(config, 'mat_pack_style', '') not in ('par_diagonal_pack', 'par_block_col_major'):
        raise ValueError('BERT CustomMultiHeadAttention requires par_diagonal_pack or par_block_col_major')

    preds = list(graph.dag.predecessors(attn_node))
    succs = list(graph.dag.successors(attn_node))
    if len(preds) != 1 or len(succs) != 1:
        raise ValueError(
            f'BERT attention node {attn_node.layer_id} must have exactly 1 input and 1 output, '
            f'got {len(preds)} inputs and {len(succs)} outputs'
        )
    x_in: FeatureNode = preds[0]
    out: FeatureNode = succs[0]

    base_id = attn_node.layer_id
    m = x_in.shape[0]
    n = x_in.shape[1]
    n_heads = max(1, int(getattr(attn_node, 'num_heads', config.n_heads)))
    scaling = _float_attr(attn_node, 'scaling', 1.0 / math.sqrt(max(1, config.head_dim)))

    def make_feature(name: str, shape: list[int] | None = None) -> FeatureNode:
        return _new_feature_like(x_in, name, shape)

    q = make_feature(f'{base_id}_q')
    k = make_feature(f'{base_id}_k')
    v = make_feature(f'{base_id}_v')
    kt = make_feature(f'{base_id}_kt', [n // n_heads, m * n_heads])
    qkt = make_feature(f'{base_id}_qkt', [m, m * n_heads])
    qktv = make_feature(f'{base_id}_qktv', list(out.shape))

    pcmm_type = 'pdmpcmm' if _is_diagonal_mat_pack() else 'parcpmm'
    transpose_type = 'pdmtranspose' if _is_diagonal_mat_pack() else 'partranspose'
    ccmm_type = 'pdmccmm' if _is_diagonal_mat_pack() else 'parccmm'

    q_node = ComputeNode(f'{base_id}_q_layer', pcmm_type, 1, 1)
    q_node.path = getattr(attn_node, 'q_weight_path', f'{base_id}.q.weight')
    q_node.bias_path = getattr(attn_node, 'q_bias_path', '')
    q_node.weight_shape = [n, n]
    q_node.weight_multiplier = scaling
    q_node.bias_multiplier = scaling

    k_node = ComputeNode(f'{base_id}_k_layer', pcmm_type, 1, 1)
    k_node.path = getattr(attn_node, 'k_weight_path', f'{base_id}.k.weight')
    k_node.bias_path = getattr(attn_node, 'k_bias_path', '')
    k_node.weight_shape = [n, n]

    v_node = ComputeNode(f'{base_id}_v_layer', pcmm_type, 1, 1)
    v_node.path = getattr(attn_node, 'v_weight_path', f'{base_id}.v.weight')
    v_node.bias_path = getattr(attn_node, 'v_bias_path', '')
    v_node.weight_shape = [n, n]

    kt_node = ComputeNode(f'{base_id}_kt_layer', transpose_type, 1, 1)
    qkt_node = ComputeNode(f'{base_id}_qkt_layer', ccmm_type, 1, 1)
    qktv_node = ComputeNode(f'{base_id}_qktv_layer', ccmm_type, 1, 1)

    out_node = ComputeNode(f'{base_id}_out', pcmm_type, 1, 1)
    out_node.weight_shape = [n, n]
    out_node.path = getattr(attn_node, 'proj_weight_path', f'{base_id}.proj.weight')
    out_node.bias_path = getattr(attn_node, 'proj_bias_path', '')

    if not _is_diagonal_mat_pack():
        for node in (q_node, k_node, v_node, out_node):
            node.weight_transposed = True

    graph.dag.remove_node(attn_node)

    for node in (q_node, k_node, v_node):
        graph.dag.add_node(node, **_compute_attrs(2, node.layer_id))
    graph.dag.add_node(kt_node, **_compute_attrs(1, kt_node.layer_id))
    graph.dag.add_node(qkt_node, **_compute_attrs(3, qkt_node.layer_id))
    graph.dag.add_node(qktv_node, **_compute_attrs(3, qktv_node.layer_id))
    graph.dag.add_node(out_node, **_compute_attrs(2, out_node.layer_id))

    for feature in (q, k, v, kt, qkt, qktv):
        graph.dag.add_node(feature, **_feature_attrs_like(graph, x_in, feature))

    graph.dag.add_edge(x_in, q_node)
    graph.dag.add_edge(q_node, q)
    graph.dag.add_edge(x_in, k_node)
    graph.dag.add_edge(k_node, k)
    graph.dag.add_edge(x_in, v_node)
    graph.dag.add_edge(v_node, v)

    graph.dag.add_edge(k, kt_node)
    graph.dag.add_edge(kt_node, kt)

    _add_ccmm_inputs(graph, q, kt, qkt_node)
    graph.dag.add_edge(qkt_node, qkt)

    softmax_out = _append_bert_softmax(graph, base_id, qkt, attn_node)

    _add_ccmm_inputs(graph, softmax_out, v, qktv_node)
    graph.dag.add_edge(qktv_node, qktv)

    graph.dag.add_edge(qktv, out_node)
    graph.dag.add_edge(out_node, out)


def expand_multi_head_attention(graph: LayerAbstractGraph):
    for vit_node in list(graph.dag.nodes):
        if not (isinstance(vit_node, ComputeNode) and vit_node.layer_type == 'CustomMultiHeadAttention'):
            continue

        if getattr(vit_node, 'model_type', '') == 'bert' or getattr(config, 'model_type', '') == 'bert':
            _expand_bert_multi_head_attention(graph, vit_node)
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

        pcmm_type = 'pdmpcmm' if _is_diagonal_mat_pack() else 'parcpmm'
        transpose_type = 'pdmtranspose' if _is_diagonal_mat_pack() else 'partranspose'
        ccmm_type = 'pdmccmm' if _is_diagonal_mat_pack() else 'parccmm'

        q_node = ComputeNode(f'{base_id}_q_layer', pcmm_type, 1, 1)
        q_node.path = getattr(vit_node, 'q_weight_path', f'{base_id}.q.weight')
        q_node.bias_path = getattr(vit_node, 'q_bias_path', '')
        q_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        k_node = ComputeNode(f'{base_id}_k_layer', pcmm_type, 1, 1)
        k_node.path = getattr(vit_node, 'k_weight_path', f'{base_id}.k.weight')
        k_node.bias_path = getattr(vit_node, 'k_bias_path', '')
        k_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        v_node = ComputeNode(f'{base_id}_v_layer', pcmm_type, 1, 1)
        v_node.path = getattr(vit_node, 'v_weight_path', f'{base_id}.v.weight')
        v_node.bias_path = getattr(vit_node, 'v_bias_path', '')
        v_node.weight_shape = [n, config.base_feat_dim] if config.base_feat_dim > 0 else [n, n]
        kt_node = ComputeNode(f'{base_id}_kt_layer', transpose_type, 1, 1)
        qkt_node = ComputeNode(f'{base_id}_qkt_layer', ccmm_type, 1, 1)
        poly_node = ComputeNode(f'{base_id}_poly', _pack_pcm_type('pcmpoly'), 1, 1)
        poly_node.path = getattr(vit_node, 'poly_weight_path', f'{base_id}.poly.weight')
        poly_node.coeffs_path = poly_node.path
        poly_node.gamma_path = getattr(vit_node, 'gamma_path', f'{base_id}.gamma')
        poly_node.order = getattr(vit_node, 'poly_order', 4)
        qktv_node = ComputeNode(f'{base_id}_qktv_layer', ccmm_type, 1, 1)
        out_node = ComputeNode(f'{base_id}_out', pcmm_type, 1, 1)
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

        def add_ccmm_inputs(lhs: FeatureNode, rhs: FeatureNode, node: ComputeNode):
            if node.layer_type == 'pdmccmm':
                graph.dag.add_edge(lhs, node, input_index=1)
                graph.dag.add_edge(rhs, node, input_index=0)
            else:
                graph.dag.add_edge(lhs, node, input_index=0)
                graph.dag.add_edge(rhs, node, input_index=1)

        add_ccmm_inputs(q, kt, qkt_node)
        graph.dag.add_edge(qkt_node, qkt)

        graph.dag.add_edge(qkt, poly_node)
        graph.dag.add_edge(poly_node, qkt_poly)

        add_ccmm_inputs(qkt_poly, v, qktv_node)
        graph.dag.add_edge(qktv_node, qktv)

        graph.dag.add_edge(qktv, out_node)
        graph.dag.add_edge(out_node, out)


def _expand_custom_gelu(graph: LayerAbstractGraph, node: ComputeNode):
    preds = list(graph.dag.predecessors(node))
    succs = list(graph.dag.successors(node))
    if len(preds) != 1 or len(succs) != 1:
        raise ValueError(
            f'CustomGELU node {node.layer_id} must have exactly 1 input and 1 output, '
            f'got {len(preds)} inputs and {len(succs)} outputs'
        )
    x_in: FeatureNode = preds[0]
    out: FeatureNode = succs[0]
    base_id = node.layer_id

    scale = _float_attr(node, 'scale', 64.0)
    f2_input_scale = _float_attr(node, 'f2_input_scale', 0.5)
    f3_input_scale = _float_attr(node, 'f3_input_scale', 0.5)
    p1 = _parse_float_list(getattr(node, 'p1_coefficients', []))
    p2 = _parse_float_list(getattr(node, 'p2_coefficients', []))
    p3 = _parse_float_list(getattr(node, 'p3_coefficients', []))
    if not p1 or not p2 or not p3:
        raise ValueError(f'CustomGELU node {node.layer_id} is missing polynomial coefficients')

    input_edge_attrs = copy.deepcopy(graph.dag.edges[x_in, node])
    input_edge_attrs.pop('input_index', None)
    output_edge_attrs = copy.deepcopy(graph.dag.edges[node, out])
    graph.dag.remove_node(node)

    x_scaled = _append_unary_compute(
        graph,
        x_in,
        _make_scalar_gamma(f'{base_id}_gelu_input_scale', 1.0 / scale,
                           secret_source('calibration', f'{base_id}.scale')),
        f'{base_id}_gelu_x_scaled',
        1,
    )
    f1 = _append_unary_compute(
        graph,
        x_scaled,
        _make_upper_poly(f'{base_id}_gelu_poly1', p1),
        f'{base_id}_gelu_f1',
        PolyBase.compute_stockmeyer_level_cost(len(p1) - 1),
    )
    f1_scaled = _append_unary_compute(
        graph,
        f1,
        _make_scalar_gamma(f'{base_id}_gelu_f2_input_scale', 1.0 / f2_input_scale),
        f'{base_id}_gelu_f1_scaled',
        1,
    )
    half_tanh = _append_unary_compute(
        graph,
        f1_scaled,
        _make_upper_poly(f'{base_id}_gelu_poly2', p2),
        f'{base_id}_gelu_half_tanh',
        PolyBase.compute_stockmeyer_level_cost(len(p2) - 1),
    )
    half_tanh_scaled = _append_unary_compute(
        graph,
        half_tanh,
        _make_scalar_gamma(f'{base_id}_gelu_f3_input_scale', 1.0 / f3_input_scale),
        f'{base_id}_gelu_half_tanh_scaled',
        1,
    )
    refined = _append_unary_compute(
        graph,
        half_tanh_scaled,
        _make_upper_poly(f'{base_id}_gelu_poly3', p3),
        f'{base_id}_gelu_refined',
        PolyBase.compute_stockmeyer_level_cost(len(p3) - 1),
    )

    gelu_node = ComputeNode(f'{base_id}_gelu_mult_ct', _pack_pcm_type('pcmpolymultct'), 1, 1)
    graph.dag.add_node(gelu_node, **_compute_attrs(2, gelu_node.layer_id))
    graph.dag.add_edge(refined, gelu_node, input_index=0)
    graph.dag.add_edge(x_in, gelu_node, input_index=1, **input_edge_attrs)
    graph.dag.add_edge(gelu_node, out, **output_edge_attrs)


def _expand_custom_tanh(graph: LayerAbstractGraph, node: ComputeNode):
    preds = list(graph.dag.predecessors(node))
    succs = list(graph.dag.successors(node))
    if len(preds) != 1 or len(succs) != 1:
        raise ValueError(
            f'CustomTanh node {node.layer_id} must have exactly 1 input and 1 output, '
            f'got {len(preds)} inputs and {len(succs)} outputs'
        )
    x_in: FeatureNode = preds[0]
    out: FeatureNode = succs[0]
    base_id = node.layer_id

    scale = _float_attr(node, 'scale', 40.0)
    f1_input_scale = _float_attr(node, 'f1_input_scale', 1.0)
    f2_input_scale = _float_attr(node, 'f2_input_scale', 1.0)
    p1 = _parse_float_list(getattr(node, 'p1_coefficients', []))
    p2 = _parse_float_list(getattr(node, 'p2_coefficients', []))
    if not p1 or not p2:
        raise ValueError(f'CustomTanh node {node.layer_id} is missing polynomial coefficients')

    output_edge_attrs = copy.deepcopy(graph.dag.edges[node, out])
    graph.dag.remove_node(node)

    x_scaled = _append_unary_compute(
        graph,
        x_in,
        _make_scalar_gamma(f'{base_id}_tanh_input_scale', 1.0 / (scale * f1_input_scale),
                           secret_source('calibration', f'{base_id}.scale')),
        f'{base_id}_tanh_x_scaled',
        1,
    )
    f1 = _append_unary_compute(
        graph,
        x_scaled,
        _make_upper_poly(f'{base_id}_tanh_poly1', p1),
        f'{base_id}_tanh_f1',
        PolyBase.compute_stockmeyer_level_cost(len(p1) - 1),
    )
    f1_scaled = _append_unary_compute(
        graph,
        f1,
        _make_scalar_gamma(f'{base_id}_tanh_f2_input_scale', 1.0 / f2_input_scale),
        f'{base_id}_tanh_f1_scaled',
        1,
    )
    tanh_node = _make_upper_poly(f'{base_id}_tanh_poly2', p2)
    graph.dag.add_node(
        tanh_node,
        **_compute_attrs(PolyBase.compute_stockmeyer_level_cost(len(p2) - 1), tanh_node.layer_id),
    )
    graph.dag.add_edge(f1_scaled, tanh_node)
    graph.dag.add_edge(tanh_node, out, **output_edge_attrs)


def expand_bert_custom_poly_functions(graph: LayerAbstractGraph):
    if getattr(config, 'model_type', '') != 'bert':
        return
    for node in list(graph.dag.nodes):
        if not isinstance(node, ComputeNode):
            continue
        if node.layer_type == 'CustomGELU':
            _expand_custom_gelu(graph, node)
        elif node.layer_type == 'CustomTanh':
            _expand_custom_tanh(graph, node)


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

        poly_node = ComputeNode(base_id, _pack_pcm_type('pcmpoly'), node.channel_input, node.channel_output)
        poly_node.depth = node.depth
        poly_node.path = coeffs_path or running_max_path
        poly_node.running_max_path = running_max_path
        poly_node.gamma_path = gamma_path
        poly_node.coeffs_path = coeffs_path
        poly_node.order = order
        poly_node.parameter_sources = {'coeff_pt': secret_source(
            'derived', running_max_path, gamma_path, coeffs_path)}

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
        weight_path = ln_node.weight_path
        bias_path = ln_node.bias_path
        if getattr(config, 'model_type', '') == 'bert':
            layernorm_param_attrs = _bert_layernorm_runtime_attrs(ln_node, n_iter)
            epsilon = layernorm_param_attrs['epsilon']
            node_n_iter = int(layernorm_param_attrs['num_iters'])
        else:
            epsilon = ln_node.epsilon
            node_n_iter = int(getattr(ln_node, 'num_iters', n_iter))
            layernorm_param_attrs = {
                'epsilon': epsilon,
                'inv_std_scale': getattr(ln_node, 'inv_std_scale', config.layernorm_inv_std_scale),
                'inv_var_scale': getattr(ln_node, 'inv_var_scale', config.layernorm_inv_var_scale),
                'c0': getattr(ln_node, 'c0', config.layernorm_c0),
                'c1': getattr(ln_node, 'c1', config.layernorm_c1),
                'c2': getattr(ln_node, 'c2', config.layernorm_c2),
                'num_iters': node_n_iter,
            }

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
        y_nodes = [y0] + [make_feature(f'{base_id}_y{i + 1}') for i in range(node_n_iter)]

        # Sub-compute nodes
        stats_type = _pack_pcm_type('pcmstats')
        center_type = _pack_pcm_type('pcmcenter')
        init_type = _pack_pcm_type('pcminit')
        gs_type = _pack_pcm_type('pcmgs')
        affine_type = _pack_pcm_type('pcmaffine')

        pcmstats = ComputeNode(f'{base_id}_{stats_type}', stats_type, 1, 1)
        pcmstats.epsilon = epsilon
        pcmcenter = ComputeNode(f'{base_id}_{center_type}', center_type, 1, 1)
        pcminit = ComputeNode(f'{base_id}_{init_type}', init_type, 1, 1)
        pcmgs_nodes = [ComputeNode(f'{base_id}_{gs_type}_{i}', gs_type, 1, 1) for i in range(node_n_iter)]
        pcmaffine = ComputeNode(f'{base_id}_{affine_type}', affine_type, 1, 1)
        pcmaffine.weight_path = weight_path
        pcmaffine.bias_path = bias_path
        for stage_node in (pcmstats, pcmcenter, pcminit, *pcmgs_nodes, pcmaffine):
            for attr_key, attr_value in layernorm_param_attrs.items():
                setattr(stage_node, attr_key, attr_value)

        pcmstats.parameter_sources = {
            f'pt:pt_idx={idx}': secret_source('calibration', f'{base_id}.variance_range')
            for idx in (2, 3)}
        pcminit.parameter_sources = {
            f'pt:pt_idx={idx}': public_source('approximation_config',
                {f'c{idx}': float(layernorm_param_attrs[f'c{idx}'])}) for idx in range(3)}
        pcmaffine.parameter_sources = {
            'pt:pt_idx=0': secret_source('derived', weight_path, f'{base_id}.variance_range'),
            'pt:pt_idx=1': secret_source('onnx_initializer', bias_path)}

        # Remove the original layernorm node (keeps x_in and out in the graph)
        graph.dag.remove_node(ln_node)

        stats_level_cost = 3 if stats_type == 'pdmstats' else 4
        # 1a. x_in → stats → a  (computes mean/variance stats)
        graph.dag.add_node(pcmstats, **c_attrs(stats_level_cost, pcmstats.layer_id))
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
    """Set K attribute on pcmgamma/pcmpoly nodes and their pdm aliases."""
    base_feat_dim = config.n_heads * config.matmul_block_size
    gamma_types = ('pcmgamma', 'pdmgamma')
    poly_types = ('pcmpoly', 'pdmpoly', 'pcmgeneralpoly', 'pcmpolymultct', 'pcmctmul', 'pcminviter')
    for node in graph.dag.nodes:
        if not isinstance(node, ComputeNode):
            continue
        if node.layer_type not in gamma_types + poly_types:
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
        if not isinstance(node, ComputeNode) or node.layer_type not in gamma_types + poly_types:
            continue
        preds = list(graph.dag.predecessors(node))
        if not preds:
            continue
        prev_compute = get_prev_compute(preds[0])
        if (
            node.layer_type in gamma_types
            and prev_compute is not None
            and prev_compute.layer_type in ('parccmm', 'pdmccmm')
        ):
            node.K = 1
        elif node.layer_type in poly_types and prev_compute is not None:
            if prev_compute.layer_type in ('parccmm', 'pdmccmm'):
                node.K = 1
            elif prev_compute.layer_type in gamma_types:
                prev_preds = list(graph.dag.predecessors(prev_compute))
                if prev_preds:
                    prev_prev_compute = get_prev_compute(prev_preds[0])
                    if prev_prev_compute is not None and prev_prev_compute.layer_type in ('parccmm', 'pdmccmm'):
                        node.K = 1
