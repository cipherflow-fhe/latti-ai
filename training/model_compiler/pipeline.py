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


from pathlib import Path
import copy

import components
from components import (
    ComputeNode,
    FeatureNode,
    LayerAbstractGraph,
    config,
    PN13QP218,
    PN14QP438,
    PN15QP880,
    PN16QP1761,
    N16QP1546H192H32,
)
import processor
from processor import *
from graph_partition_dp import *


def prepare_graph(raw_graph: LayerAbstractGraph) -> LayerAbstractGraph:
    """
    Prepare graph for compilation (common preparation steps)

    Args:
        raw_graph: Raw LayerAbstractGraph loaded from json

    Returns:
        Prepared LayerAbstractGraph
    """
    pt_graph = copy.deepcopy(raw_graph)

    substitute_layers_for_btp(pt_graph)
    # transforms.init_levels(pt_graph)
    # update_shape_for_btp(pt_graph)
    # update_skip_for_btp(pt_graph)
    # update_level_cost_for_btp(pt_graph)
    set_is_adaptive_avgpool(pt_graph)
    transforms.expand_multi_head_attention(pt_graph)
    transforms.expand_layer_norm(pt_graph)
    transforms.expand_poly_act_rn(pt_graph)
    transforms.expand_parcpmm_add_pt(pt_graph)
    transforms.split_upsampling_layers(pt_graph)
    transforms.infer_shapes_skips_and_pack_num(pt_graph)
    transforms.set_pcm_K(pt_graph)
    transforms.combine_convs_with_upsamples(pt_graph)
    transforms.process_polyact(pt_graph)
    transforms.set_level_costs(pt_graph)

    transforms.absorb_scale(pt_graph)

    return pt_graph


def set_fhe_param(params):
    config.fhe_param = copy.deepcopy(params)
    if config.set_btp_scale is not None:
        config.fhe_param.max_level -= 1


def set_block_shape(params, raw_graph: LayerAbstractGraph):
    """Set config.block_shape based on the input graph's leading feature node shape and N.

    Rules:
      (1) If leading node is not 2D, use sqrt(N/2) as a square default.
      (2) If shape0 * shape1 <= N / 2, block_shape = [shape0, shape1]
      (3) Otherwise, divide both shape0 and shape1 by 2, 4, 8, 16, ...,
          until shape0 * shape1 < N / 2
    """
    slot_num = params.poly_modulus_degree // 2
    leading_nodes = raw_graph.get_leading_feature_nodes()
    if not leading_nodes or len(leading_nodes[0].shape) == 0:
        side = 1 << (slot_num.bit_length() // 2)
        config.block_shape = [side, side]
        return
    leading_shape = leading_nodes[0].shape
    block = list(leading_shape)
    slot_num = params.poly_modulus_degree // 2
    # threshold = N / 2
    import math

    while math.prod(block) > slot_num:
        block = [s // 2 for s in block]

    config.block_shape = block
    print('block_shape=', config.block_shape)


def try_no_btp(raw_graph: LayerAbstractGraph) -> tuple[bool, LayerAbstractGraph | None, float]:
    """
    Try no-BTP mode compilation with prepared graph

    Args:
        raw_graph: Raw LayerAbstractGraph

    Returns:
        (succeeded, graph, score): succeeded=True if no-BTP succeeded, graph and score are set on success
    """
    print('Step 2: Trying no-BTP mode...')

    no_btp_params = [PN13QP218, PN14QP438, PN15QP880, PN16QP1761]

    for params in no_btp_params:
        set_fhe_param(params)
        set_block_shape(config.fhe_param, raw_graph)
        print(f'Trying FheParam {config.fhe_param.name}')

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        result = process_with_no_btp(pt_graph)

        # (3) Post-process
        if result is not None:
            print(f'Success! Using FheParam {config.fhe_param.name}')
            print('✓ No-BTP mode succeeded! Saving results...')
            restore_node_attributes(result.dag)
            result = post_process(result)
            return True, result, 0.0
        else:
            print(f'Level exceeded with POLY_N={config.fhe_param.poly_modulus_degree}, trying next level...')

    print(f'Warning: Even with POLY_N=65536, level still exceeds limit!')
    print('✗ No-BTP mode failed, switching to BTP mode...')
    return False, None, float('inf')


def process_with_no_btp(graph: LayerAbstractGraph):
    return reset_level_and_check_level(graph)


def try_btp(
    num_experiments: int,
    raw_graph: LayerAbstractGraph,
    temperature: float,
    num_workers: int,
) -> tuple[bool, LayerAbstractGraph | None, float]:
    btp_param_list = [N16QP1546H192H32]
    valid_results = []
    for params in btp_param_list:
        set_fhe_param(params)
        set_block_shape(config.fhe_param, raw_graph)

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        graph, score = run_btp_compilation(num_experiments, pt_graph, temperature, num_workers)

        # (3) Post-process
        if graph is not None:
            graph = post_process(graph)
            valid_results.append((score, graph))

    if not valid_results:
        return False, None, float('inf')

    best_score, best_graph = min(valid_results, key=lambda x: x[0])
    return True, best_graph, best_score


def run_btp_compilation(
    num_experiments: int,
    pt_graph: LayerAbstractGraph,
    temperature: float,
    num_workers: int,
) -> tuple[LayerAbstractGraph | None, float]:
    """
    Run BTP mode parallel compilation with prepared graph

    Args:
        num_experiments: Number of parallel compilation runs
        temperature: Temperature parameter for randomization
        pt_graph: Prepared graph for BTP compilation
        num_workers: Number of parallel worker processes

    Returns:
        (best_graph, best_score): best_graph is None if all runs failed
    """
    print(f'Step 4: Starting DP compilation of pt_graph with temperature={temperature}')

    # Find the best result
    score, graph = run_single_compile((pt_graph, temperature))

    print(f'\n=== Results ===')
    print(f'Final score: {score}')
    return graph, score


def post_process(graph: LayerAbstractGraph):
    slot_num = config.fhe_param.poly_modulus_degree / 2
    for node in list(graph.dag.nodes):
        if isinstance(node, ComputeNode):
            node.up_scale_str = list()
            node.down_scale_str = list()
            transforms.populate_pack_num(graph.dag, node, slot_num)
            if node.layer_type == 'reshape':
                f_node = list(graph.dag.successors(node))[0]
                if graph.dag.out_degree(f_node) == 0:
                    graph.dag.remove_node(f_node)
                    graph.dag.remove_node(node)

    transforms.set_graph_scale(graph)
    process_levels(graph)

    return graph


def _unique_graph_node_id(graph: LayerAbstractGraph, base_id: str, attr_name: str) -> str:
    existing_ids = {getattr(node, attr_name, None) for node in graph.dag.nodes}
    node_id = base_id
    idx = 1
    while node_id in existing_ids:
        node_id = f'{base_id}_{idx}'
        idx += 1
    return node_id


def _clone_feature_node(feature: FeatureNode, node_id: str) -> FeatureNode:
    cloned = copy.deepcopy(feature)
    cloned.node_id = node_id
    return cloned


def insert_btp_scale_gamma_layers(graph: LayerAbstractGraph):
    if config.set_btp_scale is None:
        return

    btp_scale = float(config.set_btp_scale)
    if btp_scale == 0:
        raise ValueError('set_btp_scale cannot be 0 when inserting BTP scale gamma layers')

    dag = graph.dag
    for btp_node in list(dag.nodes):
        if not isinstance(btp_node, ComputeNode) or btp_node.layer_type != 'bootstrapping':
            continue

        preds = list(dag.predecessors(btp_node))
        succs = list(dag.successors(btp_node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(f'Expected bootstrapping node {btp_node.layer_id} to have one input and one output')

        pred_feature = preds[0]
        succ_feature = succs[0]

        pre_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_pre_pcmgamma', 'layer_id')
        post_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_post_pcmgamma', 'layer_id')
        pre_feature_id = _unique_graph_node_id(graph, f'{pre_gamma_id}_output', 'node_id')
        post_gamma_input_feature_id = _unique_graph_node_id(graph, f'{post_gamma_id}_input', 'node_id')

        pre_gamma = ComputeNode(pre_gamma_id, 'pcmgamma', btp_node.channel_input, btp_node.channel_input)
        pre_gamma.depth = btp_node.depth
        pre_gamma.path = f'{pre_gamma_id}.weight'
        pre_gamma.btp_scale = btp_scale

        post_gamma = ComputeNode(post_gamma_id, 'pcmgamma', btp_node.channel_output, btp_node.channel_output)
        post_gamma.depth = btp_node.depth
        post_gamma.path = f'{post_gamma_id}.weight'
        post_gamma.btp_scale = 1 / btp_scale

        pre_feature = _clone_feature_node(pred_feature, pre_feature_id)
        post_gamma_input_feature = _clone_feature_node(succ_feature, post_gamma_input_feature_id)

        pred_to_btp_attrs = copy.deepcopy(dag.edges[pred_feature, btp_node])
        btp_to_succ_attrs = copy.deepcopy(dag.edges[btp_node, succ_feature])
        pre_feature_attrs = copy.deepcopy(dag.nodes[pred_feature])
        pre_feature_attrs['name'] = pre_feature.node_id
        pre_feature_attrs['level'] = 0
        post_gamma_input_feature_attrs = copy.deepcopy(dag.nodes[succ_feature])
        post_gamma_input_feature_attrs['name'] = post_gamma_input_feature.node_id
        post_gamma_input_feature_attrs['level'] = config.fhe_param.max_level + 1

        dag.nodes[btp_node]['level_cost'] = 9

        dag.remove_edge(pred_feature, btp_node)
        dag.remove_edge(btp_node, succ_feature)

        dag.add_node(pre_gamma, name=pre_gamma_id, level_cost=1)
        dag.add_node(pre_feature, **pre_feature_attrs)
        dag.add_edge(pred_feature, pre_gamma, **pred_to_btp_attrs)
        dag.add_edge(pre_gamma, pre_feature)
        dag.add_edge(pre_feature, btp_node, **pred_to_btp_attrs)

        dag.add_node(post_gamma, name=post_gamma_id, level_cost=1)
        dag.add_node(post_gamma_input_feature, **post_gamma_input_feature_attrs)
        dag.add_edge(btp_node, post_gamma_input_feature, **btp_to_succ_attrs)
        dag.add_edge(post_gamma_input_feature, post_gamma)
        dag.add_edge(post_gamma, succ_feature)


def _fuse_pcmgamma_attrs_into_parcpmm(pcmgamma_node: ComputeNode, parcpmm_node: ComputeNode):
    def _value_or_empty(attr_name: str):
        value = getattr(pcmgamma_node, attr_name, '')
        return '' if value is None else value

    fuse_gama_info = {
        'weight_path': _value_or_empty('path'),
        'K': _value_or_empty('K'),
        'gamma_path': _value_or_empty('gamma_path'),
        'running_max_path': _value_or_empty('running_max_path'),
        'btp_scale': _value_or_empty('btp_scale'),
    }

    existing_fuse_gama_info = getattr(parcpmm_node, 'fuse_gama_info', None)
    if existing_fuse_gama_info is not None and existing_fuse_gama_info != fuse_gama_info:
        raise ValueError(f'Cannot fuse multiple pcmgamma layers into parcpmm node {parcpmm_node.layer_id}')
    parcpmm_node.fuse_gama_info = fuse_gama_info


def _try_fuse_pcmgamma_before_parcpmm(dag, pcmgamma_node: ComputeNode) -> bool:
    preds = list(dag.predecessors(pcmgamma_node))
    succs = list(dag.successors(pcmgamma_node))
    if len(preds) != 1 or len(succs) != 1:
        return False

    input_feature = preds[0]
    mid_feature = succs[0]
    if not isinstance(input_feature, FeatureNode) or not isinstance(mid_feature, FeatureNode):
        return False
    if dag.in_degree(mid_feature) != 1 or dag.out_degree(mid_feature) != 1:
        return False

    parcpmm_node = next(iter(dag.successors(mid_feature)), None)
    if not isinstance(parcpmm_node, ComputeNode) or parcpmm_node.layer_type != 'parcpmm':
        return False

    edge_attrs = copy.deepcopy(dag.edges[mid_feature, parcpmm_node])
    _fuse_pcmgamma_attrs_into_parcpmm(pcmgamma_node, parcpmm_node)
    dag.remove_node(pcmgamma_node)
    dag.remove_node(mid_feature)
    dag.add_edge(input_feature, parcpmm_node, **edge_attrs)
    return True


def _try_fuse_pcmgamma_after_parcpmm(dag, pcmgamma_node: ComputeNode) -> bool:
    preds = list(dag.predecessors(pcmgamma_node))
    succs = list(dag.successors(pcmgamma_node))
    if len(preds) != 1 or len(succs) != 1:
        return False

    mid_feature = preds[0]
    output_feature = succs[0]
    if not isinstance(mid_feature, FeatureNode) or not isinstance(output_feature, FeatureNode):
        return False
    if dag.in_degree(mid_feature) != 1 or dag.out_degree(mid_feature) != 1:
        return False

    parcpmm_node = next(iter(dag.predecessors(mid_feature)), None)
    if not isinstance(parcpmm_node, ComputeNode) or parcpmm_node.layer_type != 'parcpmm':
        return False

    edge_attrs = copy.deepcopy(dag.edges[parcpmm_node, mid_feature])
    _fuse_pcmgamma_attrs_into_parcpmm(pcmgamma_node, parcpmm_node)
    dag.remove_node(pcmgamma_node)
    dag.remove_node(mid_feature)
    dag.add_edge(parcpmm_node, output_feature, **edge_attrs)
    return True


def fuse_pcmgamma_parcpmm_layers(graph: LayerAbstractGraph):
    dag = graph.dag
    changed = True
    while changed:
        changed = False
        for node in list(dag.nodes):
            if not isinstance(node, ComputeNode) or node.layer_type != 'pcmgamma':
                continue
            if _try_fuse_pcmgamma_before_parcpmm(dag, node) or _try_fuse_pcmgamma_after_parcpmm(dag, node):
                changed = True
                break


def recompute_final_level(graph: LayerAbstractGraph):
    dag = graph.dag
    min_feature_level = get_min_feature_level()
    reset_layer_types = {'bootstrapping', 'mpc_refresh'}
    anchors: dict[FeatureNode, int] = {}

    def set_anchor(feature: FeatureNode, level: int):
        level = int(level)
        existing_level = anchors.get(feature)
        if existing_level is not None and existing_level != level:
            raise ValueError(
                f'Conflicting fixed levels for feature {feature.node_id}: {existing_level} vs {level}'
            )
        anchors[feature] = level

    for node in dag.nodes:
        if not isinstance(node, ComputeNode) or node.layer_type not in reset_layer_types:
            continue
        preds = [pred for pred in dag.predecessors(node) if isinstance(pred, FeatureNode)]
        succs = [succ for succ in dag.successors(node) if isinstance(succ, FeatureNode)]
        for feature in preds + succs:
            if 'level' not in dag.nodes[feature]:
                raise ValueError(f'Feature {feature.node_id} missing level before final level recompute')
            set_anchor(feature, dag.nodes[feature]['level'])

    for node in dag.nodes:
        if not isinstance(node, FeatureNode) or dag.out_degree(node) != 0 or node in anchors:
            continue
        set_anchor(node, min_feature_level)

    feature_levels: dict[FeatureNode, int] = {}
    for node in reversed(list(nx.topological_sort(dag))):
        if not isinstance(node, FeatureNode):
            continue

        regular_consumers = [
            consumer
            for consumer in dag.successors(node)
            if isinstance(consumer, ComputeNode) and consumer.layer_type not in reset_layer_types
        ]
        downstream_req = min_feature_level if dag.out_degree(node) == 0 or regular_consumers else 0

        for consumer in regular_consumers:
            output_features = [succ for succ in dag.successors(consumer) if isinstance(succ, FeatureNode)]
            if not output_features:
                continue
            missing_outputs = [feature.node_id for feature in output_features if feature not in feature_levels]
            if missing_outputs:
                raise ValueError(
                    f'Cannot recompute level for {node.node_id}: outputs of {consumer.layer_id} not ready: '
                    f'{missing_outputs}'
                )
            output_level = max(feature_levels[feature] for feature in output_features)
            downstream_req = max(downstream_req, output_level + dag.nodes[consumer].get('level_cost', 0))

        if node in anchors:
            anchor_level = anchors[node]
            if downstream_req > anchor_level:
                raise ValueError(
                    f'Fixed level of feature {node.node_id} is {anchor_level}, but downstream requires {downstream_req}'
                )
            feature_levels[node] = anchor_level
        else:
            feature_levels[node] = downstream_req

    max_allowed_level = config.fhe_param.max_level + 1
    for feature, level in feature_levels.items():
        if level < 0 or level > max_allowed_level:
            raise ValueError(
                f'Final level of feature {feature.node_id} is out of range: {level}, max allowed {max_allowed_level}'
            )
        dag.nodes[feature]['level'] = int(level)


def dump_graph(
    graph: LayerAbstractGraph,
    output_dir: Path,
    score: float,
    use_btp: bool,
):
    task_dir = output_dir / 'task'
    server_dir = task_dir / 'server'
    client_dir = task_dir / 'client'
    ergs_dir = server_dir

    ergs_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    erg0_path = ergs_dir / 'nn_layers_ct_0.json'
    insert_btp_scale_gamma_layers(graph)
    fuse_pcmgamma_parcpmm_layers(graph)
    recompute_final_level(graph)
    transforms.insert_drop_level_layers(graph)
    graph.to_json(dict(), str(erg0_path), score=score)

    if use_btp:
        graph_to_task_config(graph, str(server_dir))
    else:
        graph_to_task_config(graph, str(server_dir), False)

    server_task_config = server_dir / 'task_config.json'
    client_task_config = client_dir / 'task_config.json'
    if server_task_config.exists():
        shutil.copy(str(server_task_config), str(client_task_config))

    ckks_param = {'param0': {**config.fhe_param.to_dict()}}

    with open(server_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)

    with open(client_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)


import os


def run_pipeline(
    num_experiments: int,
    input_file_path: Path,
    output_dir: Path,
    temperature: float = 0.0,
    num_workers: int = os.cpu_count(),
    style: str | None = None,
    graph_type: str | None = None,
    is_use_btp: bool = False,
    n_heads: int | None = None,
    head_dim: int | None = None,
    matmul_block_size: int | None = None,
    set_btp_scale: float | None = None,
):
    """
    Run multiple compilations in parallel and select the best result

    This is the main entry point for compilation. It tries no-BTP mode first,
    and falls back to BTP mode if needed.

    Args:
        num_experiments: Number of parallel compilation runs
        input_file_path: Input pt.json file path
        output_dir: Output directory (will contain erg0.json, task_config.json)
        temperature: Temperature parameter for randomization
        num_workers: Number of parallel worker processes
        style: Computation style (STYLE)
        graph_type: Graph type (GRAPH_TYPE)
        set_btp_scale: if not None, wrap BTP with pcmgamma scales and enable special level handling
    """
    if style is not None:
        config.style = style
    if graph_type is not None:
        config.graph_type = graph_type
    if n_heads is not None:
        config.n_heads = n_heads
    if head_dim is not None:
        config.head_dim = head_dim
    if matmul_block_size is not None:
        config.matmul_block_size = matmul_block_size
    config.set_btp_scale = set_btp_scale
    print(
        f'Configuration initialized: STYLE={config.style}, GRAPH_TYPE={config.graph_type}, '
        f'N_HEADS={config.n_heads}, HEAD_DIM={config.head_dim}, MATMUL_BLOCK_SIZE={config.matmul_block_size}, '
        f'SET_BTP_SCALE={config.set_btp_scale}'
    )

    raw_graph = LayerAbstractGraph.from_json(input_file_path)

    if not is_use_btp:
        use_btp = False
        succeeded, graph, score = try_no_btp(raw_graph)
        if not succeeded:
            use_btp = True
            succeeded, graph, score = try_btp(num_experiments, raw_graph, temperature, num_workers)
            if not succeeded:
                raise ValueError('Compilation failed.')
    else:
        use_btp = True
        succeeded, graph, score = try_btp(num_experiments, raw_graph, temperature, num_workers)
        if not succeeded:
            raise ValueError('Compilation failed.')
    dump_graph(graph, output_dir, score, use_btp=use_btp)

    return graph, score
