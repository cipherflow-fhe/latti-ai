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
import shutil
import time

import networkx as nx
import numpy as np

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
from mpc_graph_partition_dp import compile_graph_mpc_skip_aware
import transforms

MAT_PACK_STYLES = {'', 'par_block_col_major', 'par_diagonal_pack'}
PCM_TO_PDM_TYPES = {
    'pcmgamma': 'pdmgamma',
}


def _pack_pcm_type(layer_type: str) -> str:
    if getattr(config, 'mat_pack_style', '') == 'par_diagonal_pack':
        return PCM_TO_PDM_TYPES.get(layer_type, layer_type)
    return layer_type


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


def get_restore_param_candidates():
    if is_mpc_flow():
        return [PN13QP218]
        # return [PN14QP438]
        # [N16QP1546H192H32]
    return [N16QP1546H192H32]


def try_btp(
    num_experiments: int,
    raw_graph: LayerAbstractGraph,
    temperature: float,
    num_workers: int,
    mpc_skip_max_states: int = 4096,
    enable_score_cache: bool = True,
) -> tuple[bool, LayerAbstractGraph | None, float]:
    btp_param_list = get_restore_param_candidates()
    valid_results = []
    for params in btp_param_list:
        set_fhe_param(params)
        set_block_shape(config.fhe_param, raw_graph)
        print(f'Trying FheParam {config.fhe_param.name}')

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        if is_mpc_flow():
            graph, score = run_mpc_skip_compilation(pt_graph, mpc_skip_max_states)
        else:
            graph, score = run_btp_compilation(num_experiments, pt_graph, temperature, num_workers, enable_score_cache=enable_score_cache)

        # (3) Post-process
        if graph is not None:
            if is_mpc_flow():
                run_mpc_metadata_pass(graph)
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
    enable_score_cache: bool = True,
) -> tuple[LayerAbstractGraph | None, float]:
    """
    Run BTP mode parallel compilation with prepared graph

    Args:
        num_experiments: Number of parallel compilation runs
        temperature: Temperature parameter for randomization
        pt_graph: Prepared graph for BTP compilation
        num_workers: Number of parallel worker processes
        enable_score_cache: If True, cache per-compute score in DP compilation

    Returns:
        (best_graph, best_score): best_graph is None if all runs failed
    """
    print(f'Step 4: Starting DP compilation of pt_graph with temperature={temperature}')
    dp_start_time = time.perf_counter()

    # Find the best result
    score, graph = run_single_compile((pt_graph, temperature, enable_score_cache))

    dp_elapsed_time = time.perf_counter() - dp_start_time
    print(f'DP compilation time: {dp_elapsed_time:.3f}s')
    print(f'\n=== Results ===')
    print(f'Final score: {score}')
    return graph, score


def run_mpc_skip_compilation(
    pt_graph: LayerAbstractGraph,
    max_states_per_frontier: int,
) -> tuple[LayerAbstractGraph | None, float]:
    print(
        'Step 4: Starting skip-aware MPC DP compilation '
        f'with max_states_per_frontier={max_states_per_frontier}'
    )
    score, graph = compile_graph_mpc_skip_aware(
        pt_graph,
        max_states_per_frontier=max_states_per_frontier,
    )
    print(f'\n=== Results ===')
    print(f'Final score: {score}')
    return graph, score


def run_mpc_metadata_pass(graph: LayerAbstractGraph):
    transforms.infer_shapes_skips_and_pack_num(graph)
    transforms.set_level_costs(graph, trust_adaptive_avgpool_attr=True)


def run_mpc_post_dp_passes(graph: LayerAbstractGraph):
    update_btp_to_mpc_refresh(graph)
    run_mpc_metadata_pass(graph)
    processor.change_skip_for_graph(graph)


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

        gamma_type = _pack_pcm_type('pcmgamma')
        pre_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_pre_{gamma_type}', 'layer_id')
        post_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_post_{gamma_type}', 'layer_id')
        pre_feature_id = _unique_graph_node_id(graph, f'{pre_gamma_id}_output', 'node_id')
        post_gamma_input_feature_id = _unique_graph_node_id(graph, f'{post_gamma_id}_input', 'node_id')

        pre_gamma = ComputeNode(pre_gamma_id, gamma_type, btp_node.channel_input, btp_node.channel_input)
        pre_gamma.depth = btp_node.depth
        pre_gamma.path = f'{pre_gamma_id}.weight'
        pre_gamma.btp_scale = btp_scale

        post_gamma = ComputeNode(post_gamma_id, gamma_type, btp_node.channel_output, btp_node.channel_output)
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


PCMGAMMA_ABSORB_TARGETS = {'pcmpoly', 'pdmpoly', 'parcpmm', 'pdmpcmm'}
PCMGAMMA_PASS_THROUGH_TYPES = {'pcmgamma', 'pdmgamma'}


def _pcmgamma_fuse_info(pcmgamma_node: ComputeNode, direction: str) -> dict:
    def _value_or_empty(attr_name: str):
        value = getattr(pcmgamma_node, attr_name, '')
        return '' if value is None else value

    return {
        'weight_path': _value_or_empty('path'),
        'K': _value_or_empty('K'),
        'gamma_path': _value_or_empty('gamma_path'),
        'running_max_path': _value_or_empty('running_max_path'),
        'btp_scale': _value_or_empty('btp_scale'),
        'direction': direction,
    }


def _append_fuse_gama_info(node: ComputeNode, fuse_gama_info: dict):
    existing_fuse_gama_info = getattr(node, 'fuse_gama_info', None)
    if existing_fuse_gama_info is None:
        node.fuse_gama_info = [fuse_gama_info]
    elif isinstance(existing_fuse_gama_info, list):
        existing_fuse_gama_info.append(fuse_gama_info)
    else:
        node.fuse_gama_info = [existing_fuse_gama_info, fuse_gama_info]


def _fuse_pcmgamma_attrs_into_parcpmm(pcmgamma_node: ComputeNode, parcpmm_node: ComputeNode, direction: str):
    _append_fuse_gama_info(parcpmm_node, _pcmgamma_fuse_info(pcmgamma_node, direction))


def _fuse_pcmgamma_attrs_into_pcmpoly(pcmgamma_node: ComputeNode, pcmpoly_node: ComputeNode, direction: str):
    _append_fuse_gama_info(pcmpoly_node, _pcmgamma_fuse_info(pcmgamma_node, direction))


def _next_node_on_single_path(dag, node, search_direction: str):
    if search_direction == 'up':
        if dag.in_degree(node) != 1:
            return None
        return next(dag.predecessors(node))
    if dag.out_degree(node) != 1:
        return None
    return next(dag.successors(node))


def _find_pcmgamma_absorb_target(dag, pcmgamma_node: ComputeNode, search_direction: str) -> ComputeNode | None:
    node = pcmgamma_node
    while True:
        node = _next_node_on_single_path(dag, node, search_direction)
        if node is None:
            return None

        if isinstance(node, FeatureNode):
            if dag.in_degree(node) != 1 or dag.out_degree(node) != 1:
                return None
            continue

        if not isinstance(node, ComputeNode):
            return None
        if node.layer_type in PCMGAMMA_ABSORB_TARGETS:
            return node
        if node.layer_type not in PCMGAMMA_PASS_THROUGH_TYPES:
            return None


def _remove_pcmgamma_from_linear_path(dag, pcmgamma_node: ComputeNode, search_direction: str) -> bool:
    preds = list(dag.predecessors(pcmgamma_node))
    succs = list(dag.successors(pcmgamma_node))
    if len(preds) != 1 or len(succs) != 1:
        return False

    input_feature = preds[0]
    output_feature = succs[0]
    if not isinstance(input_feature, FeatureNode) or not isinstance(output_feature, FeatureNode):
        return False

    if search_direction == 'up':
        if dag.in_degree(input_feature) != 1 or dag.out_degree(input_feature) != 1:
            return False
        prev_compute = next(dag.predecessors(input_feature))
        if not isinstance(prev_compute, ComputeNode):
            return False
        edge_attrs = copy.deepcopy(dag.edges[prev_compute, input_feature])
        dag.remove_node(pcmgamma_node)
        dag.remove_node(input_feature)
        dag.add_edge(prev_compute, output_feature, **edge_attrs)
    else:
        if dag.in_degree(output_feature) != 1 or dag.out_degree(output_feature) != 1:
            return False
        next_compute = next(dag.successors(output_feature))
        if not isinstance(next_compute, ComputeNode):
            return False
        edge_attrs = copy.deepcopy(dag.edges[output_feature, next_compute])
        dag.remove_node(pcmgamma_node)
        dag.remove_node(output_feature)
        dag.add_edge(input_feature, next_compute, **edge_attrs)

    return True


def _absorb_pcmgamma_into_target(
    dag, pcmgamma_node: ComputeNode, target_node: ComputeNode, search_direction: str
) -> bool:
    direction = f'after_{target_node.layer_type}' if search_direction == 'up' else f'before_{target_node.layer_type}'
    if target_node.layer_type in ('parcpmm', 'pdmpcmm'):
        _fuse_pcmgamma_attrs_into_parcpmm(pcmgamma_node, target_node, direction)
    elif target_node.layer_type in ('pcmpoly', 'pdmpoly'):
        _fuse_pcmgamma_attrs_into_pcmpoly(pcmgamma_node, target_node, direction)
    else:
        return False
    return _remove_pcmgamma_from_linear_path(dag, pcmgamma_node, search_direction)


def _try_absorb_pcmgamma(dag, pcmgamma_node: ComputeNode) -> bool:
    for search_direction in ('up', 'down'):
        target_node = _find_pcmgamma_absorb_target(dag, pcmgamma_node, search_direction)
        if target_node is not None:
            return _absorb_pcmgamma_into_target(dag, pcmgamma_node, target_node, search_direction)
    return False


def absorb_pcmgamma_layers(graph: LayerAbstractGraph):
    dag = graph.dag
    changed = True
    while changed:
        changed = False
        for node in list(dag.nodes):
            if (
                isinstance(node, ComputeNode)
                and node.layer_type in PCMGAMMA_PASS_THROUGH_TYPES
                and _try_absorb_pcmgamma(dag, node)
            ):
                changed = True
                break


def fuse_pcmgamma_parcpmm_layers(graph: LayerAbstractGraph):
    absorb_pcmgamma_layers(graph)


def recompute_final_level(graph: LayerAbstractGraph):
    dag = graph.dag
    min_feature_level = get_min_feature_level()
    reset_layer_types = {'bootstrapping', 'mpc_refresh'}
    if config.graph_type == 'mpc_compute':
        reset_layer_types |= {'relu2d', 'polyact'}
    anchors: dict[FeatureNode, int] = {}

    def set_anchor(feature: FeatureNode, level: int):
        level = int(level)
        existing_level = anchors.get(feature)
        if existing_level is not None and existing_level != level:
            raise ValueError(f'Conflicting fixed levels for feature {feature.node_id}: {existing_level} vs {level}')
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


def _write_ckks_parameter(output_dir: Path, ckks_param: dict):
    with open(output_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)


def _subgraph_has_refresh_layer(subgraph: LayerAbstractGraph) -> bool:
    refresh_layer_types = {'bootstrapping', 'mpc_refresh'}
    if config.graph_type == 'mpc_compute':
        refresh_layer_types |= {'polyact', 'relu2d'}
    return any(isinstance(node, ComputeNode) and node.layer_type in refresh_layer_types for node in subgraph.dag.nodes)


def _rewrite_split_graph_io(ct_path: Path, inputs: list[FeatureNode], outputs: list[FeatureNode]) -> dict:
    with open(ct_path, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)
    graph_json['input_feature'] = [feature.node_id for feature in inputs]
    graph_json['output_feature'] = [feature.node_id for feature in outputs]
    with open(ct_path, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=4, ensure_ascii=False)
    return graph_json


def _rewrite_split_task_config_io(
    task_config_path: Path,
    graph_json: dict,
    inputs: list[FeatureNode],
    outputs: list[FeatureNode],
):
    with open(task_config_path, 'r', encoding='utf-8') as f:
        task_config = json.load(f)
    features = graph_json['feature']
    input_ids = [feature.node_id for feature in inputs]
    output_ids = [feature.node_id for feature in outputs]
    task_config['task_input_id'] = input_ids
    task_config['task_output_id'] = output_ids
    task_config['task_input_param'] = {feature_id: features[feature_id] for feature_id in input_ids}
    task_config['task_output_param'] = {feature_id: features[feature_id] for feature_id in output_ids}
    with open(task_config_path, 'w', encoding='utf-8') as f:
        json.dump(task_config, f, indent=4, ensure_ascii=False)


def dump_split_tasks(graph: LayerAbstractGraph, task_dir: Path) -> list[dict[str, str]]:
    splitter = transforms.GraphSplitSorter(graph)
    subgraphs = splitter.split_graph()
    sorted_subgraphs, _, _ = splitter.sort_graph(subgraphs)

    split_tasks_dir = task_dir / 'split_tasks'
    if split_tasks_dir.exists():
        shutil.rmtree(split_tasks_dir)
    split_tasks_dir.mkdir(parents=True, exist_ok=True)

    split_boundary_types = {'mpc_refresh'}
    if config.graph_type == 'mpc_compute':
        split_boundary_types |= {'polyact', 'relu2d'}
    last_mpc_refresh_subgraph_index = None
    for index, subgraph in enumerate(sorted_subgraphs):
        if any(
            isinstance(node, ComputeNode) and node.layer_type in split_boundary_types
            for node in subgraph.dag.nodes
        ):
            last_mpc_refresh_subgraph_index = index

    hybrid_pipeline = []
    for index, subgraph in enumerate(sorted_subgraphs):
        subtask_dir = split_tasks_dir / str(index)
        subtask_dir.mkdir(parents=True, exist_ok=True)
        ct_path = subtask_dir / 'ct.json'
        subgraph.to_json(
            dict(),
            str(ct_path),
            mark_last_mpc_refresh=index == last_mpc_refresh_subgraph_index,
        )
        inputs, outputs = splitter.find_input_and_output(subgraph)
        graph_json = _rewrite_split_graph_io(ct_path, inputs, outputs)
        subgraph_has_refresh = _subgraph_has_refresh_layer(subgraph)
        graph_to_task_config(subgraph, str(subtask_dir), use_btp=subgraph_has_refresh)
        _rewrite_split_task_config_io(subtask_dir / 'task_config.json', graph_json, inputs, outputs)

        mode = 'direct_layer' if subgraph_has_refresh else 'mega_lazy'
        pipeline_item = {
            'name': str(index),
            'mode': mode,
            'json': f'../split_tasks/{index}/ct.json',
        }
        if mode == 'mega_lazy':
            pipeline_item['runner_path'] = f'../split_tasks/{index}'
        hybrid_pipeline.append(pipeline_item)

    return hybrid_pipeline


def _add_hybrid_pipeline_to_task_config(task_config_path: Path, hybrid_pipeline: list[dict[str, str]]):
    with open(task_config_path, 'r', encoding='utf-8') as f:
        task_config = json.load(f)
    task_config['hybrid_pipeline'] = hybrid_pipeline
    with open(task_config_path, 'w', encoding='utf-8') as f:
        json.dump(task_config, f, indent=4, ensure_ascii=False)


def dump_graph(
    graph: LayerAbstractGraph,
    output_dir: Path,
    score: float,
    use_btp: bool,
    dump_split_subgraphs: bool = False,
):
    task_dir = output_dir / 'task'
    server_dir = task_dir / 'server'
    client_dir = task_dir / 'client'
    ergs_dir = server_dir

    ergs_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)
    ckks_param = {'param0': {**config.fhe_param.to_dict()}}

    erg0_path = ergs_dir / 'nn_layers_ct_0.json'
    insert_btp_scale_gamma_layers(graph)
    absorb_pcmgamma_layers(graph)
    recompute_final_level(graph)
    transforms.insert_drop_level_layers(graph)
    graph.to_json(dict(), str(erg0_path), score=score)
    hybrid_pipeline = None
    if dump_split_subgraphs:
        hybrid_pipeline = dump_split_tasks(graph, task_dir)

    if use_btp:
        graph_to_task_config(graph, str(server_dir))
    else:
        graph_to_task_config(graph, str(server_dir), False)

    server_task_config = server_dir / 'task_config.json'
    if hybrid_pipeline is not None:
        _add_hybrid_pipeline_to_task_config(server_task_config, hybrid_pipeline)

    client_task_config = client_dir / 'task_config.json'
    if server_task_config.exists():
        shutil.copy(str(server_task_config), str(client_task_config))

    _write_ckks_parameter(server_dir, ckks_param)
    _write_ckks_parameter(client_dir, ckks_param)


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
    use_gpu: bool = True,
    dump_split_subgraphs: bool = False,
    mpc_skip_max_states: int = 4096,
    mat_pack_style: str = '',
    enable_score_cache: bool = True,
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
        use_gpu: If True, use GPU primitive timing tables for FHE score; otherwise use CPU timing
        enable_score_cache: If True, cache per-compute score in DP compilation
    """
    compile_start_time = time.perf_counter()

    if mat_pack_style not in MAT_PACK_STYLES:
        raise ValueError(f'Unsupported mat_pack_style: {mat_pack_style!r}. Expected one of {sorted(MAT_PACK_STYLES)}')
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
    config.mat_pack_style = mat_pack_style
    config.set_btp_scale = set_btp_scale
    config.use_gpu = use_gpu
    if is_mpc_flow():
        is_use_btp = True
    print(
        f'Configuration initialized: STYLE={config.style}, GRAPH_TYPE={config.graph_type}, '
        f'N_HEADS={config.n_heads}, HEAD_DIM={config.head_dim}, MATMUL_BLOCK_SIZE={config.matmul_block_size}, '
        f'SET_BTP_SCALE={config.set_btp_scale}, BACKEND={"gpu" if config.use_gpu else "cpu"}, '
        f'MPC_SKIP_MAX_STATES={mpc_skip_max_states}, '
        f'SCORE_CACHE={enable_score_cache}'
    )

    raw_graph = LayerAbstractGraph.from_json(input_file_path)

    compile_start = time.perf_counter()
    if not is_use_btp:
        use_btp = False
        succeeded, graph, score = try_no_btp(raw_graph)
        if not succeeded:
            use_btp = True
            succeeded, graph, score = try_btp(
                num_experiments,
                raw_graph,
                temperature,
                num_workers,
                mpc_skip_max_states=mpc_skip_max_states,
                enable_score_cache=enable_score_cache
            )
            if not succeeded:
                raise ValueError('Compilation failed.')
    else:
        use_btp = True
        succeeded, graph, score = try_btp(
            num_experiments,
            raw_graph,
            temperature,
            num_workers,
            mpc_skip_max_states=mpc_skip_max_states,
            enable_score_cache=enable_score_cache
        )
        if not succeeded:
            raise ValueError('Compilation failed.')
    compile_elapsed = time.perf_counter() - compile_start
    print(f'[Compile] Core compile time: {compile_elapsed:.3f}s')
    dump_graph(graph, output_dir, score, use_btp=use_btp, dump_split_subgraphs=dump_split_subgraphs)

    return graph, score
