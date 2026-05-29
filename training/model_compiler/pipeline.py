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

import networkx as nx
import numpy as np

import components
from components import LayerAbstractGraph, config, PN13QP218, PN14QP438, PN15QP880, PN16QP1761, N16QP1546H192H32
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

        dag.remove_edge(pred_feature, btp_node)
        dag.remove_edge(btp_node, succ_feature)

        dag.add_node(pre_gamma, name=pre_gamma_id, level_cost=0)
        dag.add_node(pre_feature, **pre_feature_attrs)
        dag.add_edge(pred_feature, pre_gamma, **pred_to_btp_attrs)
        dag.add_edge(pre_gamma, pre_feature)
        dag.add_edge(pre_feature, btp_node, **pred_to_btp_attrs)

        dag.add_node(post_gamma, name=post_gamma_id, level_cost=0)
        dag.add_node(post_gamma_input_feature, **post_gamma_input_feature_attrs)
        dag.add_edge(btp_node, post_gamma_input_feature, **btp_to_succ_attrs)
        dag.add_edge(post_gamma_input_feature, post_gamma)
        dag.add_edge(post_gamma, succ_feature)


def _ensure_finite_scale(scale: np.float64, context: str) -> np.float64:
    scale = np.float64(scale)
    if not np.isfinite(scale):
        raise ValueError(f'Non-finite ckks_scale while propagating {context}: {scale}')
    return scale


def _default_ckks_scale() -> np.float64:
    return _ensure_finite_scale(np.float64(2**config.fhe_param.log_default_scale), 'default_scale')


def _q(level: int) -> np.float64:
    if level < 0 or level >= len(config.fhe_param.q):
        raise ValueError(f'Invalid CKKS q level {level}; q chain has {len(config.fhe_param.q)} levels')
    return _ensure_finite_scale(np.float64(config.fhe_param.q[level]), f'q[{level}]')


def _mult(lhs_scale, rhs_scale) -> np.float64:
    return _ensure_finite_scale(np.float64(lhs_scale) * np.float64(rhs_scale), 'mult')


def _rescale(product_scale, q_level) -> np.float64:
    return _ensure_finite_scale(np.float64(product_scale) / np.float64(q_level), 'rescale')


def _scale_farthest_from_default(scales: list[np.float64], default_scale: np.float64) -> np.float64:
    if not scales:
        raise ValueError('Cannot choose ckks_scale from an empty scale list')
    return _ensure_finite_scale(
        max(scales, key=lambda scale: abs(np.float64(scale) - default_scale)), 'farthest_from_default'
    )


def _ordered_feature_preds(graph: LayerAbstractGraph, node: ComputeNode) -> list[FeatureNode]:
    preds = [pred for pred in graph.dag.predecessors(node) if isinstance(pred, FeatureNode)]
    edge_indices = {pred: graph.dag.edges[pred, node].get('input_index') for pred in preds}
    if preds and all(index is not None for index in edge_indices.values()):
        return sorted(preds, key=lambda pred: edge_indices[pred])
    return preds


def _feature_level(graph: LayerAbstractGraph, feature: FeatureNode) -> int:
    return int(graph.dag.nodes[feature]['level'])


def _node_input_level(graph: LayerAbstractGraph, preds: list[FeatureNode]) -> int:
    if not preds:
        raise ValueError('Cannot compute input level without predecessor features')
    return max(_feature_level(graph, pred) for pred in preds)


def _passthrough_scale(preds: list[FeatureNode], default_scale: np.float64) -> np.float64:
    return _scale_farthest_from_default([np.float64(pred.ckks_scale) for pred in preds], default_scale)


def _propagate_pcmpoly_scale(input_scale: np.float64, level: int, default_scale: np.float64, order: int) -> np.float64:
    q_l = _q(level)
    q_l1 = _q(level - 1)

    x_sq = _rescale(_mult(input_scale, input_scale), q_l)
    c2_scale = _mult(_rescale(q_l, default_scale), q_l1)
    c2x2 = _rescale(_mult(x_sq, c2_scale), q_l1)
    c1x = _rescale(_mult(input_scale, q_l), q_l)
    low = _scale_farthest_from_default([c2x2, c1x], default_scale)

    if order == 2:
        return low
    if order != 4:
        raise ValueError(f'Unsupported pcmpoly order {order}; expected 2 or 4')

    q_l2 = _q(level - 2)
    c4_scale = _mult(
        _mult(_rescale(q_l, default_scale), _rescale(q_l, default_scale)), _mult(_rescale(q_l1, default_scale), q_l2)
    )
    c3_scale = _mult(_mult(_rescale(q_l, default_scale), _rescale(q_l, default_scale)), q_l2)
    c4x2 = _rescale(_mult(x_sq, c4_scale), q_l1)
    c3x = _rescale(_mult(input_scale, c3_scale), q_l)
    high = _scale_farthest_from_default([c4x2, c3x], default_scale)
    final = _rescale(_mult(x_sq, high), q_l2)
    return _scale_farthest_from_default([low, final], default_scale)


def _propagate_layer_ckks_scale(
    graph: LayerAbstractGraph, node: ComputeNode, preds: list[FeatureNode], default_scale: np.float64
) -> np.float64:
    layer_type = node.layer_type
    level = _node_input_level(graph, preds)
    input_scale = np.float64(preds[0].ckks_scale)

    if layer_type == 'bootstrapping':
        return default_scale

    if layer_type in ('partranspose', 'pcmgamma'):
        q_l = _q(level)
        return _rescale(_mult(input_scale, q_l), q_l)

    if layer_type == 'parcpmm':
        q_l = _q(level)
        q_l1 = _q(level - 1)
        scale = _rescale(_mult(input_scale, q_l), q_l)
        return _rescale(_mult(scale, q_l1), q_l1)

    if layer_type == 'parccmm':
        if len(preds) < 2:
            raise ValueError(f'parccmm node {node.layer_id} requires two inputs')
        q_l = _q(level)
        q_l1 = _q(level - 1)
        q_l2 = _q(level - 2)
        a_scale = np.float64(preds[0].ckks_scale)
        b_scale = np.float64(preds[1].ckks_scale)
        a_sigma = _rescale(_mult(a_scale, q_l), q_l)
        b_tau = _rescale(_mult(b_scale, q_l), q_l)
        psi_pt_scale = _mult(_rescale(q_l2, default_scale), q_l1)
        b_psi = _rescale(_mult(b_tau, psi_pt_scale), q_l1)
        return _rescale(_mult(a_sigma, b_psi), q_l2)

    if layer_type == 'pcmpoly':
        return _propagate_pcmpoly_scale(input_scale, level, default_scale, getattr(node, 'order', 4))

    if layer_type == 'pcmstats':
        q_l = _q(level)
        q_l1 = _q(level - 1)
        q_l2 = _q(level - 2)
        q_l3 = _q(level - 3)
        x_sq = _rescale(_mult(input_scale, input_scale), q_l)
        sum_x = _rescale(_mult(input_scale, q_l), q_l)
        mean = _rescale(_mult(sum_x, q_l1), q_l1)
        sum_x_sq = _rescale(_mult(x_sq, q_l), q_l1)
        e_x_sq = _rescale(_mult(sum_x_sq, q_l1), q_l2)
        mean_sq = _rescale(_mult(mean, mean), q_l2)
        var = _scale_farthest_from_default([e_x_sq, mean_sq], default_scale)
        iv_scale = _mult(_rescale(q_l2, default_scale), q_l3)
        return _rescale(_mult(var, iv_scale), q_l3)

    if layer_type == 'pcminit':
        q_l = _q(level)
        q_l1 = _q(level - 1)
        a_sq = _rescale(_mult(input_scale, input_scale), q_l)
        c2_scale = _mult(_rescale(q_l, default_scale), q_l1)
        c2a2 = _rescale(_mult(a_sq, c2_scale), q_l1)
        c1a = _rescale(_mult(input_scale, q_l), q_l)
        return _scale_farthest_from_default([c2a2, c1a], default_scale)

    if layer_type == 'pcmgs':
        if len(preds) < 2:
            raise ValueError(f'pcmgs node {node.layer_id} requires two inputs')
        y_scale = np.float64(preds[0].ckks_scale)
        a_scale = np.float64(preds[1].ckks_scale)
        y_level = _feature_level(graph, preds[0])
        q_l = _q(y_level)
        q_l1 = _q(y_level - 1)
        q_l2 = _q(y_level - 2)
        ya = _rescale(_mult(y_scale, a_scale), q_l)
        yy = _rescale(_mult(y_scale, y_scale), q_l)
        ya_yy = _rescale(_mult(ya, yy), q_l1)
        three_scale = _mult(_mult(_rescale(default_scale, q_l), _rescale(default_scale, q_l1)), default_scale)
        three_y = _rescale(_mult(y_scale, three_scale), q_l)
        diff = _scale_farthest_from_default([ya_yy, three_y], default_scale)
        half_scale = _mult(
            _mult(_rescale(q_l, default_scale), _rescale(q_l, default_scale)),
            _mult(_rescale(q_l1, default_scale), q_l2),
        )
        return _rescale(_mult(diff, half_scale), q_l2)

    if layer_type == 'pcmaffine':
        if len(preds) < 2:
            raise ValueError(f'pcmaffine node {node.layer_id} requires two inputs')
        x_centered_scale = np.float64(preds[0].ckks_scale)
        y_scale = np.float64(preds[1].ckks_scale)
        y_level = _feature_level(graph, preds[1])
        q_l = _q(y_level)
        q_l1 = _q(y_level - 1)
        yw_scale = _mult(_rescale(q_l, default_scale), q_l1)
        yw = _rescale(_mult(y_scale, yw_scale), q_l)
        return _rescale(_mult(x_centered_scale, yw), q_l1)

    return _passthrough_scale(preds, default_scale)


def propagate_ckks_scales(graph: LayerAbstractGraph):
    if not nx.is_directed_acyclic_graph(graph.dag):
        raise ValueError('Cycle exists in graph, cannot propagate ckks_scale')

    default_scale = _default_ckks_scale()
    for node in graph.dag.nodes:
        if isinstance(node, FeatureNode) and graph.dag.in_degree(node) == 0:
            node.ckks_scale = default_scale

    for node in nx.topological_sort(graph.dag):
        if not isinstance(node, ComputeNode):
            continue
        preds = _ordered_feature_preds(graph, node)
        succs = [succ for succ in graph.dag.successors(node) if isinstance(succ, FeatureNode)]
        if not preds or not succs:
            continue
        output_scale = _propagate_layer_ckks_scale(graph, node, preds, default_scale)
        output_scale = _ensure_finite_scale(output_scale, node.layer_id)
        for succ in succs:
            succ.ckks_scale = output_scale


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
    propagate_ckks_scales(graph)
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
