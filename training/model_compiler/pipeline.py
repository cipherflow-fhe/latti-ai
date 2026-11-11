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
import time
import json
import logging

import networkx as nx
import numpy as np

import components
from components import (
    ComputeNode,
    FeatureNode,
    LayerAbstractGraph,
    config,
    PN12Default,
    PN13Default,
    PN14Default,
    PN15Default,
    FHE_PARAMS,
)
import processor
from processor import *
from graph_partition_dp import *

from score import encrypted_parameter_storage_enabled, get_encrypted_parameter_storage_report
from inference.model_generator.parameter_sources import public_source

MAT_PACK_STYLES = {'', 'par_block_col_major', 'par_diagonal_pack'}
PCM_TO_PDM_TYPES = {
    'pcmgamma': 'pdmgamma',
}


def _pack_pcm_type(layer_type: str) -> str:
    if getattr(config, 'mat_pack_style', '') == 'par_diagonal_pack':
        return PCM_TO_PDM_TYPES.get(layer_type, layer_type)
    return layer_type


def set_seed_weight_subgrid_stride(graph: LayerAbstractGraph) -> None:
    stride = getattr(config, 'seed_weight_subgrid_stride', [1, 1])
    if config.style != 'multiplexed' or stride == [1, 1]:
        return
    for node in graph.dag.nodes:
        if (
            isinstance(node, components.ConvComputeNode)
            and node.layer_type == 'conv2d'
            and node.dim == 2
            and (node.groups == 1 or node.groups == node.channel_input == node.channel_output)
            and not node.is_conv_transpose
        ):
            input_feature = next(graph.dag.predecessors(node))
            seed_shape = config.block_shape if node.is_big_size else input_feature.shape
            if all(size % step == 0 for size, step in zip(seed_shape, stride)):
                node.seed_weight_subgrid_stride = list(stride)
                node.seed_weight_layout = 'canonical' if node.is_big_size else 'joint'
            else:
                logging.debug(
                    'Seed weights disabled for %s: stride %s does not divide %s', node.layer_id, stride, seed_shape
                )


def prepare_graph(raw_graph: LayerAbstractGraph) -> LayerAbstractGraph:
    """
    Prepare graph for compilation (common preparation steps)

    Args:
        raw_graph: Raw LayerAbstractGraph loaded from json

    Returns:
        Prepared LayerAbstractGraph
    """
    pt_graph = copy.deepcopy(raw_graph)

    transforms.apply_bert_l_prepad_to_feature_mat_shapes(pt_graph)
    substitute_layers_for_btp(pt_graph)
    # transforms.init_levels(pt_graph)
    # update_shape_for_btp(pt_graph)
    # update_skip_for_btp(pt_graph)
    # update_level_cost_for_btp(pt_graph)
    set_is_adaptive_avgpool(pt_graph)
    transforms.expand_multi_head_attention(pt_graph)
    transforms.expand_layer_norm(pt_graph)
    transforms.expand_bert_custom_poly_functions(pt_graph)
    transforms.expand_poly_act_rn(pt_graph)
    transforms.expand_parcpmm_add_pt(pt_graph)
    transforms.replace_avgpool1d_with_depthwise_conv(pt_graph)
    transforms.split_upsampling_layers(pt_graph)
    transforms.infer_shapes_skips_and_pack_num(pt_graph)
    transforms.set_pcm_K(pt_graph)
    transforms.combine_convs_with_upsamples(pt_graph)
    transforms.process_polyact(pt_graph)
    transforms.set_level_costs(pt_graph)
    set_seed_weight_subgrid_stride(pt_graph)

    transforms.absorb_scale(pt_graph)

    return pt_graph


def set_fhe_param(params):
    config.fhe_param = copy.deepcopy(params)
    if config.set_btp_scale is not None:
        config.fhe_param.max_level -= 1


def set_block_shape(params, raw_graph: LayerAbstractGraph):
    """Set config.block_shape based on the input graph's leading feature node shape and real slot capacity.

    Rules:
      (0) BERT/feature_mat graphs do not use block_shape-based spatial tiling.
      (1) If leading node is not 2D, use sqrt(n_real_num_slot) as a square default.
      (2) If shape0 * shape1 <= n_real_num_slot, block_shape = [shape0, shape1]
      (3) Otherwise, divide both shape0 and shape1 by 2, 4, 8, 16, ...,
          until shape0 * shape1 <= n_real_num_slot. If this leaves half the slots
          unused, double the width (dimension 1).
    """
    slot_num = params.n_slots
    leading_nodes = raw_graph.get_leading_feature_nodes()
    if getattr(config, 'model_type', '') == 'bert' or any(node.data_type == 'feature_mat' for node in leading_nodes):
        config.block_shape = [1, 1]
        return
    if not leading_nodes or len(leading_nodes[0].shape) == 0:
        side = 1 << ((slot_num.bit_length() - 1) // 2)
        config.block_shape = [side, side]
        return
    leading_shape = leading_nodes[0].shape
    block = list(leading_shape)
    slot_num = params.n_slots
    # Threshold is the v6 real slot capacity.
    import math

    while math.prod(block) > slot_num:
        block = [s // 2 for s in block]
    if len(block) == 2 and math.prod(leading_shape) > slot_num and math.prod(block) * 2 == slot_num:
        block[1] *= 2

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

    selected = getattr(config, 'fhe_param_name', None)
    no_btp_params = [FHE_PARAMS[selected]] if selected else [PN12Default, PN13Default, PN14Default, PN15Default]

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
    enable_score_cache: bool = True,
) -> tuple[bool, LayerAbstractGraph | None, float]:
    selected = getattr(config, 'fhe_param_name', None)
    btp_param_list = [FHE_PARAMS[selected]] if selected else [PN15Default]
    if any(p.name not in ('PN15Default', 'N16QP1546H192H32') for p in btp_param_list):
        raise ValueError('Bootstrapping requires PN15Default or N16QP1546H192H32')
    valid_results = []
    for params in btp_param_list:
        set_fhe_param(params)
        set_block_shape(config.fhe_param, raw_graph)

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        graph, score = run_btp_compilation(
            num_experiments, pt_graph, temperature, num_workers, enable_score_cache=enable_score_cache
        )

        # (3) Post-process
        if graph is not None:
            if encrypted_parameter_storage_enabled():
                graph.encrypted_parameter_dp_report = get_encrypted_parameter_storage_report(
                    graph.dag, generate_param_dict_for_graph()
                )
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


def post_process(graph: LayerAbstractGraph):
    slot_num = config.fhe_param.n_slots
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


def bert_softmax_denominator_scale_for_layer(layer_id: str) -> float:
    if '_softmax_normalize_initial_' in layer_id:
        scale = getattr(config, 'bert_softmax_initial_denominator_scale', 16.0)
    elif '_softmax_normalize_refine_0_' in layer_id:
        scale = getattr(config, 'bert_softmax_first_refinement_denominator_scale', 2.0)
    else:
        scale = getattr(config, 'bert_softmax_later_refinement_denominator_scale', 1.0)
    scale = float(scale)
    if scale == 0:
        raise ValueError(f'BERT softmax denominator scale cannot be 0 for {layer_id}')
    return scale


def _softmax_normalize_base_id_from_btp_layer(layer_id: str) -> str | None:
    for marker in (
        '_denominator_scaled_bootstrap',
        '_denominator_bootstrap',
        '_probabilities_unscaled_bootstrap',
        '_inverse_scaled_bootstrap',
    ):
        if marker in layer_id:
            return layer_id.split(marker, 1)[0]
    if '_inverse_iter_' in layer_id and '_out_bootstrap' in layer_id:
        return layer_id.split('_inverse_iter_', 1)[0]
    return None


def _find_compute_node_by_layer_id(dag, layer_id: str) -> ComputeNode | None:
    for node in dag.nodes:
        if isinstance(node, ComputeNode) and node.layer_id == layer_id:
            return node
    return None


def insert_btp_scale_gamma_layers(graph: LayerAbstractGraph):
    if config.set_btp_scale is None:
        return

    btp_scale = float(config.set_btp_scale)
    if btp_scale == 0:
        raise ValueError('set_btp_scale cannot be 0 when inserting BTP scale gamma layers')

    dag = graph.dag

    def normalize_denominator_scale_for_btp_node(layer_id: str) -> float:
        base_id = _softmax_normalize_base_id_from_btp_layer(layer_id)
        if base_id is not None:
            init_node = _find_compute_node_by_layer_id(dag, f'{base_id}_inverse_init')
            if init_node is not None and getattr(init_node, 'coefficients', None):
                return 2.0 / init_node.coefficients[0]
            scale_node = _find_compute_node_by_layer_id(dag, f'{base_id}_denominator_scale')
            if scale_node is not None and getattr(scale_node, 'scalar_value', None) is not None:
                return float(scale_node.scalar_value)
        return bert_softmax_denominator_scale_for_layer(layer_id)

    def pre_btp_scale_for_node(layer_id: str) -> float:
        if getattr(config, 'model_type', '') == 'bert':
            if ('_softmax_exp_poly_out_bootstrap' in layer_id) or (
                '_softmax_delta1_square_' in layer_id and '_out_bootstrap' in layer_id
            ):
                return float(getattr(config, 'bert_softmax_values_btp_scale', btp_scale))
            if '_softmax_normalize_' in layer_id:
                if '_probabilities_unscaled_bootstrap' in layer_id:
                    return normalize_denominator_scale_for_btp_node(layer_id)
                if '_inverse_scaled_bootstrap' in layer_id:
                    denominator_scale = normalize_denominator_scale_for_btp_node(layer_id)
                    return 1.0 / denominator_scale
                if '_denominator_scaled_bootstrap' in layer_id:
                    return float(getattr(config, 'bert_softmax_scaled_denominator_btp_scale', btp_scale))
                if '_denominator_bootstrap' in layer_id:
                    scale = float(getattr(config, 'bert_softmax_denominator_btp_scale', btp_scale))
                    if '_softmax_normalize_refine_' in layer_id:
                        scale *= 16.0 / normalize_denominator_scale_for_btp_node(layer_id)
                    return scale
                if '_inverse_iter_' in layer_id and '_out_bootstrap' in layer_id:
                    inverse_btp_scale = float(getattr(config, 'bert_softmax_inverse_btp_scale', btp_scale))
                    # From a0=1, positive denominators give a_i <= 2**i.
                    # The graph carries a_i/s, including for tiny thor-wide sums.
                    iteration = int(layer_id.split('_inverse_iter_', 1)[1].split('_', 1)[0]) + 1
                    denominator_scale = normalize_denominator_scale_for_btp_node(layer_id)
                    return min(512.0, inverse_btp_scale * denominator_scale / 2**iteration)
            if 'CustomLayerNorm_y' in layer_id and '_bootstrap' in layer_id:
                return float(getattr(config, 'bert_layernorm_inverse_btp_scale', btp_scale))
        return btp_scale

    for btp_node in list(dag.nodes):
        if not isinstance(btp_node, ComputeNode) or btp_node.layer_type != 'bootstrapping':
            continue

        preds = list(dag.predecessors(btp_node))
        succs = list(dag.successors(btp_node))
        if len(preds) != 1 or len(succs) != 1:
            raise ValueError(f'Expected bootstrapping node {btp_node.layer_id} to have one input and one output')

        pred_feature = preds[0]
        succ_feature = succs[0]
        pre_btp_scale = pre_btp_scale_for_node(btp_node.layer_id)
        if pre_btp_scale == 0:
            raise ValueError(f'BTP scale cannot be 0 for {btp_node.layer_id}')

        gamma_type = _pack_pcm_type('pcmgamma')
        pre_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_pre_{gamma_type}', 'layer_id')
        post_gamma_id = _unique_graph_node_id(graph, f'{btp_node.layer_id}_post_{gamma_type}', 'layer_id')
        pre_feature_id = _unique_graph_node_id(graph, f'{pre_gamma_id}_output', 'node_id')
        post_gamma_input_feature_id = _unique_graph_node_id(graph, f'{post_gamma_id}_input', 'node_id')

        pre_gamma = ComputeNode(pre_gamma_id, gamma_type, btp_node.channel_input, btp_node.channel_input)
        pre_gamma.depth = btp_node.depth
        pre_gamma.path = f'{pre_gamma_id}.weight'
        pre_gamma.btp_scale = pre_btp_scale
        pre_gamma.parameter_sources = {
            'gamma_pt': public_source('math_constant', dict(formula='bootstrap_scale', scalar_value=pre_btp_scale))
        }

        post_gamma = ComputeNode(post_gamma_id, gamma_type, btp_node.channel_output, btp_node.channel_output)
        post_gamma.depth = btp_node.depth
        post_gamma.path = f'{post_gamma_id}.weight'
        post_gamma.btp_scale = 1 / pre_btp_scale
        post_gamma.parameter_sources = {
            'gamma_pt': public_source('math_constant', dict(formula='bootstrap_scale', scalar_value=1 / pre_btp_scale))
        }

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
    sources = getattr(node, 'parameter_sources', {})
    key = 'coeff_pt' if node.layer_type in ('pcmpoly', 'pdmpoly') else 'row_chunk_seed'
    if node.layer_type in ('pcmpoly', 'pdmpoly', 'parcpmm'):
        previous = sources.get(key, {})
        refs = list(previous.get('source_refs', []))
        refs.extend(getattr(node, name, '') for name in ('weight_path', 'coeffs_path'))
        sources[key] = dict(
            visibility='secret', source_kind='derived', source_refs=[ref for ref in refs if ref] + [fuse_gama_info]
        )
    node.parameter_sources = sources
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
    absorb_pcmgamma_layers(graph)
    recompute_final_level(graph)
    if encrypted_parameter_storage_enabled() or getattr(config, 'pcm_encrypted_parameters', False):
        report = get_encrypted_parameter_storage_report(graph.dag, generate_param_dict_for_graph())
        previous = getattr(graph, 'encrypted_parameter_dp_report', report)
        before = {item['layer_id']: item['counts'] for item in previous['layers']}
        after = {item['layer_id']: item['counts'] for item in report['layers']}
        if encrypted_parameter_storage_enabled() and before != after:
            raise ValueError('Final level recompute changed encrypted parameter storage from the DP candidate')
        (output_dir / 'encrypted_parameter_storage_report.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8'
        )
        print(
            f'Encrypted parameter storage: {report["storage_bytes"]} bytes; '
            f'runtime={report["runtime_score"]}, refresh={report["refresh_score"]}, '
            f'storage={report["storage_score"]}; unaccounted={report["unaccounted"]}'
        )
    transforms.insert_drop_level_layers(graph)
    if getattr(config, 'seed_weight_subgrid_stride', [1, 1]) != [1, 1]:
        final_report = get_encrypted_parameter_storage_report(graph.dag, generate_param_dict_for_graph())
        convolution_layouts = {
            item['layer_id']: item['convolution_seed_layout']
            for item in final_report['layers']
            if 'convolution_seed_layout' in item
        }
        for node in graph.dag.nodes:
            if isinstance(node, components.ConvComputeNode) and node.layer_id in convolution_layouts:
                layout = convolution_layouts[node.layer_id]
                node.seed_weight_subgrid_stride = layout['stride']
        layouts = {
            item['layer_id']: item['dense_seed_tiles'] for item in final_report['layers'] if 'dense_seed_tiles' in item
        }
        for node in graph.dag.nodes:
            if isinstance(node, components.DenseComputeNode) and node.layer_id in layouts:
                node.dense_seed_tiles = layouts[node.layer_id]
        poly_layouts = {
            item['layer_id']: item['poly_seed_layout'] for item in final_report['layers'] if 'poly_seed_layout' in item
        }
        for node in graph.dag.nodes:
            if isinstance(node, components.ActivationComputeNode) and node.layer_id in poly_layouts:
                node.poly_seed_layout = poly_layouts[node.layer_id]
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
    mat_pack_style: str = '',
    model_type: str = '',
    set_btp_scale: float | None = None,
    bert_softmax_values_btp_scale: float | None = None,
    bert_softmax_denominator_btp_scale: float | None = None,
    bert_softmax_scaled_denominator_btp_scale: float | None = None,
    bert_softmax_inverse_btp_scale: float | None = None,
    bert_layernorm_inverse_btp_scale: float | None = None,
    bert_softmax_initial_denominator_scale: float | None = None,
    bert_softmax_wide_initial_denominator_scale: float | None = None,
    bert_softmax_first_refinement_denominator_scale: float | None = None,
    bert_softmax_later_refinement_denominator_scale: float | None = None,
    use_gpu: bool = True,
    enable_score_cache: bool = True,
    encrypted_parameter_storage_weight: float = 0.0,
    encrypted_parameter_storage_normalization_bytes: int = 1 << 30,
    deployment_mode: str = 'client_encrypted_input',
    seed_weight_subgrid_stride: list[int] | tuple[int, int] = (1, 1),
    fhe_param_name: str | None = None,
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
    if deployment_mode == 'server_provisioned_runner' and mat_pack_style == 'par_diagonal_pack':
        raise ValueError('PDM encrypted parameters are unsupported; use par_block_col_major (PCM)')

    if fhe_param_name is not None and fhe_param_name not in FHE_PARAMS:
        raise ValueError(f'Unsupported FHE parameter name: {fhe_param_name!r}')
    config.fhe_param_name = fhe_param_name
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
    config.pcm_encrypted_parameters = (
        deployment_mode == 'server_provisioned_runner' and mat_pack_style == 'par_block_col_major'
    )
    config.model_type = model_type
    seed_stride = seed_weight_subgrid_stride
    if not (
        isinstance(seed_stride, (list, tuple))
        and len(seed_stride) == 2
        and all(type(step) is int and step > 0 and (step & (step - 1)) == 0 for step in seed_stride)
    ):
        logging.debug('Invalid seed_weight_subgrid_stride %r; using [1, 1]', seed_stride)
        seed_stride = [1, 1]
    if deployment_mode != 'server_provisioned_runner':
        seed_stride = [1, 1]
    config.seed_weight_subgrid_stride = list(seed_stride)
    config.set_btp_scale = set_btp_scale
    if bert_softmax_values_btp_scale is not None:
        config.bert_softmax_values_btp_scale = float(bert_softmax_values_btp_scale)
    if bert_softmax_denominator_btp_scale is not None:
        config.bert_softmax_denominator_btp_scale = float(bert_softmax_denominator_btp_scale)
    if bert_softmax_scaled_denominator_btp_scale is not None:
        config.bert_softmax_scaled_denominator_btp_scale = float(bert_softmax_scaled_denominator_btp_scale)
    if bert_softmax_inverse_btp_scale is not None:
        config.bert_softmax_inverse_btp_scale = float(bert_softmax_inverse_btp_scale)
    if bert_layernorm_inverse_btp_scale is not None:
        config.bert_layernorm_inverse_btp_scale = float(bert_layernorm_inverse_btp_scale)
    if bert_softmax_initial_denominator_scale is not None:
        config.bert_softmax_initial_denominator_scale = float(bert_softmax_initial_denominator_scale)
    if bert_softmax_wide_initial_denominator_scale is not None:
        config.bert_softmax_wide_initial_denominator_scale = float(bert_softmax_wide_initial_denominator_scale)
    if bert_softmax_first_refinement_denominator_scale is not None:
        config.bert_softmax_first_refinement_denominator_scale = float(bert_softmax_first_refinement_denominator_scale)
    if bert_softmax_later_refinement_denominator_scale is not None:
        config.bert_softmax_later_refinement_denominator_scale = float(bert_softmax_later_refinement_denominator_scale)
    config.use_gpu = use_gpu
    print(
        f'Configuration initialized: STYLE={config.style}, GRAPH_TYPE={config.graph_type}, '
        f'N_HEADS={config.n_heads}, HEAD_DIM={config.head_dim}, MATMUL_BLOCK_SIZE={config.matmul_block_size}, '
        f'MAT_PACK_STYLE={config.mat_pack_style}, MODEL_TYPE={config.model_type}, '
        f'SEED_WEIGHT_SUBGRID_STRIDE={config.seed_weight_subgrid_stride}, '
        f'SET_BTP_SCALE={config.set_btp_scale}, '
        f'BERT_SOFTMAX_VALUES_BTP_SCALE={config.bert_softmax_values_btp_scale}, '
        f'BERT_SOFTMAX_DENOMINATOR_BTP_SCALE={config.bert_softmax_denominator_btp_scale}, '
        f'BERT_SOFTMAX_SCALED_DENOMINATOR_BTP_SCALE={config.bert_softmax_scaled_denominator_btp_scale}, '
        f'BERT_SOFTMAX_INVERSE_BTP_SCALE={config.bert_softmax_inverse_btp_scale}, '
        f'BERT_LAYERNORM_INVERSE_BTP_SCALE={config.bert_layernorm_inverse_btp_scale}, '
        f'BERT_SOFTMAX_INITIAL_DENOMINATOR_SCALE={config.bert_softmax_initial_denominator_scale}, '
        f'BERT_SOFTMAX_WIDE_INITIAL_DENOMINATOR_SCALE='
        f'{config.bert_softmax_wide_initial_denominator_scale}, '
        f'BERT_SOFTMAX_FIRST_REFINEMENT_DENOMINATOR_SCALE='
        f'{config.bert_softmax_first_refinement_denominator_scale}, '
        f'BERT_SOFTMAX_LATER_REFINEMENT_DENOMINATOR_SCALE='
        f'{config.bert_softmax_later_refinement_denominator_scale}, '
        f'BACKEND={"gpu" if config.use_gpu else "cpu"}, '
        f'SCORE_CACHE={enable_score_cache}'
    )

    if not math.isfinite(encrypted_parameter_storage_weight) or encrypted_parameter_storage_weight < 0:
        raise ValueError('encrypted_parameter_storage_weight must be finite and non-negative')
    if encrypted_parameter_storage_normalization_bytes <= 0:
        raise ValueError('encrypted_parameter_storage_normalization_bytes must be positive')
    if encrypted_parameter_storage_weight and deployment_mode != 'server_provisioned_runner':
        raise ValueError('Storage scoring requires server_provisioned_runner')
    config.encrypted_parameter_storage_weight = encrypted_parameter_storage_weight
    config.encrypted_parameter_storage_normalization_bytes = encrypted_parameter_storage_normalization_bytes

    raw_graph = LayerAbstractGraph.from_json(input_file_path)

    if not is_use_btp:
        use_btp = False
        succeeded, graph, score = try_no_btp(raw_graph)
        if not succeeded:
            use_btp = True
            succeeded, graph, score = try_btp(
                num_experiments, raw_graph, temperature, num_workers, enable_score_cache=enable_score_cache
            )
            if not succeeded:
                raise ValueError('Compilation failed.')
    else:
        use_btp = True
        succeeded, graph, score = try_btp(
            num_experiments, raw_graph, temperature, num_workers, enable_score_cache=enable_score_cache
        )
        if not succeeded:
            raise ValueError('Compilation failed.')
    dump_graph(graph, output_dir, score, use_btp=use_btp)

    compile_elapsed_time = time.perf_counter() - compile_start_time
    print(f'Total compilation time: {compile_elapsed_time:.3f}s')

    return graph, score
