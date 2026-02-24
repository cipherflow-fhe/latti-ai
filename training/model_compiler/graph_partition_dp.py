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

import argparse
import sys

sys.path.append('.')

import cProfile
import pstats

import copy
import json
import shutil

import numpy as np
import random
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
from tqdm import tqdm

from components import LayerAbstractGraph, ComputeNode, FeatureNode, load_config
import components
import processor

# Load configuration here
config = load_config()


def init_config_with_args(poly_n=None, style=None, graph_type=None):
    """
    Initialize configuration based on command line arguments

    Args:
        poly_n: Polynomial modulus degree (POLY_N)
        style: Computation style (STYLE)
        graph_type: Graph type (GRAPH_TYPE)
    """
    # If command line arguments are provided, override config file values
    if poly_n is not None:
        config['POLY_N'] = poly_n
    if style is not None:
        config['STYLE'] = style
    if graph_type is not None:
        config['GRAPH_TYPE'] = graph_type

    # Get current values from config
    current_poly_n = config.get('POLY_N', 65536)
    current_style = config.get('STYLE', 'multiplexed')
    current_graph_type = config.get('GRAPH_TYPE', 'btp')

    # Automatically set MAX_LEVEL based on POLY_N and GRAPH_TYPE
    if current_graph_type == 'btp':
        # BTP version configuration
        poly_n_to_max_level = {8192: 5, 16384: 7, 65536: 9}
    else:
        # Non-BTP version configuration
        poly_n_to_max_level = {8192: 5, 16384: 9, 65536: 24}

    max_level = poly_n_to_max_level.get(current_poly_n)
    if max_level is not None:
        config['MAX_LEVEL'] = max_level
        print(
            f'Automatically set MAX_LEVEL={max_level} based on GRAPH_TYPE={current_graph_type}, POLY_N={current_poly_n}'
        )
    else:
        print(f'Warning: No MAX_LEVEL mapping for POLY_N={current_poly_n}, using value from config.json')

    # Automatically set block_shape based on POLY_N
    poly_n_to_block_shape = {65536: [128, 256], 16384: [64, 64], 8192: [64, 64]}
    block_shape = poly_n_to_block_shape.get(current_poly_n, [64, 64])
    config['block_shape'] = block_shape
    print(f'Automatically set block_shape={block_shape} based on POLY_N={current_poly_n}')

    # Set configuration to components and processor modules
    components.config = config
    processor.config = config

    # Initialize configuration variables for each module
    components._init_config_vars()
    processor._init_config_vars()

    print(
        f'Configuration initialized: POLY_N={current_poly_n}, STYLE={current_style}, GRAPH_TYPE={current_graph_type}, MAX_LEVEL={config["MAX_LEVEL"]}, block_shape={config["block_shape"]}'
    )


# Import required functions and classes from processor
from processor import (
    substitute_layers_for_btp,
    process_level_for_graph,
    EncryptParameterNode,
    BtpScoreParam,
    MpcScoreParam,
    FheScoreParam,
    add_drop_level_for_graph,
    update_subgraph_node_param,
    get_slot_num,
    update_skip_for_btp,
    update_shape_for_btp,
    balance_scale_for_graph,
    update_level_cost_for_btp,
    absorb_scale_for_approx_poly,
    change_skip_for_graph,
    set_graph_scale,
    set_is_adaptive_avgpool,
    graph_to_task_config,
)

from typing import Callable


def add_btp_layer_in_graph(dag: nx.DiGraph, upstream_feature: FeatureNode, param_dict: dict, restore_lv: int):
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
    if config.get('MPC_REFRESH', False):
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

    slot_num = get_slot_num(btp_node.ckks_parameter_id_input, param_dict)
    dag.nodes[refreshed_feature]['pack_num'] = dag.nodes[upstream_feature]['pack_num']

    return btp_node


def update_bd_node_in_sub(node: FeatureNode, subgraph: nx.DiGraph, remaining_dag: nx.DiGraph) -> FeatureNode:
    pre_computes_sub = list(subgraph.predecessors(node))
    succ_computes_remain = list(remaining_dag.successors(node))
    is_refreshed = False
    for succ_c in succ_computes_remain:
        if 'bootstrapping' in succ_c.layer_type:
            is_refreshed = True
    if is_refreshed and len(pre_computes_sub) == 0:
        refreshed_node = list(remaining_dag.successors(succ_c))[0]
        subgraph.add_node(refreshed_node, **remaining_dag.nodes[refreshed_node])
        for s in list(subgraph.successors(node)):
            subgraph.remove_edge(node, s)
            subgraph.add_edge(refreshed_node, s)
        subgraph.remove_node(node)

    return is_refreshed


def generate_param_dict_for_graph():
    param_dict = dict()
    poly_to_mod = {8192: 31, 16384: 34, 65536: 41}
    poly_n = config.get('POLY_N', 65536)
    mod_bits = poly_to_mod.get(poly_n, 41)
    param_dict[f'param0'] = EncryptParameterNode(poly_n, mod_bits, mod_bits)

    return param_dict


def calculate_compute_score_for_graph(
    enclosing_graph: nx.DiGraph, grow: nx.DiGraph, param_dict: dict[str, EncryptParameterNode]
) -> float:
    compute_score = 0.0
    for compute in grow.nodes:
        if not isinstance(compute, ComputeNode):
            continue
        if compute.layer_type in ['conv2d', 'fc0', 'add2d', 'simple_polyrelu', 'avgpool2d']:
            pred = next(enclosing_graph.predecessors(compute))
            s_param = FheScoreParam(enclosing_graph, compute, param_dict, enclosing_graph.nodes[pred]['level'])
            score = s_param.get_score()
            enclosing_graph.nodes[compute]['score'] = score
            compute_score += score
    return compute_score


def update_btp_to_mpc_refresh(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type == 'bootstrapping':
                node.layer_type = 'mpc_refresh'


class GraphPartitioner:
    def __init__(self, entire_graph: nx.DiGraph, temperature: float = 1.0):
        self.entire_graph = entire_graph
        self.param_dict = generate_param_dict_for_graph()

        self.temperature = temperature
        self.pbar = tqdm(desc=f'Subgraph explorations (temperature={self.temperature})', unit='it')

    def inspect_level_backward(self, subgraph: nx.DiGraph):
        max_level = -1
        level_dict: dict[FeatureNode, int] = {}
        subg_nodes = subgraph.nodes
        for node in reversed(list(nx.topological_sort(subgraph))):
            if isinstance(node, ComputeNode):
                continue

            succ_c = list(subgraph.successors(node))
            if len(succ_c) == 0:
                if config.get('MPC_REFRESH', False) or config.get('GRAPH_TYPE', 'btp') == 'mpc':
                    level_dict[node] = 1
                elif config.get('GRAPH_TYPE', 'btp') == 'btp' and not config.get('MPC_REFRESH', False):
                    level_dict[node] = 0
            else:
                successing_subg_compute_nodes = [c for c in succ_c if c in subg_nodes]
                input_feature_lv: list[int] = []
                for c in successing_subg_compute_nodes:
                    assert isinstance(c, ComputeNode)
                    for feat in subgraph.successors(c):
                        assert isinstance(feat, FeatureNode)

                        input_feature_lv.append(level_dict[feat] + subgraph.nodes[c]['level_cost'])

                level_dict[node] = max(input_feature_lv)
                if level_dict[node] > config['MAX_LEVEL']:
                    return False, -1, level_dict

            max_level = max(max_level, level_dict[node])
        return True, max_level, level_dict

    def split_graph_and_set_level(self, graph_with_btp: nx.DiGraph):
        splitted_graph = LayerAbstractGraph()
        splitted_graph.dag = graph_with_btp.copy()
        btp_nodes = list()
        for compute in splitted_graph.dag.nodes:
            if isinstance(compute, ComputeNode):
                if compute.layer_type == 'bootstrapping':
                    btp_nodes.append(compute)
        splitted_graph.dag.remove_nodes_from(btp_nodes)

        weak_components = list(nx.weakly_connected_components(splitted_graph.dag))
        subgraphs: list[LayerAbstractGraph] = list()
        for component in weak_components:
            if len(component) > 1:
                sub = LayerAbstractGraph()
                sub.dag = splitted_graph.dag.subgraph(component).copy()
                subgraphs.append(sub)
        res_dict = dict()
        for sub in subgraphs:
            res = self.inspect_level_backward(sub.dag)
            if not res[0]:
                return False, dict()
            res_dict.update(res[2])
        return True, res_dict

    def remove_small_subgraphs(self, subgraphs: set[frozenset], H: nx.DiGraph) -> list[frozenset]:
        def boltzmann_weighted_probabilities(depths: list[int], temperature: float = 1.0) -> list[float]:
            if temperature <= 0:
                raise ValueError('Temperature must be positive.')

            depths = np.asarray(depths, dtype=float)

            scaled = depths / temperature
            scaled -= np.max(scaled)

            weights = np.exp(scaled)
            probs = weights / np.sum(weights)
            return probs.tolist()

        subgraphs_in_depths: dict[int, list[frozenset]] = {}
        for subgraph in subgraphs:
            subgraph_depth = self.inspect_level_backward(H.subgraph(subgraph))[1]
            if subgraph_depth not in subgraphs_in_depths:
                subgraphs_in_depths[subgraph_depth] = []
            subgraphs_in_depths[subgraph_depth].append(subgraph)

        max_depth = max(subgraphs_in_depths.keys())

        level_threshold = max_depth - 4
        depths = []
        for i, depth in enumerate(sorted(list(subgraphs_in_depths.keys()), reverse=True)):
            if i > 0 and depth < level_threshold:
                break
            depths.append(depth)

        chosen_depth = random.choices(depths, weights=boltzmann_weighted_probabilities(depths, self.temperature), k=1)[
            0
        ]
        candidates = sorted(subgraphs_in_depths[chosen_depth], key=lambda x: len(x), reverse=True)[:8]
        result = [
            random.choices(
                candidates,
                weights=boltzmann_weighted_probabilities([len(c) for c in candidates], self.temperature),
                k=1,
            )[0]
        ]

        return result

    def grow_connected_until_maximal(
        self,
        curr_nodes: frozenset,
        dag: frozenset,
        le_maximal_subg_memo: dict[frozenset, set[frozenset]],
        H: nx.DiGraph,
        memsize=8192,
    ) -> frozenset:
        def retrieve_boundary_compute_candidates(nodes: set, subgraph: nx.DiGraph, traversed_nodes: set) -> set:
            boundary_candidates = set()
            for u in nodes:
                if u in traversed_nodes:
                    continue
                for nbr in list(subgraph.predecessors(u)) + list(subgraph.successors(u)):
                    if nbr not in nodes and isinstance(nbr, ComputeNode):
                        boundary_candidates.add(nbr)
                    traversed_nodes.add(nbr)
                traversed_nodes.add(u)
            return boundary_candidates

        curr_sub = H.subgraph(curr_nodes)
        update_subgraph_node_param(curr_sub, self.param_dict, 'param0')
        level_is_below_max, max_level, _ = self.inspect_level_backward(curr_sub)
        if not level_is_below_max:
            return frozenset()

        results: list[frozenset] = [curr_nodes]

        traversed_nodes = set()
        new_curr_nodes = set(curr_nodes)
        boundary_candidates = retrieve_boundary_compute_candidates(curr_nodes, H, traversed_nodes)

        while boundary_candidates:
            v = boundary_candidates.pop()
            assert v not in new_curr_nodes
            frozen_nodes_to_inspect = frozenset(
                list(new_curr_nodes) + [v] + list(H.successors(v)) + list(H.predecessors(v))
            )

            if frozen_nodes_to_inspect in le_maximal_subg_memo:
                return le_maximal_subg_memo[frozen_nodes_to_inspect]

            frozen_nodes_sub = H.subgraph(frozen_nodes_to_inspect)
            update_subgraph_node_param(frozen_nodes_sub, self.param_dict, 'param0')
            level_is_below_max, _, _ = self.inspect_level_backward(frozen_nodes_sub)
            if not level_is_below_max:
                continue

            new_curr_nodes |= set([v]) | set(H.successors(v)) | set(H.predecessors(v))
            results.append(frozen_nodes_to_inspect)
            if len(results) >= memsize:
                results.pop(0)
            boundary_candidates |= retrieve_boundary_compute_candidates(new_curr_nodes, H, traversed_nodes)

        le_maximal_subgs = frozenset(results)
        for res in le_maximal_subgs:
            le_maximal_subg_memo[res] = le_maximal_subgs

        return le_maximal_subgs

    def process_btp_level_cost(self, dag: nx.DiGraph):
        for node in dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                preds: list[FeatureNode] = list(dag.predecessors(node))
                succs: list[FeatureNode] = list(dag.successors(node))
                dag.nodes[node]['level_cost'] = dag.nodes[preds[0]]['level'] - dag.nodes[succs[0]]['level']

    def solve(self, H: nx.DiGraph, recursion_depth: int = 0) -> tuple[float, nx.DiGraph]:
        self.pbar.update(1)
        if len(H.nodes) == 0:
            return 0.0, nx.DiGraph()

        leaf_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
        H_nodes = H.nodes
        le_maximal_subg_memo: dict[frozenset, set[frozenset]] = {}
        all_subgraphs_less_than_capacity: set[frozenset] = set()
        for leaf_data in leaf_nodes:
            assert isinstance(leaf_data, FeatureNode)
            immediate_comp_nodes = H.predecessors(leaf_data)
            for comp in immediate_comp_nodes:
                end = list(H.predecessors(comp)) + list(H.successors(comp)) + [comp]
                all_subgraphs_less_than_capacity |= self.grow_connected_until_maximal(
                    frozenset(end), frozenset(H_nodes), le_maximal_subg_memo, H
                )
            assert len(all_subgraphs_less_than_capacity) > 0

        subgraphs = self.remove_small_subgraphs(all_subgraphs_less_than_capacity, H)

        best_cost = float('inf')
        best_graph = None
        for subgraph_nodes in subgraphs:
            remaining_H_nodes = set(H_nodes) - set(subgraph_nodes)
            for node in remaining_H_nodes.copy():
                if isinstance(node, ComputeNode):
                    remaining_H_nodes |= set(H.successors(node)) | set(H.predecessors(node))

            refresh_boundary = remaining_H_nodes & subgraph_nodes
            subgraph = H.subgraph(subgraph_nodes).copy()

            rest_cost, remaining_modifed_graph = self.solve(H.subgraph(remaining_H_nodes), recursion_depth + 1)

            if remaining_modifed_graph is None:
                continue

            btp_score = 0.0

            for bd_node in list(refresh_boundary):
                is_refreshed = update_bd_node_in_sub(bd_node, subgraph, remaining_modifed_graph)
                if is_refreshed:
                    refresh_boundary.remove(bd_node)

            new_graph = nx.compose(subgraph, remaining_modifed_graph)
            btp_node_list = list()
            for bd_node in refresh_boundary:
                lv_to_restore = 1
                btp_node = add_btp_layer_in_graph(new_graph, bd_node, self.param_dict, lv_to_restore)
                btp_node_list.append(btp_node)

            new_graph_ab = LayerAbstractGraph()
            new_graph_ab.dag = new_graph

            if config.get('MPC_REFRESH', False):
                absorb_scale_for_approx_poly(new_graph_ab, config.get('MPC_REFRESH', False))
                update_subgraph_node_param(new_graph_ab.dag, self.param_dict, 'param0')
                change_skip_for_graph(new_graph_ab)
                update_subgraph_node_param(new_graph_ab.dag, self.param_dict, 'param0', True)
            level_below_max, level_info = self.split_graph_and_set_level((new_graph_ab.dag))

            for node in level_info.keys():
                new_graph_ab.dag.nodes[node]['level'] = level_info[node]
            if not level_below_max:
                print('over level ')
                continue
            self.process_btp_level_cost(new_graph_ab.dag)
            add_drop_level_for_graph(new_graph_ab, None)
            subgraph_cost = calculate_compute_score_for_graph(new_graph, subgraph, self.param_dict)
            for node in btp_node_list:
                if not config.get('MPC_REFRESH', False):
                    s_param = BtpScoreParam(new_graph_ab.dag, node, self.param_dict)
                else:
                    s_param = MpcScoreParam(new_graph_ab.dag, node, self.param_dict)
                score = s_param.get_score()
                new_graph.nodes[node]['score'] = score
                btp_score += score

            total_cost = rest_cost + subgraph_cost + btp_score
            if total_cost < best_cost:
                best_cost = total_cost
                best_graph = new_graph

        if best_graph is None:
            print('All subgraphs exceeded level limit, no valid solution found')
            return float('inf'), None

        return best_cost, best_graph

    def run(self):
        """
        Top-down recursive partition with memoization.
        Returns (segments, min_cost).
        """

        optimal_cost, optimal_graph = self.solve(self.entire_graph)

        if optimal_graph is None:
            print('Failed to find valid graph partition (all attempts exceeded level limit)')
            return None, None

        print(f'Best cost: {optimal_cost}')
        return optimal_cost, optimal_graph


def optimize_task_segments(pt_graph, temperature):
    """
    Split a task graph into segments with the given capacity and fixed cost.
    Returns (segments, min_cost).
    """
    graph_partitioner = GraphPartitioner(pt_graph.dag, temperature=temperature)
    return graph_partitioner.run()


def restore_node_attributes(G: nx.DiGraph):
    for node in G.nodes:
        for attr in node.__dict__.keys():
            if attr in G.nodes[node]:
                node.__dict__[attr] = G.nodes[node][attr]


def init_graph_level(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, FeatureNode):
            graph.dag.nodes[node]['level'] = 0
            graph.dag.nodes[node]['pack_num'] = 1


def remove_drop_level_nodes(graph: LayerAbstractGraph) -> LayerAbstractGraph:
    """
    Remove all drop_level nodes from the graph and reconnect the graph

    The structure of drop_level nodes is typically:
    input_feature -> drop_level_compute -> output_feature (with _drop_level_output suffix)

    Structure after removal:
    input_feature -> (directly connected to subsequent nodes)

    Args:
        graph: LayerAbstractGraph object

    Returns:
        LayerAbstractGraph: Graph after removing drop_level nodes
    """
    nodes_to_remove = []

    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode) and 'drop_level' in node.layer_type:
            nodes_to_remove.append(node)

    for drop_node in nodes_to_remove:
        input_features = list(graph.dag.predecessors(drop_node))
        output_features = list(graph.dag.successors(drop_node))

        if not input_features or not output_features:
            print(f'Warning: drop_level node {drop_node.layer_id} has no input or output')
            continue

        input_feature = input_features[0]
        output_feature = output_features[0]

        downstream_nodes = list(graph.dag.successors(output_feature))

        for downstream in downstream_nodes:
            graph.dag.add_edge(input_feature, downstream)

        graph.dag.remove_node(drop_node)
        graph.dag.remove_node(output_feature)

    return graph


def compile_graph(
    input_file_path: str,
    output_dir: str | None = None,
    temperature=1.0,
):
    pt_graph = LayerAbstractGraph.from_json(input_file_path)

    substitute_layers_for_btp(pt_graph)
    init_graph_level(pt_graph)

    # pt_graph.to_json(dict(), str(fhe_friendly_path))
    set_is_adaptive_avgpool(pt_graph)
    update_shape_for_btp(pt_graph)
    update_skip_for_btp(pt_graph)
    update_level_cost_for_btp(pt_graph)
    # init_graph_level(pt_graph)
    absorb_scale_for_approx_poly(pt_graph)

    score, compiled_graph = optimize_task_segments(pt_graph, temperature=temperature)

    if compiled_graph is None:
        return None, None

    return score, compiled_graph


def reset_level_and_check_level(total_graph: LayerAbstractGraph):
    g = GraphPartitioner(total_graph.dag)
    level_below_max, level_info = g.split_graph_and_set_level((total_graph.dag))

    for node in level_info.keys():
        total_graph.dag.nodes[node]['level'] = level_info[node]
    if not level_below_max:
        print('over level ')
        return None
    return total_graph


def compile_model_btp(
    input_file_path: Path, output_dir: Path, temperature=1.0, stdout=False
) -> tuple[float, LayerAbstractGraph]:
    """
    Compile model with bootstrapping

    Returns:
        tuple[float, LayerAbstractGraph]: (score, total_graph) if successful, (inf, None) if failed
    """
    seed = np.random.randint(1, 1000000)

    random.seed(seed)
    np.random.seed(seed)

    score, compiled_graph = compile_graph(
        input_file_path=str(input_file_path),
        output_dir=str(output_dir),
        temperature=temperature,
    )

    if compiled_graph is None:
        print(f'Compilation failed due to level limit exceeded (seed={seed})')
        return float('inf'), None

    total_graph = LayerAbstractGraph()
    total_graph.dag = compiled_graph
    restore_node_attributes(total_graph.dag)

    for node in total_graph.dag.nodes:
        if isinstance(node, ComputeNode):
            node.up_scale_str = list()
            node.down_scale_str = list()
    if config.get('GRAPH_TYPE', 'btp') == 'btp':
        set_graph_scale(total_graph)
        # process_batch_norm(total_graph)
        if config.get('SET_LEVEL_MAX', False):
            process_level_for_graph(total_graph)

    if config.get('MPC_REFRESH', False):
        balance_scale_for_graph(total_graph)
        if not reset_level_and_check_level(total_graph):
            print('level over')
            return float('inf'), None
        update_btp_to_mpc_refresh(total_graph)

    return score, total_graph


def run_single_compile(args):
    """Wrapper function for multiprocessing - runs a single compilation"""
    input_file_path, output_dir, temperature = args
    score, graph = compile_model_btp(input_file_path, output_dir, temperature, stdout=True)
    return score, graph


def run_parallel(
    num_experiments: int,
    input_file_path: Path,
    output_dir: Path,
    temperature: float,
    num_workers: int = 16,
):
    """
    Run multiple compilations in parallel and select the best result

    Simplified version: directly calls compile_model_btp in parallel processes

    Args:
        num_experiments: Number of parallel compilation runs
        input_file_path: Input pt.json file path
        output_dir: Output directory (will contain erg0.json, task_config.json)
        temperature: Temperature parameter for randomization
        num_workers: Number of parallel worker processes
    """
    print(f'Starting {num_experiments} parallel runs with {num_workers} processes...')

    # Prepare arguments for each run
    args_list = [(input_file_path, output_dir, temperature) for _ in range(num_experiments)]

    # Run compilations in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_single_compile, args_list))

    # Filter out failed results (score == inf)
    valid_results = [(score, graph) for score, graph in results if graph is not None]
    failed_count = num_experiments - len(valid_results)

    print(f'\n=== Summary ===')
    print(f'Total runs: {num_experiments}')
    print(f'Successful: {len(valid_results)}')
    print(f'Failed (level limit exceeded): {failed_count}')

    if not valid_results:
        print('ERROR: All runs failed! No valid results to save.')
        return

    # Find the best result (minimum score)
    best_score, best_graph = min(valid_results, key=lambda x: x[0])

    # Create directory structure: output_dir/task/server/ergs/ and output_dir/task/client/
    task_dir = output_dir / 'task'
    server_dir = task_dir / 'server'
    client_dir = task_dir / 'client'
    ergs_dir = server_dir / 'ergs'

    # Create all directories
    ergs_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    # Save files
    # 1. erg0.json in server/ergs/ (this is the ct.json)
    erg0_path = ergs_dir / 'erg0.json'
    best_graph.to_json(dict(), str(erg0_path), score=best_score)

    # 2. task_config.json in both server and client
    # graph_to_task_config will create task_config.json in server_dir
    graph_to_task_config([best_graph], str(server_dir))

    # Copy task_config.json to client as well
    server_task_config = server_dir / 'task_config.json'
    client_task_config = client_dir / 'task_config.json'
    if server_task_config.exists():
        shutil.copy(str(server_task_config), str(client_task_config))

    # 3. Create ckks_parameter.json for both server and client
    # Get POLY_N from config
    poly_n = config.get('POLY_N', 65536)
    poly_to_mod = {8192: 31, 16384: 34, 65536: 41}
    mod_bit = poly_to_mod[poly_n]
    ckks_param = {
        'param0': {
            'poly_modulus_degree': poly_n,
            'n_mult_level': config.get('MAX_LEVEL'),
            'coeff_modulus_bit_length': mod_bit,
            'special_prime_bit_length': mod_bit,
            'pack_num': 4.0,
        }
    }

    # Save to server
    with open(server_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)

    # Save to client
    with open(client_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)

    print(f'\n=== Results ===')
    print(f'Best score: {best_score}')
    print(f'Output directory: {output_dir}')
    print(f'Generated structure:')
    print(f'  task/')
    print(f'    ├── server/')
    print(f'    │   ├── ergs/erg0.json')
    print(f'    │   ├── task_config.json')
    print(f'    │   └── ckks_parameter.json')
    print(f'    └── client/')
    print(f'        ├── task_config.json')
    print(f'        └── ckks_parameter.json')


if __name__ == '__main__':
    # Default parameter configuration
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_NUM_EXPERIMENTS = 128
    DEFAULT_NUM_WORKERS = 16

    argparser = argparse.ArgumentParser()
    argparser.add_argument('input_file', type=str, help='Input file path (pt.json)')
    argparser.add_argument(
        'output_path',
        type=str,
        nargs='?',  # Optional positional parameter
        default=None,
        help='Output directory path (will contain erg0.json, task_config.json)',
    )
    # Configuration arguments
    argparser.add_argument(
        '--poly_n',
        type=int,
        choices=[8192, 16384, 65536],
        default=None,
        help='Polynomial modulus degree (POLY_N): 8192, 16384, or 65536',
    )
    argparser.add_argument(
        '--style',
        type=str,
        choices=['ordinary', 'multiplexed'],
        default=None,
        help="Computation style (STYLE): 'ordinary' or 'multiplexed'",
    )
    argparser.add_argument(
        '--graph_type', type=str, choices=['btp'], default=None, help="Graph type (GRAPH_TYPE): 'btp'"
    )
    args = argparser.parse_args()

    # Initialize configuration based on command line arguments (or use defaults)
    init_config_with_args(poly_n=args.poly_n, style=args.style, graph_type=args.graph_type)

    # Main process mode: run multi-process parallel compilation
    print(f'Using temperature: {DEFAULT_TEMPERATURE}')
    print(f'Running {DEFAULT_NUM_EXPERIMENTS} parallel compilations with {DEFAULT_NUM_WORKERS} processes')

    input_path = Path(args.input_file)

    # Determine output directory from command line argument
    if args.output_path:
        output_dir = Path(args.output_path)
    else:
        # Use input file's parent directory as default
        output_dir = input_path.parent

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nInput file: {input_path}')
    print(f'Output directory: {output_dir}')
    print(f'Will generate: erg0.json, task_config.json\n')

    run_parallel(
        num_experiments=DEFAULT_NUM_EXPERIMENTS,
        input_file_path=input_path,
        output_dir=output_dir,
        temperature=DEFAULT_TEMPERATURE,
        num_workers=DEFAULT_NUM_WORKERS,
    )
