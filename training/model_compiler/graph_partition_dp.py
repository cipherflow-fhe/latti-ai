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

from itertools import product
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
from tqdm import tqdm
from typing import Final, NamedTuple

from components import LayerAbstractGraph, ComputeNode, FeatureNode, config
import components
import processor
import transforms

from processor import (
    substitute_layers_for_btp,
    process_levels,
    FheParameter,
    BtpScoreParam,
    MpcScoreParam,
    FheScoreParam,
    update_subgraph_node_param,
    get_slot_num,
    change_skip_for_graph,
    set_is_adaptive_avgpool,
    graph_to_task_config,
)


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
    param_dict['param0'] = FheParameter(
        name=config.fhe_param.name,
        poly_modulus_degree=config.fhe_param.poly_modulus_degree,
        q=config.fhe_param.q,
        p=config.fhe_param.p,
        n_slots=config.fhe_param.n_slots,
        max_level=config.fhe_param.max_level,
        log_default_scale=config.fhe_param.log_default_scale,
    )
    return param_dict


def calculate_compute_score_for_graph(
    enclosing_graph: nx.DiGraph, grow: nx.DiGraph, param_dict: dict[str, FheParameter]
) -> float:
    compute_score = 0.0
    for compute in grow.nodes:
        if not isinstance(compute, ComputeNode):
            continue
        compute_score += get_compute_score(enclosing_graph, compute, param_dict)
    return compute_score


def get_compute_score(
    enclosing_graph: nx.DiGraph,
    compute: ComputeNode,
    param_dict: dict[str, FheParameter],
) -> float:
    supported_fhe_score_layers = {
        'conv1d',
        'conv2d',
        'fc0',
        'avgpool1d',
        'avgpool2d',
        'polyact',
        'mult_scalar',
        'add',
        'add2d',
        'add_pt',
        'pcm_add_pt',
        'parcpmm',
        'partranspose',
        'parccmm',
        'pcmgamma',
        'pcmpoly',
        'pcmstats',
        'pcmcenter',
        'pcminit',
        'pcmgs',
        'pcmaffine',
        'upsample_nearest',
        'resize',
    }
    if compute.layer_type in supported_fhe_score_layers:
        preds = list(enclosing_graph.predecessors(compute))
        level = min(enclosing_graph.nodes[p]['level'] for p in preds)
        s_param = FheScoreParam(enclosing_graph, compute, param_dict, level)
        score = s_param.get_score()
        return score
    return 0.0


def get_restoring_score(dag, restore_node, param_dict):
    if not config.mpc_refresh:
        s_param = BtpScoreParam(dag, restore_node, param_dict)
    else:
        s_param = MpcScoreParam(dag, restore_node, param_dict)
    return s_param.get_score()


def get_min_feature_level() -> int:
    return 1 if config.mpc_refresh or config.graph_type == 'mpc' or config.set_btp_scale is not None else 0


def restore_level_at(new_graph: nx.DiGraph, node: FeatureNode, param_dict):
    restore_node = transforms.add_btp_layer(
        new_graph, node, param_dict, config.fhe_param.max_level - new_graph.nodes[node]['level']
    )
    score = get_restoring_score(new_graph, restore_node, param_dict)
    new_graph.nodes[restore_node]['score'] = score
    succ = list(new_graph.successors(restore_node))[0]
    new_graph.nodes[succ]['level'] = config.fhe_param.max_level
    return score


def reconstruct_graph_from_vec(
    graph_vec: np.ndarray,
    template_graph: nx.DiGraph,
    node_to_idx: dict[FeatureNode, int],
    param_dict: dict[str, FheParameter],
) -> nx.DiGraph:
    new_graph = template_graph.copy()
    for node in template_graph.nodes:
        if not isinstance(node, FeatureNode):
            continue

        node_idx = node_to_idx[node]
        lv = int(graph_vec[node_idx])
        if lv < AUX_LV:
            new_graph.nodes[node]['level'] = lv
        else:
            new_graph.nodes[node]['level'] = get_min_feature_level()
            restore_level_at(new_graph, node, param_dict)

    return new_graph


def update_btp_to_mpc_refresh(graph: LayerAbstractGraph):
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            if node.layer_type == 'bootstrapping':
                node.layer_type = 'mpc_refresh'


class NodeLevel(NamedTuple):
    node_idx: int
    level: int


# Auxiliary level used to indicate the node is refreshed to max level by a restore node,
# and can be used for absorbing later nodes without generating new restore nodes.
AUX_LV = 255


class GraphPartitioner:
    def __init__(self, entire_graph: nx.DiGraph, temperature: float = 1.0):
        self.entire_graph = entire_graph
        self.param_dict = generate_param_dict_for_graph()

        if temperature < 0:
            raise ValueError('Temperature must be non-negative. If set to 0, a greedy algorithm will be used.')
        self.temperature = temperature

    def inspect_level_backward(self, subgraph: nx.DiGraph):
        max_level = -1
        level_dict: dict[FeatureNode, int] = {}
        subg_nodes = subgraph.nodes
        for node in reversed(list(nx.topological_sort(subgraph))):
            if isinstance(node, ComputeNode):
                continue

            succ_c = list(subgraph.successors(node))
            if len(succ_c) == 0:
                level_dict[node] = get_min_feature_level()
            else:
                successing_subg_compute_nodes = [c for c in succ_c if c in subg_nodes]
                input_feature_lv: list[int] = []
                for c in successing_subg_compute_nodes:
                    assert isinstance(c, ComputeNode)
                    for feat in subgraph.successors(c):
                        assert isinstance(feat, FeatureNode)

                        input_feature_lv.append(level_dict[feat] + subgraph.nodes[c]['level_cost'])

                level_dict[node] = max(input_feature_lv)
                if level_dict[node] > config.fhe_param.max_level:
                    return False, -1, level_dict

            max_level = max(max_level, level_dict[node])
        return True, max_level, level_dict

    def process_btp_level_cost(self, dag: nx.DiGraph):
        for node in dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                preds: list[FeatureNode] = list(dag.predecessors(node))
                succs: list[FeatureNode] = list(dag.successors(node))
                dag.nodes[node]['level_cost'] = dag.nodes[preds[0]]['level'] - dag.nodes[succs[0]]['level']

    def generate_solutions(
        self,
        new_node: FeatureNode,
        frontier: list[NodeLevel],
        frontier_solutions: dict[tuple[int], tuple[float, np.ndarray]],
        processed_feature_nodes: set[FeatureNode],
        node_to_idx: dict[FeatureNode, int],
        idx_to_node: dict[int, FeatureNode],
        dag: nx.DiGraph,
    ):
        leading_comp: ComputeNode = next(dag.predecessors(new_node))
        predecessors: list[FeatureNode] = list(dag.predecessors(leading_comp))
        pred_frontier = [f for f in frontier if idx_to_node[f.node_idx] in predecessors]
        other_frontier = [f for f in frontier if idx_to_node[f.node_idx] not in predecessors]
        frontier = pred_frontier + other_frontier

        min_feature_level = get_min_feature_level()
        new_frontier = frontier.copy()
        new_frontier.append(NodeLevel(node_to_idx[new_node], min_feature_level))
        processed_feature_nodes.add(new_node)
        nodes_became_internal: list[int] = []
        for node_max_lv in frontier:
            internal_flag = True
            for comp in dag.successors(idx_to_node[node_max_lv.node_idx]):
                for succ in dag.successors(comp):
                    if succ not in processed_feature_nodes:
                        internal_flag = False
            if internal_flag:
                nodes_became_internal.append(node_max_lv.node_idx)
                new_frontier.remove(node_max_lv)

        new_frontier_solutions = dict()

        for terminal_lv in range(min_feature_level, config.fhe_param.max_level + 1):
            if dag.nodes[leading_comp]['level_cost'] + terminal_lv > config.fhe_param.max_level:
                continue

            frontier_lvs = []
            dag.nodes[new_node]['level'] = terminal_lv
            for node_max_lv in pred_frontier:
                frontier_lvs.append(
                    list(range(dag.nodes[leading_comp]['level_cost'] + terminal_lv, node_max_lv.level + 1)) + [AUX_LV]
                )
            for node_max_lv in other_frontier:
                frontier_lvs.append(list(range(min_feature_level, node_max_lv.level + 1)) + [AUX_LV])

            frontier_lv_product = product(*frontier_lvs)

            for lv_comb in frontier_lv_product:
                frontier_key = []
                new_frontier_key = []
                for node_max_lv, lv in zip(frontier, lv_comb):
                    frontier_key.append(NodeLevel(node_max_lv.node_idx, lv))
                    if node_max_lv.node_idx not in nodes_became_internal:
                        new_frontier_key.append(NodeLevel(node_max_lv.node_idx, lv))
                new_frontier_key.append(NodeLevel(node_to_idx[new_node], terminal_lv))
                new_frontier_key.sort(key=lambda x: x.node_idx)
                frontier_key.sort(key=lambda x: x.node_idx)

                if tuple(frontier_key) not in frontier_solutions:
                    continue

                initial_score, sol_graph_vec = frontier_solutions[tuple(frontier_key)]

                for node_max_lv, lv in zip(frontier, lv_comb):
                    dag.nodes[idx_to_node[node_max_lv.node_idx]]['level'] = (
                        lv if lv < AUX_LV else config.fhe_param.max_level
                    )

                sol_cost = initial_score + get_compute_score(dag, leading_comp, self.param_dict)

                new_frontier_key_tuple = tuple(new_frontier_key)
                if (
                    new_frontier_key_tuple not in new_frontier_solutions
                    or sol_cost < new_frontier_solutions[new_frontier_key_tuple][0]
                ):
                    new_sol_graph_vec = sol_graph_vec.copy()
                    new_sol_graph_vec[node_to_idx[new_node]] = terminal_lv
                    new_frontier_solutions[new_frontier_key_tuple] = (sol_cost, new_sol_graph_vec)

            # leaf nodes only need the minimum output-level solution.
            if len(list(dag.successors(new_node))) == 0:
                break

            new_frontier[-1] = NodeLevel(node_to_idx[new_node], terminal_lv)

            if terminal_lv == min_feature_level:
                aux_lv_solutions = {}
                for k, solution in new_frontier_solutions.items():
                    new_node_lv_idx = k.index(NodeLevel(node_to_idx[new_node], terminal_lv))
                    assert k[new_node_lv_idx].level == min_feature_level
                    sol_key = list(k)
                    sol_key[new_node_lv_idx] = NodeLevel(node_to_idx[new_node], AUX_LV)

                    sol_graph_vec_aux_lv = solution[1].copy()
                    sol_graph_vec_aux_lv[node_to_idx[new_node]] = AUX_LV
                    sol_aux_lv_score = get_restoring_score(dag, leading_comp, self.param_dict)
                    aux_lv_solutions[tuple(sol_key)] = (
                        solution[0] + sol_aux_lv_score,
                        sol_graph_vec_aux_lv,
                    )

                new_frontier_solutions |= aux_lv_solutions

        return new_frontier, new_frontier_solutions

    def solve(self, H: nx.DiGraph) -> tuple[float, nx.DiGraph]:
        if len(H.nodes) == 0:
            return 0.0, nx.DiGraph()

        topo_nodes = list(nx.topological_sort(H))
        topo_rank = {node: idx for idx, node in enumerate(topo_nodes)}

        source_feature_nodes = sorted(
            [node for node in H.nodes if isinstance(node, FeatureNode) and len(list(H.predecessors(node))) == 0],
            key=lambda node: topo_rank[node],
        )
        all_feature_nodes = [node for node in topo_nodes if isinstance(node, FeatureNode)]

        sorted_nodes: list[FeatureNode] = []
        activated_feature_nodes: set[FeatureNode] = set()

        def activate_feature_node(node: FeatureNode):
            if node in activated_feature_nodes:
                return

            activated_feature_nodes.add(node)
            sorted_nodes.append(node)

            ready_successors = sorted(list(H.successors(node)), key=lambda comp: topo_rank[comp])
            for comp in ready_successors:
                leading_features = list(H.predecessors(comp))
                if not all(pred in activated_feature_nodes for pred in leading_features):
                    continue

                output_features = list(H.successors(comp))
                activate_feature_node(output_features[0])

        for node in source_feature_nodes:
            activate_feature_node(node)

        while len(sorted_nodes) < len(all_feature_nodes):
            progressed = False
            for node in all_feature_nodes:
                if node in activated_feature_nodes:
                    continue

                leading_computes = list(H.predecessors(node))
                if len(leading_computes) == 0:
                    activate_feature_node(node)
                    progressed = True
                    break

                leading_features = list(H.predecessors(leading_computes[0]))
                if all(pred in activated_feature_nodes for pred in leading_features):
                    activate_feature_node(node)
                    progressed = True
                    break

            if not progressed:
                raise RuntimeError('Failed to construct a depth-first feature traversal order for the DAG')

        idx = 0
        node_to_idx = {}
        idx_to_node = {}
        for node in sorted_nodes:
            if isinstance(node, FeatureNode):
                node_to_idx[node] = idx
                idx_to_node[idx] = node
                idx += 1
        frontier: list[NodeLevel] = []
        processed_feature_nodes: set[FeatureNode] = set()

        # the frontier_solutions dict stores the best solution for each combination of levels (plus an auxiliary lv) of the frontier nodes,
        # e.g. {(node1_index, level2, node2_index, level3, node3_index, level1): (cost, graph_vec)},
        # where the nodes are sorted by their id to ensure unique representation of the frontier state.
        frontier_solutions: dict[tuple, float] = {}
        for node in source_feature_nodes:
            frontier.append(NodeLevel(node_to_idx[node], config.fhe_param.max_level))
            processed_feature_nodes.add(node)

        min_feature_level = get_min_feature_level()
        frontier_indices = [x.node_idx for x in frontier]
        for lv_comb in product(range(min_feature_level, config.fhe_param.max_level + 1), repeat=len(frontier)):
            init_graph_vec = np.zeros(len(node_to_idx), dtype=np.uint8)
            node_lv: list[NodeLevel] = []
            for idx, lv in zip(frontier_indices, lv_comb):
                node_lv.append(NodeLevel(idx, lv))
                init_graph_vec[idx] = lv

            node_lv.sort(key=lambda x: x.node_idx)
            frontier_solutions[tuple(node_lv)] = (0.0, init_graph_vec)

        pbar = tqdm(
            desc=f'Traversing through graph',
            unit='nodes',
            total=len(sorted_nodes) - len(source_feature_nodes),
        )

        for idx, node in enumerate(sorted_nodes):
            if node in source_feature_nodes:
                continue

            frontier, frontier_solutions = self.generate_solutions(
                node, frontier, frontier_solutions, processed_feature_nodes, node_to_idx, idx_to_node, H
            )
            pbar.update(1)

        final_solution_frontier = tuple(
            sorted((NodeLevel(x.node_idx, min_feature_level) for x in frontier), key=lambda x: x.node_idx)
        )
        final_score, final_dag_vec = frontier_solutions[final_solution_frontier]

        final_dag = reconstruct_graph_from_vec(final_dag_vec, H, node_to_idx, self.param_dict)

        temp_ab = LayerAbstractGraph()
        temp_ab.dag = final_dag
        # transforms.insert_drop_level_layers(temp_ab)

        return final_score, temp_ab.dag

    def run(self):
        """
        Top-down recursive partition with memoization.
        Returns (segments, min_cost).
        """

        result = []
        optimal_cost = 0.0
        for sub in nx.weakly_connected_components(self.entire_graph):
            sub = self.entire_graph.subgraph(sub).copy()
            cost, graph = self.solve(sub)
            optimal_cost += cost
            result.append(graph)

            if graph is None:
                print('Failed to find valid graph partition (all attempts exceeded level limit)')
                return None, None

        print(f'Best cost: {optimal_cost}')
        return optimal_cost, nx.compose_all(result)


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


def compile_graph(
    pt_graph: LayerAbstractGraph | None = None,
    temperature=1.0,
):
    score, compiled_graph = optimize_task_segments(pt_graph, temperature=temperature)

    if compiled_graph is None:
        return None, None

    return score, compiled_graph


def reset_level_and_check_level(total_graph: LayerAbstractGraph):
    g = GraphPartitioner(total_graph.dag)
    level_below_max, max_level, level_info = g.inspect_level_backward((total_graph.dag))

    for node in level_info.keys():
        total_graph.dag.nodes[node]['level'] = level_info[node]
    if not level_below_max:
        print('over level ')
        return None
    return total_graph


def compile_model_btp(
    pt_graph_prepared: LayerAbstractGraph | None = None,
    temperature=1.0,
    stdout=False,
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
        pt_graph=pt_graph_prepared,
        temperature=temperature,
    )

    if compiled_graph is None:
        print(f'Compilation failed due to level limit exceeded (seed={seed})')
        return float('inf'), None

    total_graph = LayerAbstractGraph()
    total_graph.dag = compiled_graph
    restore_node_attributes(total_graph.dag)

    return score, total_graph


def run_single_compile(args):
    """Wrapper function for multiprocessing - runs a single compilation"""
    pt_graph_prepared, temperature = args
    score, graph = compile_model_btp(pt_graph_prepared, temperature, stdout=True)
    return score, graph


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
    # init_config_with_args(poly_n=args.poly_n, style=args.style, graph_type=args.graph_type)

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

    # run_pipeline(
    #     num_experiments=DEFAULT_NUM_EXPERIMENTS,
    #     input_file_path=input_path,
    #     output_dir=output_dir,
    #     temperature=DEFAULT_TEMPERATURE,
    #     num_workers=DEFAULT_NUM_WORKERS,
    # )
