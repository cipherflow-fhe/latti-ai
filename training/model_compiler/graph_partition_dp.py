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
        if compute.layer_type in ['conv2d', 'fc0', 'add2d', 'polyact', 'avgpool1d', 'avgpool2d']:
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


class NodeLevel(NamedTuple):
    node_id: str
    level: int


AUX_LV = 99999


class GraphPartitioner:
    def __init__(self, entire_graph: nx.DiGraph, temperature: float = 1.0):
        self.entire_graph = entire_graph
        self.param_dict = generate_param_dict_for_graph()

        if temperature < 0:
            raise ValueError('Temperature must be non-negative. If set to 0, a greedy algorithm will be used.')
        self.temperature = temperature
        self.pbar = tqdm(
            desc=f'Subgraph explorations (temperature={self.temperature})',
            unit='nodes',
            total=self.entire_graph.number_of_nodes(),
        )

    def inspect_level_backward(self, subgraph: nx.DiGraph):
        max_level = -1
        level_dict: dict[FeatureNode, int] = {}
        subg_nodes = subgraph.nodes
        for node in reversed(list(nx.topological_sort(subgraph))):
            if isinstance(node, ComputeNode):
                continue

            succ_c = list(subgraph.successors(node))
            if len(succ_c) == 0:
                if config.mpc_refresh or config.graph_type == 'mpc':
                    level_dict[node] = 1
                elif config.graph_type == 'btp' and not config.mpc_refresh:
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
                if level_dict[node] > config.fhe_param.max_level:
                    return False, -1, level_dict

            max_level = max(max_level, level_dict[node])
        return True, max_level, level_dict

    # def split_graph_and_set_level(self, graph_with_btp: nx.DiGraph):
    #     splitted_graph = LayerAbstractGraph()
    #     splitted_graph.dag = graph_with_btp.copy()
    #     btp_nodes = list()
    #     for compute in splitted_graph.dag.nodes:
    #         if isinstance(compute, ComputeNode):
    #             if compute.layer_type == 'bootstrapping':
    #                 btp_nodes.append(compute)
    #     splitted_graph.dag.remove_nodes_from(btp_nodes)

    #     weak_components = list(nx.weakly_connected_components(splitted_graph.dag))
    #     subgraphs: list[LayerAbstractGraph] = list()
    #     for component in weak_components:
    #         if len(component) > 1:
    #             sub = LayerAbstractGraph()
    #             sub.dag = splitted_graph.dag.subgraph(component).copy()
    #             subgraphs.append(sub)
    #     res_dict = dict()
    #     for sub in subgraphs:
    #         res = self.inspect_level_backward(sub.dag)
    #         if not res[0]:
    #             return False, dict()
    #         res_dict.update(res[2])
    #     return True, res_dict

    def process_btp_level_cost(self, dag: nx.DiGraph):
        for node in dag.nodes:
            if isinstance(node, ComputeNode) and node.layer_type == 'bootstrapping':
                preds: list[FeatureNode] = list(dag.predecessors(node))
                succs: list[FeatureNode] = list(dag.successors(node))
                dag.nodes[node]['level_cost'] = dag.nodes[preds[0]]['level'] - dag.nodes[succs[0]]['level']

    def generate_solutions(
        self,
        new_node: FeatureNode,
        frontier: list[FeatureNode],
        frontier_solutions: dict[tuple[NodeLevel, ...], tuple[float, nx.DiGraph]],
        processed_feature_nodes: set[FeatureNode],
        dag: nx.DiGraph,
    ):
        leading_comp: ComputeNode = next(dag.predecessors(new_node))
        predecessors: list[FeatureNode] = list(dag.predecessors(leading_comp))
        new_frontier = frontier.copy()
        new_frontier.append(new_node)
        processed_feature_nodes.add(new_node)
        nodes_became_internal: list[FeatureNode] = []
        for node in frontier:
            internal_flag = True
            for comp in dag.successors(node):
                for succ in dag.successors(comp):
                    if succ not in processed_feature_nodes:
                        internal_flag = False
            if internal_flag:
                nodes_became_internal.append(node)

        for n in nodes_became_internal:
            new_frontier.remove(n)
        new_frontier_ids = [nd.node_id for nd in new_frontier]

        new_frontier_solutions = dict()

        for terminal_lv in range(config.fhe_param.max_level + 1):
            for node_lv_tuple in frontier_solutions.keys():
                admissible = True
                min_level = config.fhe_param.max_level
                for node_lv in node_lv_tuple:
                    if node_lv.node_id not in (node.node_id for node in predecessors):
                        continue
                    min_level = min(min_level, node_lv.level)
                    if min_level - dag.nodes[leading_comp]['level_cost'] < terminal_lv:
                        admissible = False
                        break
                if not admissible:
                    continue

                initial_score = frontier_solutions[node_lv_tuple][0]
                sol_graph = frontier_solutions[node_lv_tuple][1].copy()
                sol_graph.add_node(leading_comp, **dag.nodes[leading_comp])
                sol_graph.add_node(new_node, **dag.nodes[new_node])
                for pred in predecessors:
                    pred_lv = next((lv for n, lv in node_lv_tuple if n == pred.node_id), None)
                    if pred_lv is None:
                        raise ValueError('Predecessor node level must exist in the frontier state key')

                    # if the predecessor node is at auxiliary level, it means we have added a restoring node immediately after it,
                    # so we should connect the leading compute node to the restoring node instead of the original predecessor node.
                    if pred_lv == AUX_LV:
                        restoring_node = next(sol_graph.successors(pred))
                        restored_node = next(sol_graph.successors(restoring_node))
                        sol_graph.add_edge(restored_node, leading_comp)
                    else:
                        sol_graph.add_edge(pred, leading_comp)
                sol_graph.add_edge(leading_comp, new_node)

                frontier_key = []
                for node_lv in node_lv_tuple:
                    if node_lv.node_id in new_frontier_ids:
                        frontier_key.append(node_lv)

                sol_graph_ab = LayerAbstractGraph()
                sol_graph_ab.dag = sol_graph
                sol_graph_ab.dag.nodes[new_node]['level'] = terminal_lv
                frontier_key.append(NodeLevel(new_node.node_id, terminal_lv))
                frontier_key.sort(key=lambda x: x[0])
                frontier_key_tuple = tuple(frontier_key)

                if config.mpc_refresh:
                    transforms.absorb_scale(sol_graph_ab, config.mpc_refresh)
                    update_subgraph_node_param(sol_graph, self.param_dict, 'param0')
                    change_skip_for_graph(sol_graph_ab)
                    update_subgraph_node_param(sol_graph, self.param_dict, 'param0', True)

                self.process_btp_level_cost(sol_graph)

                grow = sol_graph.subgraph(predecessors + [leading_comp, new_node]).copy()
                sol_cost = initial_score + calculate_compute_score_for_graph(sol_graph, grow, self.param_dict)

                if (
                    frontier_key_tuple not in new_frontier_solutions
                    or sol_cost < new_frontier_solutions[frontier_key_tuple][0]
                ):
                    new_frontier_solutions[frontier_key_tuple] = (sol_cost, sol_graph)

            if terminal_lv == 0:
                aux_lv_solutions = {}
                for k in new_frontier_solutions.keys():
                    new_node_lv_idx = [node_lv.node_id for node_lv in k].index(new_node.node_id)
                    if k[new_node_lv_idx].level != 0:
                        continue

                    sol_key = list(k)
                    sol_key[new_node_lv_idx] = NodeLevel(new_node.node_id, AUX_LV)
                    sol_graph_aux_lv = new_frontier_solutions[k][1].copy()
                    sol_aux_lv_score = self.restore_level_at(sol_graph_aux_lv, new_node)
                    aux_lv_solutions[tuple(sol_key)] = (
                        new_frontier_solutions[k][0] + sol_aux_lv_score,
                        sol_graph_aux_lv,
                    )

                new_frontier_solutions |= aux_lv_solutions

        return new_frontier, new_frontier_solutions

    def restore_level_at(self, new_graph: nx.DiGraph, node: FeatureNode):
        restore_node = transforms.add_btp_layer(
            new_graph, node, self.param_dict, config.fhe_param.max_level - config.fhe_param.max_level
        )
        if not config.mpc_refresh:
            s_param = BtpScoreParam(new_graph, restore_node, self.param_dict)
        else:
            s_param = MpcScoreParam(new_graph, restore_node, self.param_dict)
        score = s_param.get_score()
        new_graph.nodes[restore_node]['score'] = score
        succ = list(new_graph.successors(restore_node))[0]
        new_graph.nodes[succ]['level'] = config.fhe_param.max_level
        return score

    def solve(self, H: nx.DiGraph) -> tuple[float, nx.DiGraph]:
        self.pbar.update(1)
        if len(H.nodes) == 0:
            return 0.0, nx.DiGraph()

        sorted_nodes = list(nx.topological_sort(H))
        frontier: list[FeatureNode] = []
        processed_feature_nodes: set[FeatureNode] = set()

        # the frontier_solutions dict stores the best solution for each combination of levels (plus an auxiliary lv) of the frontier nodes,
        # e.g. {(NodeLevel(node1,level2), NodeLevel(node2,level3), NodeLevel(node3, level1)): (cost, modified_graph)},
        # where the nodes are sorted by their id to ensure unique representation of the frontier state.
        frontier_solutions: dict[tuple[NodeLevel, ...], tuple[float, nx.DiGraph]] = {}
        for cur_idx, node in enumerate(sorted_nodes):
            if isinstance(node, FeatureNode) and len(list(H.predecessors(node))) == 0:
                frontier.append(node)
                processed_feature_nodes.add(node)
            else:
                break

        for lv_comb in product(range(config.fhe_param.max_level + 1), repeat=len(frontier)):
            nodes_and_lv = sorted(zip(frontier, lv_comb), key=lambda x: x[0])
            frontier_state_key = tuple(NodeLevel(node.node_id, lv) for node, lv in nodes_and_lv)

            new_graph = H.subgraph(frontier).copy()
            for node, lv in nodes_and_lv:
                new_graph.nodes[node]['level'] = lv
            frontier_solutions[frontier_state_key] = (0.0, new_graph)

        for node in sorted_nodes[cur_idx + 1 :]:
            if isinstance(node, FeatureNode):
                frontier, frontier_solutions = self.generate_solutions(
                    node, frontier, frontier_solutions, processed_feature_nodes, H
                )

        final_solution_frontier = tuple(sorted((NodeLevel(node.node_id, 0) for node in frontier), key=lambda x: x[0]))
        final_score, final_dag = frontier_solutions[final_solution_frontier]

        temp_ab = LayerAbstractGraph()
        temp_ab.dag = final_dag
        transforms.insert_drop_level_layers(temp_ab)

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
