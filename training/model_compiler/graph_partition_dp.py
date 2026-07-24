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
import multiprocessing
import os
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
        'pdm_add_pt',
        'parcpmm',
        'partranspose',
        'parccmm',
        'pdmpcmm',
        'pdmtranspose',
        'pdmccmm',
        'pcmgamma',
        'pcmpoly',
        'pcmstats',
        'pcmcenter',
        'pcminit',
        'pcmgs',
        'pcmaffine',
        'pdmgamma',
        'pdmpoly',
        'pdmstats',
        'pdmcenter',
        'pdminit',
        'pdmgs',
        'pdmaffine',
        'upsample_nearest',
        'resize',
    }
    if compute.layer_type in supported_fhe_score_layers:
        preds = list(enclosing_graph.predecessors(compute))
        level = min(enclosing_graph.nodes[p]['level'] for p in preds)
        s_param = FheScoreParam(
            enclosing_graph,
            compute,
            param_dict,
            level,
            use_gpu=getattr(config, 'use_gpu', True),
        )
        score = s_param.get_score()
        return score
    return 0.0


def is_mpc_flow() -> bool:
    return config.graph_type in ('mpc_refresh', 'mpc_compute')


def get_restoring_score(dag, restore_node, param_dict):
    if is_mpc_flow():
        s_param = MpcScoreParam(dag, restore_node, param_dict)
    else:
        s_param = BtpScoreParam(
            dag,
            restore_node,
            param_dict,
            use_gpu=getattr(config, 'use_gpu', True),
        )
    return s_param.get_score()


def get_min_feature_level() -> int:
    return 1 if is_mpc_flow() or config.set_btp_scale is not None else 0


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


class FanoutRegion(NamedTuple):
    entry: FeatureNode
    exit: FeatureNode
    merge: ComputeNode
    nodes: frozenset
    internal_features: frozenset


class RegionPlan(NamedTuple):
    cost: float
    feature_levels: dict[FeatureNode, int]


# Auxiliary level used to indicate the node is refreshed to max level by a restore node,
# and can be used for absorbing later nodes without generating new restore nodes.
AUX_LV = 255


class GraphPartitioner:
    def __init__(self, entire_graph: nx.DiGraph, temperature: float = 1.0, enable_score_cache: bool = True):
        self.entire_graph = entire_graph
        self.param_dict = generate_param_dict_for_graph()
        self.enable_score_cache = enable_score_cache
        self.compute_score_cache: dict[tuple[ComputeNode, tuple[int, ...], tuple[int, ...]], float] = {}
        self.region_plan_cache: dict[FanoutRegion, dict[tuple[int, int], RegionPlan]] = {}

        if temperature < 0:
            raise ValueError('Temperature must be non-negative. If set to 0, a greedy algorithm will be used.')
        self.temperature = temperature

    def get_compute_score_cached(self, enclosing_graph: nx.DiGraph, compute: ComputeNode) -> float:
        if not self.enable_score_cache:
            return get_compute_score(enclosing_graph, compute, self.param_dict)

        preds = tuple(enclosing_graph.predecessors(compute))
        succs = tuple(enclosing_graph.successors(compute))
        pred_levels = tuple(int(enclosing_graph.nodes[p]['level']) for p in preds)
        succ_levels = tuple(int(enclosing_graph.nodes[s]['level']) for s in succs)
        cache_key = (compute, pred_levels, succ_levels)

        if cache_key not in self.compute_score_cache:
            self.compute_score_cache[cache_key] = get_compute_score(enclosing_graph, compute, self.param_dict)

        return self.compute_score_cache[cache_key]

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

    def _trace_linear_branch_to_source(
        self,
        dag: nx.DiGraph,
        start_feature: FeatureNode,
    ) -> tuple[list[FeatureNode], list[ComputeNode]] | None:
        features = [start_feature]
        computes = []
        cur_feature = start_feature
        seen = {cur_feature}

        while True:
            producers = [node for node in dag.predecessors(cur_feature) if isinstance(node, ComputeNode)]
            if not producers:
                return features, computes
            if len(producers) != 1:
                return features, computes

            producer = producers[0]
            producer_inputs = [node for node in dag.predecessors(producer) if isinstance(node, FeatureNode)]
            producer_outputs = [node for node in dag.successors(producer) if isinstance(node, FeatureNode)]
            if len(producer_inputs) != 1 or len(producer_outputs) != 1:
                return features, computes

            prev_feature = producer_inputs[0]
            if prev_feature in seen:
                return None

            computes.append(producer)
            features.append(prev_feature)
            seen.add(prev_feature)
            cur_feature = prev_feature

    def _collect_merge_leaf_features(
        self,
        dag: nx.DiGraph,
        merge: ComputeNode,
    ) -> tuple[list[FeatureNode], set] | None:
        if merge.layer_type not in {'add', 'add2d', 'concat2d'}:
            return None

        merge_outputs = [node for node in dag.successors(merge) if isinstance(node, FeatureNode)]
        if len(merge_outputs) != 1:
            return None

        chain_nodes = {merge, merge_outputs[0]}
        leaf_features: list[FeatureNode] = []
        seen_adds = set()

        def collect_from_add(add_node: ComputeNode) -> bool:
            if add_node in seen_adds:
                return False
            seen_adds.add(add_node)

            add_inputs = [node for node in dag.predecessors(add_node) if isinstance(node, FeatureNode)]
            if len(add_inputs) < 2:
                return False

            for feature in add_inputs:
                producers = [node for node in dag.predecessors(feature) if isinstance(node, ComputeNode)]
                producer = producers[0] if len(producers) == 1 else None
                if (
                    producer is not None
                    and producer.layer_type in {'add', 'add2d'}
                    and len([node for node in dag.successors(producer) if isinstance(node, FeatureNode)]) == 1
                    and dag.out_degree(feature) == 1
                ):
                    chain_nodes.add(feature)
                    chain_nodes.add(producer)
                    if not collect_from_add(producer):
                        return False
                else:
                    leaf_features.append(feature)
            return True

        if merge.layer_type in {'add', 'add2d'}:
            if not collect_from_add(merge):
                return None
        else:
            leaf_features.extend([node for node in dag.predecessors(merge) if isinstance(node, FeatureNode)])

        deduped_leaf_features = []
        seen_leaf_features = set()
        for feature in leaf_features:
            if feature in seen_leaf_features:
                continue
            seen_leaf_features.add(feature)
            deduped_leaf_features.append(feature)

        if len(deduped_leaf_features) < 2:
            return None

        return deduped_leaf_features, chain_nodes

    def _try_build_fanout_region(self, dag: nx.DiGraph, merge: ComputeNode) -> FanoutRegion | None:
        if merge.layer_type not in {'add', 'add2d', 'concat2d'}:
            return None

        merge_inputs_and_chain = self._collect_merge_leaf_features(dag, merge)
        if merge_inputs_and_chain is None:
            return None
        merge_inputs, chain_nodes = merge_inputs_and_chain
        merge_outputs = [node for node in dag.successors(merge) if isinstance(node, FeatureNode)]
        if len(merge_outputs) != 1:
            return None

        traced_branches = []
        common_features = None
        for feature in merge_inputs:
            traced = self._trace_linear_branch_to_source(dag, feature)
            if traced is None:
                return None
            branch_features, branch_computes = traced
            traced_branches.append((branch_features, branch_computes))
            feature_set = set(branch_features)
            common_features = feature_set if common_features is None else common_features & feature_set

        if not common_features:
            return None

        topo_rank = {node: index for index, node in enumerate(nx.topological_sort(dag))}
        entry = min(common_features, key=lambda node: topo_rank[node])
        if len([node for node in dag.successors(entry) if isinstance(node, ComputeNode)]) < 2:
            return None

        region_nodes = {entry, *chain_nodes}
        internal_features = set()
        internal_features.update(
            node for node in chain_nodes
            if isinstance(node, FeatureNode) and node is not entry
        )
        for branch_features, branch_computes in traced_branches:
            if entry not in branch_features:
                return None
            entry_pos = branch_features.index(entry)
            path_features = branch_features[:entry_pos]
            path_computes = branch_computes[:entry_pos]
            if not path_features:
                return None
            region_nodes.update(path_features)
            region_nodes.update(path_computes)
            internal_features.update(path_features)

        exit_feature = merge_outputs[0]
        internal_features.add(exit_feature)

        for node in region_nodes:
            if node is entry or node is exit_feature:
                continue
            for pred in dag.predecessors(node):
                if pred not in region_nodes:
                    return None
            for succ in dag.successors(node):
                if succ not in region_nodes:
                    return None

        return FanoutRegion(
            entry=entry,
            exit=exit_feature,
            merge=merge,
            nodes=frozenset(region_nodes),
            internal_features=frozenset(internal_features),
        )

    def _find_fanout_regions(self, dag: nx.DiGraph) -> list[FanoutRegion]:
        candidates = []
        topo_rank = {node: index for index, node in enumerate(nx.topological_sort(dag))}
        for node in topo_rank:
            if not isinstance(node, ComputeNode):
                continue
            region = self._try_build_fanout_region(dag, node)
            if region is not None:
                candidates.append(region)

        candidates.sort(key=lambda region: (-len(region.nodes), topo_rank[region.exit]))
        regions = []
        claimed_internal_features = set()
        for region in candidates:
            if region.internal_features & claimed_internal_features:
                continue
            regions.append(region)
            claimed_internal_features.update(region.internal_features)
        return sorted(regions, key=lambda region: topo_rank[region.exit])

    def _internalized_frontier_indices(
        self,
        dag: nx.DiGraph,
        frontier: list[NodeLevel],
        processed_feature_nodes: set[FeatureNode],
        idx_to_node: dict[int, FeatureNode],
    ) -> set[int]:
        internal = set()
        for node_max_lv in frontier:
            feature = idx_to_node[node_max_lv.node_idx]
            internal_flag = True
            for comp in dag.successors(feature):
                for succ in dag.successors(comp):
                    if succ not in processed_feature_nodes:
                        internal_flag = False
            if internal_flag:
                internal.add(node_max_lv.node_idx)
        return internal

    def _build_region_plan_table(
        self,
        dag: nx.DiGraph,
        region: FanoutRegion,
    ) -> dict[tuple[int, int], RegionPlan]:
        if region in self.region_plan_cache:
            return self.region_plan_cache[region]

        if region.merge.layer_type in {'add', 'add2d'}:
            plan_table = self._build_add_chain_region_plan_table(dag, region)
            self.region_plan_cache[region] = plan_table
            return plan_table

        region_dag = dag.subgraph(region.nodes).copy()
        sorted_features = [
            node for node in nx.topological_sort(region_dag)
            if isinstance(node, FeatureNode)
        ]
        plan_table = self._build_region_plan_table_by_input_levels(region_dag, sorted_features, region)
        self.region_plan_cache[region] = plan_table
        return plan_table

    def _build_region_plan_table_for_input_level(
        self,
        region_dag: nx.DiGraph,
        sorted_features: list[FeatureNode],
        region: FanoutRegion,
        input_level: int,
    ) -> dict[tuple[int, int], RegionPlan]:
        node_to_idx = {node: idx for idx, node in enumerate(sorted_features)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        entry_idx = node_to_idx[region.entry]
        exit_idx = node_to_idx[region.exit]
        min_feature_level = get_min_feature_level()

        frontier = [NodeLevel(entry_idx, input_level)]
        processed_feature_nodes = {region.entry}
        init_graph_vec = np.zeros(len(node_to_idx), dtype=np.uint8)
        init_graph_vec[entry_idx] = input_level
        frontier_solutions = {(NodeLevel(entry_idx, input_level),): (0.0, init_graph_vec)}

        for feature in sorted_features:
            if feature is region.entry:
                continue
            frontier, frontier_solutions = self.generate_solutions(
                feature,
                frontier,
                frontier_solutions,
                processed_feature_nodes,
                node_to_idx,
                idx_to_node,
                region_dag,
                leaf_min_only=feature is not region.exit,
            )

        plan_table: dict[tuple[int, int], RegionPlan] = {}
        for output_level in range(min_feature_level, config.fhe_param.max_level + 1):
            final_key = (NodeLevel(exit_idx, output_level),)
            if final_key not in frontier_solutions:
                continue
            cost, graph_vec = frontier_solutions[final_key]
            feature_levels = {
                feature: int(graph_vec[node_to_idx[feature]])
                for feature in sorted_features
                if feature is not region.entry
            }
            plan_table[(input_level, output_level)] = RegionPlan(cost, feature_levels)

        return plan_table

    def _region_input_level_worker_count(self, input_level_count: int, region_feature_count: int) -> int:
        if input_level_count < 2 or multiprocessing.current_process().name != 'MainProcess':
            return 1

        configured_workers = os.environ.get('LATTI_REGION_INPUT_LEVEL_WORKERS')
        if configured_workers is not None:
            try:
                return max(1, min(input_level_count, int(configured_workers)))
            except ValueError:
                print(f'Ignoring invalid LATTI_REGION_INPUT_LEVEL_WORKERS={configured_workers!r}')

        estimated_feature_steps = input_level_count * max(0, region_feature_count - 1)
        if estimated_feature_steps < 256:
            return 1

        return max(1, min(input_level_count, os.cpu_count() or 1, 4))

    def _build_region_plan_table_by_input_levels(
        self,
        region_dag: nx.DiGraph,
        sorted_features: list[FeatureNode],
        region: FanoutRegion,
    ) -> dict[tuple[int, int], RegionPlan]:
        min_feature_level = get_min_feature_level()
        input_levels = list(range(min_feature_level, config.fhe_param.max_level + 1))
        worker_count = self._region_input_level_worker_count(len(input_levels), len(sorted_features))

        if worker_count <= 1:
            plan_table = {}
            for input_level in input_levels:
                plan_table.update(
                    self._build_region_plan_table_for_input_level(region_dag, sorted_features, region, input_level)
                )
            return plan_table

        try:
            tasks = [
                (region_dag, sorted_features, region, input_level, self.enable_score_cache)
                for input_level in input_levels
            ]
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                serialized_tables = list(executor.map(_build_region_input_level_plan_table_worker, tasks))
        except Exception as exc:
            print(f'Region input-level parallel precompute failed ({exc}); falling back to serial precompute')
            plan_table = {}
            for input_level in input_levels:
                plan_table.update(
                    self._build_region_plan_table_for_input_level(region_dag, sorted_features, region, input_level)
                )
            return plan_table

        feature_by_id = {node.node_id: node for node in sorted_features}
        plan_table = {}
        for serialized_table in serialized_tables:
            plan_table.update(self._deserialize_region_plan_table(serialized_table, feature_by_id))
        return plan_table

    def _deserialize_region_plan_table(
        self,
        serialized_plan_table: dict[tuple[int, int], tuple[float, dict[str, int]]],
        feature_by_id: dict[str, FeatureNode],
    ) -> dict[tuple[int, int], RegionPlan]:
        plan_table = {}
        for boundary_levels, (cost, feature_levels_by_id) in serialized_plan_table.items():
            plan_table[boundary_levels] = RegionPlan(
                cost,
                {feature_by_id[node_id]: level for node_id, level in feature_levels_by_id.items()},
            )
        return plan_table

    def _region_plan_precompute_worker_count(self, region_count: int) -> int:
        if region_count < 2:
            return 1

        configured_workers = os.environ.get('LATTI_FANOUT_REGION_WORKERS')
        if configured_workers is not None:
            try:
                return max(1, min(region_count, int(configured_workers)))
            except ValueError:
                print(f'Ignoring invalid LATTI_FANOUT_REGION_WORKERS={configured_workers!r}')

        return max(1, min(region_count, os.cpu_count() or 1, 8))

    def _precompute_region_plan_tables(
        self,
        dag: nx.DiGraph,
        regions: list[FanoutRegion],
    ):
        pending_regions = [region for region in regions if region not in self.region_plan_cache]
        if not pending_regions:
            return

        worker_count = self._region_plan_precompute_worker_count(len(pending_regions))
        if worker_count <= 1:
            for region in pending_regions:
                self._build_region_plan_table(dag, region)
            return

        try:
            tasks = [(dag, region, self.enable_score_cache) for region in pending_regions]
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                serialized_tables = list(executor.map(_build_fanout_region_plan_table_worker, tasks))
        except Exception as exc:
            print(f'Fan-out region parallel precompute failed ({exc}); falling back to serial precompute')
            for region in pending_regions:
                self._build_region_plan_table(dag, region)
            return

        feature_by_id = {
            node.node_id: node
            for node in dag.nodes
            if isinstance(node, FeatureNode)
        }
        for region, serialized_table in zip(pending_regions, serialized_tables):
            self.region_plan_cache[region] = self._deserialize_region_plan_table(serialized_table, feature_by_id)

    def _build_linear_branch_plan_table(
        self,
        dag: nx.DiGraph,
        entry: FeatureNode,
        leaf: FeatureNode,
    ) -> dict[tuple[int, int], RegionPlan]:
        traced = self._trace_linear_branch_to_source(dag, leaf)
        if traced is None:
            return {}
        branch_features, branch_computes = traced
        if entry not in branch_features:
            return {}

        entry_pos = branch_features.index(entry)
        path_features = list(reversed(branch_features[: entry_pos + 1]))
        path_computes = list(reversed(branch_computes[:entry_pos]))
        branch_nodes = set(path_features) | set(path_computes)
        branch_dag = dag.subgraph(branch_nodes).copy()

        sorted_features = [
            node for node in nx.topological_sort(branch_dag)
            if isinstance(node, FeatureNode)
        ]
        node_to_idx = {node: idx for idx, node in enumerate(sorted_features)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        entry_idx = node_to_idx[entry]
        leaf_idx = node_to_idx[leaf]
        min_feature_level = get_min_feature_level()

        plan_table: dict[tuple[int, int], RegionPlan] = {}
        for input_level in range(min_feature_level, config.fhe_param.max_level + 1):
            frontier = [NodeLevel(entry_idx, input_level)]
            processed_feature_nodes = {entry}
            init_graph_vec = np.zeros(len(node_to_idx), dtype=np.uint8)
            init_graph_vec[entry_idx] = input_level
            frontier_solutions = {(NodeLevel(entry_idx, input_level),): (0.0, init_graph_vec)}

            for feature in sorted_features:
                if feature is entry:
                    continue
                frontier, frontier_solutions = self.generate_solutions(
                    feature,
                    frontier,
                    frontier_solutions,
                    processed_feature_nodes,
                    node_to_idx,
                    idx_to_node,
                    branch_dag,
                    leaf_min_only=feature is not leaf,
                )

            for output_level in range(min_feature_level, config.fhe_param.max_level + 1):
                final_key = (NodeLevel(leaf_idx, output_level),)
                if final_key not in frontier_solutions:
                    continue
                cost, graph_vec = frontier_solutions[final_key]
                feature_levels = {
                    feature: int(graph_vec[node_to_idx[feature]])
                    for feature in sorted_features
                    if feature is not entry
                }
                plan_table[(input_level, output_level)] = RegionPlan(cost, feature_levels)

            min_plan = plan_table.get((input_level, min_feature_level))
            producers = [node for node in dag.predecessors(leaf) if isinstance(node, ComputeNode)]
            if min_plan is not None and len(producers) == 1:
                aux_feature_levels = dict(min_plan.feature_levels)
                aux_feature_levels[leaf] = AUX_LV
                aux_cost = min_plan.cost + get_restoring_score(dag, producers[0], self.param_dict)
                plan_table[(input_level, AUX_LV)] = RegionPlan(aux_cost, aux_feature_levels)

        return plan_table

    def _build_add_chain_region_plan_table(
        self,
        dag: nx.DiGraph,
        region: FanoutRegion,
    ) -> dict[tuple[int, int], RegionPlan]:
        merge_inputs_and_chain = self._collect_merge_leaf_features(dag, region.merge)
        if merge_inputs_and_chain is None:
            return {}
        _, chain_nodes = merge_inputs_and_chain
        chain_compute_set = {
            node for node in chain_nodes
            if isinstance(node, ComputeNode) and node.layer_type in {'add', 'add2d'}
        }

        min_feature_level = get_min_feature_level()
        subtree_cache = {}

        def score_add(add_node: ComputeNode, pred_levels: list[int], output_level: int) -> float:
            pred_features = [node for node in dag.predecessors(add_node) if isinstance(node, FeatureNode)]
            succ_feature = next(node for node in dag.successors(add_node) if isinstance(node, FeatureNode))
            touched = set(pred_features + [succ_feature, add_node])
            saved = {node: dict(dag.nodes[node]) for node in touched}
            try:
                for feature, level in zip(pred_features, pred_levels):
                    dag.nodes[feature]['level'] = level
                dag.nodes[succ_feature]['level'] = output_level
                return self.get_compute_score_cached(dag, add_node)
            finally:
                for node, attrs in saved.items():
                    dag.nodes[node].clear()
                    dag.nodes[node].update(attrs)

        def producer_add_for_feature(feature: FeatureNode) -> ComputeNode | None:
            producers = [node for node in dag.predecessors(feature) if isinstance(node, ComputeNode)]
            if len(producers) != 1:
                return None
            producer = producers[0]
            if producer not in chain_compute_set or dag.out_degree(feature) != 1:
                return None
            return producer

        def build_input_table(feature: FeatureNode) -> dict[tuple[int, int], RegionPlan]:
            producer = producer_add_for_feature(feature)
            if producer is not None:
                return build_add_table(producer)
            return self._build_linear_branch_plan_table(dag, region.entry, feature)

        def candidate_input_levels(output_level: int) -> list[int]:
            return list(range(output_level, config.fhe_param.max_level + 1)) + [AUX_LV]

        def effective_level(level: int) -> int:
            return config.fhe_param.max_level if level == AUX_LV else level

        def build_add_table(add_node: ComputeNode) -> dict[tuple[int, int], RegionPlan]:
            if add_node in subtree_cache:
                return subtree_cache[add_node]

            inputs = [node for node in dag.predecessors(add_node) if isinstance(node, FeatureNode)]
            if len(inputs) != 2:
                subtree_cache[add_node] = {}
                return subtree_cache[add_node]
            succ_feature = next(node for node in dag.successors(add_node) if isinstance(node, FeatureNode))
            left_table = build_input_table(inputs[0])
            right_table = build_input_table(inputs[1])
            table: dict[tuple[int, int], RegionPlan] = {}

            for input_level in range(min_feature_level, config.fhe_param.max_level + 1):
                for output_level in range(min_feature_level, config.fhe_param.max_level + 1):
                    best_plan = None
                    best_cost = float('inf')
                    for left_level in candidate_input_levels(output_level):
                        left_plan = left_table.get((input_level, left_level))
                        if left_plan is None:
                            continue
                        for right_level in candidate_input_levels(output_level):
                            right_plan = right_table.get((input_level, right_level))
                            if right_plan is None:
                                continue
                            feature_levels = {}
                            feature_levels.update(left_plan.feature_levels)
                            feature_levels.update(right_plan.feature_levels)
                            feature_levels[succ_feature] = output_level
                            add_score = score_add(
                                add_node,
                                [effective_level(left_level), effective_level(right_level)],
                                output_level,
                            )
                            cost = left_plan.cost + right_plan.cost + add_score
                            if cost < best_cost:
                                best_cost = cost
                                best_plan = RegionPlan(cost, dict(feature_levels))
                    if best_plan is not None:
                        table[(input_level, output_level)] = best_plan

                min_plan = table.get((input_level, min_feature_level))
                if min_plan is not None:
                    aux_feature_levels = dict(min_plan.feature_levels)
                    aux_feature_levels[succ_feature] = AUX_LV
                    aux_cost = min_plan.cost + get_restoring_score(dag, add_node, self.param_dict)
                    table[(input_level, AUX_LV)] = RegionPlan(aux_cost, aux_feature_levels)

            subtree_cache[add_node] = table
            return table

        return build_add_table(region.merge)

    def generate_region_solutions(
        self,
        region: FanoutRegion,
        frontier: list[NodeLevel],
        frontier_solutions: dict[tuple[int], tuple[float, np.ndarray]],
        processed_feature_nodes: set[FeatureNode],
        node_to_idx: dict[FeatureNode, int],
        idx_to_node: dict[int, FeatureNode],
        dag: nx.DiGraph,
    ):
        entry_idx = node_to_idx[region.entry]
        exit_idx = node_to_idx[region.exit]
        if entry_idx not in {node.node_idx for node in frontier}:
            return None

        min_feature_level = get_min_feature_level()
        plan_table = self._build_region_plan_table(dag, region)

        processed_feature_nodes.update(region.internal_features)
        internal_indices = self._internalized_frontier_indices(dag, frontier, processed_feature_nodes, idx_to_node)
        new_frontier = [node for node in frontier if node.node_idx not in internal_indices]
        new_frontier.append(NodeLevel(exit_idx, min_feature_level))

        new_frontier_solutions = {}
        frontier_solution_entries = [
            (
                {node_lv.node_idx: node_lv.level for node_lv in frontier_key},
                initial_score,
                sol_graph_vec,
            )
            for frontier_key, (initial_score, sol_graph_vec) in frontier_solutions.items()
        ]

        for output_level in range(min_feature_level, config.fhe_param.max_level + 1):
            for frontier_level_by_idx, initial_score, sol_graph_vec in frontier_solution_entries:
                input_level = frontier_level_by_idx[entry_idx]
                effective_input_level = config.fhe_param.max_level if input_level == AUX_LV else input_level
                plan = plan_table.get((effective_input_level, output_level))
                if plan is None:
                    continue

                new_frontier_key = []
                for node_max_lv in frontier:
                    lv = frontier_level_by_idx[node_max_lv.node_idx]
                    if node_max_lv.node_idx not in internal_indices:
                        new_frontier_key.append(NodeLevel(node_max_lv.node_idx, lv))
                new_frontier_key.append(NodeLevel(exit_idx, output_level))
                new_frontier_key.sort(key=lambda x: x.node_idx)

                sol_cost = initial_score + plan.cost
                new_frontier_key_tuple = tuple(new_frontier_key)
                if (
                    new_frontier_key_tuple not in new_frontier_solutions
                    or sol_cost < new_frontier_solutions[new_frontier_key_tuple][0]
                ):
                    new_sol_graph_vec = sol_graph_vec.copy()
                    for feature, level in plan.feature_levels.items():
                        new_sol_graph_vec[node_to_idx[feature]] = level
                    new_frontier_solutions[new_frontier_key_tuple] = (sol_cost, new_sol_graph_vec)

            if len(list(dag.successors(region.exit))) == 0:
                break

            new_frontier[-1] = NodeLevel(exit_idx, output_level)
            if output_level == min_feature_level:
                aux_lv_solutions = {}
                for k, solution in new_frontier_solutions.items():
                    exit_lv_idx = k.index(NodeLevel(exit_idx, output_level))
                    sol_key = list(k)
                    sol_key[exit_lv_idx] = NodeLevel(exit_idx, AUX_LV)

                    sol_graph_vec_aux_lv = solution[1].copy()
                    sol_graph_vec_aux_lv[exit_idx] = AUX_LV
                    sol_aux_lv_score = get_restoring_score(dag, region.merge, self.param_dict)
                    aux_lv_solutions[tuple(sol_key)] = (
                        solution[0] + sol_aux_lv_score,
                        sol_graph_vec_aux_lv,
                    )

                new_frontier_solutions |= aux_lv_solutions

        return new_frontier, new_frontier_solutions

    def generate_solutions(
        self,
        new_node: FeatureNode,
        frontier: list[NodeLevel],
        frontier_solutions: dict[tuple[int], tuple[float, np.ndarray]],
        processed_feature_nodes: set[FeatureNode],
        node_to_idx: dict[FeatureNode, int],
        idx_to_node: dict[int, FeatureNode],
        dag: nx.DiGraph,
        leaf_min_only: bool = True,
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
        # Iterate only reachable frontier states; wide fan-out graphs make the raw frontier level product very sparse.
        frontier_solution_entries = [
            (
                {node_lv.node_idx: node_lv.level for node_lv in frontier_key},
                initial_score,
                sol_graph_vec,
            )
            for frontier_key, (initial_score, sol_graph_vec) in frontier_solutions.items()
        ]

        for terminal_lv in range(min_feature_level, config.fhe_param.max_level + 1):
            if dag.nodes[leading_comp]['level_cost'] + terminal_lv > config.fhe_param.max_level:
                continue

            dag.nodes[new_node]['level'] = terminal_lv
            min_pred_level = dag.nodes[leading_comp]['level_cost'] + terminal_lv

            for frontier_level_by_idx, initial_score, sol_graph_vec in frontier_solution_entries:
                valid = True
                new_frontier_key = []

                for node_max_lv in pred_frontier:
                    lv = frontier_level_by_idx[node_max_lv.node_idx]
                    if lv != AUX_LV and lv < min_pred_level:
                        valid = False
                        break

                if not valid:
                    continue

                for node_max_lv in frontier:
                    lv = frontier_level_by_idx[node_max_lv.node_idx]
                    if node_max_lv.node_idx not in nodes_became_internal:
                        new_frontier_key.append(NodeLevel(node_max_lv.node_idx, lv))
                new_frontier_key.append(NodeLevel(node_to_idx[new_node], terminal_lv))
                new_frontier_key.sort(key=lambda x: x.node_idx)

                for node_max_lv in pred_frontier:
                    lv = frontier_level_by_idx[node_max_lv.node_idx]
                    dag.nodes[idx_to_node[node_max_lv.node_idx]]['level'] = (
                        lv if lv < AUX_LV else config.fhe_param.max_level
                    )

                try:
                    sol_cost = initial_score + self.get_compute_score_cached(dag, leading_comp)
                except KeyError as exc:
                    preds = list(dag.predecessors(leading_comp))
                    missing = [p.node_id for p in preds if 'level' not in dag.nodes[p]]
                    raise KeyError(f'{exc} while scoring {leading_comp.layer_id}; missing levels for {missing}') from exc

                new_frontier_key_tuple = tuple(new_frontier_key)
                if (
                    new_frontier_key_tuple not in new_frontier_solutions
                    or sol_cost < new_frontier_solutions[new_frontier_key_tuple][0]
                ):
                    new_sol_graph_vec = sol_graph_vec.copy()
                    new_sol_graph_vec[node_to_idx[new_node]] = terminal_lv
                    new_frontier_solutions[new_frontier_key_tuple] = (sol_cost, new_sol_graph_vec)

            # leaf nodes only need the minimum output-level solution.
            is_leaf = len(list(dag.successors(new_node))) == 0
            if is_leaf and leaf_min_only:
                break

            new_frontier[-1] = NodeLevel(node_to_idx[new_node], terminal_lv)

            if (not is_leaf) and terminal_lv == min_feature_level and not (
                leading_comp.layer_type in {'avgpool1d', 'avgpool2d'}
                and getattr(leading_comp, 'is_adaptive_avgpool', False)
            ):
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
        fanout_regions = [] if is_mpc_flow() else self._find_fanout_regions(H)
        regions_by_exit = {region.exit: region for region in fanout_regions}
        region_exit_features = set(regions_by_exit)
        region_internal_features = set()
        for region in fanout_regions:
            region_internal_features.update(region.internal_features)
        region_internal_features -= region_exit_features
        if fanout_regions:
            print(f'Using fan-out region DP summaries: {len(fanout_regions)} regions')
            self._precompute_region_plan_tables(H, fanout_regions)

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

            if node in region_internal_features:
                pbar.update(1)
                continue

            region = regions_by_exit.get(node)
            if region is not None:
                region_result = self.generate_region_solutions(
                    region, frontier, frontier_solutions, processed_feature_nodes, node_to_idx, idx_to_node, H
                )
                if region_result is None:
                    frontier, frontier_solutions = self.generate_solutions(
                        node, frontier, frontier_solutions, processed_feature_nodes, node_to_idx, idx_to_node, H
                    )
                else:
                    frontier, frontier_solutions = region_result
            else:
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


def optimize_task_segments(pt_graph, temperature, enable_score_cache: bool = True):
    """
    Split a task graph into segments with the given capacity and fixed cost.
    Returns (segments, min_cost).
    """
    graph_partitioner = GraphPartitioner(pt_graph.dag, temperature=temperature, enable_score_cache=enable_score_cache)
    return graph_partitioner.run()


def _build_fanout_region_plan_table_worker(
    args: tuple[nx.DiGraph, FanoutRegion, bool],
) -> dict[tuple[int, int], tuple[float, dict[str, int]]]:
    dag, region, enable_score_cache = args
    graph_partitioner = GraphPartitioner(dag, enable_score_cache=enable_score_cache)
    plan_table = graph_partitioner._build_region_plan_table(dag, region)
    return {
        boundary_levels: (
            plan.cost,
            {feature.node_id: level for feature, level in plan.feature_levels.items()},
        )
        for boundary_levels, plan in plan_table.items()
    }


def _build_region_input_level_plan_table_worker(
    args: tuple[nx.DiGraph, list[FeatureNode], FanoutRegion, int, bool],
) -> dict[tuple[int, int], tuple[float, dict[str, int]]]:
    region_dag, sorted_features, region, input_level, enable_score_cache = args
    graph_partitioner = GraphPartitioner(region_dag, enable_score_cache=enable_score_cache)
    plan_table = graph_partitioner._build_region_plan_table_for_input_level(
        region_dag,
        sorted_features,
        region,
        input_level,
    )
    return {
        boundary_levels: (
            plan.cost,
            {feature.node_id: level for feature, level in plan.feature_levels.items()},
        )
        for boundary_levels, plan in plan_table.items()
    }


def restore_node_attributes(G: nx.DiGraph):
    for node in G.nodes:
        for attr in node.__dict__.keys():
            if attr in G.nodes[node]:
                node.__dict__[attr] = G.nodes[node][attr]


def compile_graph(
    pt_graph: LayerAbstractGraph | None = None,
    temperature=1.0,
    enable_score_cache: bool = True,
):
    score, compiled_graph = optimize_task_segments(
        pt_graph, temperature=temperature, enable_score_cache=enable_score_cache
    )

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
    enable_score_cache: bool = True,
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
        enable_score_cache=enable_score_cache,
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
    pt_graph_prepared, temperature, *rest = args
    enable_score_cache = rest[0] if rest else True
    score, graph = compile_model_btp(
        pt_graph_prepared, temperature, stdout=True, enable_score_cache=enable_score_cache
    )
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
