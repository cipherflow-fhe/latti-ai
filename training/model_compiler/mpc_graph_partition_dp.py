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

"""MPC graph partition DP with skip-aware frontier states.

This module is intentionally separate from ``graph_partition_dp.py``.  The
existing BTP DP tracks only feature levels, which is sufficient when a refresh
does not change packing layout.  MPC refresh changes both level and skip, so
this experimental partitioner carries ``(level, skip)`` in each frontier state
and computes level_cost from the current skip state.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

import networkx as nx

from components import (
    ComputeNode,
    ConvComputeNode,
    FeatureNode,
    LayerAbstractGraph,
    SpatialComputeNode,
    UpsampleComputeNode,
    config,
)
import transforms
from inference.model_generator.layers.poly_relu_base import PolyReluBase
from score import MpcScoreParam

from graph_partition_dp import (
    generate_param_dict_for_graph,
    get_compute_score,
    restore_node_attributes,
)


Skip = tuple[int, ...]
MPC_COMPUTE_CUT_LAYER_TYPES = {'relu2d', 'polyact', 'maxpool2d'}


class NodeState(NamedTuple):
    node_idx: int
    level: int
    skip: Skip


class FeatureState(NamedTuple):
    level: int
    skip: Skip


class FeatureRefreshDecision(NamedTuple):
    feature_idx: int


class EdgeRefreshDecision(NamedTuple):
    feature_idx: int
    compute_layer_id: str
    target_skip: Skip


@dataclass(frozen=True)
class MpcDpSolution:
    score: float
    feature_states: dict[int, FeatureState]
    feature_refreshes: tuple[FeatureRefreshDecision, ...] = ()
    edge_refreshes: tuple[EdgeRefreshDecision, ...] = ()


@dataclass(frozen=True)
class _Transition:
    output_state: FeatureState
    score: float
    edge_refreshes: tuple[EdgeRefreshDecision, ...] = ()


def _min_feature_level() -> int:
    return 1


def _feature_skip_dim(feature: FeatureNode) -> int:
    return max(1, int(feature.dim))


def _normalize_skip(skip, dim: int) -> Skip:
    if isinstance(skip, (int, float)):
        values = [skip]
    else:
        values = list(skip)
    if not values:
        values = [1]
    if len(values) < dim:
        values.extend([values[-1]] * (dim - len(values)))
    normalized = []
    for value in values[:dim]:
        rounded = int(round(value))
        if abs(float(value) - rounded) > 1e-9:
            raise ValueError(f'Non-integral skip is not supported by MPC skip DP: {skip}')
        normalized.append(max(1, rounded))
    return tuple(normalized)


def _unit_skip(feature: FeatureNode) -> Skip:
    return tuple([1] * _feature_skip_dim(feature))


def _target_skip(feature: FeatureNode, target: Skip | None = None) -> Skip:
    if feature.dim == 0:
        return _unit_skip(feature)
    if target is None:
        return _unit_skip(feature)
    return _normalize_skip(target, _feature_skip_dim(feature))


def _state_key(states: list[NodeState]) -> tuple[NodeState, ...]:
    return tuple(sorted(states, key=lambda state: state.node_idx))


class MpcSkipGraphPartitioner:
    """Forward DP whose frontier state includes feature level and skip.

    This is an experimental MPC-specific partitioner.  It models refresh as a
    layout-changing operation:

        (level, skip) -> (max_level, [1, 1])

    The implementation keeps the state space finite by only introducing unit
    skip refresh states plus forced resize target skips.
    """

    def __init__(
        self,
        entire_graph: nx.DiGraph,
        max_states_per_frontier: int = 4096,
        cut_compute_types: set[str] | None = None,
    ):
        if max_states_per_frontier <= 0:
            raise ValueError('max_states_per_frontier must be positive')
        self.entire_graph = entire_graph
        self.param_dict = generate_param_dict_for_graph()
        self.max_states_per_frontier = max_states_per_frontier
        self.cut_compute_types = cut_compute_types or set()

    def run(self) -> tuple[float, nx.DiGraph | None]:
        if self.cut_compute_types:
            return self._run_with_compute_cuts()
        return self._run_weak_components(self.entire_graph)

    def _run_weak_components(self, graph: nx.DiGraph) -> tuple[float, nx.DiGraph | None]:
        result = []
        total_score = 0.0
        for nodes in nx.weakly_connected_components(graph):
            subgraph = graph.subgraph(nodes).copy()
            score, dag = self.solve(subgraph)
            if dag is None:
                return float('inf'), None
            total_score += score
            result.append(dag)
        return total_score, nx.compose_all(result) if result else nx.DiGraph()

    def _run_with_compute_cuts(self) -> tuple[float, nx.DiGraph | None]:
        cut_nodes = [
            node
            for node in self.entire_graph.nodes
            if isinstance(node, ComputeNode) and node.layer_type in self.cut_compute_types
        ]
        if not cut_nodes:
            print('MPC compute split: no cut layers found; compiling original weak components')
            return self._run_weak_components(self.entire_graph)

        split_graph = self.entire_graph.copy()
        split_graph.remove_nodes_from(cut_nodes)
        n_subgraphs = nx.number_weakly_connected_components(split_graph) if len(split_graph.nodes) > 0 else 0
        print(
            'MPC compute split: '
            f'cut_layers={len(cut_nodes)}, weak_subgraphs={n_subgraphs}'
        )

        total_score, compiled_dag = self._run_weak_components(split_graph)
        if compiled_dag is None:
            return float('inf'), None

        for node in cut_nodes:
            attrs = copy.deepcopy(self.entire_graph.nodes[node])
            attrs['level_cost'] = 0
            compiled_dag.add_node(node, **attrs)

        cut_node_set = set(cut_nodes)
        for pred, succ, attrs in self.entire_graph.edges(data=True):
            if pred in cut_node_set or succ in cut_node_set:
                compiled_dag.add_edge(pred, succ, **copy.deepcopy(attrs))

        for cut_node in cut_nodes:
            for pred in self.entire_graph.predecessors(cut_node):
                if isinstance(pred, FeatureNode) and pred in compiled_dag.nodes:
                    compiled_dag.nodes[pred]['level'] = _min_feature_level()
            for succ in self.entire_graph.successors(cut_node):
                if isinstance(succ, FeatureNode) and succ in compiled_dag.nodes:
                    compiled_dag.nodes[succ]['level'] = config.fhe_param.max_level

        return total_score, compiled_dag

    def solve(self, dag: nx.DiGraph) -> tuple[float, nx.DiGraph | None]:
        if len(dag.nodes) == 0:
            return 0.0, nx.DiGraph()

        sorted_features = self._feature_traversal_order(dag)
        node_to_idx = {node: idx for idx, node in enumerate(sorted_features)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        source_features = [
            node for node in sorted_features if isinstance(node, FeatureNode) and dag.in_degree(node) == 0
        ]

        frontier_indices = [node_to_idx[node] for node in source_features]
        processed_features = set(source_features)
        initial_states = []
        initial_feature_states = {}
        for node in source_features:
            idx = node_to_idx[node]
            state = FeatureState(config.fhe_param.max_level, self._initial_skip(dag, node))
            initial_feature_states[idx] = state
            initial_states.append(NodeState(idx, state.level, state.skip))

        frontier_solutions: dict[tuple[NodeState, ...], MpcDpSolution] = {
            _state_key(initial_states): MpcDpSolution(0.0, initial_feature_states)
        }

        for output_feature in sorted_features:
            if output_feature in source_features:
                continue

            leading_computes = list(dag.predecessors(output_feature))
            if len(leading_computes) != 1:
                raise ValueError(
                    f'Expected feature {output_feature.node_id} to have one producer, got {len(leading_computes)}'
                )
            compute = leading_computes[0]
            if not isinstance(compute, ComputeNode):
                raise TypeError(f'Expected producer of {output_feature.node_id} to be ComputeNode')

            pred_features = [node for node in dag.predecessors(compute) if isinstance(node, FeatureNode)]
            pred_indices = [node_to_idx[node] for node in pred_features]
            output_idx = node_to_idx[output_feature]

            processed_features.add(output_feature)
            internal_indices = self._internalized_frontier_indices(
                dag, frontier_indices, processed_features, idx_to_node
            )
            new_frontier_indices = [
                idx for idx in frontier_indices if idx not in internal_indices
            ] + [output_idx]

            next_solutions: dict[tuple[NodeState, ...], MpcDpSolution] = {}
            for solution in frontier_solutions.values():
                input_states = [solution.feature_states[idx] for idx in pred_indices]
                transitions = self._transitions_for_compute(
                    dag,
                    compute,
                    pred_features,
                    input_states,
                    output_feature,
                    node_to_idx,
                )

                for transition in transitions:
                    feature_states = dict(solution.feature_states)
                    feature_states[output_idx] = transition.output_state

                    states = []
                    for idx in new_frontier_indices:
                        state = feature_states[idx]
                        states.append(NodeState(idx, state.level, state.skip))
                    key = _state_key(states)

                    new_solution = MpcDpSolution(
                        score=solution.score + transition.score,
                        feature_states=feature_states,
                        feature_refreshes=solution.feature_refreshes,
                        edge_refreshes=solution.edge_refreshes + transition.edge_refreshes,
                    )
                    self._keep_best_solution(next_solutions, key, new_solution)

                    if dag.out_degree(output_feature) > 0 and output_feature.dim > 0:
                        refresh_state = FeatureState(config.fhe_param.max_level, _unit_skip(output_feature))
                        refreshed_feature_states = dict(feature_states)
                        refreshed_feature_states[output_idx] = refresh_state
                        refreshed_states = []
                        for idx in new_frontier_indices:
                            state = refreshed_feature_states[idx]
                            refreshed_states.append(NodeState(idx, state.level, state.skip))
                        refresh_key = _state_key(refreshed_states)
                        refresh_score = self._mpc_refresh_score_for_feature(
                            dag,
                            output_feature,
                            transition.output_state,
                        )
                        refresh_solution = MpcDpSolution(
                            score=solution.score + transition.score + refresh_score,
                            feature_states=refreshed_feature_states,
                            feature_refreshes=solution.feature_refreshes
                            + (FeatureRefreshDecision(output_idx),),
                            edge_refreshes=solution.edge_refreshes + transition.edge_refreshes,
                        )
                        self._keep_best_solution(next_solutions, refresh_key, refresh_solution)

            frontier_indices = new_frontier_indices
            frontier_solutions = self._trim_solutions(next_solutions)
            if not frontier_solutions:
                return float('inf'), None

        best_solution = min(frontier_solutions.values(), key=lambda solution: solution.score)
        final_dag = self.reconstruct_graph(dag, idx_to_node, best_solution)
        return best_solution.score, final_dag

    def _feature_traversal_order(self, dag: nx.DiGraph) -> list[FeatureNode]:
        topo_nodes = list(nx.topological_sort(dag))
        topo_rank = {node: idx for idx, node in enumerate(topo_nodes)}
        source_feature_nodes = sorted(
            [node for node in dag.nodes if isinstance(node, FeatureNode) and dag.in_degree(node) == 0],
            key=lambda node: topo_rank[node],
        )
        all_feature_nodes = [node for node in topo_nodes if isinstance(node, FeatureNode)]

        sorted_nodes: list[FeatureNode] = []
        activated: set[FeatureNode] = set()

        def activate(node: FeatureNode):
            if node in activated:
                return
            activated.add(node)
            sorted_nodes.append(node)
            for comp in sorted(dag.successors(node), key=lambda comp_node: topo_rank[comp_node]):
                if all(pred in activated for pred in dag.predecessors(comp)):
                    outputs = list(dag.successors(comp))
                    if len(outputs) != 1:
                        raise ValueError(f'Expected one output feature for compute {comp.layer_id}')
                    activate(outputs[0])

        for node in source_feature_nodes:
            activate(node)

        while len(sorted_nodes) < len(all_feature_nodes):
            progressed = False
            for node in all_feature_nodes:
                if node in activated:
                    continue
                producers = list(dag.predecessors(node))
                if not producers:
                    activate(node)
                    progressed = True
                    break
                pred_features = list(dag.predecessors(producers[0]))
                if all(pred in activated for pred in pred_features):
                    activate(node)
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError('Failed to construct feature traversal order')

        return sorted_nodes

    def _internalized_frontier_indices(
        self,
        dag: nx.DiGraph,
        frontier_indices: list[int],
        processed_features: set[FeatureNode],
        idx_to_node: dict[int, FeatureNode],
    ) -> set[int]:
        internal = set()
        for idx in frontier_indices:
            feature = idx_to_node[idx]
            is_internal = True
            for comp in dag.successors(feature):
                for succ in dag.successors(comp):
                    if succ not in processed_features:
                        is_internal = False
                        break
                if not is_internal:
                    break
            if is_internal:
                internal.add(idx)
        return internal

    def _transitions_for_compute(
        self,
        dag: nx.DiGraph,
        compute: ComputeNode,
        pred_features: list[FeatureNode],
        input_states: list[FeatureState],
        output_feature: FeatureNode,
        node_to_idx: dict[FeatureNode, int],
    ) -> list[_Transition]:
        candidates: list[tuple[list[FeatureState], tuple[EdgeRefreshDecision, ...], float]] = []
        candidates.extend(self._direct_input_candidates(compute, input_states))
        candidates.extend(self._forced_layout_repair_candidates(dag, compute, pred_features, input_states, node_to_idx))

        transitions = []
        seen = set()
        for states, edge_refreshes, refresh_score in candidates:
            key = tuple(states)
            if key in seen:
                continue
            seen.add(key)
            level_cost = self._dynamic_level_cost(dag, compute, pred_features, states, output_feature)
            max_output_level = min(state.level for state in states) - level_cost
            if max_output_level < _min_feature_level():
                continue

            output_skip = self._transfer_skip(dag, compute, pred_features, states, output_feature)
            if output_skip is None:
                continue

            for output_level in range(_min_feature_level(), min(max_output_level, config.fhe_param.max_level) + 1):
                actual_input_level = output_level + level_cost
                actual_input_states = [
                    FeatureState(actual_input_level, state.skip)
                    for state in states
                ]
                output_state = FeatureState(output_level, output_skip)
                compute_score = self._compute_score_with_states(
                    dag,
                    compute,
                    pred_features,
                    actual_input_states,
                    output_feature,
                    output_state,
                    level_cost,
                )
                transitions.append(_Transition(output_state, refresh_score + compute_score, edge_refreshes))
        return transitions

    def _direct_input_candidates(
        self,
        compute: ComputeNode,
        input_states: list[FeatureState],
    ) -> list[tuple[list[FeatureState], tuple[EdgeRefreshDecision, ...], float]]:
        if compute.layer_type in {'add', 'add2d', 'concat2d'} and not self._all_skips_equal(input_states):
            return []
        if compute.layer_type == 'resize' and self._resize_requires_refresh(compute, input_states[0]):
            return []
        return [(list(input_states), tuple(), 0.0)]

    def _forced_layout_repair_candidates(
        self,
        dag: nx.DiGraph,
        compute: ComputeNode,
        pred_features: list[FeatureNode],
        input_states: list[FeatureState],
        node_to_idx: dict[FeatureNode, int],
    ) -> list[tuple[list[FeatureState], tuple[EdgeRefreshDecision, ...], float]]:
        if compute.layer_type == 'resize' and self._resize_requires_refresh(compute, input_states[0]):
            target = _normalize_skip(compute.upsample_factor_in, _feature_skip_dim(pred_features[0]))
            state = FeatureState(config.fhe_param.max_level, target)
            decision = EdgeRefreshDecision(node_to_idx[pred_features[0]], compute.layer_id, target)
            score = self._mpc_refresh_score_for_feature(dag, pred_features[0], input_states[0])
            return [([state], (decision,), score)]

        if compute.layer_type not in {'add', 'add2d', 'concat2d'} or self._all_skips_equal(input_states):
            return []

        repaired_states = []
        decisions = []
        refresh_score = 0.0
        for feature, state in zip(pred_features, input_states):
            unit = _unit_skip(feature)
            if state.skip == unit:
                repaired_states.append(state)
                continue
            repaired = FeatureState(config.fhe_param.max_level, unit)
            repaired_states.append(repaired)
            decisions.append(EdgeRefreshDecision(node_to_idx[feature], compute.layer_id, unit))
            refresh_score += self._mpc_refresh_score_for_feature(dag, feature, state)
        return [(repaired_states, tuple(decisions), refresh_score)]

    def _dynamic_level_cost(
        self,
        dag: nx.DiGraph,
        compute: ComputeNode,
        pred_features: list[FeatureNode],
        input_states: list[FeatureState],
        output_feature: FeatureNode,
    ) -> int:
        pred = pred_features[0]
        pred_state = input_states[0]

        if isinstance(compute, ConvComputeNode):
            if config.style == 'ordinary':
                return 1
            if config.style != 'multiplexed':
                raise ValueError('Unsupported config.style')
            if any(pred.shape[i] > config.block_shape[i] for i in range(pred.dim)):
                if any(output_feature.shape[i] < config.block_shape[i] for i in range(output_feature.dim)):
                    return 2
                return 1
            if compute.groups == 1:
                if all(compute.stride[i] == 1 for i in range(compute.dim)) and all(s == 1 for s in pred_state.skip):
                    return 1
                return 2
            if all(compute.stride[i] == 1 for i in range(compute.dim)):
                return 1
            return 2

        if compute.layer_type in {'avgpool1d', 'avgpool2d'}:
            if getattr(compute, 'is_adaptive_avgpool', False):
                return 0
            if any(pred.shape[i] > config.block_shape[i] for i in range(pred.dim)):
                return 1 if any(output_feature.shape[i] < config.block_shape[i] for i in range(output_feature.dim)) else 0
            succs_sub = list(dag.successors(output_feature))
            if succs_sub and succs_sub[0].layer_type == 'reshape':
                return 0
            return 1

        if compute.layer_type == config.approx_poly_type:
            return PolyReluBase.compute_bsgs_level_cost(compute.order)
        if isinstance(compute, UpsampleComputeNode):
            return 0 if all(compute.upsample_factor[i] == 1 for i in range(compute.dim)) else 1
        if compute.layer_type.startswith('fc'):
            return 1
        if 'mult_scalar' in compute.layer_type:
            return 1
        if compute.layer_type == 'resize':
            return 1
        if compute.layer_type == 'concat2d':
            has_uneven = any(
                feature.channel % self._pack_num_for_state(dag, feature, state) != 0
                for feature, state in zip(pred_features, input_states)
            )
            return 1 if has_uneven else 0
        if compute.layer_type == 'parcpmm':
            return 2
        if compute.layer_type in {'add_pt', 'pcm_add_pt'}:
            return 0
        if compute.layer_type == 'partranspose':
            return 1
        if compute.layer_type == 'parccmm':
            return 3
        if compute.layer_type == 'pcmgamma':
            return 1
        if compute.layer_type == 'pcmpoly':
            return 2 if compute.order == 2 else 3
        if compute.layer_type == 'pcmstats':
            return 4
        if compute.layer_type == 'pcmcenter':
            return 2
        if compute.layer_type == 'pcminit':
            return 2
        if compute.layer_type == 'pcmgs':
            return 3
        if compute.layer_type == 'pcmaffine':
            return 2
        return 0

    def _transfer_skip(
        self,
        dag: nx.DiGraph,
        compute: ComputeNode,
        pred_features: list[FeatureNode],
        input_states: list[FeatureState],
        output_feature: FeatureNode,
    ) -> Skip | None:
        if output_feature.dim == 0:
            return _normalize_skip(input_states[0].skip, _feature_skip_dim(output_feature))

        pred = pred_features[0]
        pred_state = input_states[0]
        dim = _feature_skip_dim(output_feature)

        if isinstance(compute, SpatialComputeNode):
            values = []
            for i in range(compute.dim):
                value = pred_state.skip[i] * compute.stride[i] / compute.upsample_factor_in[i]
                if abs(value - round(value)) > 1e-9:
                    return None
                values.append(int(round(value)))
            if any(pred.shape[i] > config.block_shape[i] for i in range(pred.dim)):
                return _unit_skip(output_feature)
            return _normalize_skip(values, dim)

        if compute.layer_type == 'upsample':
            return _unit_skip(output_feature)
        if compute.layer_type == 'mpc_refresh':
            if compute.change_skip_to != 0:
                return _normalize_skip([compute.change_skip_to] * dim, dim)
            return _unit_skip(output_feature)
        if (
            'batchnorm' in compute.layer_type
            or 'drop_level' in compute.layer_type
            or 'mult_scalar' in compute.layer_type
            or compute.layer_type == config.approx_poly_type
            or compute.layer_type == 'relu2d'
            or compute.layer_type == 'identity'
        ):
            return _normalize_skip(pred_state.skip, dim)
        if compute.layer_type in {'add', 'add2d', 'concat2d'}:
            if not self._all_skips_equal(input_states):
                return None
            return _normalize_skip(input_states[0].skip, dim)
        if compute.layer_type in {'avgpool1d', 'avgpool2d'}:
            if getattr(compute, 'is_adaptive_avgpool', False):
                return _normalize_skip(pred_state.skip, dim)
            values = [pred_state.skip[i] * compute.stride[i] for i in range(compute.dim)]
            if any(pred.shape[i] > config.block_shape[i] for i in range(pred.dim)):
                return _unit_skip(output_feature)
            return _normalize_skip(values, dim)
        if compute.layer_type == 'resize':
            if self._resize_requires_refresh(compute, pred_state):
                return None
            values = [pred_state.skip[i] / compute.upsample_factor_in[i] for i in range(compute.dim)]
            return _normalize_skip(values, dim)
        return _normalize_skip(pred_state.skip, dim)

    def _resize_requires_refresh(self, compute: ComputeNode, state: FeatureState) -> bool:
        return any(state.skip[i] < compute.upsample_factor_in[i] for i in range(compute.dim))

    def _all_skips_equal(self, states: list[FeatureState]) -> bool:
        if not states:
            return True
        return all(state.skip == states[0].skip for state in states[1:])

    def _initial_skip(self, dag: nx.DiGraph, feature: FeatureNode) -> Skip:
        return _normalize_skip(dag.nodes[feature].get('skip', _unit_skip(feature)), _feature_skip_dim(feature))

    def _pack_num_for_state(self, dag: nx.DiGraph, feature: FeatureNode, state: FeatureState) -> int:
        slot_num = config.fhe_param.poly_modulus_degree // 2
        if feature.dim == 0:
            return math.ceil(slot_num / state.skip[0])
        denom = math.prod(feature.shape) * math.prod(feature.invalid_fill)
        return math.ceil(slot_num / denom)

    def _compute_score_with_states(
        self,
        dag: nx.DiGraph,
        compute: ComputeNode,
        pred_features: list[FeatureNode],
        input_states: list[FeatureState],
        output_feature: FeatureNode,
        output_state: FeatureState,
        level_cost: int,
    ) -> float:
        touched = list(pred_features) + [output_feature, compute]
        saved = {node: dict(dag.nodes[node]) for node in touched}
        try:
            for feature, state in zip(pred_features, input_states):
                dag.nodes[feature]['level'] = state.level
                dag.nodes[feature]['skip'] = list(state.skip)
                dag.nodes[feature]['pack_num'] = self._pack_num_for_state(dag, feature, state)
            dag.nodes[output_feature]['level'] = output_state.level
            dag.nodes[output_feature]['skip'] = list(output_state.skip)
            dag.nodes[output_feature]['pack_num'] = self._pack_num_for_state(dag, output_feature, output_state)
            dag.nodes[compute]['level_cost'] = level_cost
            return get_compute_score(dag, compute, self.param_dict)
        finally:
            for node, attrs in saved.items():
                dag.nodes[node].clear()
                dag.nodes[node].update(attrs)

    def _mpc_refresh_score_for_feature(
        self,
        dag: nx.DiGraph,
        feature: FeatureNode,
        state: FeatureState,
    ) -> float:
        output = copy.deepcopy(feature)
        output.node_id = f'{feature.node_id}_mpc_refresh_cost_output'
        refresh = ComputeNode(
            layer_id=f'{feature.node_id}_mpc_refresh_cost',
            layer_type='mpc_refresh',
            channel_input=feature.channel,
            channel_output=feature.channel,
        )

        temp_dag = nx.DiGraph()
        temp_dag.add_node(
            feature,
            level=state.level,
            skip=list(state.skip),
            pack_num=self._pack_num_for_state(dag, feature, state),
        )
        output_state = FeatureState(config.fhe_param.max_level, _unit_skip(output))
        temp_dag.add_node(
            output,
            level=output_state.level,
            skip=list(output_state.skip),
            pack_num=self._pack_num_for_state(dag, output, output_state),
        )
        temp_dag.add_node(refresh, level_cost=0)
        temp_dag.add_edge(feature, refresh)
        temp_dag.add_edge(refresh, output)
        return MpcScoreParam(temp_dag, refresh, self.param_dict).get_score()

    def _keep_best_solution(
        self,
        solutions: dict[tuple[NodeState, ...], MpcDpSolution],
        key: tuple[NodeState, ...],
        solution: MpcDpSolution,
    ):
        if key not in solutions or solution.score < solutions[key].score:
            solutions[key] = solution

    def _trim_solutions(
        self,
        solutions: dict[tuple[NodeState, ...], MpcDpSolution],
    ) -> dict[tuple[NodeState, ...], MpcDpSolution]:
        pruned = self._dominance_prune_solutions(solutions)
        if len(pruned) <= self.max_states_per_frontier:
            return pruned
        return dict(sorted(pruned.items(), key=lambda item: item[1].score)[: self.max_states_per_frontier])

    def _dominance_prune_solutions(
        self,
        solutions: dict[tuple[NodeState, ...], MpcDpSolution],
    ) -> dict[tuple[NodeState, ...], MpcDpSolution]:
        groups: dict[tuple[tuple[int, Skip], ...], list[tuple[tuple[NodeState, ...], MpcDpSolution]]] = {}
        for key, solution in solutions.items():
            layout_key = tuple((state.node_idx, state.skip) for state in key)
            groups.setdefault(layout_key, []).append((key, solution))

        kept: dict[tuple[NodeState, ...], MpcDpSolution] = {}
        eps = 1e-12
        for items in groups.values():
            for key, solution in items:
                levels = [state.level for state in key]
                dominated = False
                for other_key, other_solution in items:
                    if other_key == key:
                        continue
                    other_levels = [state.level for state in other_key]
                    score_not_worse = other_solution.score <= solution.score + eps
                    levels_not_worse = all(
                        other_level >= level
                        for other_level, level in zip(other_levels, levels)
                    )
                    strictly_better = (
                        other_solution.score < solution.score - eps
                        or any(other_level > level for other_level, level in zip(other_levels, levels))
                    )
                    if score_not_worse and levels_not_worse and strictly_better:
                        dominated = True
                        break
                if not dominated:
                    kept[key] = solution
        return kept

    def reconstruct_graph(
        self,
        template_dag: nx.DiGraph,
        idx_to_node: dict[int, FeatureNode],
        solution: MpcDpSolution,
    ) -> nx.DiGraph:
        graph = LayerAbstractGraph()
        graph.dag = template_dag.copy()

        feature_by_id = {
            node.node_id: node for node in graph.dag.nodes if isinstance(node, FeatureNode)
        }
        compute_by_id = {
            node.layer_id: node for node in graph.dag.nodes if isinstance(node, ComputeNode)
        }
        refreshed_feature_by_idx: dict[int, FeatureNode] = {}

        for decision in solution.feature_refreshes:
            original_feature = feature_by_id[idx_to_node[decision.feature_idx].node_id]
            graph.dag.nodes[original_feature]['level'] = _min_feature_level()
            btp_node = transforms.add_btp_layer(
                graph.dag,
                original_feature,
                self.param_dict,
                config.fhe_param.max_level - _min_feature_level(),
            )
            btp_node.layer_type = 'mpc_refresh'
            refreshed_feature = next(graph.dag.successors(btp_node))
            refreshed_feature_by_idx[decision.feature_idx] = refreshed_feature

        for decision in solution.edge_refreshes:
            feature = refreshed_feature_by_idx.get(
                decision.feature_idx,
                feature_by_id[idx_to_node[decision.feature_idx].node_id],
            )
            compute = compute_by_id[decision.compute_layer_id]
            preds = list(graph.dag.predecessors(compute))
            if feature not in preds:
                continue
            state = solution.feature_states.get(decision.feature_idx)
            if state is not None:
                graph.dag.nodes[feature]['level'] = state.level
                graph.dag.nodes[feature]['skip'] = list(state.skip)
                graph.dag.nodes[feature]['pack_num'] = self._pack_num_for_state(graph.dag, feature, state)
            pred_index = preds.index(feature)
            refresh = transforms.add_layer(graph, compute, 0, pred_index, 'mpc_refresh', preds, None)
            if decision.target_skip and any(skip != 1 for skip in decision.target_skip):
                refresh.change_skip_to = decision.target_skip[0]
            refreshed_output = next(graph.dag.successors(refresh))
            refreshed_state = FeatureState(config.fhe_param.max_level, decision.target_skip)
            graph.dag.nodes[refreshed_output]['level'] = refreshed_state.level
            graph.dag.nodes[refreshed_output]['skip'] = list(refreshed_state.skip)
            graph.dag.nodes[refreshed_output]['pack_num'] = self._pack_num_for_state(
                graph.dag,
                refreshed_output,
                refreshed_state,
            )

        for idx, state in solution.feature_states.items():
            feature = refreshed_feature_by_idx.get(idx, feature_by_id[idx_to_node[idx].node_id])
            if feature in graph.dag.nodes:
                graph.dag.nodes[feature]['level'] = state.level
                graph.dag.nodes[feature]['skip'] = list(state.skip)

        transforms.infer_shapes_skips_and_pack_num(graph)
        transforms.set_level_costs(graph, trust_adaptive_avgpool_attr=True)
        restore_node_attributes(graph.dag)
        return graph.dag


def compile_graph_mpc_skip_aware(
    pt_graph_prepared: LayerAbstractGraph,
    max_states_per_frontier: int = 4096,
) -> tuple[float, LayerAbstractGraph | None]:
    cut_compute_types = MPC_COMPUTE_CUT_LAYER_TYPES if config.graph_type == 'mpc_compute' else None
    partitioner = MpcSkipGraphPartitioner(
        pt_graph_prepared.dag,
        max_states_per_frontier=max_states_per_frontier,
        cut_compute_types=cut_compute_types,
    )
    score, compiled_dag = partitioner.run()
    if compiled_dag is None:
        return float('inf'), None
    graph = LayerAbstractGraph()
    graph.dag = compiled_dag
    restore_node_attributes(graph.dag)
    return score, graph
