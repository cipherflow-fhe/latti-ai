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

"""Greedy bootstrapping insertion.

This is an experimental fast compiler pass. It ignores operator timing cost and
inserts bootstrapping greedily when a feature's current level cannot support the
largest level_cost among its immediate successor compute nodes.

Current intentionally unsupported pieces:
- drop_level insertion;
- multi-input level alignment;
- edge-specific refresh;
- global optimality of bootstrapping count.
"""

import copy

import networkx as nx

from components import ComputeNode, FeatureNode, LayerAbstractGraph, config


SPECIAL_COMPUTE_LAYERS = {'bootstrapping', 'drop_level'}


def get_min_feature_level() -> int:
    return 1 if config.mpc_refresh or config.graph_type == 'mpc' or config.set_btp_scale is not None else 0


def _is_regular_compute(node) -> bool:
    return isinstance(node, ComputeNode) and node.layer_type not in SPECIAL_COMPUTE_LAYERS


def _make_unique_feature_id(dag: nx.DiGraph, base_id: str) -> str:
    new_id = base_id
    counter = 0
    existing_ids = {n.node_id for n in dag.nodes if isinstance(n, FeatureNode)}
    while new_id in existing_ids:
        counter += 1
        new_id = f'{base_id}_{counter}'
    return new_id


def _make_unique_layer_id(dag: nx.DiGraph, base_id: str) -> str:
    new_id = base_id
    counter = 0
    existing_ids = {n.layer_id for n in dag.nodes if isinstance(n, ComputeNode)}
    while new_id in existing_ids:
        counter += 1
        new_id = f'{base_id}_{counter}'
    return new_id


def _set_feature_level(dag: nx.DiGraph, feature: FeatureNode, level: int):
    dag.nodes[feature]['level'] = level
    feature.level = level


def insert_btp_after_feature(
    dag: nx.DiGraph,
    f_node: FeatureNode,
    *,
    input_level: int,
    btp_out_level: int,
) -> tuple[ComputeNode, FeatureNode]:
    """Insert one shared BTP after f_node and move all successor computes behind it."""
    refreshed_f = copy.deepcopy(f_node)
    refreshed_f.node_id = _make_unique_feature_id(dag, f'{f_node.node_id}_refreshed')

    btp_node = ComputeNode(
        layer_id=_make_unique_layer_id(dag, f'{f_node.node_id}_bootstrap'),
        layer_type='bootstrapping',
        channel_input=f_node.channel,
        channel_output=f_node.channel,
    )

    old_successors = list(dag.successors(f_node))
    refreshed_attrs = dict(dag.nodes[f_node])
    refreshed_attrs['level'] = btp_out_level

    dag.add_node(
        btp_node,
        name=btp_node.layer_id,
        level_cost=input_level - btp_out_level,
    )
    dag.add_node(refreshed_f, **refreshed_attrs)
    refreshed_f.level = btp_out_level

    for c_node in old_successors:
        edge_attrs = dict(dag.edges[f_node, c_node])
        dag.remove_edge(f_node, c_node)
        dag.add_edge(refreshed_f, c_node, **edge_attrs)

    dag.add_edge(f_node, btp_node)
    dag.add_edge(btp_node, refreshed_f)

    return btp_node, refreshed_f


def _try_process_compute(
    dag: nx.DiGraph,
    c_node: ComputeNode,
    feature_level: dict[FeatureNode, int],
    prepared_features: set[FeatureNode],
    processed_compute: set[ComputeNode],
    *,
    min_level: int,
) -> FeatureNode | None:
    if c_node in processed_compute:
        return None

    preds: list[FeatureNode] = list(dag.predecessors(c_node))
    if not preds or not all(p in feature_level and p in prepared_features for p in preds):
        return None

    succs: list[FeatureNode] = list(dag.successors(c_node))
    if len(succs) != 1:
        raise RuntimeError(f'compute node {c_node.layer_id} should have exactly one output feature')

    level_cost = dag.nodes[c_node].get('level_cost', 0)
    input_level = min(feature_level[p] for p in preds)
    if input_level - level_cost < min_level:
        raise RuntimeError(
            f'insufficient level before {c_node.layer_id}: input_level={input_level}, level_cost={level_cost}, '
            f'min_level={min_level}'
        )

    out_f = succs[0]
    out_level = input_level - level_cost
    _set_feature_level(dag, out_f, out_level)
    feature_level[out_f] = out_level
    processed_compute.add(c_node)
    return out_f


def greedy_insert_btp(
    graph: LayerAbstractGraph,
    *,
    copy_graph: bool = True,
    min_level: int | None = None,
    btp_out_level: int | None = None,
) -> tuple[int, LayerAbstractGraph]:
    """Insert BTP greedily and return (btp_count, graph).

    Algorithm:
    1. Initialize all source FeatureNode levels to max_level.
    2. Traverse original FeatureNodes in topological order.
    3. For each feature, inspect all immediate regular successor ComputeNodes.
    4. If current_level - max(successor.level_cost) < min_level, insert one BTP
       after this feature and move all successor computes behind the refreshed
       feature.
    5. Process ready successor computes and set output_level = min(pred_levels) - level_cost.
    """
    if graph is None:
        raise ValueError('graph must not be None')

    work_graph = copy.deepcopy(graph) if copy_graph else graph
    dag = work_graph.dag

    min_level = get_min_feature_level() if min_level is None else min_level
    max_level = config.fhe_param.max_level
    if btp_out_level is None:
        btp_out_level = max_level

    feature_level: dict[FeatureNode, int] = {}
    prepared_features: set[FeatureNode] = set()
    processed_compute: set[ComputeNode] = set()
    btp_count = 0

    topo_nodes = list(nx.topological_sort(dag))
    topo_rank = {node: idx for idx, node in enumerate(topo_nodes)}
    ready_features: list[FeatureNode] = []

    for node in topo_nodes:
        if isinstance(node, FeatureNode) and dag.in_degree(node) == 0:
            _set_feature_level(dag, node, max_level)
            feature_level[node] = max_level
            ready_features.append(node)

    while ready_features:
        ready_features.sort(key=lambda f: topo_rank.get(f, len(topo_rank)))
        node = ready_features.pop(0)
        if node not in dag or node in prepared_features or node not in feature_level:
            continue

        current_f = node
        current_level = feature_level[current_f]

        succ_computes = [succ for succ in dag.successors(current_f) if _is_regular_compute(succ)]
        if succ_computes:
            max_level_cost = max(dag.nodes[c].get('level_cost', 0) for c in succ_computes)
            if max_level_cost > btp_out_level - min_level:
                raise RuntimeError(
                    f'level_cost={max_level_cost} after {current_f.node_id} exceeds available budget '
                    f'btp_out_level={btp_out_level}, min_level={min_level}'
                )

            if current_level - max_level_cost < min_level:
                _, refreshed_f = insert_btp_after_feature(
                    dag,
                    current_f,
                    input_level=current_level,
                    btp_out_level=btp_out_level,
                )
                btp_count += 1
                feature_level[refreshed_f] = btp_out_level
                prepared_features.add(current_f)
                current_f = refreshed_f

        prepared_features.add(current_f)

        for c_node in list(dag.successors(current_f)):
            if not _is_regular_compute(c_node):
                continue
            out_f = _try_process_compute(
                dag,
                c_node,
                feature_level,
                prepared_features,
                processed_compute,
                min_level=min_level,
            )
            if out_f is not None:
                ready_features.append(out_f)

    remaining_compute = [node for node in dag.nodes if _is_regular_compute(node) and node not in processed_compute]
    if remaining_compute:
        names = ', '.join(node.layer_id for node in remaining_compute[:10])
        suffix = '...' if len(remaining_compute) > 10 else ''
        raise RuntimeError(f'failed to process {len(remaining_compute)} compute nodes: {names}{suffix}')

    return btp_count, work_graph


def compile_model_btp_greedy(pt_graph_prepared: LayerAbstractGraph) -> tuple[int, LayerAbstractGraph]:
    """Compatibility-style entrypoint for the greedy BTP pass."""
    return greedy_insert_btp(pt_graph_prepared)
