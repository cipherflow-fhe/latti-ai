# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Iterative plan-based scale absorption.

Replaces the recursive PlanBuilder with PlanBuilderIterative:
  - Pass 1a (GraphAnalysis): same as before — static properties in topo order.
  - Pass 1b (PlanBuilderIterative.build): process every ComputeNode in topo
    order (graph-input → output), computing above[C] = (cost, list[Action]).
    No recursion: above[pred_c] is always precomputed by the time C is reached.
  - Pass 2 (execute): unchanged — walk the flat action list.

Action shapes and execute() are identical to absorb_scale_dp.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Union

import networkx as nx

from components import LayerAbstractGraph, FeatureNode, ComputeNode
from transforms import (
    Direction,
    _backward_level_dict,
    _propagate_scale,
    _remove_identity_mult_scalars,
    add_mult_scalar_between_feature_and_layer,
)


# ---------------------------------------------------------------------------
# Absorber categories / cost constants
# ---------------------------------------------------------------------------
ABSORBER_TYPES = frozenset({'conv2d', 'fc0', 'fc1', 'mult_scalar', 'polyact'})
DOWN_ABSORBER_TYPES = frozenset({'conv2d', 'fc0', 'fc1', 'mult_scalar'})

_FREE_COST = 1
_NONFREE_COST = 10_000


# ---------------------------------------------------------------------------
# Action types  (identical to absorb_scale_dp.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AbsorbInto:
    compute: ComputeNode
    direction: Literal['up', 'down']


@dataclass(frozen=True)
class InsertMsOn:
    f_node: FeatureNode
    c_node: ComputeNode
    direction: Literal['up', 'down']


@dataclass(frozen=True)
class CompensateDown:
    """Pass 2 delegates to legacy _propagate_scale for the reactive walk."""

    f_node: FeatureNode
    c_node: ComputeNode


@dataclass(frozen=True)
class ClearSource:
    mc: ComputeNode


Action = Union[AbsorbInto, InsertMsOn, CompensateDown, ClearSource]


# ---------------------------------------------------------------------------
# Static graph analysis  (identical to absorb_scale_dp.py)
# ---------------------------------------------------------------------------
class GraphAnalysis:
    """Per-graph static properties consumed by PlanBuilderIterative."""

    def __init__(self, dag: nx.DiGraph):
        self.dag = dag
        self.topo = list(nx.topological_sort(dag))
        self.topo_rank = {n: i for i, n in enumerate(self.topo)}

        self._level, self.arm_level = _backward_level_dict(dag)
        self.bottleneck_succs: dict[object, frozenset] = {}
        self.nearest_convergence_up: dict[object, object | None] = {}

        self._compute_bottleneck_succs()
        self._compute_nearest_convergence_up()

    def _compute_bottleneck_succs(self):
        for node in self.dag.nodes:
            if isinstance(node, ComputeNode):
                continue
            succs = list(self.dag.successors(node))
            if not succs:
                self.bottleneck_succs[node] = frozenset()
                continue
            arm_levels = {c: self.arm_level.get((node, c), 0) for c in succs}
            max_lv = max(arm_levels.values())
            self.bottleneck_succs[node] = frozenset(c for c, lv in arm_levels.items() if lv == max_lv)

    def _compute_nearest_convergence_up(self):
        id_to_feat = {id(n): n for n in self.dag.nodes if isinstance(n, FeatureNode)}

        def upstream_feat_ids(start):
            visited: set = set()
            stack = [start]
            while stack:
                f = stack.pop()
                fid = id(f)
                if fid in visited:
                    continue
                visited.add(fid)
                for c in self.dag.predecessors(f):
                    for f_above in self.dag.predecessors(c):
                        stack.append(f_above)
            return visited

        self.nearest_convergence_up = {n: None for n in self.dag.nodes if isinstance(n, FeatureNode)}
        for node in self.topo:
            if not isinstance(node, ComputeNode):
                continue
            preds = list(self.dag.predecessors(node))
            if len(preds) < 2:
                continue
            reach = [(f, upstream_feat_ids(f)) for f in preds]
            for i in range(len(reach)):
                f_i, set_i = reach[i]
                for j in range(i + 1, len(reach)):
                    f_j, set_j = reach[j]
                    shared = set_i & set_j
                    if not shared:
                        continue
                    nearest_fid = max(
                        (fid for fid in shared if fid in id_to_feat),
                        key=lambda fid: self.topo_rank.get(id_to_feat[fid], -1),
                        default=None,
                    )
                    if nearest_fid is None:
                        continue
                    nearest_f = id_to_feat[nearest_fid]
                    for arm_f in (f_i, f_j):
                        existing = self.nearest_convergence_up[arm_f]
                        if existing is None or self.topo_rank[nearest_f] > self.topo_rank[existing]:
                            self.nearest_convergence_up[arm_f] = nearest_f


# ---------------------------------------------------------------------------
# Iterative plan builder
# ---------------------------------------------------------------------------
class PlanBuilderIterative:
    """Compute above[C] = (cost, list[Action]) for every ComputeNode C.

    Processing order: topological (graph-input first → output last).
    When we compute above[C], every predecessor ComputeNode of C is already
    in self.above, so _plan_at can look up above[pred_c] without recursion.
    """

    def __init__(self, ga: GraphAnalysis):
        self.ga = ga
        self.dag = ga.dag
        self.above: dict[object, tuple[int, list[Action]]] = {}
        self._down_memo: dict[tuple[int, int, str, int], tuple[int, list[Action]]] = {}

    # ------------------------------------------------------------------- #
    # Public entry
    # ------------------------------------------------------------------- #
    def build(self):
        """Precompute above[C] for all ComputeNodes in topo order."""
        for node in self.ga.topo:
            if isinstance(node, ComputeNode):
                self._compute_above(node)

    # ------------------------------------------------------------------- #
    # above[C] computation
    # ------------------------------------------------------------------- #
    def _compute_above(self, C: ComputeNode):
        if C.layer_type in ABSORBER_TYPES:
            self.above[C] = (0, [AbsorbInto(C, 'up')])
            return

        preds_f = list(self.dag.predecessors(C))
        if len(preds_f) == 1:
            self.above[C] = self._plan_at(preds_f[0], frozenset({C}))
        else:
            self.above[C] = self._plan_join(C)

    def _plan_at(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        """min(INSERT at F, GO_UP through F) for scale arriving via ci_set arms.

        No recursion: above[pred_c] is always in self.above because pred_c
        is upstream of F and was processed earlier in topo order.
        """
        insert_cost, insert_plan = self._plan_terminate(F, ci_set)

        preds = list(self.dag.predecessors(F))
        if not preds:
            # graph input — can only INSERT
            return insert_cost, insert_plan

        pred_c = preds[0]
        above_cost, above_plan = self.above[pred_c]  # lookup, no recursion
        comp_cost, comp_plan = self._plan_compensate(F, ci_set)
        go_cost = comp_cost + above_cost

        if insert_cost <= go_cost:
            return insert_cost, insert_plan
        return go_cost, comp_plan + above_plan

    def _plan_join(self, C: ComputeNode) -> tuple[int, list[Action]]:
        """Plan for JOIN (multi-input) node: group preds by nearest_convergence_up."""
        preds_f = list(self.dag.predecessors(C))
        total = 0
        plan: list[Action] = []
        grouped: dict[int, dict] = {}

        for f_i in preds_f:
            pred_cs = list(self.dag.predecessors(f_i))
            if not pred_cs:
                # f_i is graph input — must terminate here
                cost_i, plan_i = self._plan_terminate(f_i, frozenset({C}))
                total += cost_i
                plan.extend(plan_i)
                continue

            # Check convergence BEFORE absorber short-circuit (same reasoning as PlanBuilder)
            ncu = self.ga.nearest_convergence_up.get(f_i)
            if ncu is not None:
                gid = id(ncu)
                if gid not in grouped:
                    grouped[gid] = {'G': ncu, 'f_list': [], 'comp_sum': 0, 'ci_set': set(), 'comp_plan': []}
                grouped[gid]['f_list'].append(f_i)
                if f_i is ncu:
                    grouped[gid]['ci_set'].add(C)
                else:
                    comp_cost, comp_plan = self._plan_compensate(f_i, frozenset({C}))
                    grouped[gid]['comp_sum'] += comp_cost
                    grouped[gid]['comp_plan'].extend(comp_plan)
                continue

            c_i = pred_cs[0]
            if c_i.layer_type in ABSORBER_TYPES:
                plan.append(AbsorbInto(c_i, 'up'))
                continue

            cost_i, plan_i = self._plan_at(f_i, frozenset({C}))
            total += cost_i
            plan.extend(plan_i)

        # Resolve ci_set for each convergence group
        for grp in grouped.values():
            G = grp['G']
            for f_i in grp['f_list']:
                if f_i is G:
                    continue
                grp['ci_set'].update(self._cis_reaching_fi(G, f_i))

        for grp in grouped.values():
            cost_g, plan_g = self._plan_at(grp['G'], frozenset(grp['ci_set']))
            total += grp['comp_sum'] + cost_g
            plan.extend(grp['comp_plan'])
            plan.extend(plan_g)

        return total, plan

    # ------------------------------------------------------------------- #
    # Helpers
    # ------------------------------------------------------------------- #
    def _plan_terminate(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        """INSERT: terminate scale at F, act only on ci_set arms."""
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c not in ci_set:
                continue
            sub_cost, sub_plan = self._plan_find_sink(F, c, inject_from='up')
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def _plan_compensate(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        """GO_UP: F is lifted; emit 1/s on every non-ci_set output arm."""
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c in ci_set:
                continue
            sub_cost, sub_plan = self._plan_find_sink(F, c, inject_from='down')
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def plan_down(self, F: FeatureNode, c: ComputeNode) -> tuple[int, list[Action]]:
        """Public wrapper kept for the top-level entry call."""
        return self._plan_find_sink(F, c, inject_from='down')

    def _edge_arm_cost(self, F: FeatureNode, c: ComputeNode) -> int:
        return _NONFREE_COST if c in self.ga.bottleneck_succs.get(F, frozenset()) else _FREE_COST

    def _plan_find_sink(
        self,
        F: FeatureNode,
        c: ComputeNode,
        inject_from: Literal['up', 'down'],
        carried_cost: int | None = None,
    ) -> tuple[int, list[Action]]:
        """Find the nearest sink(s) reachable from F→c and emit absorb/insert actions.

        Walks the DAG downward regardless of inject_from; inject_from only
        controls which action tag (and which node field) is used at the sink.

        Terminals (forced stop):
          1. Absorber (conv/fc/ms)  → AbsorbInto(c, inject_from), cost 0
          2. polyact + inject 'up'  → AbsorbInto(c, 'up'),        cost 0
          3. polyact + inject 'down'→ InsertMsOn(F, c, 'down'),   cost carried_cost
          4. multi-input node       → InsertMsOn(F, c, inject_from), cost carried_cost
          5. graph output           → InsertMsOn(F, c, inject_from), cost carried_cost
        Single-output pass-through inherits carried_cost; new fan-out recomputes per child arm.
        Memoised by (id(F), id(c), inject_from, carried_cost).
        """
        if carried_cost is None:
            carried_cost = self._edge_arm_cost(F, c)
        key = (id(F), id(c), inject_from, carried_cost)
        if key in self._down_memo:
            return self._down_memo[key]
        result = self._plan_find_sink_impl(F, c, inject_from, carried_cost)
        self._down_memo[key] = result
        return result

    def _plan_find_sink_impl(
        self,
        F: FeatureNode,
        c: ComputeNode,
        inject_from: Literal['up', 'down'],
        carried_cost: int,
    ) -> tuple[int, list[Action]]:
        # Terminal 1: absorber — free in both directions
        if c.layer_type in DOWN_ABSORBER_TYPES:
            return 0, [AbsorbInto(c, inject_from)]

        # Terminal 2/3: polyact
        if c.layer_type == 'polyact':
            if inject_from == 'up':
                return 0, [AbsorbInto(c, 'up')]  # polyact absorbs scale from above
            return carried_cost, [InsertMsOn(F, c, 'down')]

        # Terminal 4: multi-input node — must stop on this arm's edge
        if len(list(self.dag.predecessors(c))) != 1:
            return carried_cost, [InsertMsOn(F, c, inject_from)]

        # Single-input pass-through: recurse into c's output subtree
        out_feats = list(self.dag.successors(c))
        if not out_feats:
            return carried_cost, [InsertMsOn(F, c, inject_from)]
        F_out = out_feats[0]
        succs_out = list(self.dag.successors(F_out))

        # Terminal 5: graph output — nowhere further to go
        if not succs_out:
            return carried_cost, [InsertMsOn(F, c, inject_from)]

        if len(succs_out) == 1:
            return self._plan_find_sink(F_out, succs_out[0], inject_from, carried_cost)

        total_cost = 0
        total_plan: list[Action] = []
        for c2 in succs_out:
            sub_cost, sub_plan = self._plan_find_sink(
                F_out,
                c2,
                inject_from,
                self._edge_arm_cost(F_out, c2),
            )
            total_cost += sub_cost
            total_plan.extend(sub_plan)
        return total_cost, total_plan

    def _cis_reaching_fi(self, G: FeatureNode, f_i: FeatureNode) -> set:
        """Direct succs of G whose downstream subgraph contains f_i."""
        result: set = set()
        for c in self.dag.successors(G):
            if self._reaches(c, f_i):
                result.add(c)
        return result

    def _reaches(self, start, target) -> bool:
        visited: set = set()
        stack = [start]
        while stack:
            n = stack.pop()
            nid = id(n)
            if nid in visited:
                continue
            visited.add(nid)
            if n is target:
                return True
            for s in self.dag.successors(n):
                stack.append(s)
        return False


# ---------------------------------------------------------------------------
# Plan executor  (identical to absorb_scale_dp.py)
# ---------------------------------------------------------------------------
def execute(graph: LayerAbstractGraph, plan: list[Action], s: float):
    s_up = s
    s_down = 1.0 / s
    g = LayerAbstractGraph()
    g.dag = graph.dag

    _, arm_level = _backward_level_dict(graph.dag)

    for a in plan:
        if isinstance(a, AbsorbInto):
            if a.direction == 'up':
                a.compute.scale_down *= s_up
            else:
                a.compute.scale_up *= s_down
        elif isinstance(a, InsertMsOn):
            ms = add_mult_scalar_between_feature_and_layer(g, a.f_node, a.c_node)
            if a.direction == 'up':
                ms.scale_down *= s_up
            else:
                ms.scale_up *= s_down
        elif isinstance(a, CompensateDown):
            _propagate_scale(graph.dag, a.f_node, a.c_node, Direction.DOWN, s_down, arm_level=arm_level)
        elif isinstance(a, ClearSource):
            a.mc.coeff = 1.0
        else:
            raise TypeError(f'Unknown action: {a!r}')


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def absorb_scale_dp(graph: LayerAbstractGraph):
    """Iterative plan-based scale absorption (drop-in replacement for absorb_scale_dp.py)."""
    layers_to_absorb = ('avgpool1d', 'avgpool2d', 'mult_coeff')
    processed: set[str] = set()

    while True:
        node = next(
            (
                n
                for n in graph.dag.nodes
                if isinstance(n, ComputeNode) and n.layer_type in layers_to_absorb and n.layer_id not in processed
            ),
            None,
        )
        if node is None:
            break
        processed.add(node.layer_id)

        if node.layer_type in ('avgpool1d', 'avgpool2d'):
            if not (node.is_adaptive_avgpool or node.is_big_size):
                continue
            scale = 1.0 / math.prod(node.kernel_shape)
        else:
            scale = node.coeff

        pre_f = list(graph.dag.predecessors(node))[0]
        out_f = next(graph.dag.successors(node))
        preds_of_pre_f = list(graph.dag.predecessors(pre_f))

        if not preds_of_pre_f:
            _, arm_level = _backward_level_dict(graph.dag)
            _propagate_scale(graph.dag, node, out_f, Direction.DOWN, scale, arm_level=arm_level)
            continue

        # Pass 1a: static analysis
        ga = GraphAnalysis(graph.dag)
        # Pass 1b: precompute above[C] for all ComputeNodes in topo order
        builder = PlanBuilderIterative(ga)
        builder.build()

        # Build plan from precomputed table
        plan: list[Action] = []
        exclude_source = node.layer_type == 'mult_coeff'
        for c in graph.dag.successors(pre_f):
            if c is node and exclude_source:
                continue
            _, sub_plan = builder.plan_down(pre_f, c)
            plan.extend(sub_plan)

        pred_c = preds_of_pre_f[0]
        plan.extend(builder.above[pred_c][1])  # lookup, no recursion

        if node.layer_type == 'mult_coeff':
            plan.append(ClearSource(node))

        # Pass 2: execute
        execute(graph, plan, scale)

    for n in graph.dag.nodes:
        if isinstance(n, ComputeNode) and n.layer_type == 'mult_scalar':
            print('layer=', n.layer_id, 'scale_up=', n.scale_up, n.scale_down)

    _remove_identity_mult_scalars(graph)
