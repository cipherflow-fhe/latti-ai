# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, Union

import networkx as nx

from components import ComputeNode, FeatureNode, LayerAbstractGraph
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
        self.nearest_convergence_down: dict[object, object | None] = {}

        self._compute_bottleneck_succs()
        self._compute_nearest_convergence_up()
        self._compute_nearest_convergence_down()

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

    def _compute_nearest_convergence_down(self):
        id_to_feat = {id(n): n for n in self.dag.nodes if isinstance(n, FeatureNode)}

        def downstream_feat_ids(start):
            visited: set = set()
            result: set = set()
            stack = [start]
            while stack:
                n = stack.pop()
                nid = id(n)
                if nid in visited:
                    continue
                visited.add(nid)
                if isinstance(n, FeatureNode):
                    result.add(nid)
                stack.extend(self.dag.successors(n))
            return result

        self.nearest_convergence_down = {n: None for n in self.dag.nodes if isinstance(n, ComputeNode)}
        for node in self.topo:
            if not isinstance(node, FeatureNode):
                continue
            succs = list(self.dag.successors(node))
            if len(succs) < 2:
                continue
            reach = [(c, downstream_feat_ids(c)) for c in succs]
            for i in range(len(reach)):
                c_i, set_i = reach[i]
                for j in range(i + 1, len(reach)):
                    c_j, set_j = reach[j]
                    shared = set_i & set_j
                    if not shared:
                        continue
                    nearest_fid = min(
                        (fid for fid in shared if fid in id_to_feat),
                        key=lambda fid: self.topo_rank.get(id_to_feat[fid], math.inf),
                        default=None,
                    )
                    if nearest_fid is None:
                        continue
                    nearest_f = id_to_feat[nearest_fid]
                    for arm_c in (c_i, c_j):
                        existing = self.nearest_convergence_down[arm_c]
                        if existing is None or self.topo_rank[nearest_f] < self.topo_rank[existing]:
                            self.nearest_convergence_down[arm_c] = nearest_f


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
        self.down: dict[tuple[FeatureNode, ComputeNode, int], tuple[int, list[Action]]] = {}
        self._down_memo: dict[tuple, tuple[int, list[Action]]] = {}

    # ------------------------------------------------------------------- #
    # Public entry
    # ------------------------------------------------------------------- #
    def build(self):
        """Precompute UP then DOWN plans."""
        for node in self.ga.topo:
            if isinstance(node, ComputeNode):
                self._compute_above(node)
        self._build_down()

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

    def _plan_at(
        self,
        F: FeatureNode,
        ci_set: frozenset,
        targets: frozenset[FeatureNode] | None = None,
    ) -> tuple[int, list[Action]]:
        """min(INSERT at F, GO_UP through F) for scale arriving via ci_set arms.

        No recursion: above[pred_c] is always in self.above because pred_c
        is upstream of F and was processed earlier in topo order.
        """
        insert_cost, insert_plan = self._plan_terminate(F, ci_set, targets)

        preds = list(self.dag.predecessors(F))
        if not preds:
            # graph input — can only INSERT
            return insert_cost, insert_plan

        pred_c = preds[0]
        above_cost, above_plan = self.above[pred_c]  # lookup, no recursion
        comp_cost, comp_plan = self._plan_compensate(F, ci_set)
        if targets is not None:
            off_cost, off_plan = self._plan_compensate_off_target_branches(F, ci_set, targets)
            comp_cost += off_cost
            comp_plan.extend(off_plan)
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
                    grouped[gid] = {
                        'G': ncu,
                        'f_list': [],
                        'target_fis': set(),
                        'comp_sum': 0,
                        'ci_set': set(),
                        'comp_plan': [],
                    }
                grouped[gid]['f_list'].append(f_i)
                grouped[gid]['target_fis'].add(f_i)
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
            cost_g, plan_g = self._plan_at(
                grp['G'],
                frozenset(grp['ci_set']),
                frozenset(grp['target_fis']),
            )
            total += grp['comp_sum'] + cost_g
            plan.extend(grp['comp_plan'])
            plan.extend(plan_g)

        return total, plan

    # ------------------------------------------------------------------- #
    # Helpers
    # ------------------------------------------------------------------- #
    def _plan_terminate(
        self,
        F: FeatureNode,
        ci_set: frozenset,
        targets: frozenset[FeatureNode] | None = None,
    ) -> tuple[int, list[Action]]:
        """INSERT: terminate scale at F, act only on ci_set arms."""
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c not in ci_set:
                continue
            if targets is None:
                cost += self._edge_arm_cost(F, c)
                plan.append(InsertMsOn(F, c, 'up'))
                continue
            sub_cost, sub_plan = self._plan_find_sink(F, c, inject_from='up', targets=targets)
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

    def _plan_compensate_off_target_branches(
        self,
        F: FeatureNode,
        ci_set: frozenset,
        targets: frozenset[FeatureNode],
    ) -> tuple[int, list[Action]]:
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c not in ci_set:
                continue
            sub_cost, sub_plan = self._plan_compensate_off_target_edge(F, c, targets)
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def _plan_compensate_off_target_edge(
        self,
        F: FeatureNode,
        c: ComputeNode,
        targets: frozenset[FeatureNode],
    ) -> tuple[int, list[Action]]:
        if not any(self._reaches(c, target) for target in targets):
            return self._plan_find_sink(F, c, inject_from='down')

        out_feats = list(self.dag.successors(c))
        if not out_feats:
            return 0, []
        F_out = out_feats[0]
        if F_out in targets:
            return 0, []

        cost = 0
        plan: list[Action] = []
        for c2 in self.dag.successors(F_out):
            if any(self._reaches(c2, target) for target in targets):
                sub_cost, sub_plan = self._plan_compensate_off_target_edge(F_out, c2, targets)
            else:
                sub_cost, sub_plan = self._plan_find_sink(F_out, c2, inject_from='down')
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def plan_down(self, F: FeatureNode, c: ComputeNode) -> tuple[int, list[Action]]:
        """Full DOWN plan for scale compensation along F→c."""
        carried_cost = self._edge_arm_cost(F, c)
        return self._get_down(F, c, carried_cost)

    def _get_down(self, F: FeatureNode, c: ComputeNode, carried_cost: int) -> tuple[int, list[Action]]:
        key = (F, c, carried_cost)
        if key in self.down:
            return self.down[key]
        return self._compute_down_edge(F, c, carried_cost)

    def _build_down(self):
        for node in reversed(self.ga.topo):
            if not isinstance(node, FeatureNode):
                continue
            for c in self.dag.successors(node):
                self.down[(node, c, _FREE_COST)] = self._compute_down_edge(node, c, _FREE_COST)
                self.down[(node, c, _NONFREE_COST)] = self._compute_down_edge(node, c, _NONFREE_COST)

    def _compute_down_edge(
        self,
        F: FeatureNode,
        c: ComputeNode,
        carried_cost: int,
    ) -> tuple[int, list[Action]]:
        return self._plan_down_at(c, {F: carried_cost})

    def _plan_down_at(
        self,
        c: ComputeNode,
        fi_cost: dict[FeatureNode, int],
    ) -> tuple[int, list[Action]]:
        if c.layer_type in DOWN_ABSORBER_TYPES:
            return 0, [AbsorbInto(c, 'down')]

        insert_cost, insert_plan = self._plan_down_terminate(c, fi_cost)

        if c.layer_type == 'polyact':
            return insert_cost, insert_plan

        out_feats = list(self.dag.successors(c))
        if not out_feats:
            return insert_cost, insert_plan

        F_out = out_feats[0]
        if not list(self.dag.successors(F_out)):
            return insert_cost, insert_plan

        comp_cost, comp_plan = self._plan_down_compensate(c, frozenset(fi_cost))
        down_cost, down_plan = self._plan_down_from_feature(F_out, max(fi_cost.values()))
        go_cost = comp_cost + down_cost

        if insert_cost <= go_cost:
            return insert_cost, insert_plan
        return go_cost, comp_plan + down_plan

    def _plan_down_terminate(
        self,
        c: ComputeNode,
        fi_cost: dict[FeatureNode, int],
    ) -> tuple[int, list[Action]]:
        cost = 0
        plan: list[Action] = []
        for f, arm_cost in fi_cost.items():
            cost += arm_cost
            plan.append(InsertMsOn(f, c, 'down'))
        return cost, plan

    def _plan_down_compensate(self, c: ComputeNode, fi_set: frozenset) -> tuple[int, list[Action]]:
        cost = 0
        plan: list[Action] = []
        for f_other in self.dag.predecessors(c):
            if f_other in fi_set:
                continue
            cost_i, plan_i = self._plan_at(f_other, frozenset({c}))
            cost += cost_i
            plan.extend(plan_i)
        return cost, plan

    def _plan_down_from_feature(self, F: FeatureNode, carried_cost: int) -> tuple[int, list[Action]]:
        succs = list(self.dag.successors(F))
        if not succs:
            return math.inf, []
        if len(succs) == 1:
            c = succs[0]
            return self._get_down(F, c, carried_cost)
        return self._plan_down_fork(F)

    def _plan_down_fork(self, F: FeatureNode) -> tuple[int, list[Action]]:
        total_cost = 0
        total_plan: list[Action] = []
        grouped: dict[int, dict] = {}

        for c in self.dag.successors(F):
            ncd = self.ga.nearest_convergence_down.get(c)
            if ncd is None:
                cost_i, plan_i = self._get_down(F, c, self._edge_arm_cost(F, c))
                total_cost += cost_i
                total_plan.extend(plan_i)
                continue

            gid = id(ncd)
            if gid not in grouped:
                grouped[gid] = {'H': ncd, 'arms': []}
            grouped[gid]['arms'].append(c)

        for grp in grouped.values():
            cost_g, plan_g = self._plan_down_convergence_group(F, grp['H'], grp['arms'])
            total_cost += cost_g
            total_plan.extend(plan_g)

        return total_cost, total_plan

    def _plan_down_convergence_group(
        self,
        F: FeatureNode,
        H: FeatureNode,
        arms: list[ComputeNode],
    ) -> tuple[int, list[Action]]:
        independent_cost = 0
        independent_plan: list[Action] = []
        for c in arms:
            cost_i, plan_i = self._get_down(F, c, self._edge_arm_cost(F, c))
            independent_cost += cost_i
            independent_plan.extend(plan_i)

        join_preds = list(self.dag.predecessors(H))
        if len(join_preds) != 1:
            return independent_cost, independent_plan
        join_c = join_preds[0]
        if len(list(self.dag.predecessors(join_c))) <= 1:
            return independent_cost, independent_plan

        fi_cost = self._fis_reached_from_arms(join_c, arms)
        if not fi_cost:
            return independent_cost, independent_plan

        shared_cost, shared_plan = self._plan_down_at(join_c, fi_cost)
        if independent_cost <= shared_cost:
            return independent_cost, independent_plan
        return shared_cost, shared_plan

    def _fis_reached_from_arms(
        self,
        join_c: ComputeNode,
        arms: list[ComputeNode],
    ) -> dict[FeatureNode, int]:
        result: dict[FeatureNode, int] = {}
        for f in self.dag.predecessors(join_c):
            if any(self._reaches(arm_c, f) for arm_c in arms):
                result[f] = self._edge_arm_cost(f, join_c)
        return result

    def _edge_arm_cost(self, F: FeatureNode, c: ComputeNode) -> int:
        return _NONFREE_COST if c in self.ga.bottleneck_succs.get(F, frozenset()) else _FREE_COST

    def _plan_find_sink(
        self,
        F: FeatureNode,
        c: ComputeNode,
        inject_from: Literal['up', 'down'],
        carried_cost: int | None = None,
        targets: frozenset[FeatureNode] | None = None,
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
        With targets, fan-out only follows arms that can reach the target join inputs.
        Memoised by (id(F), id(c), inject_from, carried_cost, targets).
        """
        if carried_cost is None:
            carried_cost = self._edge_arm_cost(F, c)
        target_key = None if targets is None else frozenset(id(target) for target in targets)
        key = (id(F), id(c), inject_from, carried_cost, target_key)
        if key in self._down_memo:
            return self._down_memo[key]
        result = self._plan_find_sink_impl(F, c, inject_from, carried_cost, targets)
        self._down_memo[key] = result
        return result

    def _plan_find_sink_impl(
        self,
        F: FeatureNode,
        c: ComputeNode,
        inject_from: Literal['up', 'down'],
        carried_cost: int,
        targets: frozenset[FeatureNode] | None,
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
            return self._plan_find_sink(F_out, succs_out[0], inject_from, carried_cost, targets)

        if targets is None:
            target_succs = succs_out
        else:
            target_succs = [c2 for c2 in succs_out if any(self._reaches(c2, target) for target in targets)]
            if not target_succs:
                return 0, []

        split_cost = 0
        split_plan: list[Action] = []
        for c2 in target_succs:
            sub_cost, sub_plan = self._plan_find_sink(
                F_out,
                c2,
                inject_from,
                self._edge_arm_cost(F_out, c2),
                targets,
            )
            split_cost += sub_cost
            split_plan.extend(sub_plan)

        if targets is None or inject_from != 'up' or c.layer_type == 'polyact':
            return split_cost, split_plan

        hoist_cost = carried_cost
        hoist_plan: list[Action] = [InsertMsOn(F, c, 'up')]
        for c2 in succs_out:
            if c2 in target_succs:
                continue
            sub_cost, sub_plan = self._plan_find_sink(
                F_out,
                c2,
                inject_from='down',
                carried_cost=self._edge_arm_cost(F_out, c2),
            )
            hoist_cost += sub_cost
            hoist_plan.extend(sub_plan)

        if split_cost <= hoist_cost:
            return split_cost, split_plan
        return hoist_cost, hoist_plan

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


# _DEBUG_MEMO_HIT = os.environ.get('ABSORB_SCALE_DEBUG_MEMO') == '1'
_DEBUG_MEMO_HIT = True


def _node_label(node) -> str:
    return getattr(node, 'node_id', getattr(node, 'layer_id', repr(node)))


def _nodes_label(nodes) -> str:
    return '[' + ','.join(_node_label(n) for n in sorted(nodes, key=_node_label)) + ']'


def _debug_memo_hit(kind: str, **fields):
    if not _DEBUG_MEMO_HIT:
        return
    details = ' '.join(f'{k}={v}' for k, v in fields.items())
    print(f'[absorb_scale_target_aware memo hit] {kind} {details}')


@dataclass
class PlanChoice:
    cost: int
    plan: list[Action]


@dataclass
class MergeInputSink:
    compute: ComputeNode
    input_feature: FeatureNode
    targets: frozenset[FeatureNode]
    carried_cost: int


ArmResult = Union[PlanChoice, MergeInputSink]


class TargetAwarePlanner:
    def __init__(
        self,
        ga: GraphAnalysis,
        above: dict[object, tuple[int, list[Action]]],
        target_consumers: dict[FeatureNode, frozenset[ComputeNode]] | None = None,
    ):
        self.ga = ga
        self.dag = ga.dag
        self.above = above
        self.target_consumers = target_consumers or {}
        self._feature_memo: dict[tuple[int, frozenset[int]], PlanChoice] = {}
        self._edge_memo: dict[tuple[int, int, frozenset[int], int], ArmResult] = {}
        self._merge_memo: dict[tuple[int, frozenset[int], frozenset[int], tuple[tuple[int, int], ...]], PlanChoice] = {}
        self._down_memo: dict[tuple[int, int, int], PlanChoice] = {}
        self._reach_memo: dict[tuple[int, int], bool] = {}

    def plan_feature(self, F: FeatureNode, targets: frozenset[FeatureNode]) -> tuple[int, list[Action]]:
        targets = self._filter_reachable_targets(F, targets)
        if not targets:
            return 0, []

        key = (id(F), self._target_key(targets))
        if key in self._feature_memo:
            choice = self._feature_memo[key]
            _debug_memo_hit('feature', F=_node_label(F), targets=_nodes_label(targets), cost=choice.cost)
            return choice.cost, list(choice.plan)

        split = self._plan_feature_split(F, targets)
        go_up = self._plan_feature_go_up(F, targets)
        best = self._min_choice(split, go_up)
        self._feature_memo[key] = best
        return best.cost, list(best.plan)

    def plan_edge(
        self,
        F: FeatureNode,
        c: ComputeNode,
        targets: frozenset[FeatureNode],
        carried_cost: int | None = None,
    ) -> ArmResult:
        targets = self._filter_reachable_targets(c, targets)
        if not targets:
            return PlanChoice(0, [])
        if carried_cost is None:
            carried_cost = self._edge_arm_cost(F, c)

        key = (id(F), id(c), self._target_key(targets), carried_cost)
        if key in self._edge_memo:
            result = self._edge_memo[key]
            result_cost = result.cost if isinstance(result, PlanChoice) else 'merge'
            _debug_memo_hit(
                'edge',
                F=_node_label(F),
                c=_node_label(c),
                targets=_nodes_label(targets),
                carried_cost=carried_cost,
                cost=result_cost,
            )
            return result

        result = self._plan_edge_impl(F, c, targets, carried_cost)
        self._edge_memo[key] = result
        return result

    def plan_merge_node(
        self,
        C: ComputeNode,
        covered_inputs: frozenset[FeatureNode],
        targets: frozenset[FeatureNode],
        input_costs: dict[FeatureNode, int],
    ) -> tuple[int, list[Action]]:
        targets = self._filter_reachable_targets(C, targets)
        if not targets:
            return 0, []

        cost_key = tuple(sorted((id(f), input_costs[f]) for f in covered_inputs))
        key = (id(C), frozenset(id(f) for f in covered_inputs), self._target_key(targets), cost_key)
        if key in self._merge_memo:
            choice = self._merge_memo[key]
            _debug_memo_hit(
                'merge',
                C=_node_label(C),
                inputs=_nodes_label(covered_inputs),
                targets=_nodes_label(targets),
                cost=choice.cost,
            )
            return choice.cost, list(choice.plan)

        stop = PlanChoice(
            sum(input_costs[f] for f in covered_inputs),
            [InsertMsOn(f, C, 'up') for f in covered_inputs],
        )
        best = stop

        all_inputs = frozenset(self.dag.predecessors(C))
        out_feats = list(self.dag.successors(C))
        if covered_inputs == all_inputs and out_feats:
            F_out = out_feats[0]
            out_targets = self._filter_reachable_targets(F_out, targets)
            if out_targets:
                cost, plan = self.plan_feature(F_out, out_targets)
                best = self._min_choice(best, PlanChoice(cost, plan))

        self._merge_memo[key] = best
        return best.cost, list(best.plan)

    def _plan_feature_split(self, F: FeatureNode, targets: frozenset[FeatureNode]) -> PlanChoice:
        cost = 0
        plan: list[Action] = []
        merge_groups: dict[ComputeNode, dict] = {}

        if F in targets:
            target_cost, target_plan = self._plan_target_feature(F)
            cost += target_cost
            plan.extend(target_plan)
            targets = frozenset(t for t in targets if t is not F)

        for c in self.dag.successors(F):
            targets_c = self._filter_reachable_targets(c, targets)
            if not targets_c:
                continue

            result = self.plan_edge(F, c, targets_c, self._edge_arm_cost(F, c))
            if isinstance(result, PlanChoice):
                cost += result.cost
                plan.extend(result.plan)
                continue

            grp = merge_groups.setdefault(result.compute, {'inputs': set(), 'targets': set(), 'input_costs': {}})
            grp['inputs'].add(result.input_feature)
            grp['targets'].update(result.targets)
            grp['input_costs'][result.input_feature] = result.carried_cost

        for C, grp in merge_groups.items():
            sub_cost, sub_plan = self.plan_merge_node(
                C,
                frozenset(grp['inputs']),
                frozenset(grp['targets']),
                grp['input_costs'],
            )
            cost += sub_cost
            plan.extend(sub_plan)

        return PlanChoice(cost, plan)

    def _plan_feature_go_up(self, F: FeatureNode, targets: frozenset[FeatureNode]) -> PlanChoice:
        preds = list(self.dag.predecessors(F))
        if not preds:
            return PlanChoice(math.inf, [])

        pred_c = preds[0]
        if pred_c not in self.above:
            return PlanChoice(math.inf, [])

        above_cost, above_plan = self.above[pred_c]
        cost = above_cost
        plan = list(above_plan)

        for c in self.dag.successors(F):
            if self._filter_reachable_targets(c, targets):
                sub_cost, sub_plan = self._compensate_off_target_edge(F, c, targets)
            else:
                sub = self._plan_down_sink(F, c, self._edge_arm_cost(F, c))
                sub_cost, sub_plan = sub.cost, sub.plan
            cost += sub_cost
            plan.extend(sub_plan)

        return PlanChoice(cost, plan)

    def _plan_edge_impl(
        self,
        F: FeatureNode,
        c: ComputeNode,
        targets: frozenset[FeatureNode],
        carried_cost: int,
    ) -> ArmResult:
        if c.layer_type in ABSORBER_TYPES:
            return PlanChoice(0, [AbsorbInto(c, 'up')])

        preds = list(self.dag.predecessors(c))
        if len(preds) != 1:
            return MergeInputSink(c, F, targets, carried_cost)

        out_feats = list(self.dag.successors(c))
        if not out_feats:
            return PlanChoice(carried_cost, [InsertMsOn(F, c, 'up')])

        F_out = out_feats[0]
        out_targets = self._filter_reachable_targets(F_out, targets)
        if not out_targets:
            return PlanChoice(carried_cost, [InsertMsOn(F, c, 'up')])

        if F_out in targets:
            cost, plan = self.plan_feature(F_out, out_targets)
            return PlanChoice(cost, plan)

        succs_out = list(self.dag.successors(F_out))
        if len(succs_out) == 1:
            return self.plan_edge(F_out, succs_out[0], out_targets, carried_cost)

        cost, plan = self.plan_feature(F_out, out_targets)
        pass_choice = PlanChoice(cost, plan)
        off_cost, off_plan = self._compensate_off_target_edge(F, c, out_targets)
        hoist_choice = PlanChoice(carried_cost + off_cost, [InsertMsOn(F, c, 'up'), *off_plan])
        return self._min_choice(pass_choice, hoist_choice)

    def _plan_target_feature(self, F: FeatureNode) -> tuple[int, list[Action]]:
        consumers = self.target_consumers.get(F)
        if consumers is None:
            consumers = frozenset(self.dag.successors(F))
        if not consumers:
            return 0, []

        cost = 0
        plan: list[Action] = []
        for c in consumers:
            if c.layer_type in ABSORBER_TYPES:
                plan.append(AbsorbInto(c, 'up'))
            else:
                cost += self._edge_arm_cost(F, c)
                plan.append(InsertMsOn(F, c, 'up'))
        return cost, plan

    def _compensate_off_target_edge(
        self,
        F: FeatureNode,
        c: ComputeNode,
        targets: frozenset[FeatureNode],
    ) -> tuple[int, list[Action]]:
        if not self._filter_reachable_targets(c, targets):
            choice = self._plan_down_sink(F, c, self._edge_arm_cost(F, c))
            return choice.cost, choice.plan

        out_feats = list(self.dag.successors(c))
        if not out_feats:
            return 0, []
        F_out = out_feats[0]
        if F_out in targets:
            return 0, []

        cost = 0
        plan: list[Action] = []
        for c2 in self.dag.successors(F_out):
            if self._filter_reachable_targets(c2, targets):
                sub_cost, sub_plan = self._compensate_off_target_edge(F_out, c2, targets)
            else:
                sub = self._plan_down_sink(F_out, c2, self._edge_arm_cost(F_out, c2))
                sub_cost, sub_plan = sub.cost, sub.plan
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def _plan_down_sink(self, F: FeatureNode, c: ComputeNode, carried_cost: int) -> PlanChoice:
        key = (id(F), id(c), carried_cost)
        if key in self._down_memo:
            choice = self._down_memo[key]
            _debug_memo_hit('down', F=_node_label(F), c=_node_label(c), carried_cost=carried_cost, cost=choice.cost)
            return choice

        if c.layer_type in DOWN_ABSORBER_TYPES:
            choice = PlanChoice(0, [AbsorbInto(c, 'down')])
        elif c.layer_type == 'polyact':
            choice = PlanChoice(carried_cost, [InsertMsOn(F, c, 'down')])
        elif len(list(self.dag.predecessors(c))) != 1:
            choice = PlanChoice(carried_cost, [InsertMsOn(F, c, 'down')])
        else:
            out_feats = list(self.dag.successors(c))
            if not out_feats:
                choice = PlanChoice(carried_cost, [InsertMsOn(F, c, 'down')])
            else:
                F_out = out_feats[0]
                succs = list(self.dag.successors(F_out))
                if not succs:
                    choice = PlanChoice(carried_cost, [InsertMsOn(F, c, 'down')])
                elif len(succs) == 1:
                    choice = self._plan_down_sink(F_out, succs[0], carried_cost)
                else:
                    total = 0
                    plan: list[Action] = []
                    for c2 in succs:
                        sub = self._plan_down_sink(F_out, c2, self._edge_arm_cost(F_out, c2))
                        total += sub.cost
                        plan.extend(sub.plan)
                    choice = PlanChoice(total, plan)

        self._down_memo[key] = choice
        return choice

    def _filter_reachable_targets(self, start, targets: frozenset[FeatureNode]) -> frozenset[FeatureNode]:
        return frozenset(t for t in targets if start is t or self._reaches(start, t))

    def _edge_arm_cost(self, F: FeatureNode, c: ComputeNode) -> int:
        return _NONFREE_COST if c in self.ga.bottleneck_succs.get(F, frozenset()) else _FREE_COST

    def _reaches(self, start, target) -> bool:
        key = (id(start), id(target))
        if key in self._reach_memo:
            return self._reach_memo[key]

        visited: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            if node is target:
                self._reach_memo[key] = True
                return True
            stack.extend(self.dag.successors(node))

        self._reach_memo[key] = False
        return False

    def _target_key(self, targets: frozenset[FeatureNode]) -> frozenset[int]:
        return frozenset(id(t) for t in targets)

    def _min_choice(self, a: PlanChoice, b: PlanChoice) -> PlanChoice:
        if a.cost <= b.cost:
            return a
        return b


def _upstream_features(builder: PlanBuilderIterative, start: FeatureNode) -> set[FeatureNode]:
    visited: set[int] = set()
    result: set[FeatureNode] = set()
    stack = [start]
    while stack:
        f = stack.pop()
        fid = id(f)
        if fid in visited:
            continue
        visited.add(fid)
        result.add(f)
        for c in builder.dag.predecessors(f):
            stack.extend(builder.dag.predecessors(c))
    return result


def _maximal_common_upstream_groups(
    builder: PlanBuilderIterative,
    targets: list[FeatureNode],
) -> list[tuple[FeatureNode | None, frozenset[FeatureNode]]]:
    remaining = set(targets)
    upstream_by_target = {target: _upstream_features(builder, target) for target in targets}
    groups: list[tuple[FeatureNode | None, frozenset[FeatureNode]]] = []

    while remaining:
        covered_by_anchor: dict[FeatureNode, set[FeatureNode]] = {}
        for target in remaining:
            for anchor in upstream_by_target[target]:
                covered_by_anchor.setdefault(anchor, set()).add(target)

        best_anchor = None
        best_targets: set[FeatureNode] = set()
        for anchor, covered in covered_by_anchor.items():
            if len(covered) < 2:
                continue
            if len(covered) > len(best_targets):
                best_anchor = anchor
                best_targets = covered
                continue
            if len(covered) == len(best_targets) and best_anchor is not None:
                if builder.ga.topo_rank.get(anchor, -1) > builder.ga.topo_rank.get(best_anchor, -1):
                    best_anchor = anchor
                    best_targets = covered

        if best_anchor is None:
            for target in sorted(remaining, key=_node_label):
                groups.append((None, frozenset({target})))
            break

        targets_group = frozenset(best_targets)
        groups.append((best_anchor, targets_group))
        remaining -= best_targets

    return groups


def _plan_single_input_above_target_aware(
    builder: PlanBuilderIterative,
    C: ComputeNode,
    f_i: FeatureNode,
) -> tuple[int, list[Action]]:
    pred_cs = list(builder.dag.predecessors(f_i))
    if not pred_cs:
        return builder._plan_terminate(f_i, frozenset({C}))

    c_i = pred_cs[0]
    if c_i.layer_type in ABSORBER_TYPES:
        return 0, [AbsorbInto(c_i, 'up')]

    return builder._plan_at(f_i, frozenset({C}))


def _plan_multi_input_above_target_aware(
    builder: PlanBuilderIterative,
    C: ComputeNode,
) -> tuple[int, list[Action]]:
    preds_f = list(builder.dag.predecessors(C))
    target_consumers = {f: frozenset({C}) for f in preds_f}
    target_planner = TargetAwarePlanner(builder.ga, builder.above, target_consumers)

    total = 0
    plan: list[Action] = []
    for anchor, targets in _maximal_common_upstream_groups(builder, preds_f):
        if anchor is None:
            target = next(iter(targets))
            cost_i, plan_i = _plan_single_input_above_target_aware(builder, C, target)
        else:
            cost_i, plan_i = target_planner.plan_feature(anchor, targets)
        total += cost_i
        plan.extend(plan_i)

    return total, plan


def absorb_scale_target_aware(graph: LayerAbstractGraph):
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

        ga = GraphAnalysis(graph.dag)
        builder = PlanBuilderIterative(ga)
        builder.build()

        if not preds_of_pre_f:
            plan: list[Action] = []
            for c in graph.dag.successors(out_f):
                _, sub_plan = builder.plan_down(out_f, c)
                plan.extend(sub_plan)
            if node.layer_type == 'mult_coeff' and plan:
                plan.append(ClearSource(node))
            if plan:
                execute(graph, plan, 1.0 / scale)
            else:
                _, arm_level = _backward_level_dict(graph.dag)
                _propagate_scale(graph.dag, node, out_f, Direction.DOWN, scale, arm_level=arm_level)
            continue

        plan: list[Action] = []
        exclude_source = node.layer_type == 'mult_coeff'
        for c in graph.dag.successors(pre_f):
            if c is node and exclude_source:
                continue
            _, sub_plan = builder.plan_down(pre_f, c)
            plan.extend(sub_plan)

        pred_c = preds_of_pre_f[0]
        if len(list(graph.dag.predecessors(pred_c))) > 1:
            _, above_plan = _plan_multi_input_above_target_aware(builder, pred_c)
            plan.extend(above_plan)
        else:
            plan.extend(builder.above[pred_c][1])

        if node.layer_type == 'mult_coeff':
            plan.append(ClearSource(node))

        execute(graph, plan, scale)

    _remove_identity_mult_scalars(graph)
