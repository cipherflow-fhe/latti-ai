# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Plan-based scale absorption.

Pass 1 (`PlanBuilder`) is a top-down memoized DP that returns
`(cost, list[Action])` per (F, ci_set) and per ComputeNode.  Pass 2
(`execute`) walks the flat action list.

State graph (all DAG — no cycle):

    plan_above(C)  ──→  plan_up(F, {C})              # single-input C
                   ──→  plan_up(G, ci_set)           # JOIN convergence groups
                   ──→  plan_up(f_i, {C})            # JOIN singletons

    plan_up(F, Σ) ──→  plan_above(pred_C(F))         # GO_UP option
                   ──→  plan_down(F, c)               # local arm: depth-1 only

    plan_down(F, c) — FLAT, no recursion:
      c is DOWN absorber  →  AbsorbInto(c, 'down')
      c is polyact        →  InsertMsOn(F, c, 'down') (polyact can't absorb DOWN)
      otherwise           →  CompensateDown(F, c)    (Pass 2 expands via
                                                      legacy _propagate_scale)

Action shapes:
  AbsorbInto(c, 'up')      : c.scale_down *= s
  AbsorbInto(c, 'down')    : c.scale_up   *= 1/s
  InsertMsOn(f, c, 'up')   : new ms on f→c; ms.scale_down *= s
  InsertMsOn(f, c, 'down') : new ms on f→c; ms.scale_up   *= 1/s
  CompensateDown(f, c)     : Pass 2 calls _propagate_scale(... DOWN, 1/s)
  ClearSource(mc)          : mc.coeff = 1.0
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
# Action types
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
    """Compensation flow: 1/s DOWN from f_node via c_node.  Executor
    delegates to legacy `_propagate_scale` for the recursive walk that
    finds the actual absorber / dead-end (with loop detection)."""

    f_node: FeatureNode
    c_node: ComputeNode


@dataclass(frozen=True)
class ClearSource:
    mc: ComputeNode  # mc.coeff = 1.0


Action = Union[AbsorbInto, InsertMsOn, CompensateDown, ClearSource]


# ---------------------------------------------------------------------------
# Static graph analysis
# ---------------------------------------------------------------------------
class GraphAnalysis:
    """Per-graph static properties consumed by PlanBuilder."""

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
        """Detect convergence between sibling arms of every JOIN.

        Two key differences from the legacy `absorber-stop` walk:
          1. `upstream_feat_ids` traverses **through** absorbers — so we can
             see when one arm of a JOIN sits on another arm's upstream path
             across an absorber boundary (e.g. DeepNested: F_b1 is downstream
             of F_c1a via conv1_b absorber).
          2. ncu candidates are filtered to FeatureNodes that are **also**
             direct preds of the current JOIN.  This rejects "deep" common
             ancestors (e.g. ThreeBranch: F3 and F5 share F1 deep upstream
             but F1 isn't a sibling at Add — no convergence at Add).
        """
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
                # Walk through ALL preds (including absorbers) so absorber-bracketed
                # sibling relationships become visible at JOIN level.
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
                    # Take ANY shared upstream FeatureNode — not filtered to
                    # JOIN preds.  Reason: even when the convergence point G
                    # is not itself a pred of the current JOIN (e.g. cat-cat
                    # NewModel where cat2's three arms all share F_inner via
                    # different routes), grouping them at G via _cis_reaching_fi
                    # is correct and often cheaper.
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
# Plan builder (top-down recursive DP, memoized)
# ---------------------------------------------------------------------------
class PlanBuilder:
    def __init__(self, ga: GraphAnalysis):
        self.ga = ga
        self.dag = ga.dag
        self._above_memo: dict[ComputeNode, tuple[int, list[Action]]] = {}
        self._up_memo: dict[tuple, tuple[int, list[Action]]] = {}

    # ------------------------------------------------------------------- #
    # T(C): min cost when scale has arrived above ComputeNode C
    # ------------------------------------------------------------------- #
    def plan_above(self, C: ComputeNode) -> tuple[int, list[Action]]:
        if C in self._above_memo:
            return self._above_memo[C]

        if C.layer_type in ABSORBER_TYPES:
            result = (0, [AbsorbInto(C, 'up')])
        else:
            preds_f = list(self.dag.predecessors(C))
            if len(preds_f) == 1:
                result = self.plan_up(preds_f[0], frozenset({C}))
            else:
                result = self._plan_join(C)

        self._above_memo[C] = result
        return result

    # ------------------------------------------------------------------- #
    # U(F, Σ): min cost when scale arrives at F via Σ arms going UP
    # ------------------------------------------------------------------- #
    def plan_up(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        key = (id(F), ci_set)
        if key in self._up_memo:
            return self._up_memo[key]

        insert_cost, insert_plan = self._plan_terminate(F, ci_set)

        preds = list(self.dag.predecessors(F))
        if not preds:
            # Graph input — can only INSERT
            result = (insert_cost, insert_plan)
        else:
            pred_c = preds[0]
            above_cost, above_plan = self.plan_above(pred_c)
            comp_cost, comp_plan = self._plan_compensate(F, ci_set)
            go_cost = comp_cost + above_cost

            if insert_cost <= go_cost:
                result = (insert_cost, insert_plan)
            else:
                result = (go_cost, comp_plan + above_plan)

        self._up_memo[key] = result
        return result

    # ------------------------------------------------------------------- #
    # D(F, c): FLAT — depth-1 only.  Pass 2 expands CompensateDown via the
    # legacy reactive walker that finds the actual landing point.
    # ------------------------------------------------------------------- #
    def plan_down(self, F: FeatureNode, c: ComputeNode) -> tuple[int, list[Action]]:
        if c.layer_type in DOWN_ABSORBER_TYPES:
            return 0, [AbsorbInto(c, 'down')]
        if c.layer_type == 'polyact':
            arm_cost = _NONFREE_COST if c in self.ga.bottleneck_succs.get(F, frozenset()) else _FREE_COST
            return arm_cost, [InsertMsOn(F, c, 'down')]
        arm_cost = _NONFREE_COST if c in self.ga.bottleneck_succs.get(F, frozenset()) else _FREE_COST
        return arm_cost, [CompensateDown(F, c)]

    # ------------------------------------------------------------------- #
    # internals
    # ------------------------------------------------------------------- #
    def _arm_cost(self, F: FeatureNode, c: ComputeNode) -> int:
        if c.layer_type in DOWN_ABSORBER_TYPES:
            return 0
        return _NONFREE_COST if c in self.ga.bottleneck_succs[F] else _FREE_COST

    def _plan_terminate(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        """INSERT branch: terminate at F, act only on ci_set arms (F unchanged)."""
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c not in ci_set:
                continue
            if c.layer_type in DOWN_ABSORBER_TYPES:
                plan.append(AbsorbInto(c, 'up'))
            else:
                plan.append(InsertMsOn(F, c, 'up'))
                cost += self._arm_cost(F, c)
        return cost, plan

    def _plan_compensate(self, F: FeatureNode, ci_set: frozenset) -> tuple[int, list[Action]]:
        """GO_UP branch: F is lifted; emit 1/s on every non-Σ output arm."""
        cost = 0
        plan: list[Action] = []
        for c in self.dag.successors(F):
            if c in ci_set:
                continue
            sub_cost, sub_plan = self.plan_down(F, c)
            cost += sub_cost
            plan.extend(sub_plan)
        return cost, plan

    def _plan_join(self, C: ComputeNode) -> tuple[int, list[Action]]:
        preds_f = list(self.dag.predecessors(C))
        total = 0
        plan: list[Action] = []
        grouped: dict[int, dict] = {}

        for f_i in preds_f:
            pred_cs = list(self.dag.predecessors(f_i))
            if not pred_cs:
                cost_i, plan_i = self._plan_terminate(f_i, frozenset({C}))
                total += cost_i
                plan.extend(plan_i)
                continue

            # Check convergence BEFORE absorber short-circuit: a JOIN pred
            # whose own pred is an absorber may still be part of a convergence
            # group (e.g. DeepNested: F_c1a's pred is conv1_a absorber, but
            # F_c1a is itself the ncu shared with F_b1 — must enter group).
            ncu = self.ga.nearest_convergence_up.get(f_i)
            if ncu is not None:
                gid = id(ncu)
                if gid not in grouped:
                    grouped[gid] = {
                        'G': ncu,
                        'f_list': [],
                        'comp_sum': 0,
                        'ci_set': set(),
                        'comp_plan': [],
                    }
                grouped[gid]['f_list'].append(f_i)
                if f_i is ncu:
                    grouped[gid]['ci_set'].add(C)
                else:
                    comp_cost, comp_plan = self._plan_compensate(f_i, frozenset({C}))
                    grouped[gid]['comp_sum'] += comp_cost
                    grouped[gid]['comp_plan'].extend(comp_plan)
                continue

            # No convergence — fall back to absorber short-circuit or singleton.
            c_i = pred_cs[0]
            if c_i.layer_type in ABSORBER_TYPES:
                plan.append(AbsorbInto(c_i, 'up'))
                continue
            cost_i, plan_i = self.plan_up(f_i, frozenset({C}))
            total += cost_i
            plan.extend(plan_i)

        for grp in grouped.values():
            G = grp['G']
            for f_i in grp['f_list']:
                if f_i is G:
                    continue
                # All succs of G whose downstream subgraph contains f_i must
                # be in ci_set. For single-input chains this is one c; for
                # forks that re-converge below G (e.g. BranchInBranch: F1 →
                # {conv_a, conv_b} → ... → Add_inner → F_add_inner) it's the
                # whole set of c's that lead to f_i.
                grp['ci_set'].update(self._cis_reaching_fi(G, f_i))

        for grp in grouped.values():
            cost_g, plan_g = self.plan_up(grp['G'], frozenset(grp['ci_set']))
            total += grp['comp_sum'] + cost_g
            plan.extend(grp['comp_plan'])
            plan.extend(plan_g)

        return total, plan

    def _cis_reaching_fi(self, G: FeatureNode, f_i: FeatureNode) -> set:
        """Return every direct succ ComputeNode of G whose downstream subgraph
        contains f_i.  Handles forks-that-reconverge below G — unlike the old
        single-input-chain walker, which gave up at multi-input nodes."""
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
# Plan executor
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
    """Plan-based scale absorption.

    Trigger conditions identical to legacy absorb_scale_new_dp:
      - mult_coeff   : always; scale = node.coeff
      - avgpool1d/2d : only when is_adaptive_avgpool or is_big_size
    """
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

        ga = GraphAnalysis(graph.dag)
        builder = PlanBuilder(ga)

        plan: list[Action] = []
        exclude_source = node.layer_type == 'mult_coeff'
        for c in graph.dag.successors(pre_f):
            if c is node and exclude_source:
                continue
            _, sub_plan = builder.plan_down(pre_f, c)
            plan.extend(sub_plan)

        pred_c = preds_of_pre_f[0]
        _, above_plan = builder.plan_above(pred_c)
        plan.extend(above_plan)

        if node.layer_type == 'mult_coeff':
            plan.append(ClearSource(node))

        execute(graph, plan, scale)

    for n in graph.dag.nodes:
        if isinstance(n, ComputeNode) and n.layer_type == 'mult_scalar':
            print('layer=', n.layer_id, 'scale_up=', n.scale_up, n.scale_down)

    _remove_identity_mult_scalars(graph)
