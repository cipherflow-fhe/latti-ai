"""Inspectable primitive DAG export before backend compound-node formation.

This is a frontend graph audit, not a LayerAbstractGraph bootstrap scheduler.
Depth follows dependencies: parallel multiplications do not add their depths.
"""
import json
from collections import Counter
import networkx as nx


def export_primitive_graph(path, stages):
    from frontend import custom_task
    from frontend.types import FheComputeNode, CiphertextNode
    dag = custom_task.g_dag
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError('Training graph must be acyclic')
    depths = {}
    rows = []
    counts = Counter()
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        depth = max((depths[p] for p in predecessors), default=0)
        if isinstance(node, FheComputeNode):
            op = node.type.value
            ciphertext_inputs = sum(isinstance(p, CiphertextNode) for p in predecessors)
            # A square has one unique graph predecessor but degree-2 output.
            ct_mult = op == 'mult' and any(
                getattr(getattr(out, 'metadata', None), 'degree', 0) == 2
                for out in dag.successors(node))
            depth += int(ct_mult)
            counts[op] += 1
            row = dict(id=node.id, kind='compute', operation=op,
                       ciphertext_inputs=ciphertext_inputs,
                       ciphertext_multiplication=ct_mult)
        else:
            row = dict(id=node.id, kind='data', data_type=node.type.value)
            metadata = getattr(node, 'metadata', None)
            if metadata:
                row.update(level=metadata.level, scale=metadata.scale)
        depths[node] = depth
        row.update(inputs=[p.id for p in predecessors], ct_multiplication_depth=depth)
        rows.append(row)
    report = dict(schema_version=1, stages=stages, nodes=rows,
                  operation_counts=dict(counts),
                  ct_multiplication_depth=max(depths.values(), default=0),
                  bootstrap_scheduling='not yet integrated with LayerAbstractGraph')
    path.write_text(json.dumps(report, indent=2))
    return report
