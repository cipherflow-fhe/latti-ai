"""Structural checks for the manually unrolled real CKKS frontend graph."""
import json
import tempfile
import unittest
from pathlib import Path
from training.model_compiler.logistic_training import build


class TrainingGraphTests(unittest.TestCase):
    def test_twenty_round_refresh_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build(directory, features=2, steps=20, checkpoints=True)
            graph = json.loads(Path(directory, 'training_dag.json').read_text())
            self.assertEqual(len(graph['stages']), 20)
            self.assertEqual(manifest['bootstrap_batch_count'], 9)
            self.assertEqual(manifest['bootstrapping_count'], 27)
            self.assertEqual([r['before_step'] for r in manifest['bootstrap_placements']], list(range(3, 21, 2)))
            self.assertTrue(all(r['input_gain'] == 2 for r in manifest['bootstrap_placements']))
            self.assertEqual(manifest['outputs'][-1]['level'], 1)
            self.assertEqual(graph['operation_counts']['bootstrap'], 9)
            self.assertEqual(len(manifest['outputs']), 60)
            self.assertEqual(graph['ct_multiplication_depth'], 80)
            self.assertTrue(all(o['level'] >= 0 for o in manifest['outputs']))
            nodes = {n['id']: n for n in graph['nodes']}
            for first, second in zip(graph['stages'], graph['stages'][1:]):
                seen, pending = set(), [second['logits']]
                while pending:
                    node = pending.pop()
                    for predecessor in nodes[node]['inputs']:
                        if predecessor not in seen:
                            seen.add(predecessor)
                            pending.append(predecessor)
                self.assertTrue(set(first['weights']) <= seen)

    def test_two_round_dependency_and_parallel_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            build(directory, features=2, steps=2)
            graph = json.loads(Path(directory, 'training_dag.json').read_text())
            nodes = {n['id']: n for n in graph['nodes']}
            def ancestors(node_id):
                seen = set()
                pending = [node_id]
                while pending:
                    node = pending.pop()
                    for predecessor in nodes[node]['inputs']:
                        if predecessor not in seen:
                            seen.add(predecessor)
                            pending.append(predecessor)
                return seen
            first, second = graph['stages']
            self.assertTrue(set(first['weights']) <= ancestors(second['logits']))
            self.assertEqual(graph['ct_multiplication_depth'], 8)
            self.assertFalse(any(n.get('operation') == 'bootstrap' for n in nodes.values()))
            self.assertEqual(len(second['weights']), 3)


if __name__ == '__main__':
    unittest.main()
