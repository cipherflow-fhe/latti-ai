"""Check cross-batch values and final backend-to-handle outputs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lattisense'))
from frontend import custom_task as task
from frontend import types


class StreamingLowLevelTest(unittest.TestCase):
    def test_shared_key_bridges_publish_the_final_level(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                task.set_fhe_param(task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant))
                task.g_dag.start_stream(directory, task.Processor.GPU)
                x = task.CkksCiphertextNode(3, id='x')
                low = task.drop_level(x, 2)
                low = task.rotate_cols(low, [1])[0]
                task.g_dag.flush()
                high = task.rotate_cols(x, [1])[0]
                high = task.drop_level(high, 2)
                y = task.add(low, high)
                task.process_custom_task(
                    [task.Argument('x', [x])], [task.Argument('y', [y])], directory, task.Processor.GPU
                )
                graph = json.loads(Path(directory, 'compiled_mega_ag.json').read_text())
                keys = [value for value in graph['data'].values() if value['type'] == 'glk']
                self.assertTrue(keys)
                self.assertTrue(all(value['level'] == 3 for value in keys))
            finally:
                task.g_dag.close_stream()
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                task.g_param = None
                types.reset_node_state()

    def test_public_mask_encoding_waits_for_its_first_ciphertext_use(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                task.set_fhe_param(task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant))
                task.g_dag.start_stream(directory, task.Processor.GPU)
                x = task.CkksCiphertextNode(3, id='x')
                mask = task.CkksPlaintextNode('mask', scale=float(task.g_param.q[3]))
                task.custom_compute([], mask, 'encode_pt', {'kind': 'public'})
                self.assertEqual(len(task.g_dag), 0)
                y = task.mult(x, mask)
                task.process_custom_task(
                    [task.Argument('x', [x])], [task.Argument('y', [y])], directory, task.Processor.GPU
                )
                graph = json.loads(Path(directory, 'mega_ag.json').read_text())
                encode = [op for op in graph['compute'].values() if op['type'] == 'encode_pt']
                self.assertEqual(len(encode), 1)
                self.assertEqual(encode[0]['inputs'], ['x'])
                compiled = json.loads(Path(directory, 'compiled_mega_ag.json').read_text())
                self.assertTrue(all(op['priority'] > 0 for op in compiled['compute'].values()))
            finally:
                task.g_dag.close_stream()
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                task.g_param = None
                types.reset_node_state()

    def test_reused_intermediate_survives_batches_and_outputs_have_one_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            try:
                task.set_fhe_param(task.CkksParam.create_default_param(15, task.RingType.ConjugateInvariant))
                task.g_dag.start_stream(directory, task.Processor.GPU)
                pt = task.CkksPlaintextNode('pt', level=3, is_ringt=False)
                x = task.CkksCiphertextNode(3, id='x')
                with patch.object(type(task.g_dag), 'batch_node_limit', 64):
                    saved = task.rotate_cols(task.add(x, 2), [1])[0]
                    y = saved
                    for index in range(100):
                        y = task.rotate_cols(task.add(y, 1), [1])[0] if index % 20 == 0 else task.add(y, 1)
                    y = task.mult(task.add(y, saved), pt)
                    task.process_custom_task(
                        [task.Argument('x', [x]), task.Argument('pt', [pt])],
                        [task.Argument('y', [y])],
                        directory,
                        task.Processor.GPU,
                    )
                graph = json.loads(Path(directory, 'compiled_mega_ag.json').read_text())
                producers = {}
                for name, compute in graph['compute'].items():
                    for datum in compute['outputs']:
                        self.assertNotIn(datum, producers, datum)
                        producers[datum] = name
                dag = nx.DiGraph()
                dag.add_nodes_from(graph['compute'])
                for name, compute in graph['compute'].items():
                    for datum in compute['inputs']:
                        self.assertTrue(datum in graph['inputs'] or datum in producers, datum)
                        if datum in producers:
                            dag.add_edge(producers[datum], name)
                key_ids = {name for name in graph['inputs'] if graph['data'][name]['type'] == 'glk'}
                exports = [
                    op['ops'][0]
                    for op in graph['compute'].values()
                    if op['ops'][0]['type'] == 'export_to_abi' and op['ops'][0]['inputs'][0] in key_ids
                ]
                self.assertEqual(len(exports), len(key_ids))
                values = {name: 0 for name in graph['inputs']}
                values['x'] = 3
                values['pt'] = 2
                for name in nx.topological_sort(dag):
                    compute = graph['compute'][name]
                    self.assertEqual(len(compute['ops']), 1)
                    op = compute['ops'][0]
                    args = [values[d] for d in op['inputs']]
                    if op['type'] == 'add':
                        result = sum(args) + op.get('scalar', 0)
                    elif op['type'] == 'mult':
                        self.assertEqual([graph['data'][d]['type'] for d in op['inputs']], ['ct', 'pt'])
                        result = args[0] * args[1]
                    else:
                        self.assertIn(
                            op['type'],
                            ('export_to_abi', 'load_to_backend', 'rotate_col', 'store_from_backend', 'import_from_abi'),
                        )
                        result = args[0]
                    values[op['outputs'][0]] = result
                self.assertEqual(values[graph['outputs'][0]], 220)
                self.assertEqual(len(list(Path(directory).glob('.stream-*'))), 0)
            finally:
                task.g_dag.close_stream()
                task.g_dag.clear()
                task.g_evk_node_dict.clear()
                task.g_param = None
                types.reset_node_state()
        self.assertIsNone(types.stream_node_counter)
        self.assertIsNone(task.g_dag.stream)


if __name__ == '__main__':
    unittest.main()
