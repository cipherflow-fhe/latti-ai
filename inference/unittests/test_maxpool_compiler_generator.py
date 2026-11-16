"""Real ONNX MaxPool detection -> preparation -> DP -> CPU/GPU instruction generation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
from onnx import helper
import components
import transforms
from pipeline import prepare_graph, recompute_final_level
from graph_partition_dp import compile_model_btp
from processor import graph_to_task_config
from training.model_export.operations import FeatureNode as ExportFeature
from training.model_export.operations.MaxPool import MaxPoolComputeNode
from inference.model_generator.deploy_cmds import gen_custom_task
from training.model_compiler.unittests import test_maxpool_dp as fixture


class MaxPoolCompilerGeneratorTest(unittest.TestCase):
    def test_onnx_auto_detection_and_real_frontend(self):
        state = fixture.MaxPoolDPTest()
        state.setUp()
        self.addCleanup(state.tearDown)
        for alpha in (7, 14):
            with self.subTest(alpha=alpha), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                features = {name: ExportFeature(name, 2, 2, 1., [1, 1], 'param0', shape)
                            for name, shape in [('x', [256, 256]), ('y', [128, 128])]}
                onnx_node = helper.make_node('MaxPool', ['x'], ['y'], name='retina/maxpool',
                                             kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
                exported = MaxPoolComputeNode.from_onnx_node(onnx_node, features)
                layer = exported.to_json()
                self.assertEqual(layer['type'], 'maxpool2d')
                # Calibration is supplied separately; layer recognition and
                # lowering must not require manually constructing stage nodes.
                layer.update(running_max=2.5, alpha=alpha, upper_bound=.5, B=1.3)
                raw_path = path / 'raw.json'
                raw_path.write_text(json.dumps({'layer': {exported.layer_id: layer},
                    'feature': {k: v.to_json() for k, v in features.items()}}))
                raw = components.LayerAbstractGraph.from_json(str(raw_path))
                graph = prepare_graph(raw)
                output = next(n for n in graph.dag if getattr(n, 'node_id', None) == 'y')
                self.assertEqual(graph.dag.nodes[output]['pack_num'], 2)
                pack_rotate = next(n for n in graph.dag if getattr(n, 'layer_id', '').endswith('_pack_rotate'))
                self.assertEqual(pack_rotate.maxpool_primitive['steps'], [0, -16384])
                primitives = [n for n in graph.dag if hasattr(n, 'maxpool_primitive')]
                self.assertTrue(primitives)
                self.assertFalse(any(getattr(n, 'layer_type', '') == 'maxpool2d' for n in graph.dag))
                self.assertEqual(sum(n.maxpool_primitive['op'] == 'mult' for n in primitives), 8)
                for node in primitives:
                    if node.maxpool_primitive['op'] in ('mult', 'add', 'sub'):
                        indices = [graph.dag.edges[p, node]['input_index'] for p in graph.dag.predecessors(node)]
                        self.assertEqual(sorted(indices), [0, 1])
                score, graph = compile_model_btp(graph, temperature=0)
                self.assertGreater(score, 0)
                recompute_final_level(graph)
                transforms.insert_drop_level_layers(graph)
                for node in graph.dag:
                    if getattr(node, 'layer_type', '') == 'drop_level':
                        before = next(graph.dag.predecessors(node))
                        after = next(graph.dag.successors(node))
                        self.assertEqual(getattr(before, 'maxpool_state', None),
                                         getattr(after, 'maxpool_state', None))
                graph.to_json({'param0': components.config.fhe_param}, str(path/'nn_layers_ct_0.json'))
                graph_to_task_config(graph, directory)
                raw = json.loads((path/'nn_layers_ct_0.json').read_text())
                restored = components.LayerAbstractGraph.from_json(str(path/'nn_layers_ct_0.json'))
                self.assertEqual(sum(hasattr(n, 'maxpool_primitive') for n in restored.dag), len(primitives))
                for gpu, mode in [(False, 'plaintext_eager'), (True, 'plaintext_eager'),
                                  (True, 'plaintext_lazy'), (True, 'encrypted_offline')]:
                    with self.subTest(gpu=gpu, mode=mode):
                        target = path / f'{gpu}_{mode}'
                        gen_custom_task(directory, param_name='PN15Default', style='multiplexed',
                                        use_gpu=gpu, parameter_mode=mode, output_dir=str(target))
                        dag = json.loads((target/'mega_ag.json').read_text())
                        signature = json.loads((target/'task_signature.json').read_text())
                        args = {a['id']: a for a in signature['online']}
                        self.assertEqual(args['x']['size'], [8])
                        self.assertEqual(args['y']['size'], [1])
                        self.assertFalse(any(a['id'].startswith('maxpool_')
                                             for a in signature['online'] + signature.get('offline', [])))
                        topology = nx.DiGraph()
                        produced = set()
                        for op in dag['compute'].values():
                            self.assertNotEqual(op['type'], 'custom')
                            self.assertFalse(produced.intersection(op['outputs']))
                            produced.update(op['outputs'])
                            topology.add_edges_from((i, o) for i in op['inputs'] for o in op['outputs'])
                        self.assertTrue(nx.is_directed_acyclic_graph(topology))
                        available = set(dag['inputs']) | produced
                        self.assertTrue(all(set(op['inputs']) <= available for op in dag['compute'].values()))
                        self.assertTrue(set(dag['outputs']) <= produced)
                        expected_btp = 0
                        for layer in raw['layer'].values():
                            if layer['type'] == 'bootstrapping':
                                feature = raw['feature'][layer['feature_input'][0]]
                                live = feature.get('maxpool_state')
                                expected_btp += live['groups'] * live['channels'] if live else 1
                        self.assertEqual(sum(op['type'] == 'bootstrap' for op in dag['compute'].values()), expected_btp)
                        self.assertTrue(any(op['type'] == 'encode_pt' for op in dag['compute'].values()))


if __name__ == '__main__':
    unittest.main()
