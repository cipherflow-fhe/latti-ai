"""Compiler DP -> serialized stages -> real Generator DAG integration."""
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import components
import transforms
from graph_partition_dp import compile_model_btp
from pipeline import recompute_final_level
from processor import graph_to_task_config
from training.model_compiler.unittests import test_maxpool_dp as fixture
from inference.model_generator.deploy_cmds import gen_custom_task, _FHE_PARAMS


class MaxPoolCompilerGeneratorTest(unittest.TestCase):
    def test_split_profile_real_frontend_and_manifest(self):
        state = fixture.MaxPoolDPTest()
        state.setUp()
        self.addCleanup(state.tearDown)
        graph = state.graph(alpha=14, channels=2)
        score, graph = compile_model_btp(graph, temperature=0)
        self.assertGreater(score, 0)
        recompute_final_level(graph)
        transforms.insert_drop_level_layers(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            graph.to_json({'param0': components.config.fhe_param}, str(path / 'nn_layers_ct_0.json'))
            graph_to_task_config(graph, directory)
            gen_custom_task(directory, param_name='PN15Default', style='multiplexed', use_gpu=False)
            raw = json.loads((path / 'nn_layers_ct_0.json').read_text())
            dag = json.loads((path / 'mega_ag.json').read_text())
            manifest = json.loads((path / 'maxpool_stage_parameters.json').read_text())
            parameter = json.loads((path / 'fhe_parameter.json').read_text())
            signature = json.loads((path / 'task_signature.json').read_text())
            for mode, loading in [('plaintext_lazy', 'eager'), ('encrypted_offline', 'eager'),
                                  ('encrypted_offline', 'runtime_lazy')]:
                with self.subTest(mode=mode, loading=loading):
                    generated = path / f'{mode}_{loading}'
                    gen_custom_task(directory, param_name='PN15Default', style='multiplexed',
                                    use_gpu=False, parameter_mode=mode, output_dir=str(generated),
                                    deployment_mode='server_provisioned_runner',
                                    parameter_loading_mode=loading)
                    public_dag = json.loads((generated / 'mega_ag.json').read_text())
                    public_signature = json.loads((generated / 'task_signature.json').read_text())
                    self.assertFalse(any(arg['id'].startswith('maxpool_')
                                         for arg in public_signature['online'] + public_signature['offline']))
                    encoders = [op for op in public_dag['compute'].values()
                                if op['type'] == 'encode_pt' and
                                op.get('attributes', {}).get('op_class') == 'PolyMaxPool']
                    self.assertTrue(encoders)
                    for op in encoders:
                        self.assertEqual(op['inputs'], [])
                    sources = set(public_dag['inputs'])
                    for op in public_dag['compute'].values():
                        sources.update(op['outputs'])
                    for op in public_dag['compute'].values():
                        self.assertTrue(set(op['inputs']) <= sources)
        self.assertEqual(len(manifest), 19)
        self.assertTrue(all(layer.get('alpha') == 14 for layer in raw['layer'].values()
                            if layer['type'].startswith('maxpool2d_')))
        arguments = {arg['id']: arg for arg in signature['online']}
        for layer_id, declarations in manifest.items():
            self.assertEqual(bool(declarations), not layer_id.endswith('_finish'))
            for name, desc in declarations.items():
                self.assertEqual(arguments[name]['size'], [desc['count']])
        bootstraps = [op for op in dag['compute'].values() if op['type'] == 'bootstrap']
        expected = 0
        internal = 0
        for layer in raw['layer'].values():
            if layer['type'] != 'bootstrapping':
                continue
            feature = raw['feature'][layer['feature_input'][0]]
            state_info = feature.get('maxpool_state')
            expected += state_info['groups'] * 2 if state_info else 1
            internal += state_info is not None and 'layout' in state_info
        self.assertGreater(internal, 0)
        self.assertEqual(len(bootstraps), expected)
        topology = nx.DiGraph()
        for op in dag['compute'].values():
            self.assertNotEqual(op['type'], 'custom')
            topology.add_edges_from((source, target) for source in op['inputs'] for target in op['outputs'])
        self.assertTrue(nx.is_directed_acyclic_graph(topology))

    def test_retina_spatial_dag_and_runtime_manifest(self):
        state = fixture.MaxPoolDPTest()
        state.setUp()
        self.addCleanup(state.tearDown)
        graph = state.graph()
        score, graph = compile_model_btp(graph, temperature=0)
        self.assertGreater(score, 0)
        recompute_final_level(graph)
        transforms.insert_drop_level_layers(graph)
        # Generic adapters must preserve the physical live-set description.
        drops = []
        for node in graph.dag:
            if getattr(node, 'layer_type', '') == 'drop_level':
                before = next(graph.dag.predecessors(node))
                after = next(graph.dag.successors(node))
                self.assertEqual(before.maxpool_state, after.maxpool_state)
                self.assertIsNot(before.maxpool_state, after.maxpool_state)
                drops.append(node)
        self.assertTrue(drops)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            graph.to_json({'param0': components.config.fhe_param}, str(path / 'nn_layers_ct_0.json'))
            graph_to_task_config(graph, directory)
            # Real frontend arithmetic, Bootstrap construction and serialization;
            # no mocked stage operators or process_custom_task.
            gen_custom_task(directory, param_name='PN15Default', style='multiplexed', use_gpu=False)
            raw = json.loads((path / 'nn_layers_ct_0.json').read_text())
            dag = json.loads((path / 'mega_ag.json').read_text())
            signature = json.loads((path / 'task_signature.json').read_text())
            manifest = json.loads((path / 'maxpool_stage_parameters.json').read_text())
            parameter = json.loads((path / 'fhe_parameter.json').read_text())
        arguments = {arg['id']: arg for arg in signature['online']}
        self.assertEqual(arguments['x']['size'], [256])
        self.assertEqual(arguments['y']['size'], [32])
        self.assertEqual(arguments['y']['level'], 0)
        self.assertEqual(len(manifest), 7)
        declared = {}
        for layer_id, descriptors in manifest.items():
            self.assertTrue(raw['layer'][layer_id]['type'].startswith('maxpool2d_'))
            for name, desc in descriptors.items():
                self.assertNotIn(name, declared)
                declared[name] = desc
                self.assertEqual(arguments[name]['type'], 'pt')
                self.assertEqual(arguments[name]['size'], [desc['count']])
                self.assertGreaterEqual(desc['level'], 0)
                self.assertLessEqual(desc['level'], components.config.fhe_param.max_level)
        self.assertEqual(set(declared), {name for name in arguments if name.startswith('maxpool_')})
        data = dag['data']
        topology = nx.DiGraph()
        supplied = set(dag['inputs'])
        produced = set()
        bootstraps = []
        for op in dag['compute'].values():
            self.assertNotEqual(op['type'], 'custom')
            for index in op['outputs']:
                self.assertNotIn(index, produced)
                produced.add(index)
                for source in op['inputs']:
                    topology.add_edge(source, index)
            if op['type'] == 'bootstrap':
                bootstraps.append(op)
                self.assertEqual(data[str(op['inputs'][0])]['level'], 0)
                self.assertEqual(data[str(op['outputs'][0])]['level'], components.config.fhe_param.max_level)
        self.assertTrue(nx.is_directed_acyclic_graph(topology))
        available = supplied | produced
        for op in dag['compute'].values():
            self.assertTrue(set(op['inputs']) <= available)
        self.assertTrue(set(dag['outputs']) <= produced)
        expected_refreshes = sum(
            raw['feature'][layer['feature_input'][0]]['maxpool_state']['groups'] * 64
            for layer in raw['layer'].values() if layer['type'] == 'bootstrapping')
        self.assertEqual(len(bootstraps), expected_refreshes)
        self.assertGreater(expected_refreshes, 0)
        param = _FHE_PARAMS['PN15Default']
        self.assertEqual(1 << parameter['log_n'], param.poly_modulus_degree)
        self.assertEqual(parameter['q'], param.q)
        self.assertEqual(parameter['p'], param.p)


if __name__ == '__main__':
    unittest.main()
