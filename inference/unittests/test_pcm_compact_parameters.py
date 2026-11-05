# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np

from inference.model_generator.deploy_cmds import gen_custom_task
from inference.model_generator.parameter_sources import (
    pcm_parameter_sources, public_source, resolve_pcm_parameter_source, secret_source,
)
from inference.unittests.test_server_provisioned_runner_generator import _data_with_missing_source


class PCMCompactParametersTest(unittest.TestCase):
    def test_pdm_encrypted_parameter_tasks_are_rejected_before_export(self):
        for loading_mode in ['eager', 'runtime_lazy']:
            with self.subTest(loading_mode=loading_mode), tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                root = Path(tmp)
                (root/'task_config.json').write_text(json.dumps({'mat_pack_style': 'par_diagonal_pack'}))
                (root/'nn_layers_ct_0.json').write_text(json.dumps({'layer': {}}))
                with self.assertRaisesRegex(ValueError, 'PDM encrypted parameters are unsupported'):
                    gen_custom_task(str(root), param_name='PN13Default',
                                    parameter_mode='encrypted_offline', input_mode='plaintext',
                                    deployment_mode='server_provisioned_runner',
                                    parameter_loading_mode=loading_mode, output_dir=str(root/'runner'))
                self.assertFalse((root/'runner').exists())

    def test_affine_provenance_hashes_gamma_and_beta_separately(self):
        from training.nn_tools.feature_mat_h5 import _audit_pcm_parameter_sources
        gamma, beta = np.array([1.0, 2.0]), np.array([0.1, 0.2])
        layer = dict(type='pcmaffine', weight_path='gamma', bias_path='beta')
        audit = _audit_pcm_parameter_sources({'affine': layer}, {'gamma': gamma, 'beta': beta})
        hashes = {v['parameter']: v['value_sha256'] for v in audit}
        self.assertEqual(hashes['pt:pt_idx=0'], hashlib.sha256(gamma.astype('<f8').tobytes()).hexdigest())
        self.assertEqual(hashes['pt:pt_idx=1'], hashlib.sha256(beta.astype('<f8').tobytes()).hexdigest())

    def test_mixed_stats_and_missing_provenance(self):
        layer = {'parameter_sources': {'pt:pt_idx=2': secret_source('calibration', 'ln.max_var')}}
        def resolve(idx):
            return resolve_pcm_parameter_source(layer, dict(op_class='ParBlockColMajorLNStats', type='pt', pt_idx=idx))
        self.assertEqual(resolve(0)['visibility'], 'public')
        self.assertEqual(resolve(1)['visibility'], 'public')
        self.assertEqual(resolve(2)['visibility'], 'secret')
        self.assertTrue(resolve(3)['missing_provenance'])

    def test_public_recipe_integrity_and_private_h5_defaults(self):
        attrs = dict(op_class='ParBlockColMajorPoly', type='stockmeyer_coeff_pt', coeff_idx=2)
        layer = dict(type='pcmgeneralpoly', weight_path='private.weight')
        layer['parameter_sources'] = pcm_parameter_sources(layer)
        self.assertEqual(resolve_pcm_parameter_source(layer, attrs)['visibility'], 'secret')
        layer['parameter_sources']['stockmeyer_coeff_pt'] = public_source(
            'approximation_config', {'coefficients': [1.0, 2.0, 3.0]})
        self.assertEqual(resolve_pcm_parameter_source(layer, attrs)['visibility'], 'public')
        layer['parameter_sources']['stockmeyer_coeff_pt']['public_recipe']['content']['coefficients'][0] = 4.0
        with self.assertRaisesRegex(ValueError, 'hash/version'):
            resolve_pcm_parameter_source(layer, attrs)
        layer['parameter_sources']['stockmeyer_coeff_pt'] = dict(visibility='public')
        with self.assertRaisesRegex(ValueError, 'lacks recipe'):
            resolve_pcm_parameter_source(layer, attrs)

    def test_compact_graph_stores_each_seed_once_and_gates_recovery(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = Path(tmp)
            feature = dict(dim=2, channel=1, scale=1.0, ckks_scale=1, shape=[19, 48], skip=[1, 1],
                           ckks_parameter_id='param0', level=3, depth=-1, pack_num=1,
                           invalid_fill=[1, 1], is_transposed=False, data_type='feature_mat', head_shape=[19, 16])
            output = dict(feature, level=1)
            linear = dict(type='parcpmm', feature_input=['input'], feature_output=['output'],
                          weight_shape=[48, 48], weight_path='w', bias_path='b')
            linear['parameter_sources'] = pcm_parameter_sources(linear)
            graph = dict(feature={'input': feature, 'output': output}, layer={'linear': linear},
                         input_feature=['input'], output_feature=['output'])
            config = dict(task_type='fhe', task_num=1, pack_style='multiplexed',
                          mat_pack_style='par_block_col_major', n_heads=3, head_dim=16, matmul_block_size=16,
                          block_shape=[64, 64], task_input_id=['input'], task_output_id=['output'],
                          task_input_param={'input': feature}, task_output_param={'output': output}, use_btp=False)
            (root/'nn_layers_ct_0.json').write_text(json.dumps(graph))
            (root/'task_config.json').write_text(json.dumps(config))
            gen_custom_task(str(root), param_name='PN13Default', style='multiplexed',
                            parameter_mode='encrypted_offline', input_mode='plaintext',
                            deployment_mode='server_provisioned_runner', parameter_loading_mode='runtime_lazy',
                            output_dir=str(root/'runner'))
            signature = json.loads((root/'runner/task_signature.json').read_text())
            mega = json.loads((root/'runner/mega_ag.json').read_text())
            params = signature['encrypted_parameters']
            # T=4096, padded heads=4, C=1024, rho=4: ceil(3/4)=1 seed.
            self.assertEqual(Counter(p['generator']['type'] for p in params),
                             {'zero': 1, 'row_chunk_seed': 1, 'bias_pt': 2})
            seed = next(p for p in params if p['generator']['type'] == 'row_chunk_seed')
            self.assertEqual(seed['level'], 4)
            self.assertEqual(seed['use_site_level'], 3)
            self.assertEqual(seed['layout']['seed_count'], 1)
            masks = [c for c in mega['compute'].values()
                     if c['type'] == 'encode_pt' and c['attributes']['type'] == 'row_chunk_selection']
            self.assertEqual(len(masks), 48)
            self.assertTrue(all(c['inputs'] for c in masks))
            self.assertEqual(_data_with_missing_source(mega), [])

    def test_structural_mac_parameters_have_producers(self):
        for kind, cost in [('partranspose', 1), ('parccmm', 3)]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                root = Path(tmp)
                feature = dict(dim=2, channel=1, scale=1.0, ckks_scale=1, shape=[16, 32],
                               skip=[1, 1], ckks_parameter_id='param0', level=4, depth=-1,
                               pack_num=1, invalid_fill=[1, 1], is_transposed=False,
                               data_type='feature_mat', head_shape=[16, 16])
                inputs = ['a'] if kind == 'partranspose' else ['a', 'b']
                features = {name: dict(feature) for name in inputs}
                features['output'] = dict(feature, level=4-cost)
                graph = dict(feature=features, input_feature=inputs, output_feature=['output'],
                             layer={'op': dict(type=kind, feature_input=inputs, feature_output=['output'])})
                config = dict(task_type='fhe', task_num=1, pack_style='multiplexed',
                              mat_pack_style='par_block_col_major', n_heads=2, head_dim=16,
                              matmul_block_size=16, block_shape=[64, 64], task_input_id=inputs,
                              task_output_id=['output'], task_input_param={n: features[n] for n in inputs},
                              task_output_param={'output': features['output']}, use_btp=False)
                (root/'nn_layers_ct_0.json').write_text(json.dumps(graph))
                (root/'task_config.json').write_text(json.dumps(config))
                gen_custom_task(str(root), param_name='PN13Default', style='multiplexed',
                                parameter_mode='encrypted_offline', input_mode='plaintext',
                                deployment_mode='server_provisioned_runner', parameter_loading_mode='runtime_lazy',
                                output_dir=str(root/'runner'))
                mega = json.loads((root/'runner/mega_ag.json').read_text())
                self.assertEqual(_data_with_missing_source(mega), [])
                masks = [c for c in mega['compute'].values() if c['type'] == 'encode_pt']
                self.assertTrue(masks)
                self.assertTrue(all(c['inputs'] for c in masks))


if __name__ == '__main__':
    unittest.main()
