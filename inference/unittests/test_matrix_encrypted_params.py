# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Matrix parameter graphs must retain packing while loading weights lazily."""
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter

from inference.model_generator.deploy_cmds import gen_custom_task
from inference.model_generator.layers.matrix_param_ops import _resolve, custom_compute
from inference.lattisense.frontend.custom_task import CkksCiphertextNode, CkksPlaintextRingtNode, CustomDataNode
from inference.unittests.test_server_provisioned_runner_generator import _data_with_missing_source


class MatrixEncryptedParameterTest(unittest.TestCase):
    def test_deferred_parameters_are_cached_at_each_consuming_level(self):
        source = CustomDataNode(type='matrix_source', id='source')
        calls = []
        def build(attrs, name, level):
            calls.append((attrs, level))
            return CkksCiphertextNode(f'{name}_{level}', level=level)
        source.matrix_parameter_builder = build
        parameter = CkksPlaintextRingtNode('weight')
        custom_compute(inputs=[source], output=parameter, type='encode_pt',
                       attributes={'op_class': 'ParLowerDiagPCMM', 'type': 'pt_A'})
        at3 = CkksCiphertextNode('x3', level=3)
        at2 = CkksCiphertextNode('x2', level=2)
        self.assertIs(_resolve(parameter, at3), _resolve(parameter, at3))
        self.assertEqual(_resolve(parameter, at2).level, 2)
        self.assertEqual([level for _, level in calls], [3, 2])

    def test_runtime_lazy_pcmm_has_encrypted_weights_and_local_public_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature = dict(dim=2, channel=1, scale=1.0, ckks_scale=1, shape=[32, 128], skip=[1, 1],
                           ckks_parameter_id='param0', level=3, depth=-1, pack_num=1,
                           invalid_fill=[1, 1], is_transposed=True, data_type='feature_mat', head_shape=[16, 128])
            output = dict(feature, level=1)
            graph = dict(feature={'input': feature, 'output': output},
                         layer={'linear': dict(type='pdmpcmm', feature_input=['input'], feature_output=['output'],
                                              weight_shape=[32, 32], weight_path='w', bias_path='b')},
                         input_feature=['input'], output_feature=['output'])
            config = dict(task_type='fhe', task_num=1, pack_style='multiplexed',
                          mat_pack_style='par_diagonal_pack', n_heads=2, head_dim=16, matmul_block_size=16,
                          block_shape=[64, 64], task_input_id=['input'], task_output_id=['output'],
                          task_input_param={'input': feature}, task_output_param={'output': output}, use_btp=False)
            (root/'nn_layers_ct_0.json').write_text(json.dumps(graph))
            (root/'task_config.json').write_text(json.dumps(config))
            gen_custom_task(str(root), param_name='PN13QP218', style='multiplexed',
                            parameter_mode='encrypted_offline', input_mode='plaintext',
                            deployment_mode='server_provisioned_runner', parameter_loading_mode='runtime_lazy',
                            output_dir=str(root/'runner'))
            signature = json.loads((root/'runner/task_signature.json').read_text())
            mega = json.loads((root/'runner/mega_ag.json').read_text())
            parameters = signature['encrypted_parameters']
            self.assertTrue(all(p['type'] == 'ct' for p in parameters))
            self.assertEqual(Counter(p['generator']['type'] for p in parameters),
                             {'zero': 1, 'pt_A': 32, 'bias_pt': 1})
            self.assertTrue(all(p['level'] == (1 if p['generator']['type'] == 'bias_pt' else 3)
                                for p in parameters))
            masks = [c for c in mega['compute'].values() if c['type'] == 'encode_pt']
            self.assertTrue(masks)
            self.assertTrue(all(m['inputs'] == [] and m['attributes']['type'] == 'mask_wrap_pt' for m in masks))
            self.assertEqual(_data_with_missing_source(mega), [])
            self.assertEqual([p['id'] for p in signature['offline']], ['encrypted_parameter_store'])
            self.assertTrue(all(p['type'] != 'custom' for p in signature['online']))
            loads = [c for c in mega['compute'].values() if c['type'] == 'load_encrypted_param_ct']
            self.assertEqual(len(loads), len(parameters))
            # Every CT weight product has the relinearization required by CKKS.
            self.assertGreater(sum(c['type'] == 'relin' for c in mega['compute'].values()), 0)


if __name__ == '__main__':
    unittest.main()
