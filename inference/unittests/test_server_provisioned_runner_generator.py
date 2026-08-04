# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from inference.model_generator.deploy_cmds import gen_custom_task


def _feature(level: int):
    return {
        'dim': 0,
        'channel': 1,
        'scale': 1.0,
        'ckks_scale': 1,
        'skip': 4096,
        'ckks_parameter_id': 'param0',
        'level': level,
        'depth': -1,
        'pack_num': 1,
    }


class TestServerProvisionedRunnerGenerator(unittest.TestCase):
    def test_two_dense_layers_use_plaintext_input_and_offline_ct_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_dir = root / 'server'
            runner_dir = root / 'runner'
            server_dir.mkdir()

            graph = {
                'feature': {
                    'input': _feature(2),
                    'hidden': _feature(1),
                    'output': _feature(0),
                },
                'layer': {
                    'fc1': {
                        'type': 'fc0',
                        'feature_input': ['input'],
                        'feature_output': ['hidden'],
                        'channel_input': 1,
                        'channel_output': 1,
                    },
                    'fc2': {
                        'type': 'fc0',
                        'feature_input': ['hidden'],
                        'feature_output': ['output'],
                        'channel_input': 1,
                        'channel_output': 1,
                    },
                },
                'input_feature': ['input'],
                'output_feature': ['output'],
            }
            task_config = {
                'task_type': 'fhe',
                'task_num': 1,
                'server_start_id': 0,
                'server_end_id': 0,
                'block_shape': [1, 1],
                'pack_style': 'ordinary',
                'task_input_id': ['input'],
                'task_output_id': ['output'],
                'task_input_param': {'input': graph['feature']['input']},
                'task_output_param': {'output': graph['feature']['output']},
                'server_task': {'nn_layers_ct_0': {'enable_fpga': True}},
                'use_btp': False,
            }

            with open(server_dir / 'nn_layers_ct_0.json', 'w', encoding='utf-8') as f:
                json.dump(graph, f)
            with open(server_dir / 'task_config.json', 'w', encoding='utf-8') as f:
                json.dump(task_config, f)

            gen_custom_task(
                str(server_dir),
                param_name='PN13QP218',
                style='ordinary',
                parameter_mode='encrypted_offline',
                input_mode='plaintext',
                output_dir=str(runner_dir),
            )

            with open(runner_dir / 'task_signature.json', 'r', encoding='utf-8') as f:
                signature = json.load(f)
            with open(runner_dir / 'mega_ag.json', 'r', encoding='utf-8') as f:
                mega_ag = json.load(f)

            online = signature['online']
            offline = signature['offline']
            self.assertEqual(online[0]['id'], 'input')
            self.assertEqual(online[0]['type'], 'pt_ringt')
            self.assertEqual(online[0]['phase'], 'in')
            self.assertEqual(online[-1]['id'], 'output')
            self.assertEqual(online[-1]['type'], 'ct')
            self.assertEqual(online[-1]['phase'], 'out')

            offline_by_id = {arg['id']: arg for arg in offline}
            self.assertEqual(offline_by_id['densew_fc1']['type'], 'ct')
            self.assertEqual(offline_by_id['densew_fc1']['level'], 2)
            self.assertEqual(offline_by_id['denseb_fc1']['level'], 1)
            self.assertEqual(offline_by_id['densew_fc2']['type'], 'ct')
            self.assertEqual(offline_by_id['densew_fc2']['level'], 1)
            self.assertEqual(offline_by_id['denseb_fc2']['level'], 0)
            self.assertGreaterEqual(len(mega_ag['offline_inputs']), 4)

            compute_types = {node['type'] for node in mega_ag['compute'].values()}
            self.assertIn('mult', compute_types)
            self.assertIn('relin', compute_types)
            self.assertIn('rescale', compute_types)

    def test_mult_scalar_uses_offline_ct_param(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_dir = root / 'server'
            runner_dir = root / 'runner'
            server_dir.mkdir()

            graph = {
                'feature': {
                    'input': _feature(2),
                    'hidden': _feature(1),
                    'output': _feature(0),
                },
                'layer': {
                    'fc1': {
                        'type': 'fc0',
                        'feature_input': ['input'],
                        'feature_output': ['hidden'],
                        'channel_input': 1,
                        'channel_output': 1,
                    },
                    'scale': {
                        'type': 'mult_scalar',
                        'feature_input': ['hidden'],
                        'feature_output': ['output'],
                        'channel_input': 1,
                        'channel_output': 1,
                    },
                },
                'input_feature': ['input'],
                'output_feature': ['output'],
            }
            task_config = {
                'task_type': 'fhe',
                'task_num': 1,
                'server_start_id': 0,
                'server_end_id': 0,
                'block_shape': [1, 1],
                'pack_style': 'ordinary',
                'task_input_id': ['input'],
                'task_output_id': ['output'],
                'task_input_param': {'input': graph['feature']['input']},
                'task_output_param': {'output': graph['feature']['output']},
                'server_task': {'nn_layers_ct_0': {'enable_fpga': True}},
                'use_btp': False,
            }

            with open(server_dir / 'nn_layers_ct_0.json', 'w', encoding='utf-8') as f:
                json.dump(graph, f)
            with open(server_dir / 'task_config.json', 'w', encoding='utf-8') as f:
                json.dump(task_config, f)

            gen_custom_task(
                str(server_dir),
                param_name='PN13QP218',
                style='ordinary',
                parameter_mode='encrypted_offline',
                input_mode='plaintext',
                output_dir=str(runner_dir),
            )

            with open(runner_dir / 'task_signature.json', 'r', encoding='utf-8') as f:
                signature = json.load(f)
            with open(runner_dir / 'mega_ag.json', 'r', encoding='utf-8') as f:
                mega_ag = json.load(f)

            offline_by_id = {arg['id']: arg for arg in signature['offline']}
            self.assertEqual(offline_by_id['mult_scalar_scale']['type'], 'ct')
            self.assertEqual(offline_by_id['mult_scalar_scale']['phase'], 'offline')
            self.assertEqual(offline_by_id['mult_scalar_scale']['level'], 1)
            self.assertEqual(offline_by_id['mult_scalar_scale']['size'], [1])

            compute_types = {node['type'] for node in mega_ag['compute'].values()}
            self.assertIn('mult', compute_types)
            self.assertIn('relin', compute_types)
            self.assertIn('rescale', compute_types)


if __name__ == '__main__':
    unittest.main()
