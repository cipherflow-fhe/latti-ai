#!/usr/bin/env python3
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

"""Generate mega_ag instructions for a task.

Usage:
    python gen_mega_ag.py --task-dir ./task
    python gen_mega_ag.py  # defaults to ./task
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'inference' / 'lattisense'))
from inference.lattisense.frontend.custom_task import *  # noqa: E402

# from inference.model_generator.deploy_cmds import gen_custom_task  # noqa: E402
from inference.model_generator.deploy_cmds import gen_custom_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Generate mega_ag instructions for a task.')
    parser.add_argument(
        '--task-dir',
        type=str,
        required=True,
        help='Path to the task directory',
    )
    parser.add_argument(
        '--deployment-mode',
        choices=['client_encrypted_input', 'server_provisioned_runner'],
        default='client_encrypted_input',
        help='Deployment mode for generated MegaAG. Default preserves the existing client-encrypted-input flow.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory for mega_ag.json/task_signature.json.',
    )
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    ergs_path = os.path.join(task_dir, 'server')
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            os.path.join(task_dir, 'runner') if args.deployment_mode == 'server_provisioned_runner' else ergs_path
        )
    output_dir = os.path.abspath(output_dir)

    # Read param_name from ckks_parameter.json to identify the FHE parameter set.
    ckks_param_path_candidates = [
        os.path.join(output_dir, 'ckks_parameter.json'),
        os.path.join(task_dir, 'client', 'ckks_parameter.json'),
        os.path.join(task_dir, 'server', 'ckks_parameter.json'),
    ]
    ckks_param_path = next((path for path in ckks_param_path_candidates if os.path.exists(path)), None)
    if ckks_param_path is None:
        raise FileNotFoundError('Cannot find ckks_parameter.json in runner, client, or server task directories')
    with open(ckks_param_path, 'r', encoding='utf-8') as f:
        ckks_config = json.load(f)
    first_param = next(iter(ckks_config.values()))
    param_name = first_param.get('param_name', '')

    if not param_name:
        n = first_param['poly_modulus_degree']
        n_mult_level = first_param.get('n_mult_level', 0)
        if n == 65536 and n_mult_level <= 9:
            param_name = 'N16QP1546H192H32'
        else:
            _N_TO_PARAM = {8192: 'PN13QP218', 16384: 'PN14QP438', 32768: 'PN15QP880', 65536: 'PN16QP1761'}
            param_name = _N_TO_PARAM.get(n, '')
        if not param_name:
            raise ValueError(f'Cannot determine param_name for poly_modulus_degree={n}')

    # Read pack_style from task_config.json.
    task_config_path_candidates = [
        os.path.join(output_dir, 'task_config.json'),
        os.path.join(task_dir, 'client', 'task_config.json'),
        os.path.join(task_dir, 'server', 'task_config.json'),
    ]
    task_config_path = next((path for path in task_config_path_candidates if os.path.exists(path)), None)
    if task_config_path is None:
        raise FileNotFoundError('Cannot find task_config.json in runner, client, or server task directories')
    with open(task_config_path, 'r', encoding='utf-8') as f:
        task_config = json.load(f)
    style = task_config.get('pack_style', 'ordinary')

    # Read server config to find ergs with GPU acceleration enabled.
    server_config_path = os.path.join(task_dir, 'server', 'task_config.json')
    with open(server_config_path, 'r', encoding='utf-8') as f:
        server_config = json.load(f)

    for erg_name, erg_config in server_config['server_task'].items():
        if erg_config['enable_fpga']:
            if args.deployment_mode == 'server_provisioned_runner':
                os.makedirs(output_dir, exist_ok=True)
                for filename in ('ckks_parameter.json', 'nn_layers_ct_0.json'):
                    src = os.path.join(ergs_path, filename)
                    dst = os.path.join(output_dir, filename)
                    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                        shutil.copy2(src, dst)
                runner_task_config_path = os.path.join(output_dir, 'task_config.json')
                if not os.path.exists(runner_task_config_path):
                    shutil.copy2(server_config_path, runner_task_config_path)
                with open(runner_task_config_path, 'r', encoding='utf-8') as f:
                    runner_config = json.load(f)
                runner_config.update(
                    {
                        'deployment_mode': 'server_provisioned_runner',
                        'input_mode': 'plaintext',
                        'parameter_mode': 'encrypted_offline',
                        'decryptor': 'provisioner',
                        'deployment_role': 'runner',
                    }
                )
                with open(runner_task_config_path, 'w', encoding='utf-8') as f:
                    json.dump(runner_config, f, indent=4, ensure_ascii=False)
                gen_custom_task(
                    ergs_path,
                    use_gpu=True,
                    param_name=param_name,
                    style=style,
                    lazy=False,
                    parameter_mode='encrypted_offline',
                    input_mode='plaintext',
                    output_dir=output_dir,
                )
            else:
                gen_custom_task(ergs_path, use_gpu=True, param_name=param_name, style=style, lazy=True)

    print(f'Done: mega_ag generated for {output_dir}.')


if __name__ == '__main__':
    main()
