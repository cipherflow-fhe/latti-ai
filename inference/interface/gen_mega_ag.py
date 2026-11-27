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
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'inference' / 'lattisense'))
from frontend import custom_task as frontend_state  # noqa: E402

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
    parser.add_argument(
        '--parameter-loading-mode',
        choices=['eager', 'runtime_lazy'],
        default=None,
        help='Encrypted parameter loading mode for server_provisioned_runner. Defaults to task_config or eager.',
    )
    args = parser.parse_args()

    generate_low_level(
        args.task_dir,
        deployment_mode=args.deployment_mode,
        output_dir=args.output_dir,
        parameter_loading_mode=args.parameter_loading_mode,
    )


def generate_low_level(
    task_dir,
    *,
    deployment_mode='client_encrypted_input',
    output_dir=None,
    parameter_loading_mode=None,
    random_seed=None,
    streaming=False,
):
    """Generate a task with the shared frontend/linker and actual key nodes."""
    previous_random = random.getstate()
    try:
        if random_seed is not None:
            random.seed(random_seed)
        return _generate_low_level(task_dir, deployment_mode, output_dir, parameter_loading_mode, streaming)
    finally:
        # Also reset failures that occur before process_custom_task's cleanup.
        frontend_state.g_dag.close_stream()
        frontend_state.g_dag.clear()
        frontend_state.g_evk_node_dict.clear()
        frontend_state.g_param = None
        frontend_state.reset_node_state()
        random.setstate(previous_random)


def _generate_low_level(task_dir, deployment_mode, output_dir, parameter_loading_mode, streaming):
    if deployment_mode not in ('client_encrypted_input', 'server_provisioned_runner'):
        raise ValueError(f'Unsupported deployment mode: {deployment_mode!r}')
    task_dir = os.path.abspath(task_dir)
    ergs_path = os.path.join(task_dir, 'server')
    if output_dir is None:
        output_dir = os.path.join(task_dir, 'runner') if deployment_mode == 'server_provisioned_runner' else ergs_path
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
        raise ValueError(f'Missing param_name in {ckks_param_path}')

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
    if server_config.get('deployment_mode', 'client_encrypted_input') != deployment_mode:
        raise ValueError('Deployment mode differs from compiled task')
    parameter_loading_mode = parameter_loading_mode or server_config.get('parameter_loading_mode', 'eager')

    for erg_name, erg_config in server_config['server_task'].items():
        if erg_config['enable_fpga']:
            if deployment_mode == 'server_provisioned_runner':
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
                        'input_mode': server_config.get('input_mode', 'client_ciphertext'),
                        'parameter_mode': 'encrypted_offline',
                        'parameter_loading_mode': parameter_loading_mode,
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
                    deployment_mode='server_provisioned_runner',
                    parameter_loading_mode=parameter_loading_mode,
                    streaming=streaming,
                    input_mode='plaintext' if server_config.get('input_mode') == 'plaintext' else 'ciphertext',
                    output_dir=output_dir,
                )
            else:
                gen_custom_task(ergs_path, use_gpu=True, param_name=param_name, style=style, lazy=True)

    if deployment_mode in ('client_encrypted_input', 'server_provisioned_runner'):
        # Export the actual graph's key requirements with the client task.
        # Browser binding can then reject missing keys before encryption.
        graph_path = 'graph_key_data.json' if streaming else 'mega_ag.json'
        with open(os.path.join(output_dir, graph_path), encoding='utf-8') as f:
            data = json.load(f)
        graph_data = data if streaming else data['data']
        requirements = {}
        for name, datum in graph_data.items():
            kind = datum.get('type')
            if kind not in ('rlk', 'glk', 'evk'):
                continue
            role = datum.get('key_role', 'residual')
            if kind == 'evk':
                if name not in (
                    'evk_ci_to_std',
                    'evk_std_to_ci',
                    'evk_n1_to_n2',
                    'evk_n2_to_n1',
                    'evk_dense_to_sparse',
                    'evk_sparse_to_dense',
                ):
                    raise ValueError('Unknown evaluation-key role: ' + name)
                kind = name
            element = datum.get('galois_element', 0)
            key = (role, kind, element)
            requirement = {
                'role': role,
                'kind': kind,
                'levelQ': datum['level'],
                'levelP': datum.get('sp_level', -1),
                'baseTwoDecomposition': 0,
                'compressed': False,
            }
            if element:
                requirement['galoisElement'] = str(element)
            if key in requirements:
                requirement['levelQ'] = max(requirement['levelQ'], requirements[key]['levelQ'])
                requirement['levelP'] = max(requirement['levelP'], requirements[key]['levelP'])
            requirements[key] = requirement
        roles = ('client',) if deployment_mode == 'client_encrypted_input' else ('provisioner', 'runner')
        evaluation_keys = [requirements[key] for key in sorted(requirements)]
        for role in roles:
            config_path = os.path.join(task_dir, role, 'task_config.json')
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
            config['evaluation_keys'] = evaluation_keys
            config['task_manifest_required'] = True
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
                f.write('\n')
        with open(os.path.join(output_dir, 'key_requirements.json'), 'w', encoding='utf-8') as f:
            json.dump({'deployment_mode': deployment_mode, 'evaluation_keys': evaluation_keys}, f, indent=2)
            f.write('\n')
        owner = 'client' if deployment_mode == 'client_encrypted_input' else 'provisioner'
        for filename in ('task_signature.json', 'key_requirements.json'):
            shutil.copy2(os.path.join(output_dir, filename), os.path.join(task_dir, owner, filename))
        owner_directory = Path(task_dir) / owner
        files = {}
        for name in (
            'task_config.json',
            'ckks_parameter.json',
            'task_signature.json',
            'key_requirements.json',
            'input_contract.json',
        ):
            path = owner_directory / name
            if path.exists():
                data = path.read_bytes()
                files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        (owner_directory / 'task_manifest.json').write_text(
            json.dumps(
                {
                    'schema': 'latti-ai.browser-client-task.v1',
                    'deployment_mode': deployment_mode,
                    'files': files,
                },
                indent=2,
            )
            + '\n'
        )

    print(f'Done: mega_ag generated for {output_dir}.')

    return Path(output_dir)


if __name__ == '__main__':
    main()
