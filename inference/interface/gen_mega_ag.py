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
import sys

# Resolve project root by walking up until we find the 'training' directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
_dir = script_dir
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, 'training')):
        break
    _dir = os.path.dirname(_dir)
project_root = _dir

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'inference', 'lattisense'))

from frontend.custom_task import *  # noqa: E402

from inference.model_generator.deploy_cmds import gen_custom_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Generate mega_ag instructions for a task.')
    parser.add_argument(
        '--task-dir',
        type=str,
        required=True,
        help='Path to the task directory',
    )
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    ergs_path = os.path.join(task_dir, 'server')

    # Read poly_modulus_degree from ckks_parameter.json to determine n.
    ckks_param_path = os.path.join(task_dir, 'client', 'ckks_parameter.json')
    with open(ckks_param_path, 'r', encoding='utf-8') as f:
        ckks_config = json.load(f)
    first_param = next(iter(ckks_config.values()))
    n = first_param['poly_modulus_degree']

    # Read pack_style from task_config.json.
    task_config_path = os.path.join(task_dir, 'client', 'task_config.json')
    with open(task_config_path, 'r', encoding='utf-8') as f:
        task_config = json.load(f)
    style = task_config.get('pack_style', 'ordinary')

    # Read server config to find ergs with GPU acceleration enabled.
    server_config_path = os.path.join(task_dir, 'server', 'task_config.json')
    with open(server_config_path, 'r', encoding='utf-8') as f:
        server_config = json.load(f)

    for erg_name, erg_config in server_config['server_task'].items():
        if erg_config['enable_fpga']:
            gen_custom_task(ergs_path, use_gpu=True, n=n, style=style)

    print(f'Done: mega_ag generated for {task_dir}.')


if __name__ == '__main__':
    main()
