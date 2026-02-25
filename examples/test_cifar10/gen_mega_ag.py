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

"""Generate mega_ag instructions for the CIFAR-10 (ResNet-20) example.

Usage:
    cd examples/test_cifar10
    python gen_mega_ag.py
"""

import json
import os
import sys

# TODO: Create a general gen_mega_ag.py, or gen_erg.py, script

# Resolve the directory where this script lives.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find project root by walking up until we find the 'training' directory.
_dir = script_dir
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, 'training')):
        break
    _dir = os.path.dirname(_dir)
project_root = _dir

# Add project root and LattiSense library to the Python path so that
# the frontend and training modules can be imported.
sys.path.insert(0, project_root)

from inference.lattisense.frontend.custom_task import *  # noqa: E402
from inference.model_generator.deploy_cmds import gen_custom_task  # noqa: E402

# Path to the server-side encrypted computation graph (ergs directory).
task_path = os.path.join(script_dir, 'task', 'server')

# Read the server task configuration to determine which computation
# segments (ergs) require GPU-accelerated mega_ag generation.
with open(os.path.join(task_path, 'task_config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)

# For each erg with FPGA/GPU acceleration enabled, generate the
# corresponding mega_ag instruction sequence.  mega_ag fuses multiple
# HE operations into optimized GPU kernels for faster inference.
for erg_name, erg_config in config['server_task'].items():
    if erg_config['enable_fpga']:
        gen_custom_task(task_path, use_gpu=True, n=65536, style='multiplexed')

print('Done: CIFAR-10 mega_ag generated.')
