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

# ---- 修复 frontend 双模块树问题 ----
# lattisense 的 frontend 包内部用 `from frontend.xxx` 裸导入（非相对导入）。
# sys.path 里同时有 repo_root 和 repo_root/inference/lattisense 时，
# Python 会创建两棵独立的模块树，导致 set_fhe_param 和 _current_param 读到不同的 g_param。
# 在 import custom_task 之前，把 sys.modules 里的 'frontend.*' 别名指向 inference.lattisense.frontend.*。
import inference.lattisense.frontend as _fe_pkg  # noqa: E402
sys.modules.setdefault('frontend', _fe_pkg)
import inference.lattisense.frontend.types as _fe_types  # noqa: E402
sys.modules.setdefault('frontend.types', _fe_types)
import inference.lattisense.frontend.bootstrap_params as _fe_bp  # noqa: E402
sys.modules.setdefault('frontend.bootstrap_params', _fe_bp)

from inference.lattisense.frontend.custom_task import *  # noqa: E402


# ---- v6 兼容补丁：CkksCiphertextNode / BfvCiphertextNode 旧签名 (id, level=...) ----
# v3 的调用形态是 CkksCiphertextNode('x', level=3)，v6 换成 CkksCiphertextNode(level=3, id='x')。
# latti-ai-new 侧还有 200+ 处旧调用，改用 monkey-patch 兼容，不改 lattisense 源码。
from inference.lattisense.frontend import types as _v6_types  # noqa: E402


def _patch_ciphertext_node_ctor(cls):
    _orig_init = cls.__init__

    def _compat_init(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            # 旧形态：id 是第一个位置参数
            id_arg = args[0]
            args = args[1:]
            kwargs.setdefault('id', id_arg)
        return _orig_init(self, *args, **kwargs)

    cls.__init__ = _compat_init


_patch_ciphertext_node_ctor(_v6_types.CkksCiphertextNode)
_patch_ciphertext_node_ctor(_v6_types.BfvCiphertextNode)


# ---- v6 兼容补丁：FheDataNode 加 level/scale/degree 等 property ----
# v3 里 node.level / node.scale / node.degree 是直接字段；
# v6 里收进 node.metadata。给基类加 property 转发，覆盖所有子类（Plaintext/Ciphertext/KeyNode）。
def _add_meta_forward_props(cls, props):
    for name in props:
        if not hasattr(cls, name):
            setattr(
                cls,
                name,
                property(
                    lambda self, _n=name: getattr(self.metadata, _n),
                    lambda self, v, _n=name: setattr(self.metadata, _n, v),
                ),
            )


_add_meta_forward_props(
    _v6_types.FheDataNode,
    ('level', 'degree', 'scale', 'log_slots', 'is_ringt', 'is_batched', 'is_ntt', 'mform_bits'),
)


# ---- v6 兼容补丁：custom_compute 加 wait_inputs 参数 ----
# v6 删掉了 wait_inputs，但 latti-ai-new 侧 encrypted_param_ops 还在用它。
# 语义：wait_inputs 只是额外的调度依赖，不影响输入/输出结构，可以直接加为图边。
import inference.lattisense.frontend.custom_task as _ct_module  # noqa: E402


def _compat_custom_compute(inputs, output, type, attributes=None, wait_inputs=None):
    if not inputs:
        raise ValueError('At least one input data node is required for custom compute.')
    if output is None:
        raise ValueError('Output data node is required for custom compute.')

    op = _ct_module.CustomComputeNode(type=type, attributes=attributes)
    # 主输入：严格按传入顺序加边，保证 op.input_nodes[0] 是对应的 store
    for input_node in inputs:
        _ct_module.g_dag.add_edge(input_node, op)
    # wait_inputs 是额外调度依赖，放在主输入之后
    if wait_inputs:
        for w in wait_inputs:
            _ct_module.g_dag.add_edge(w, op)
    _ct_module.g_dag.add_edge(op, output)


_ct_module.custom_compute = _compat_custom_compute


# ---- v6 兼容补丁：process_custom_task 支持 offline_input_args / extra_signature_fields ----
# v6 删除了这两个参数。latti-ai-new 侧（deploy_cmds.py）还在用。
# 策略：调用原版时忽略这两个参数，然后在生成的 mega_ag.json / task_signature.json 里后处理注入。
def _shape_of(data):
    if not isinstance(data, list):
        return []
    if not data:
        return []
    sub = _shape_of(data[0])
    return [len(data)] + sub if isinstance(sub, list) else [len(data)]


def _compat_process_custom_task(
    input_args=None,
    output_args=None,
    output_instruction_path=None,
    processor=None,
    offline_input_args=None,
    extra_signature_fields=None,
    fpga_acc=False,
):
    if processor is None:
        processor = _ct_module.Processor.CPU

    # 忽略 offline_input_args / extra_signature_fields / fpga_acc，只走 v6 原版路径
    _orig_process_custom_task(
        input_args=input_args,
        output_args=output_args,
        output_instruction_path=output_instruction_path,
        processor=processor,
    )

    out_dir = Path(output_instruction_path)

    # 后处理 1：把 offline_input_args 合并到 online 段（紧跟 input 之后）
    if offline_input_args:
        import json

        offline_entries = []
        offline_ids = []
        for arg in offline_input_args:
            nodes = []
            stack = [arg.data]
            while stack:
                x = stack.pop(0)
                if isinstance(x, list):
                    stack = list(x) + stack
                else:
                    nodes.append(x)
            if not nodes:
                continue
            first = nodes[0]
            t = first.type.value if hasattr(first.type, 'value') else first.type
            entry = {
                'id': arg.id,
                'type': t,
                'size': _shape_of(arg.data),
                'phase': 'in',
            }
            if hasattr(first, 'metadata'):
                entry['level'] = first.metadata.level
            offline_entries.append(entry)
            offline_ids.append(arg.id)

        # 修改 mega_ag.json：把 offline_ids 插到 inputs[1..]
        mag_path = out_dir / 'mega_ag.json'
        with open(mag_path, 'r', encoding='utf-8') as fp:
            mag = json.load(fp)
        for i, oid in enumerate(offline_ids):
            if oid not in mag['inputs']:
                mag['inputs'].insert(1 + i, oid)
        with open(mag_path, 'w', encoding='utf-8') as fp:
            json.dump(mag, fp, indent=2)

        # 修改 task_signature.json：把 offline_entries 插到 online[1..]
        sig_path = out_dir / 'task_signature.json'
        with open(sig_path, 'r', encoding='utf-8') as fp:
            sig = json.load(fp)
        for i, entry in enumerate(offline_entries):
            sig['online'].insert(1 + i, entry)
        with open(sig_path, 'w', encoding='utf-8') as fp:
            json.dump(sig, fp, indent=4)

    # 后处理 2：合并 extra_signature_fields 到 task_signature.json 顶层
    if extra_signature_fields:
        import json
        sig_path = out_dir / 'task_signature.json'
        with open(sig_path, 'r', encoding='utf-8') as fp:
            sig = json.load(fp)
        for k, v in extra_signature_fields.items():
            sig[k] = v
        with open(sig_path, 'w', encoding='utf-8') as fp:
            json.dump(sig, fp, indent=4)

    # 后处理 3：把 ringt plaintext 的 type 从 'pt' 改成 'pt_ringt'
    # C++ runner（inference_runner.cpp:192/238/322）硬编码检查 pt_ringt，
    # 而 v6 的 DataType.Plaintext.value == 'pt'。ringt 的特征是 level==0。
    import json
    sig_path = out_dir / 'task_signature.json'
    if sig_path.exists():
        with open(sig_path, 'r', encoding='utf-8') as fp:
            sig = json.load(fp)
        changed = False
        for entry in sig.get('online', []):
            if entry.get('type') == 'pt' and entry.get('level', 0) == 0:
                entry['type'] = 'pt_ringt'
                changed = True
        if changed:
            with open(sig_path, 'w', encoding='utf-8') as fp:
                json.dump(sig, fp, indent=4)


# 保存原版并替换
_orig_process_custom_task = _ct_module.process_custom_task
_ct_module.process_custom_task = _compat_process_custom_task


# ---- v6 兼容补丁：FheDataNode 加 level/scale/degree 等 property ----
# v3 里 node.level / node.scale / node.degree 是直接字段；
# v6 里收进 node.metadata。给基类加 property 转发，覆盖所有子类（Plaintext/Ciphertext/KeyNode）。
def _add_meta_forward_props(cls, props):
    for name in props:
        if not hasattr(cls, name):
            setattr(
                cls,
                name,
                property(
                    lambda self, _n=name: getattr(self.metadata, _n),
                    lambda self, v, _n=name: setattr(self.metadata, _n, v),
                ),
            )


_add_meta_forward_props(
    _v6_types.FheDataNode,
    ('level', 'degree', 'scale', 'log_slots', 'is_ringt', 'is_batched', 'is_ntt', 'mform_bits'),
)

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
    parameter_loading_mode = args.parameter_loading_mode or server_config.get('parameter_loading_mode', 'eager')

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
                    input_mode='plaintext',
                    output_dir=output_dir,
                )
            else:
                gen_custom_task(ergs_path, use_gpu=True, param_name=param_name, style=style, lazy=True)

    print(f'Done: mega_ag generated for {output_dir}.')


if __name__ == '__main__':
    main()
