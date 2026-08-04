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

"""
Graph Splitter Compilation Script

This script provides a convenient interface to compile models using graph_splitter_recur.py
Supports both ONNX model files (.onnx) and pre-converted JSON files (pt.json)
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'model_compiler'))
sys.path.insert(0, str(Path(__file__).parent))

from model_compiler.pipeline import run_pipeline
from model_export.onnx_to_json import onnx_to_json
from nn_tools import export_h5_from_onnx

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


MAT_PACK_STYLES = {'', 'par_block_col_major', 'par_diagonal_pack'}
DEPLOYMENT_MODES = {'client_encrypted_input', 'server_provisioned_runner'}


def deployment_metadata(deployment_mode: str) -> dict[str, str]:
    if deployment_mode == 'client_encrypted_input':
        return {
            'deployment_mode': 'client_encrypted_input',
            'input_mode': 'client_ciphertext',
            'parameter_mode': 'plaintext_lazy',
            'decryptor': 'client',
        }
    if deployment_mode == 'server_provisioned_runner':
        return {
            'deployment_mode': 'server_provisioned_runner',
            'input_mode': 'plaintext',
            'parameter_mode': 'encrypted_offline',
            'decryptor': 'provisioner',
        }
    raise ValueError(f'Unsupported deployment_mode: {deployment_mode!r}')


def read_json_file(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_server_provisioned_runner_layout(task_dir: Path, runner_output_dir: Path | None = None) -> None:
    server_dir = task_dir / 'server'
    if not server_dir.exists():
        raise FileNotFoundError(f'Cannot create runner layout: missing {server_dir}')

    runner_dir = runner_output_dir if runner_output_dir is not None else task_dir / 'runner'
    provisioner_dir = task_dir / 'provisioner'
    runner_dir.mkdir(parents=True, exist_ok=True)
    provisioner_dir.mkdir(parents=True, exist_ok=True)

    mode_config = deployment_metadata('server_provisioned_runner')
    provisioner_config = {**mode_config, 'deployment_role': 'provisioner'}
    runner_config = {**mode_config, 'deployment_role': 'runner'}

    server_task_config_path = server_dir / 'task_config.json'
    server_task_config = read_json_file(server_task_config_path)
    server_task_config.update(provisioner_config)
    write_json_file(server_task_config_path, server_task_config)

    provisioner_task_config = dict(server_task_config)
    provisioner_task_config.update(provisioner_config)
    write_json_file(provisioner_dir / 'task_config.json', provisioner_task_config)

    runner_task_config = dict(server_task_config)
    runner_task_config.update(runner_config)
    write_json_file(runner_dir / 'task_config.json', runner_task_config)

    for filename in ('ckks_parameter.json', 'nn_layers_ct_0.json'):
        src = server_dir / filename
        if src.exists():
            shutil.copy2(src, runner_dir / filename)
            shutil.copy2(src, provisioner_dir / filename)

    h5_src = server_dir / 'model_parameters.h5'
    if h5_src.exists():
        shutil.copy2(h5_src, provisioner_dir / 'model_parameters.h5')


def read_compile_config(config_path: str) -> dict[str, int | float | str]:
    if not config_path:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mat_pack_style = data.get('mat_pack_style', '')
    if mat_pack_style not in MAT_PACK_STYLES:
        raise ValueError(f'Unsupported mat_pack_style: {mat_pack_style!r}. Expected one of {sorted(MAT_PACK_STYLES)}')
    result = {
        'n_heads': int(data['n_heads']),
        'head_dim': int(data['head_dim']),
        'matmul_block_size': int(data['matmul_block_size']),
        'mat_pack_style': mat_pack_style,
        'model_type': str(data.get('model_type', '')),
    }
    for key in (
        'btp_scale',
        'set_btp_scale',
        'bert_softmax_values_btp_scale',
        'bert_softmax_denominator_btp_scale',
        'bert_softmax_scaled_denominator_btp_scale',
        'bert_softmax_inverse_btp_scale',
        'bert_layernorm_inverse_btp_scale',
        'bert_softmax_initial_denominator_scale',
        'bert_softmax_wide_initial_denominator_scale',
        'bert_softmax_use_wide_inverse_epsilon',
        'bert_softmax_first_refinement_denominator_scale',
        'bert_softmax_later_refinement_denominator_scale',
    ):
        if key in data and data[key] is not None:
            result[key] = float(data[key])
    return result


def main():
    """Main function to run graph splitter compilation"""

    parser = argparse.ArgumentParser(
        description='Compile a model using the graph splitter tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using pt.json file directly
  python run_compile.py -i pt.json
  python run_compile.py -i pt.json -o ./output
  python run_compile.py -i pt.json -o ./output --style ordinary

  # Using ONNX model file (will auto-convert to pt.json)
  python run_compile.py -i model.onnx
  python run_compile.py -i model.onnx -o ./output --style multiplexed
        """,
    )

    parser.add_argument(
        '-i', '--input', type=str, required=True, help='Input ONNX model file (.onnx) or JSON file (pt.json) (required)'
    )

    parser.add_argument(
        '-o', '--output', type=str, default=None, help='Output directory path (default: same as input file directory)'
    )

    parser.add_argument(
        '--style',
        type=str,
        choices=['ordinary', 'multiplexed'],
        default='multiplexed',
        help='Computation style: ordinary or multiplexed (default: multiplexed)',
    )

    parser.add_argument('--graph_type', type=str, choices=['btp'], default='btp', help='Graph type: btp (default: btp)')

    parser.add_argument(
        '--num_experiments', type=int, default=128, help='Number of parallel compilation experiments (default: 128)'
    )

    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel worker processes (default: 16)')

    parser.add_argument(
        '--temperature', type=float, default=1.0, help='Temperature parameter for randomization (default: 0.0)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='Compile config JSON path containing n_heads, head_dim, matmul_block_size, mat_pack_style',
    )

    parser.add_argument(
        '--set_btp_scale',
        '--btp_scale',
        dest='set_btp_scale',
        type=float,
        default=None,
        help='Wrap each bootstrapping layer with pcmgamma layers using this scale',
    )

    parser.add_argument(
        '--deployment-mode',
        choices=sorted(DEPLOYMENT_MODES),
        default='client_encrypted_input',
        help='Deployment mode for task metadata and optional runner/provisioner layout.',
    )

    parser.add_argument(
        '--runner-output-dir',
        type=str,
        default=None,
        help='Runner output directory for server_provisioned_runner mode. Defaults to <output>/task/runner.',
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'[Error] Input file not found: {input_path}')
        sys.exit(1)

    if not input_path.is_file():
        print(f'[Error] Input path is not a file: {input_path}')
        sys.exit(1)

    is_onnx = input_path.suffix.lower() == '.onnx'
    is_json = input_path.suffix.lower() == '.json'

    if not (is_onnx or is_json):
        print(f'[Error] Input file must be .onnx or .json, got: {input_path.suffix}')
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    compile_config = read_compile_config(args.config)
    mat_pack_style = compile_config.get('mat_pack_style', '')
    model_type = compile_config.get('model_type', '')
    feature_mat = mat_pack_style in ('par_block_col_major', 'par_diagonal_pack')
    set_btp_scale = args.set_btp_scale
    if set_btp_scale is None:
        set_btp_scale = compile_config.get('set_btp_scale', compile_config.get('btp_scale'))
    onnx_path = input_path if is_onnx else None

    if is_onnx:
        onnx_style = args.style if args.style else 'ordinary'
        pt_json_path = output_dir / 'pt.json'

        print(f'\n[ONNX→JSON] Input: {input_path}')
        print(f'[ONNX→JSON] Output: {pt_json_path}')
        print(f'[ONNX→JSON] Style: {onnx_style}')

        try:
            onnx_to_json(
                str(input_path),
                str(pt_json_path),
                onnx_style,
                mat_pack_style=mat_pack_style,
                model_type=model_type,
            )
            log.info('[ONNX→JSON] Done: %s → %s (style=%s)', input_path, pt_json_path, onnx_style)
        except Exception as e:
            print(f'\n[Error] ONNX to JSON conversion failed: {e}')
            import traceback

            traceback.print_exc()
            sys.exit(1)

        input_path = pt_json_path
    else:
        pt_json_path = input_path

    print(f'\n[Compile] Input: {pt_json_path}')
    print(f'[Compile] Output: {output_dir}')
    print(
        f'[Compile] Config: STYLE={args.style}, GRAPH_TYPE={args.graph_type}, '
        f'COMPILE_CONFIG={args.config or "<none>"}, MAT_PACK_STYLE={mat_pack_style}, '
        f'MODEL_TYPE={model_type}, SET_BTP_SCALE={set_btp_scale}, DEPLOYMENT_MODE={args.deployment_mode}'
    )
    print(f'[Compile] Running {args.num_experiments} experiments with {args.num_workers} workers\n')

    try:
        run_pipeline(
            num_experiments=args.num_experiments,
            input_file_path=pt_json_path,
            output_dir=output_dir,
            temperature=args.temperature,
            num_workers=args.num_workers,
            style=args.style,
            graph_type=args.graph_type,
            n_heads=compile_config.get('n_heads'),
            head_dim=compile_config.get('head_dim'),
            matmul_block_size=compile_config.get('matmul_block_size'),
            mat_pack_style=mat_pack_style,
            model_type=model_type,
            set_btp_scale=set_btp_scale,
            bert_softmax_values_btp_scale=compile_config.get('bert_softmax_values_btp_scale'),
            bert_softmax_denominator_btp_scale=compile_config.get('bert_softmax_denominator_btp_scale'),
            bert_softmax_scaled_denominator_btp_scale=compile_config.get('bert_softmax_scaled_denominator_btp_scale'),
            bert_softmax_inverse_btp_scale=compile_config.get('bert_softmax_inverse_btp_scale'),
            bert_layernorm_inverse_btp_scale=compile_config.get('bert_layernorm_inverse_btp_scale'),
            bert_softmax_initial_denominator_scale=compile_config.get('bert_softmax_initial_denominator_scale'),
            bert_softmax_wide_initial_denominator_scale=compile_config.get(
                'bert_softmax_wide_initial_denominator_scale'
            ),
            bert_softmax_use_wide_inverse_epsilon=compile_config.get('bert_softmax_use_wide_inverse_epsilon'),
            bert_softmax_first_refinement_denominator_scale=compile_config.get(
                'bert_softmax_first_refinement_denominator_scale'
            ),
            bert_softmax_later_refinement_denominator_scale=compile_config.get(
                'bert_softmax_later_refinement_denominator_scale'
            ),
            # enable_score_cache=False
        )

        print(f'\n[Compile] Success! Output: {output_dir}')

        task_dir = output_dir / 'task'
        if task_dir.exists():
            print(
                '[Compile] Structure: task/server/nn_layers_ct_0.json, '
                'task/{server,client}/{task_config,ckks_parameter}.json'
            )

        if onnx_path is not None and task_dir.exists():
            json_path = task_dir / 'server' / 'nn_layers_ct_0.json'
            h5_path = task_dir / 'server' / 'model_parameters.h5'
            if json_path.exists():
                print(f'\n[H5 Export] ONNX: {onnx_path}')
                print(f'[H5 Export] JSON: {json_path}')
                print(f'[H5 Export] H5:   {h5_path}')
                try:
                    export_h5_from_onnx(
                        onnx_path=str(onnx_path),
                        json_path=str(json_path),
                        h5_path=str(h5_path),
                        feature_mat=feature_mat,
                        model_type=model_type,
                    )
                    print(f'[H5 Export] Done: {h5_path}')
                except Exception as e:
                    print(f'\n[H5 Export] Failed: {e}')
                    import traceback

                    traceback.print_exc()
            else:
                print(f'\n[H5 Export] Skipped: {json_path} not found')

        if args.deployment_mode == 'server_provisioned_runner' and task_dir.exists():
            runner_output_dir = Path(args.runner_output_dir) if args.runner_output_dir else None
            write_server_provisioned_runner_layout(task_dir, runner_output_dir)
            runner_dir = runner_output_dir if runner_output_dir is not None else task_dir / 'runner'
            print(f'\n[Deployment] server_provisioned_runner layout written:')
            print(f'[Deployment] Provisioner: {task_dir / "provisioner"}')
            print(f'[Deployment] Runner:      {runner_dir}')
            print('[Deployment] Runner bundle config excludes secret_key.bin and model_parameters.h5.')

        return 0

    except KeyboardInterrupt:
        print('\n[Compile] Interrupted by user')
        return 130

    except Exception as e:
        print(f'\n[Compile] Failed: {e}')
        import traceback

        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
