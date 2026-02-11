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
import logging
import sys
from pathlib import Path

# Add model_compiler to Python path
sys.path.insert(0, str(Path(__file__).parent / 'model_compiler'))

from model_compiler.graph_partition_dp import run_parallel, init_config_with_args
from model_export.onnx_to_json import onnx_to_json

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


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
  python run_compile.py -i pt.json -o ./output --poly_n 65536 --style ordinary

  # Using ONNX model file (will auto-convert to pt.json)
  python run_compile.py -i model.onnx
  python run_compile.py -i model.onnx -o ./output --style multiplexed
  python run_compile.py -i model.onnx -o ./output --poly_n 65536 --style ordinary
        """,
    )

    # Required arguments
    parser.add_argument(
        '-i', '--input', type=str, required=True, help='Input ONNX model file (.onnx) or JSON file (pt.json) (required)'
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output', type=str, default=None, help='Output directory path (default: same as input file directory)'
    )

    parser.add_argument(
        '--poly_n',
        type=int,
        choices=[8192, 16384, 65536],
        default=None,
        help='Polynomial modulus degree: 8192, 16384, or 65536',
    )

    parser.add_argument(
        '--style',
        type=str,
        choices=['ordinary', 'multiplexed'],
        default=None,
        help='Computation style: ordinary or multiplexed',
    )

    parser.add_argument('--graph_type', type=str, choices=['btp'], default=None, help='Graph type: btp')

    parser.add_argument(
        '--num_experiments', type=int, default=128, help='Number of parallel compilation experiments (default: 128)'
    )

    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel worker processes (default: 16)')

    parser.add_argument(
        '--temperature', type=float, default=1.0, help='Temperature parameter for randomization (default: 1.0)'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'[Error] Input file not found: {input_path}')
        sys.exit(1)

    if not input_path.is_file():
        print(f'[Error] Input path is not a file: {input_path}')
        sys.exit(1)

    # Check if input is ONNX or JSON
    is_onnx = input_path.suffix.lower() == '.onnx'
    is_json = input_path.suffix.lower() == '.json'

    if not (is_onnx or is_json):
        print(f'[Error] Input file must be .onnx or .json, got: {input_path.suffix}')
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # If input is ONNX, convert it to pt.json first
    if is_onnx:
        # Determine style for ONNX conversion
        onnx_style = args.style if args.style else 'ordinary'

        # Generate pt.json filename
        pt_json_path = output_dir / 'pt.json'

        print(f'\n[ONNX→JSON] Input: {input_path}')
        print(f'[ONNX→JSON] Output: {pt_json_path}')
        print(f'[ONNX→JSON] Style: {onnx_style}')

        try:
            onnx_to_json(str(input_path), str(pt_json_path), onnx_style)
            log.info('[ONNX→JSON] Done: %s → %s (style=%s)', input_path, pt_json_path, onnx_style)
        except Exception as e:
            print(f'\n[Error] ONNX to JSON conversion failed: {e}')
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Update input_path to the generated pt.json
        input_path = pt_json_path
    else:
        pt_json_path = input_path

    # Print configuration
    print(f'\n[Compile] Input: {pt_json_path}')
    print(f'[Compile] Output: {output_dir}')
    if args.poly_n or args.style or args.graph_type:
        config_parts = []
        if args.poly_n:
            config_parts.append(f'POLY_N={args.poly_n}')
        if args.style:
            config_parts.append(f'STYLE={args.style}')
        if args.graph_type:
            config_parts.append(f'GRAPH_TYPE={args.graph_type}')
        print(f'[Compile] Config: {", ".join(config_parts)}')
    print(f'[Compile] Running {args.num_experiments} experiments with {args.num_workers} workers\n')

    try:
        # Initialize configuration with command line arguments
        init_config_with_args(poly_n=args.poly_n, style=args.style, graph_type=args.graph_type)

        # Run parallel compilation
        run_parallel(
            num_experiments=args.num_experiments,
            input_file_path=pt_json_path,
            output_dir=output_dir,
            temperature=args.temperature,
            num_workers=args.num_workers,
        )

        print(f'\n[Compile] Success! Output: {output_dir}')

        # List generated files in task structure
        task_dir = output_dir / 'task'
        if task_dir.exists():
            print(
                f'[Compile] Structure: task/server/ergs/erg0.json, task/{{server,client}}/{{task_config,ckks_parameter}}.json'
            )

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
