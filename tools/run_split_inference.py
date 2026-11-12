#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing GPU Client/Runner with split contracts and compare full outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_inference_csv import read_outputs
from onnx_split_contract import file_hash, validate_contract


def compare(contract, actual_path, report_path):
    manifest_path = Path(contract['manifest'])
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    actual = read_outputs(actual_path)
    mapping = contract['output_map']
    if set(actual) != set(mapping):
        raise ValueError('Decrypted suffix output IDs differ from contract')
    actual = {mapping[key]: value for key, value in actual.items()}
    with np.load(root / manifest['reference']) as reference, np.load(root / manifest['prefix_outputs']) as prefix:
        if set(reference.files) != {route['name'] for route in manifest['output_routes']}:
            raise ValueError('Original output routes do not match reference')
        reports, merged = {}, {}
        for route in manifest['output_routes']:
            name = route['name']
            expected = reference[name]
            if list(expected.shape) != route['shape'] or str(expected.dtype) != route['dtype']:
                raise ValueError(f'Reference shape/dtype differs: {name}')
            if route['route'] == 'suffix':
                values = actual[name]
            elif route['route'] == 'prefix':
                values = prefix[name].reshape(-1)
            elif route['route'] == 'input':
                entry = next(e for e in manifest['original_inputs'] if e['name'] == name)
                if file_hash(entry['csv']) != entry['sha256']:
                    raise ValueError(f'Original input CSV changed: {name}')
                values = np.loadtxt(entry['csv'], delimiter=',', dtype=entry['dtype'], ndmin=1).reshape(-1)
            else:
                # Static outputs have no FHE path; their original values are unchanged.
                values = expected.reshape(-1)
            if values.size != expected.size or not np.isfinite(values).all() or not np.isfinite(expected).all():
                raise ValueError(f'Output count or finite-value check failed: {name}')
            errors = np.abs(values.astype(np.float64) - expected.reshape(-1).astype(np.float64))
            index = int(errors.argmax())
            reports[name] = {
                'shape': route['shape'],
                'dtype': route['dtype'],
                'count': expected.size,
                'max_abs_error': float(errors[index]),
                'max_error_index': index,
                'reference_at_max_error': float(expected.reshape(-1)[index]),
                'actual_at_max_error': float(values[index]),
                'route': route['route'],
            }
            merged[name] = values.reshape(expected.shape)
    np.savez(report_path.parent / 'merged_outputs.npz', **merged)
    maximum = max(output['max_abs_error'] for output in reports.values())
    report = {
        'reference': 'original_full_onnx_csv_plaintext',
        'reference_path': str(root / manifest['reference']),
        'manifest_sha256': contract['manifest_sha256'],
        'suffix_sha256': contract['suffix_sha256'],
        'deployment_mode': contract['deployment_mode'],
        'parameters': {key: value['param_name'] for key, value in contract['ckks'].items()},
        'outputs': reports,
        'max_abs_error': maximum,
        'tolerance': 0.1,
        'pass': maximum < 0.1,
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['pass']:
        raise ValueError(f'FHE maximum absolute error {maximum} must be strictly below 0.1')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, required=True, help='Compiled task root, for either mode')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    task, out = args.task_dir.resolve(), args.output_dir.resolve()
    contract = validate_contract(task)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    parameter_mode = contract['deployment_mode'] == 'server_provisioned_runner'
    command = [
        str(repo / 'build/examples/inference'),
        '--task-dir',
        str(task / 'runner' if parameter_mode else task),
        '--gpu',
    ]
    for entry in contract['inputs']:
        path = entry['csv']
        if contract['mat_pack_style'] == 'par_diagonal_pack':
            # Existing PDM Client reads the transposed storage order; logical CSVs stay unchanged.
            values = np.loadtxt(path, delimiter=',').reshape(entry['shape'][-2:])
            path = out / (entry['id'] + '_pdm.csv')
            np.savetxt(path, values.T, delimiter=',', fmt='%.17g')
        command += ['--input', f'{entry["feature_id"]}={path}']
    decrypted = out / 'decrypted.csv'
    if parameter_mode:
        command += ['--deployment-mode', contract['deployment_mode'], '--output-cipher', str(out / 'output.ct')]
    else:
        command += ['--output-csv', str(decrypted)]
    subprocess.run(
        [sys.executable, str(repo / 'tools/measure_gpu_command.py'), '--output-dir', str(out), '--', *command],
        check=True,
    )
    if parameter_mode:
        subprocess.run(
            [
                str(repo / 'build/tools/decrypt_runner_output'),
                '--task-dir',
                str(task / 'provisioner'),
                '--cipher',
                str(out / 'output.ct'),
                '--output-csv',
                str(decrypted),
            ],
            check=True,
        )
    compare(contract, decrypted, out / 'comparison.json')


if __name__ == '__main__':
    main()
