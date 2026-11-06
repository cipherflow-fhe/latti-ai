#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Measure ciphertext payload separately from runner and full task bytes."""

import argparse
import json
import math
import struct
from pathlib import Path


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def element_lengths(path, argument):
    """Read payload lengths from the actual CT container, including old manifests."""
    file_size = path.stat().st_size
    with path.open('rb') as stream:
        def read_u64():
            data = stream.read(8)
            if len(data) != 8:
                raise ValueError(f'Truncated ciphertext container: {path}')
            return struct.unpack('<Q', data)[0]

        count = read_u64()
        if count != math.prod(argument['size']):
            raise ValueError(f'Ciphertext count does not match manifest: {path}')
        spans = []
        for _ in range(count):
            length = read_u64()
            offset = stream.tell()
            if length > file_size - offset:
                raise ValueError(f'Ciphertext span exceeds file size: {path}')
            spans.append({'offset': offset, 'length': length})
            stream.seek(length, 1)
        if stream.tell() != file_size:
            raise ValueError(f'Unexpected trailing bytes in ciphertext container: {path}')
    if 'elements' in argument and argument['elements'] != spans:
        raise ValueError(f'Ciphertext element index does not match file: {path}')
    return [span['length'] for span in spans]


def summarize(task_dir):
    task = Path(task_dir).resolve()
    runner = task / 'runner'
    parameter_root = runner / 'encrypted_model_parameters'
    manifest = json.loads((parameter_root / 'manifest.json').read_text())
    kinds, layers, arguments = {}, {}, {}
    for arg_id, arg in manifest['arguments'].items():
        if arg['type'] != 'ct':
            raise ValueError(f'Expected ciphertext argument: {arg_id}')
        lengths = element_lengths(parameter_root / arg['file'], arg)
        measurement = {'ciphertext_count': len(lengths), 'payload_bytes': sum(lengths)}
        arguments[arg_id] = dict(measurement, level=arg['level'], size=arg['size'],
                                 layer_id=arg['layer_id'], param_kind=arg['param_kind'])
        for table, key in ((kinds, arg['param_kind']), (layers, arg['layer_id'])):
            total = table.setdefault(key, {'ciphertext_count': 0, 'payload_bytes': 0})
            for field, value in measurement.items():
                total[field] += value
    return {
        'task_dir': str(task),
        'ciphertext_count': sum(a['ciphertext_count'] for a in arguments.values()),
        'encrypted_parameter_payload_bytes': sum(a['payload_bytes'] for a in arguments.values()),
        'encrypted_parameter_directory_bytes': directory_bytes(parameter_root),
        'runner_bundle_bytes': directory_bytes(runner),
        'task_directory_bytes': directory_bytes(task),
        'by_param_kind': kinds, 'by_layer': layers, 'arguments': arguments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='JSON report path')
    args = parser.parse_args()
    report = json.dumps(summarize(args.task_dir), indent=2) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')


if __name__ == '__main__':
    main()
