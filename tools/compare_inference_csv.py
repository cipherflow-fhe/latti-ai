#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare every decrypted output against an original ONNX CSV or NPZ reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_outputs(path):
    if path.suffix == '.npz':
        with np.load(path) as data:
            return {name: data[name].reshape(-1) for name in data.files}
    outputs = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            values = outputs.setdefault(row['output'], [])
            if int(row['index']) != len(values):
                raise ValueError(f'Noncontiguous or duplicated index in {path}: {row}')
            values.append(float(row['value']))
    return {name: np.asarray(values) for name, values in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--output-map', action='append', default=[], metavar='ACTUAL=REFERENCE')
    parser.add_argument('--tolerance', type=float, default=0.2)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    reference, actual = read_outputs(args.reference), read_outputs(args.actual)
    mapping = dict(binding.split('=', 1) for binding in args.output_map)
    if not set(mapping).issubset(actual):
        raise ValueError('Output mapping refers to unknown actual outputs')
    mapped = {mapping.get(name, name): values for name, values in actual.items()}
    if len(mapped) != len(actual) or set(mapped) != set(reference):
        raise ValueError(f'Output names differ: reference={list(reference)}, actual={list(mapped)}')
    reports = {}
    for name, expected in reference.items():
        values = mapped[name]
        if values.shape != expected.shape or not len(values):
            raise ValueError(f'Output size differs or is empty: {name}, {values.shape} vs {expected.shape}')
        if not np.isfinite(values).all() or not np.isfinite(expected).all():
            raise ValueError(f'Nonfinite output: {name}')
        errors = np.abs(values - expected)
        i = int(errors.argmax())
        reports[name] = {'count': len(values), 'max_abs_error': float(errors[i]),
                         'mean_abs_error': float(errors.mean()), 'max_error_index': i,
                         'reference_at_max_error': float(expected[i]), 'actual_at_max_error': float(values[i])}
    maximum = max(row['max_abs_error'] for row in reports.values())
    report = {'reference': str(args.reference.resolve()), 'actual': str(args.actual.resolve()),
              'output_map': mapping, 'outputs': reports, 'max_abs_error': maximum,
              'tolerance': args.tolerance, 'pass': maximum < args.tolerance}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
