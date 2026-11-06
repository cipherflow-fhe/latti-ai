#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run the spec's isolated legacy/compressed GPU runtime-lazy acceptance."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_onnx_csv import evaluate, write_logits
from summarize_parameter_bundle import summarize

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    'resnet20': ('runs/cifar10/model/trained_poly.onnx', 'examples/test_cifar10/task/client/img.csv', 10),
    'mobilenet': ('runs/imagenet/model/trained_poly.onnx', 'examples/test_imagenet/task/client/img.csv', 1000),
}


def read_logits(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len({(r['output'], int(r['index'])) for r in rows}) != len(rows):
        raise ValueError('Duplicate output index')
    return {(r['output'], int(r['index'])): float(r['value']) for r in rows}


def metrics(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('Incomplete logits or mismatched output names')
    keys = sorted(reference)
    p = np.array([reference[k] for k in keys])
    f = np.array([candidate[k] for k in keys])
    if not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError('Nonfinite logits')
    error = np.abs(f - p)
    return {
        'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'max_rel_error': float((error / np.maximum(np.abs(p), 1e-12)).max()),
        'cosine_similarity': float(np.dot(f, p) / (np.linalg.norm(f) * np.linalg.norm(p))),
        'top1_match': bool(np.argmax(f) == np.argmax(p)),
    }


def layout_metadata(graph, argument):
    layer = graph['layer'][argument['layer_id']]
    arg_id = argument['id']
    if arg_id.startswith('dense'):
        return layer.get('dense_seed_tiles')
    if arg_id.startswith('poly_reluw_'):
        return layer.get('poly_seed_layout')
    if arg_id.startswith('convw_') and 'seed_weight_subgrid_stride' in layer:
        stride = layer['seed_weight_subgrid_stride']
        return dict(kind=layer.get('seed_weight_layout', 'kernel'), stride=stride,
                    group_size=math.prod(stride), kernel_shape=layer['kernel_shape'],
                    storage_shape=argument['size'],
                    input_geometry=graph['feature'][layer['feature_input'][0]],
                    output_geometry=graph['feature'][layer['feature_output'][0]])
    return None


def recovery_operations(graph, manifest):
    """Count the parameter-only subgraph, stopping where it meets activations."""
    recovered = set()
    counts = Counter()
    for op in graph['compute'].values():
        if op['type'] == 'load_encrypted_param_ct':
            arg_id = op['attributes']['arg_id']
            layout = manifest[arg_id].get('layout', {})
            packed = bool(layout and layout.get('group_size', 1) > 1)
            if arg_id.startswith('dense'):
                packed = packed and layout.get('weight' if arg_id.startswith('densew_') else 'bias', False)
            if arg_id.startswith('poly_reluw_'):
                packed = packed and layout.get('packed_degrees', [])[int(arg_id.rsplit('_', 1)[1])]
            if packed:
                recovered.update(op['outputs'])
            continue
        attrs = op.get('attributes', {})
        if op['type'] == 'encode_pt' and attrs.get('type') in ('seed_mask_pt', 'tile_mask_pt'):
            counts['encode_pt'] += 1
        ct_inputs = [i for i in op['inputs'] if graph['data'][str(i)]['type'] == 'ct']
        if ct_inputs and all(i in recovered for i in ct_inputs):
            counts[op['type']] += 1
            recovered.update(op['outputs'])
    return dict(counts)


def source_provenance():
    """Hash tracked and new implementation files, including nested dependencies."""
    hashes = {}
    roots = [REPO, REPO / 'inference/lattisense', REPO / 'inference/lattisense/fhe_ops_lib/lattigo']
    revisions = {}
    for root in roots:
        revisions[str(root.relative_to(REPO))] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
        for name in names.decode().split('\0'):
            path = root / name
            if not name or not path.is_file() or path.suffix not in ('.py', '.cpp', '.h', '.cu', '.go', '.json', '.txt'):
                continue
            relative = path.relative_to(REPO).as_posix()
            if relative.startswith(('docs/report/', 'runs/', 'examples/test_')):
                continue
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'revisions': revisions, 'source_sha256': digest, 'files_sha256': hashes}


def check_structure(task):
    runner = task / 'runner'
    cfg = json.loads((runner / 'task_config.json').read_text())
    for key, expected in {'deployment_mode': 'server_provisioned_runner', 'input_mode': 'plaintext',
                          'parameter_mode': 'encrypted_offline', 'parameter_loading_mode': 'runtime_lazy'}.items():
        if cfg.get(key) != expected:
            raise ValueError(f'{key} != {expected}')
    sig = json.loads((runner / 'task_signature.json').read_text())
    manifest = json.loads((runner / 'encrypted_model_parameters/manifest.json').read_text())['arguments']
    parameters = {a['id']: a for a in sig['encrypted_parameters']}
    compiled = json.loads((runner / 'nn_layers_ct_0.json').read_text())
    if manifest.keys() != parameters.keys():
        raise ValueError('Manifest/signature parameter mismatch')
    for arg_id, arg in manifest.items():
        if arg['param_kind'] == 'mask' or arg_id.startswith(('convm_', 'repack_mask_')):
            raise ValueError(f'Public mask entered manifest: {arg_id}')
        if any(arg[k] != parameters[arg_id][k] for k in ('size', 'level', 'type')):
            raise ValueError(f'Manifest shape/level mismatch: {arg_id}')
        expected_scale = parameters[arg_id]['scale']
        if not math.isfinite(arg['scale']) or arg['scale'] <= 0 or not math.isclose(arg['scale'], expected_scale, rel_tol=1e-10):
            raise ValueError(f'Manifest scale mismatch: {arg_id}')
        if arg.get('layout') != parameters[arg_id].get('layout') or arg.get('layout') != layout_metadata(compiled, arg):
            raise ValueError(f'Manifest layout mismatch: {arg_id}')
    graph = json.loads((runner / 'mega_ag.json').read_text())
    loads = []
    ops = Counter()

    def visit(value):
        if isinstance(value, dict):
            if value.get('type') == 'load_encrypted_param_ct' and 'attributes' in value:
                loads.append(value['attributes'])
            for key, child in value.items():
                if key == 'type' and isinstance(child, str) and 'inputs' in value and 'outputs' in value:
                    ops[child] += 1
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(graph)
    actual = Counter((a['arg_id'], a['flat_index'], a['expected_level']) for a in loads)
    expected = Counter((name, i, arg['level']) for name, arg in parameters.items()
                       for i in range(int(np.prod(arg['size']))))
    if actual != expected:
        raise ValueError(f'Canonical load nodes differ from export: {len(actual)} vs {len(expected)}')
    return {'canonical_load_nodes': len(loads), 'graph_operations': dict(ops),
            'recovery_operations': recovery_operations(graph, manifest), 'public_masks_absent': True,
            'shape_level_scale_layout_match': True}


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.raw = args.report_dir / 'raw'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONPATH=f'{REPO}/inference/lattisense:{REPO}',
                        CUDA_VISIBLE_DEVICES=str(args.gpu_device))
        self.commands = []
        self.provenance = source_provenance()
        (args.report_dir / 'source_provenance.json').write_text(json.dumps(self.provenance, indent=2) + '\n')
        self.build_flags = dict(re.findall(r'^(INFERENCE_SDK_ENABLE_GPU|LATTISENSE_DEV):BOOL=(.*)$',
                                          (args.build_dir / 'CMakeCache.txt').read_text(), re.M))
        if any(self.build_flags.get(k) != 'ON' for k in ('INFERENCE_SDK_ENABLE_GPU', 'LATTISENSE_DEV')):
            raise ValueError('GPU acceptance requires INFERENCE_SDK_ENABLE_GPU=ON and LATTISENSE_DEV=ON')

    def run(self, command, label, cwd=REPO):
        command = [str(x) for x in command]
        log, timing = self.raw / f'{label}.log', self.raw / f'{label}-time.json'
        started = time.perf_counter()
        with log.open('w') as stream:
            result = subprocess.run(['/usr/bin/time', '-f', '{"peak_rss_kib":%M}', '-o', str(timing),
                                     *command], cwd=cwd, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
        record = {'command': command, 'cwd': str(cwd), 'returncode': result.returncode,
                  'wall_ms': (time.perf_counter() - started) * 1000, 'log': str(log)}
        try:
            record.update(json.loads(timing.read_text()))
        except (ValueError, FileNotFoundError):
            pass
        self.commands.append(record)
        (self.raw / f'{label}-command.json').write_text(json.dumps(record, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'Command failed; see {log}')
        return record

    def model(self, name):
        model_commands_start = len(self.commands)
        onnx, input_csv, count = MODELS[name]
        onnx, input_csv = REPO / onnx, REPO / input_csv
        results, plain_report = evaluate(onnx, input_csv, self.args.warmup, self.args.repeats)
        plain_csv = self.raw / f'{name}-plaintext.csv'
        write_logits(plain_csv, results)
        (self.raw / f'{name}-plaintext.log').write_text(json.dumps(plain_report, indent=2) + '\n')
        reference = read_logits(plain_csv)
        if len(reference) != count:
            raise ValueError('Unexpected plaintext logit count')
        report = {'model': name, 'onnx': str(onnx), 'input_csv': str(input_csv),
                  'deployment_mode': 'server_provisioned_runner', 'parameter_mode': 'encrypted_offline',
                  'parameter_loading_mode': 'runtime_lazy', 'gpu': True,
                  'seed_stride': json.loads(self.args.config.read_text())['seed_weight_subgrid_stride'],
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'source_sha256': self.provenance['source_sha256'],
                  'onnx_sha256': hashlib.sha256(onnx.read_bytes()).hexdigest(),
                  'input_csv_sha256': hashlib.sha256(input_csv.read_bytes()).hexdigest(),
                  'device': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                  'build_flags': self.build_flags,
                  'plaintext': plain_report, 'modes': {}, 'acceptance_passed': False}
        for mode in self.args.modes:
            start_command = len(self.commands)
            prefix = f'{name}-{mode}'
            output = self.args.output_root / name / mode
            task = output / 'task'
            config = json.loads(self.args.config.read_text())
            if mode == 'legacy':
                config['seed_weight_subgrid_stride'] = [1, 1]
            config_path = self.raw / f'{prefix}-config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            if not self.args.reuse_task:
                if task.exists():
                    raise ValueError(f'Use a new isolated output root or --reuse-task: {task}')
                self.run([sys.executable, REPO / 'training/run_compile.py', '--input', onnx, '--output', output,
                          '--style', 'multiplexed', '--deployment-mode', 'server_provisioned_runner',
                          '--config', config_path], f'{prefix}-compile')
                self.run([sys.executable, REPO / 'inference/interface/gen_mega_ag.py', '--task-dir', task,
                          '--deployment-mode', 'server_provisioned_runner', '--parameter-loading-mode', 'runtime_lazy'],
                         f'{prefix}-mega-ag')
                self.run([self.args.build_dir / 'tools/provision_encrypted_runner_bundle', '--task-dir', task,
                          '--out', task / 'runner'], f'{prefix}-provision')
            structure = check_structure(task)
            bundle = summarize(task)
            (self.raw / f'{prefix}-bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
            ckks = json.loads((task / 'runner/ckks_parameter.json').read_text())
            cfg = json.loads((task / 'runner/task_config.json').read_text())
            input_names = list(cfg['task_input_param'])
            if len(input_names) != 1:
                raise ValueError('Expected exactly one input')
            samples = []
            for i in range(self.args.warmup + self.args.repeats):
                label = f'{prefix}-{i}'
                cipher = output / 'output.ct'
                before = set(self.raw.glob('mem_usage_*.csv'))
                run = self.run([self.args.build_dir / 'examples/inference', '--task-dir', task / 'runner',
                                '--deployment-mode', 'server_provisioned_runner', '--input', f'{input_names[0]}={input_csv}',
                                '--output-cipher', cipher, '--gpu'], f'{label}-runner', cwd=self.raw)
                log_text = Path(run['log']).read_text()
                if 'fallback' in log_text.lower() or '[GPU-MEM]' not in log_text:
                    raise ValueError('Missing GPU execution evidence or CPU fallback')
                monitors, gpu_peak = {}, 0
                for source in set(self.raw.glob('mem_usage_*.csv')) - before:
                    kind = 'gpu' if 'gpu' in source.name else 'cpu'
                    target = self.raw / f'{label}-mem_usage_{kind}.csv'
                    source.rename(target)
                    monitors[kind] = str(target)
                    with target.open() as stream:
                        rows = list(csv.DictReader(stream))
                    if not rows:
                        raise ValueError(f'Empty monitor: {target}')
                    if kind == 'gpu':
                        gpu_peak = int(max(float(r['used_gb']) for r in rows) * 1024 ** 3)
                if monitors.keys() != {'cpu', 'gpu'}:
                    raise ValueError('Missing CPU/GPU monitor CSV')
                csv_path = self.raw / f'{label}-logits.csv'
                decrypt = self.run([self.args.build_dir / 'tools/decrypt_runner_output', '--task-dir', task / 'provisioner',
                                    '--cipher', cipher, '--output-csv', csv_path], f'{label}-decrypt')
                errors = metrics(reference, read_logits(csv_path))
                cache = re.findall(r'\[Runner\] Parameter cache: (\{.*\})', log_text)
                if not cache:
                    raise ValueError('Missing lazy cache statistics')
                core_time = re.findall(r'Run GPU time:\s*([0-9.]+)\s*ms', log_text)
                samples.append({'warmup': i < self.args.warmup, 'runner_wall_ms': run['wall_ms'],
                                'decrypt_wall_ms': decrypt['wall_ms'], 'e2e_fhe_wall_ms': run['wall_ms'] + decrypt['wall_ms'],
                                'cpu_peak_rss_bytes': run['peak_rss_kib'] * 1024, 'gpu_peak_used_bytes': gpu_peak,
                                'metrics': errors, 'cache': json.loads(cache[-1]), 'monitors': monitors,
                                'gpu_core_ms': float(core_time[-1]) if core_time else None,
                                'logits_csv': str(csv_path)})
            measured = [s for s in samples if not s['warmup']]
            result = {k: statistics.median(s[k] for s in measured) for k in (
                'runner_wall_ms', 'decrypt_wall_ms', 'e2e_fhe_wall_ms', 'cpu_peak_rss_bytes', 'gpu_peak_used_bytes')}
            result.update({k: bundle[k] for k in ('encrypted_parameter_payload_bytes', 'runner_bundle_bytes', 'task_directory_bytes')})
            result.update(samples=samples, bundle=bundle, structure=structure, ckks_parameter=ckks,
                          plaintext_wall_ms=plain_report['plaintext_wall_ms'], commands=self.commands[start_command:],
                          seed_stride=config['seed_weight_subgrid_stride'],
                          metrics=max((s['metrics'] for s in measured), key=lambda m: m['max_abs_error']))
            result['timing_scope'] = 'Process wall time includes key/model loading and CUDA initialization; decrypt is a separate process. Plaintext inference excludes session preparation (reported separately).'
            report['modes'][mode] = result
            path = self.args.report_dir / f'{name}_runtime_lazy.json'
            path.write_text(json.dumps(report, indent=2) + '\n')
        if set(report['modes']) == {'legacy', 'compressed'}:
            legacy = report['modes']['legacy']
            compressed = report['modes']['compressed']
            baseline = max(s['metrics']['max_abs_error'] for s in legacy['samples'])
            passed = all(s['metrics']['max_abs_error'] <= 1e-2 and s['metrics']['rmse'] <= 1e-3
                         and s['metrics']['cosine_similarity'] >= 0.9999 and s['metrics']['top1_match']
                         and s['metrics']['max_abs_error'] <= 2 * baseline + 1e-6
                         for s in compressed['samples'])
            report.update({k: compressed[k] for k in ('encrypted_parameter_payload_bytes', 'runner_bundle_bytes',
                          'task_directory_bytes', 'cpu_peak_rss_bytes', 'gpu_peak_used_bytes', 'runner_wall_ms',
                          'decrypt_wall_ms', 'e2e_fhe_wall_ms', 'plaintext_wall_ms', 'metrics', 'ckks_parameter')})
            report['commands'] = self.commands[model_commands_start:]
            report['acceptance_passed'] = passed and self.args.warmup >= 1 and self.args.repeats >= 3
            path.write_text(json.dumps(report, indent=2) + '\n')
            if not passed:
                raise ValueError(f'Numerical acceptance failed; see {path}')
        print(f'{name}: report saved to {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, action='append')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=REPO / 'docs/report')
    parser.add_argument('--build-dir', type=Path, default=REPO / 'build')
    parser.add_argument('--config', type=Path, default=REPO / 'training/config/storage_aware_config.json')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=['legacy', 'compressed'], default=['legacy', 'compressed'])
    parser.add_argument('--reuse-task', action='store_true', help='Reuse an already compiled/provisioned isolated task')
    parser.add_argument('--delete-tasks', action='store_true', help='Delete only generated tasks after a successful run')
    args = parser.parse_args()
    for key in ('output_root', 'report_dir', 'build_dir', 'config'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_root == REPO or (REPO / 'runs') in (args.output_root, *args.output_root.parents):
        parser.error('output-root must be an isolated directory outside existing runs')
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative and repeats positive')
    bench = Benchmark(args)
    for model in args.model or MODELS:
        bench.model(model)
        if args.delete_tasks:
            for mode in args.modes:
                shutil.rmtree(args.output_root / model / mode / 'task')


if __name__ == '__main__':
    main()
