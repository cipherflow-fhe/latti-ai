"""Prepare the requested GPU validation tasks; keep all outputs beside each model."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
python = '/home/linghm/pytorch_venv/bin/python3'
env = dict(os.environ, PYTHONPATH=f'{repo}/inference/lattisense:{repo}',
           OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
models = {
    'cifar10': ('trained_poly.onnx', 'CIFAR10', ['img.csv']),
    'imagenet': ('trained_poly.onnx', 'ImageNet', ['img.csv']),
    'retina': ('epoch041.onnx', 'Retina', ['input_0=img.csv', 'input_1=non_inv.csv', 'input_2=inv.csv']),
    'yolov5': ('best_conv_outputs.onnx', 'YOLOv5', ['img.csv']),
    'BERT': ('trained_poly.onnx', 'BERT', ['input.csv']),
    'ViT': ('trained_poly.onnx', 'ViT', ['patch_input.csv']),
}
plan = []
for name, (model, config, inputs) in models.items():
    root = repo / 'runs' / name
    variants = [('input', 'pcm'), ('parameter', 'pcm'), ('input', 'pdm')] if name in ['BERT', 'ViT'] else [('input', ''), ('parameter', '')]
    for mode, packing in variants:
        out = root / f'task_{mode}_encrypted'
        if packing:
            out = out / packing
        if packing:
            config_name = (f'{config}_{packing}_config.json' if packing == 'pcm' else
                           ('BERT_base_config.json' if name == 'BERT' else 'ViT_tiny_config.json'))
        elif mode == 'parameter' and name in ['cifar10', 'imagenet', 'retina', 'yolov5']:
            config_name = 'storage_aware_config.json'
        else:
            config_name = ''
        task = out / 'task'
        compile_cmd = [python, str(repo / 'training/run_compile.py'), '--input', str(root / 'model' / model),
                       '--output', str(out), '--style', 'multiplexed']
        if config_name:
            compile_cmd += ['--config', str(repo / 'training/config' / config_name)]
        graph_cmd = [python, str(repo / 'inference/interface/gen_mega_ag.py'), '--task-dir', str(task)]
        if mode == 'parameter':
            compile_cmd += ['--deployment-mode', 'server_provisioned_runner']
            graph_cmd += ['--deployment-mode', 'server_provisioned_runner', '--parameter-loading-mode', 'runtime_lazy']
        bindings = []
        for value in inputs:
            key, sep, filename = value.partition('=')
            bindings.append(f'{key}={root / filename}' if sep else str(root / value))
        plan.append(dict(id=f'{name}:{mode}' + (f':{packing}' if packing else ''), model=name, mode=mode,
                         packing=packing, out=str(out), task=str(task), onnx=str(root / 'model' / model),
                         inputs=bindings, compile=compile_cmd, graph=graph_cmd))
(repo / 'runs/gpu_pipeline_tasks.json').write_text(json.dumps(plan, indent=2) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('--exclude', action='append', default=[])
parser.add_argument('--only', action='append', default=[])
args = parser.parse_args()
failed = []
for item in plan:
    if item['id'] in args.exclude or (args.only and item['id'] not in args.only):
        continue
    out = Path(item['out'])
    out.mkdir(parents=True, exist_ok=True)
    for stage in ['compile', 'graph']:
        record = out / f'{stage}.command.json'
        config = {}
        if '--config' in item['compile']:
            config = json.loads(Path(item['compile'][item['compile'].index('--config') + 1]).read_text())
        target = Path(item['task']) / ('runner' if item['mode'] == 'parameter' else 'server')
        expected = Path(item['task']) / 'server/task_config.json' if stage == 'compile' else target / 'mega_ag.json'
        if record.exists() and expected.exists():
            old = json.loads(record.read_text())
            if old.get('exit_code') == 0 and old['command'] == item[stage] and old.get('config') == config:
                print(f'SKIP {item["id"]} {stage}', flush=True)
                continue
        data = dict(command=item[stage], cwd=str(out), config=config)
        record.write_text(json.dumps(data, indent=2) + '\n')
        print(f'START {item["id"]} {stage}', flush=True)
        with (out / f'{stage}.log').open('w') as log:
            result = subprocess.run(item[stage], cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        data['exit_code'] = result.returncode
        record.write_text(json.dumps(data, indent=2) + '\n')
        print(f'END {item["id"]} {stage} exit={result.returncode}', flush=True)
        if result.returncode:
            failed.append([item['id'], stage])
            break
print(json.dumps({'failed': failed}), flush=True)
sys.exit(bool(failed))
