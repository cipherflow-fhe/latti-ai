#!/usr/bin/env python3
# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run a GPU command in its artifact directory and record memory and FHE time."""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool_files_before = set(out.glob('mem_usage_gpu_*.csv'))
    record = {'command': command, 'cwd': str(out), 'env': {
        k: os.environ[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONPATH']
        if k in os.environ}}
    (out / 'inference.command.json').write_text(json.dumps(record, indent=2) + '\n')
    cpu_peak = gpu_peak = 0
    started = time.monotonic()
    with (out / 'inference.log').open('w') as log, (out / 'inference.memory.csv').open('w') as memory:
        writer = csv.writer(memory)
        writer.writerow(['elapsed_s', 'tree_rss_bytes', 'gpu_pid_used_bytes'])
        proc = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out / 'inference.time'), *command],
                                cwd=out, stdout=log, stderr=subprocess.STDOUT)
        print(f'PID {proc.pid}: {out}', flush=True)
        while True:
            processes = []
            try:
                parent = psutil.Process(proc.pid)
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                pass
            pids = {p.pid for p in processes}
            cpu = 0
            for process in processes:
                try:
                    cpu += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            gpu = 0
            records = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            for row in csv.reader(records.stdout.splitlines()):
                if len(row) == 2 and int(row[0]) in pids:
                    gpu += int(row[1]) * 1024**2
            cpu_peak, gpu_peak = max(cpu_peak, cpu), max(gpu_peak, gpu)
            writer.writerow([time.monotonic() - started, cpu, gpu])
            memory.flush()
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    text = (out / 'inference.log').read_text(errors='replace')
    times = re.findall(r'Run GPU time:\s*([\d.eE+\-]+)\s*ms', text)
    time_text = (out / 'inference.time').read_text()
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', time_text)
    pool_peak = 0.0
    pool_files = sorted(set(out.glob('mem_usage_gpu_*.csv')) - pool_files_before)
    for path in pool_files:
        with path.open() as stream:
            pool_peak = max(pool_peak, max((float(row['used_gb']) for row in csv.DictReader(stream)), default=0))
    report = dict(record, exit_code=proc.returncode, wall_s=time.monotonic() - started,
                  cpu_peak_rss_bytes=int(rss[1]) * 1024 if rss else cpu_peak,
                  sampled_cpu_tree_peak_bytes=cpu_peak, gpu_pid_peak_bytes=gpu_peak,
                  gpu_pool_peak_gib=pool_peak, gpu_pool_csv=[str(p) for p in pool_files],
                  fhe_s=sum(float(t) for t in times) / 1000 if times else None,
                  sample_interval_s=0.5, gpu_pool_sample_interval_s=0.1)
    (out / 'inference.metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
