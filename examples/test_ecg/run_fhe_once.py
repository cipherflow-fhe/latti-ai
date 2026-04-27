import os
import json
import argparse
import subprocess
from pathlib import Path

"""
Single Sample FHE Inference Execution Module.

This module runs Fully Homomorphic Encryption (FHE) inference on a single ECG sample,
executing the compiled FHE binary with verification mode enabled. It captures the complete
output including encrypted/plaintext logits and verification details, saves the execution
log to a file, and generates a summary JSON with command details and return code for
debugging and validation purposes.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=str, default='./test_ecg/runs/exp_over009/task')
    parser.add_argument('--input-csv', type=str, default='./test_ecg/runs/exp_over009/task/client/ecg_input.csv')
    parser.add_argument('--binary', type=str, default='./build/examples/inference')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--save-log', type=str, default='./test_ecg/runs/exp_over009/task/client/fhe_run.log')
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f'找不到推理二进制: {binary}')

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(args.threads)
    env['OPENBLAS_NUM_THREADS'] = str(args.threads)
    env['MKL_NUM_THREADS'] = str(args.threads)
    env['NUMEXPR_NUM_THREADS'] = str(args.threads)

    cmd = [
        str(binary),
        '--task-dir', args.task_dir,
        '--input', args.input_csv,
        '--verify'
    ]

    print('Running command:')
    print(' '.join(cmd))
    print(f'OMP_NUM_THREADS={env["OMP_NUM_THREADS"]}')
    print()

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )

    output = proc.stdout
    print(output)

    log_path = Path(args.save_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding='utf-8')

    summary = {
        'command': cmd,
        'return_code': proc.returncode,
        'log_path': str(log_path),
    }

    summary_path = log_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n========== Run Summary ==========')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
