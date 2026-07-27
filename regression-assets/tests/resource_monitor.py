from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SAMPLE_INTERVAL_SECONDS = 0.1
OPENMP_ACTUAL_THREADS_PREFIX = 'latti_ai_openmp_actual_threads='


@dataclass(frozen=True)
class GpuMemorySample:
    used_mib: int | None
    note: str


@dataclass(frozen=True)
class ResourcePeaks:
    memory_delta_peak_kib: int | None
    gpu_memory_delta_peak_mib: int | None
    gpu_sampled: bool
    gpu_note: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run one examples benchmark case and record resource deltas.'
    )
    parser.add_argument('--step', required=True, help='Benchmark step name.')
    parser.add_argument('--repeat', required=True, type=int, help='Benchmark repeat index.')
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='JSONL output file for resource metrics.',
    )
    parser.add_argument(
        '--gpu',
        choices=('true', 'false'),
        required=True,
        help='Whether to sample GPU memory for this examples case.',
    )
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Command to execute after --.',
    )
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('missing benchmark command after --')

    sample_gpu = args.gpu == 'true'
    baseline_memory_kib = _system_used_memory_kib()
    baseline_gpu_memory = _system_gpu_memory_used_mib() if sample_gpu else GpuMemorySample(None, '')

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    output_thread = threading.Thread(
        target=_forward_process_output,
        args=(process, output_lines),
        daemon=True,
    )
    output_thread.start()
    peaks = _sample_until_exit(
        process,
        sample_gpu=sample_gpu,
        baseline_memory_kib=baseline_memory_kib,
        baseline_gpu_memory_mib=baseline_gpu_memory.used_mib,
        initial_gpu_note=baseline_gpu_memory.note,
    )
    exit_code = process.wait()
    output_thread.join()
    wall_seconds = time.monotonic() - started
    openmp_actual_threads = _max_openmp_actual_threads(output_lines)

    record = {
        'step': args.step,
        'repeat': args.repeat,
        'command': _command_to_text(command),
        'exit_code': exit_code,
        'wall_seconds': wall_seconds,
        'openmp_actual_threads': openmp_actual_threads,
        'baseline_memory_kib': baseline_memory_kib,
        'memory_delta_peak_kib': peaks.memory_delta_peak_kib,
        'baseline_gpu_memory_mib': baseline_gpu_memory.used_mib,
        'gpu_memory_delta_peak_mib': peaks.gpu_memory_delta_peak_mib,
        'gpu_sampled': peaks.gpu_sampled,
        'gpu_note': peaks.gpu_note,
    }
    _append_jsonl(args.output, record)
    _print_metrics(record)
    return exit_code


def _forward_process_output(process: subprocess.Popen[str], output_lines: list[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        output_lines.append(line)
        print(line, end='', flush=True)


def _sample_until_exit(
    process: subprocess.Popen[str],
    *,
    sample_gpu: bool,
    baseline_memory_kib: int | None,
    baseline_gpu_memory_mib: int | None,
    initial_gpu_note: str,
) -> ResourcePeaks:
    memory_delta_peak_kib: int | None = None
    gpu_memory_delta_peak_mib: int | None = None
    gpu_note = initial_gpu_note

    while process.poll() is None:
        memory_delta_peak_kib = _max_optional(
            memory_delta_peak_kib,
            _memory_delta_kib(baseline_memory_kib, _system_used_memory_kib()),
        )
        if sample_gpu:
            gpu_sample = _system_gpu_memory_used_mib()
            if gpu_sample.note:
                gpu_note = gpu_sample.note
            gpu_memory_delta_peak_mib = _max_optional(
                gpu_memory_delta_peak_mib,
                _memory_delta_mib(baseline_gpu_memory_mib, gpu_sample.used_mib),
            )
        time.sleep(SAMPLE_INTERVAL_SECONDS)

    memory_delta_peak_kib = _max_optional(
        memory_delta_peak_kib,
        _memory_delta_kib(baseline_memory_kib, _system_used_memory_kib()),
    )
    if sample_gpu:
        gpu_sample = _system_gpu_memory_used_mib()
        if gpu_sample.note:
            gpu_note = gpu_sample.note
        gpu_memory_delta_peak_mib = _max_optional(
            gpu_memory_delta_peak_mib,
            _memory_delta_mib(baseline_gpu_memory_mib, gpu_sample.used_mib),
        )

    return ResourcePeaks(
        memory_delta_peak_kib=memory_delta_peak_kib,
        gpu_memory_delta_peak_mib=gpu_memory_delta_peak_mib,
        gpu_sampled=sample_gpu,
        gpu_note=gpu_note,
    )


def _max_openmp_actual_threads(lines: Sequence[str]) -> int | None:
    values: list[int] = []
    for line in lines:
        value = _parse_openmp_actual_thread_line(line)
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _parse_openmp_actual_thread_line(line: str) -> int | None:
    stripped = line.strip()
    if not stripped.startswith(OPENMP_ACTUAL_THREADS_PREFIX):
        return None
    raw_value = stripped.removeprefix(OPENMP_ACTUAL_THREADS_PREFIX)
    try:
        return int(raw_value)
    except ValueError:
        return None


def _system_used_memory_kib() -> int | None:
    values: dict[str, int] = {}
    try:
        lines = Path('/proc/meminfo').read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, raw_value = line.partition(':')
        parts = raw_value.split()
        if not parts:
            continue
        try:
            values[name] = int(parts[0])
        except ValueError:
            continue
    mem_total = values.get('MemTotal')
    mem_available = values.get('MemAvailable')
    if mem_total is None or mem_available is None:
        return None
    return max(0, mem_total - mem_available)


def _system_gpu_memory_used_mib() -> GpuMemorySample:
    try:
        completed = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return GpuMemorySample(None, 'nvidia-smi not found')
    if completed.returncode != 0:
        note = completed.stderr.strip() or f'nvidia-smi exit {completed.returncode}'
        return GpuMemorySample(None, note)

    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return GpuMemorySample(sum(values) if values else None, '')


def _memory_delta_kib(baseline_kib: int | None, current_kib: int | None) -> int | None:
    if baseline_kib is None or current_kib is None:
        return None
    return max(0, current_kib - baseline_kib)


def _memory_delta_mib(baseline_mib: int | None, current_mib: int | None) -> int | None:
    if baseline_mib is None or current_mib is None:
        return None
    return max(0, current_mib - baseline_mib)


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def _print_metrics(record: dict[str, object]) -> None:
    print(f'wall_seconds={float(record["wall_seconds"]):.3f}', flush=True)
    print(
        f'resource_openmp_actual_threads={_format_optional(record["openmp_actual_threads"])}',
        flush=True,
    )
    memory_delta_peak_kib = record['memory_delta_peak_kib']
    if isinstance(memory_delta_peak_kib, int):
        print(
            f'resource_memory_delta_peak_mib={memory_delta_peak_kib / 1024:.3f}',
            flush=True,
        )
    else:
        print('resource_memory_delta_peak_mib=N/A', flush=True)
    gpu_memory_delta_peak_mib = record['gpu_memory_delta_peak_mib']
    if isinstance(gpu_memory_delta_peak_mib, int):
        print(
            f'resource_gpu_memory_delta_peak_mib={gpu_memory_delta_peak_mib:.3f}',
            flush=True,
        )
    else:
        print('resource_gpu_memory_delta_peak_mib=N/A', flush=True)
    gpu_note = str(record.get('gpu_note') or '')
    if gpu_note:
        print(f'resource_gpu_note={gpu_note}', flush=True)


def _format_optional(value: object) -> str:
    if value is None:
        return 'N/A'
    return str(value)


def _max_optional(current: int | None, value: int | None) -> int | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _command_to_text(command: Sequence[str]) -> str:
    return ' '.join(command)


if __name__ == '__main__':
    raise SystemExit(main())
