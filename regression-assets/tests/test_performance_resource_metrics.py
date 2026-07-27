from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import resource_monitor
from tests.performance_benchmark import (
    RESOURCE_METRICS_JSONL,
    build_benchmark_steps,
    load_resource_delta_summaries,
)
from tests.regression_config import RegressionConfig
from tests.report_writer import _performance_resource_lines

pytestmark = pytest.mark.regression_scope('performance')


def _config(tmp_path: Path, *, cuda_arch: str | None = None) -> RegressionConfig:
    return RegressionConfig(
        repo_root=tmp_path,
        log_dir=tmp_path / 'logs',
        scope='performance',
        enabled_stages={'environment', 'performance', 'report'},
        cuda_arch=cuda_arch,
        continue_on_error=True,
        python_bin='python3',
        install_python_deps=False,
        baseline=None,
        target='target-commit',
        baseline_log_dir=None,
        performance_regression_threshold=10.0,
        skip_performance=False,
        skip_docker=False,
        skip_lattisense=False,
        benchmark_modules='cpu',
        benchmark_repeat=2,
        benchmark_timeout_seconds=3600,
        cccl_source_dir=None,
    )


def test_operator_benchmark_run_steps_do_not_use_resource_monitor(tmp_path: Path) -> None:
    steps = build_benchmark_steps(
        _config(tmp_path, cuda_arch='89'),
        ['cpu', 'gpu', 'conv-cpu', 'conv-gpu'],
        repeat=2,
        timeout_seconds=3600,
    )

    run_steps = [step for step in steps if step.name.endswith('-run')]

    assert run_steps
    assert all('tests/resource_monitor.py' not in str(step.command) for step in run_steps)
    assert any('/usr/bin/time -f "wall_seconds=%e"' in str(step.command) for step in run_steps)


def test_examples_benchmark_run_steps_wrap_each_repeat_with_resource_monitor(tmp_path: Path) -> None:
    steps = build_benchmark_steps(
        _config(tmp_path, cuda_arch='89'),
        ['examples-cpu', 'examples-gpu'],
        repeat=2,
        timeout_seconds=3600,
    )

    cpu_step = next(step for step in steps if step.name == 'benchmark-examples-cpu-mnist-run')
    gpu_step = next(step for step in steps if step.name == 'benchmark-examples-gpu-mnist-run')

    assert 'tests/resource_monitor.py' in str(cpu_step.command)
    assert '--step benchmark-examples-cpu-mnist-run' in str(cpu_step.command)
    assert '--repeat "$i"' in str(cpu_step.command)
    assert f'--output "$REGRESSION_LOG_DIR/{RESOURCE_METRICS_JSONL}"' in str(cpu_step.command)
    assert '--configured-threads' not in str(cpu_step.command)
    assert '--gpu false' in str(cpu_step.command)
    assert 'ctest' not in str(cpu_step.command)
    assert './examples/inference' in str(cpu_step.command)
    assert 'examples/test_mnist/task' in str(cpu_step.command)
    assert '--gpu true' in str(gpu_step.command)
    assert 'ctest' not in str(gpu_step.command)
    assert './examples/inference' in str(gpu_step.command)
    assert 'examples/test_mnist/task' in str(gpu_step.command)
    assert ' --gpu' in str(gpu_step.command)


def test_load_resource_delta_summaries_uses_openmp_thread_values(tmp_path: Path) -> None:
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    records = [
        {
            'step': 'benchmark-cpu-bfv-mult-relin-run',
            'repeat': 1,
            'exit_code': 0,
            'wall_seconds': 1.0,
            'detected_threads': 57,
            'openmp_actual_threads': 8,
            'memory_delta_peak_kib': 1000,
            'gpu_memory_delta_peak_mib': None,
            'gpu_sampled': False,
        },
        {
            'step': 'benchmark-cpu-bfv-mult-relin-run',
            'repeat': 2,
            'exit_code': 0,
            'wall_seconds': 1.2,
            'detected_threads': 64,
            'openmp_actual_threads': 16,
            'memory_delta_peak_kib': 2048,
            'gpu_memory_delta_peak_mib': None,
            'gpu_sampled': False,
        },
        {
            'step': 'benchmark-gpu-bfv-mult-relin-run',
            'repeat': 1,
            'exit_code': 0,
            'wall_seconds': 2.0,
            'detected_threads': 71,
            'openmp_actual_threads': 12,
            'memory_delta_peak_kib': 4096,
            'gpu_memory_delta_peak_mib': 512,
            'gpu_sampled': True,
        },
    ]
    (log_dir / RESOURCE_METRICS_JSONL).write_text(
        ''.join(json.dumps(record) + '\n' for record in records),
        encoding='utf-8',
    )

    summaries = load_resource_delta_summaries(log_dir)

    cpu = summaries['benchmark-cpu-bfv-mult-relin-run']
    assert cpu.repeat_count == 2
    assert cpu.openmp_actual_threads == 16
    assert cpu.memory_delta_peak_kib == 2048
    assert cpu.gpu_memory_delta_peak_mib is None
    assert cpu.gpu_sampled is False

    gpu = summaries['benchmark-gpu-bfv-mult-relin-run']
    assert gpu.repeat_count == 1
    assert gpu.openmp_actual_threads == 12
    assert gpu.gpu_memory_delta_peak_mib == 512
    assert gpu.gpu_sampled is True


def test_resource_monitor_parses_openmp_thread_markers() -> None:
    lines = [
        'regular inference output',
        'latti_ai_openmp_actual_threads=8',
        'latti_ai_openmp_actual_threads=16',
    ]

    assert resource_monitor._max_openmp_actual_threads(lines) == 16


def test_performance_resource_lines_render_resource_delta_table(tmp_path: Path) -> None:
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    (log_dir / RESOURCE_METRICS_JSONL).write_text(
        json.dumps(
            {
                'step': 'benchmark-gpu-bfv-mult-relin-run',
                'repeat': 1,
                'exit_code': 0,
                'wall_seconds': 2.0,
                'detected_threads': 57,
                'openmp_actual_threads': 16,
                'memory_delta_peak_kib': 2048,
                'gpu_memory_delta_peak_mib': 512,
                'gpu_sampled': True,
            }
        )
        + '\n',
        encoding='utf-8',
    )

    lines = _performance_resource_lines(log_dir)

    rendered = '\n'.join(lines)
    assert '### 3.3 Examples 资源增量' in rendered
    assert 'OpenMP线程数' in rendered
    assert '识别线程数' not in rendered
    assert (
        '| gpu | benchmark-gpu-bfv-mult-relin-run | 1 | 16 | 2.000 | 512.000 | `performance.log` |'
        in rendered
    )