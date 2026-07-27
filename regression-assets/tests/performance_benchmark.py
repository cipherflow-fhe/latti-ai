from __future__ import annotations

import json
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tests.regression_config import RegressionConfig
from tests.regression_runner import CommandStep, StepResult


PERFORMANCE_LOG = 'performance.log'
RESOURCE_METRICS_JSONL = 'performance-resource-metrics.jsonl'
BENCHMARK_RESULTS_JSONL = 'performance-benchmark-results.jsonl'
BENCHMARK_SUMMARY_MD = 'performance-benchmark.md'
PERFORMANCE_COMPARISON_JSON = 'performance-comparison.json'
PERFORMANCE_REPORT_MD = 'performance.md'
PERFORMANCE_OUTPUT_ARTIFACTS = (
    BENCHMARK_RESULTS_JSONL,
    BENCHMARK_SUMMARY_MD,
    PERFORMANCE_COMPARISON_JSON,
    PERFORMANCE_REPORT_MD,
)


@dataclass(frozen=True)
class PerformanceMetric:
    key: str
    scene: str
    metric: str
    value: float | None
    unit: str
    source_step: str
    status: str
    note: str = ''
    lower_is_better: bool = True


@dataclass(frozen=True)
class PerformanceComparison:
    key: str
    scene: str
    metric: str
    unit: str
    baseline_value: float | None
    target_value: float | None
    change_percent: float | None
    exceeds_threshold: bool | None
    conclusion: str
    note: str


@dataclass(frozen=True)
class PerformanceReportSummary:
    baseline_log_dir: str
    target_log_dir: str
    threshold_percent: float
    total: int
    passed: int
    regressions: int
    insufficient: int
    failed: int
    conclusion: str
    comparisons: list[PerformanceComparison]


@dataclass(frozen=True)
class ResourceDeltaSummary:
    step: str
    repeat_count: int
    openmp_actual_threads: int | None
    memory_delta_peak_kib: int | None
    gpu_memory_delta_peak_mib: int | None
    gpu_sampled: bool


ALL_BENCHMARK_MODULES = (
    'cpu',
    'gpu',
    'conv-cpu',
    'conv-gpu',
    'examples-cpu',
    'examples-gpu',
)
GPU_BENCHMARK_MODULES = {'gpu', 'conv-gpu', 'examples-gpu'}
CONVOLUTION_BENCHMARK_CONFIGS = (
    (4, 5, 1, 1),
    (8, 5, 1, 1),
    (16, 5, 1, 1),
    (32, 5, 1, 1),
    (64, 5, 1, 1),
    (32, 3, 1, 32),
    (32, 3, 4, 4),
    (32, 3, 32, 1),
    (16, 1, 1, 1),
    (16, 3, 1, 1),
    (16, 5, 1, 1),
)


def parse_benchmark_modules(raw_modules: str | None) -> list[str]:
    raw = raw_modules or 'all'
    requested = [item.strip() for item in raw.split(',') if item.strip()]
    if not requested or requested == ['all']:
        return list(ALL_BENCHMARK_MODULES)

    valid = set(ALL_BENCHMARK_MODULES)
    modules: list[str] = []
    for module in requested:
        if module == 'all':
            raise ValueError('Unsupported benchmark module mix: use either all or a comma-separated module list')
        if module not in valid:
            choices = ', '.join(('all', *ALL_BENCHMARK_MODULES))
            raise ValueError(f'Unsupported benchmark module {module!r}; expected one of: {choices}')
        if module not in modules:
            modules.append(module)
    return modules


def build_benchmark_steps(
    config: RegressionConfig,
    modules: Sequence[str],
    *,
    repeat: int,
    timeout_seconds: int | None,
) -> list[CommandStep]:
    if repeat < 1:
        raise ValueError('--benchmark-repeat must be at least 1')

    selected = list(modules)
    gpu_modules = [module for module in selected if module in GPU_BENCHMARK_MODULES]
    if gpu_modules and not config.cuda_arch:
        joined = ', '.join(gpu_modules)
        raise ValueError(f'Benchmark modules {joined} requires --cuda-arch')

    steps: list[CommandStep] = []
    lattisense_root = config.repo_root / 'inference' / 'lattisense'
    python_env = {'PYTHONPATH': str(lattisense_root)}

    cpu_targets: list[str] = []
    if 'cpu' in selected:
        cpu_targets.append('benchmark_cpu')
    if 'conv-cpu' in selected:
        cpu_targets.append('benchmark_convolution_cpu')
    if cpu_targets:
        steps.extend(_build_lattisense_cpu_steps(lattisense_root, cpu_targets, timeout_seconds))

    gpu_targets: list[str] = []
    if 'gpu' in selected:
        gpu_targets.append('benchmark_gpu')
    if 'conv-gpu' in selected:
        gpu_targets.append('benchmark_convolution_gpu')
    if gpu_targets:
        steps.extend(_build_lattisense_gpu_steps(lattisense_root, gpu_targets, config.cuda_arch or '', timeout_seconds))

    if 'cpu' in selected:
        steps.extend(_cpu_benchmark_steps(lattisense_root, config, repeat, timeout_seconds, python_env))
    if 'gpu' in selected:
        steps.extend(_gpu_benchmark_steps(lattisense_root, config, repeat, timeout_seconds, python_env))

    conv_dir = lattisense_root / 'examples' / 'benchmark_convolution'
    if 'conv-cpu' in selected or 'conv-gpu' in selected:
        steps.append(
            CommandStep(
                name='benchmark-conv-generate-tasks',
                command=f'{config.python_bin} benchmark_convolution.py --all',
                cwd=conv_dir,
                env=python_env,
                log_name=PERFORMANCE_LOG,
                timeout_seconds=timeout_seconds,
            )
        )
    if 'conv-cpu' in selected:
        steps.extend(_convolution_benchmark_steps('cpu', conv_dir, repeat, timeout_seconds))
    if 'conv-gpu' in selected:
        steps.extend(_convolution_benchmark_steps('gpu', conv_dir, repeat, timeout_seconds))

    if 'examples-cpu' in selected:
        steps.extend(
            CommandStep(
                name=f'benchmark-examples-cpu-{example}-run',
                command=_monitored_repeat_command(
                    _example_inference_command(config.repo_root, example, gpu=False),
                    repeat,
                    step_name=f'benchmark-examples-cpu-{example}-run',
                    config=config,
                    gpu=False,
                ),
                cwd=config.repo_root / 'build-cpu',
                log_name=PERFORMANCE_LOG,
                timeout_seconds=timeout_seconds,
            )
            for example in ('mnist', 'cifar10')
        )
    if 'examples-gpu' in selected:
        steps.extend(
            CommandStep(
                name=f'benchmark-examples-gpu-{example}-run',
                command=_monitored_repeat_command(
                    _example_inference_command(config.repo_root, example, gpu=True),
                    repeat,
                    step_name=f'benchmark-examples-gpu-{example}-run',
                    config=config,
                    gpu=True,
                ),
                cwd=config.repo_root / 'build-gpu',
                log_name=PERFORMANCE_LOG,
                timeout_seconds=timeout_seconds,
            )
            for example in ('mnist', 'cifar10', 'imagenet')
        )

    return steps


def is_performance_step(result: StepResult) -> bool:
    return result.name.startswith('benchmark-') or Path(result.log_path).name == PERFORMANCE_LOG


def module_for_step(step_name: str) -> str:
    return _module_for_step(step_name)


def remove_legacy_performance_artifacts(log_dir: Path) -> None:
    for name in PERFORMANCE_OUTPUT_ARTIFACTS:
        path = log_dir / name
        if path.exists():
            path.unlink()


def _load_benchmark_records(
    log_dir: Path,
    results: Sequence[StepResult] | None = None,
) -> dict[str, dict[str, object]]:
    if results is not None:
        records = _records_from_step_results(results)
        if records:
            return records

    records = _load_benchmark_records_from_log(log_dir)
    if records:
        return records

    return _load_benchmark_records_from_jsonl(log_dir)


def _records_from_step_results(results: Sequence[StepResult]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for result in results:
        if not is_performance_step(result):
            continue
        records[result.name] = {
            'module': _module_for_step(result.name),
            'step': result.name,
            'name': result.name,
            'exit_code': result.exit_code,
            'duration_seconds': result.duration_seconds,
            'critical': result.critical,
        }
    return records


def _load_benchmark_records_from_jsonl(log_dir: Path) -> dict[str, dict[str, object]]:
    records_path = log_dir / BENCHMARK_RESULTS_JSONL
    if not records_path.exists():
        return {}

    records: dict[str, dict[str, object]] = {}
    with records_path.open('r', encoding='utf-8') as records_file:
        for line in records_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            step_name = str(payload.get('step') or payload.get('name') or '')
            if step_name:
                records[step_name] = payload
    return records


def _load_benchmark_records_from_log(log_dir: Path) -> dict[str, dict[str, object]]:
    log_path = log_dir / PERFORMANCE_LOG
    if not log_path.exists():
        return {}

    records: dict[str, dict[str, object]] = {}
    current_step = ''
    current_exit_code: int | None = None
    with log_path.open('r', encoding='utf-8') as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            step_match = _STEP_RE.match(line)
            if step_match:
                current_step = step_match.group('step')
                current_exit_code = None
                continue
            if not current_step:
                continue
            if line.startswith('Exit code:'):
                try:
                    current_exit_code = int(line.split(':', 1)[1].strip())
                except ValueError:
                    current_exit_code = 1
                continue
            if line.startswith('Duration seconds:'):
                try:
                    duration_seconds = float(line.split(':', 1)[1].strip())
                except ValueError:
                    duration_seconds = 0.0
                records[current_step] = {
                    'module': _module_for_step(current_step),
                    'step': current_step,
                    'name': current_step,
                    'exit_code': 0 if current_exit_code is None else current_exit_code,
                    'duration_seconds': duration_seconds,
                    'critical': True,
                }
    return records


def _status_for_record(record: dict[str, object]) -> str:
    exit_code = int(record.get('exit_code', 1))
    critical = bool(record.get('critical', True))
    if exit_code == 0:
        return 'PASS'
    if critical:
        return 'FAIL'
    return 'NON_BLOCKING_FAIL'


def _metric_key(scene: str, metric: str, unit: str) -> str:
    return f'{scene}|{metric}|{unit}'


_OPERATOR_RE = re.compile(
    r'^(?P<scheme>BFV|CKKS) (?P<operation>[^:]+): '
    r'(?P<ops>\d+) ops, (?P<duration_ms>\d+(?:\.\d+)?) ms, (?P<ops_per_sec>\d+(?:\.\d+)?) ops/sec$'
)
_WALL_SECONDS_RE = re.compile(r'^wall_seconds=(?P<seconds>\d+(?:\.\d+)?)$')
_STEP_RE = re.compile(r'^Step: (?P<step>.+)$')


def _parse_performance_log(log_dir: Path, pass_steps: set[str]) -> dict[str, list[tuple[str, float, str, bool]]]:
    log_path = log_dir / PERFORMANCE_LOG
    if not log_path.exists():
        return {}

    values: dict[str, list[tuple[str, float, str, bool]]] = defaultdict(list)
    current_step = ''
    with log_path.open('r', encoding='utf-8') as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            step_match = _STEP_RE.match(line)
            if step_match:
                current_step = step_match.group('step')
                continue
            if current_step not in pass_steps:
                continue

            wall_match = _WALL_SECONDS_RE.match(line)
            if wall_match:
                scene = current_step
                values[_metric_key(scene, 'wall_seconds平均', 's')].append(
                    (scene, float(wall_match.group('seconds')), 's', True)
                )
                continue

            operator_match = _OPERATOR_RE.match(line)
            if operator_match:
                scene = f'{current_step}: {operator_match.group("scheme")} {operator_match.group("operation")}'
                values[_metric_key(scene, '单次耗时', 'ms')].append(
                    (scene, float(operator_match.group('duration_ms')), 'ms', True)
                )
                values[_metric_key(scene, '吞吐量', 'ops/sec')].append(
                    (scene, float(operator_match.group('ops_per_sec')), 'ops/sec', False)
                )
    return values


def load_resource_delta_summaries(log_dir: Path) -> dict[str, ResourceDeltaSummary]:
    records_path = log_dir / RESOURCE_METRICS_JSONL
    if not records_path.exists():
        return {}

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with records_path.open('r', encoding='utf-8') as records_file:
        for line in records_file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = str(payload.get('step') or '').strip()
            if step:
                grouped[step].append(payload)

    summaries: dict[str, ResourceDeltaSummary] = {}
    for step, records in grouped.items():
        summaries[step] = ResourceDeltaSummary(
            step=step,
            repeat_count=len(records),
            openmp_actual_threads=_max_record_int(records, 'openmp_actual_threads'),
            memory_delta_peak_kib=_max_record_int(records, 'memory_delta_peak_kib'),
            gpu_memory_delta_peak_mib=_max_record_int(records, 'gpu_memory_delta_peak_mib'),
            gpu_sampled=any(bool(record.get('gpu_sampled')) for record in records),
        )
    return summaries


def _max_record_int(records: Sequence[dict[str, object]], key: str) -> int | None:
    values: list[int] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, int):
            values.append(value)
        elif isinstance(value, float):
            values.append(int(value))
    return max(values) if values else None


def load_performance_metrics(
    log_dir: Path,
    results: Sequence[StepResult] | None = None,
) -> dict[str, PerformanceMetric]:
    records = _load_benchmark_records(log_dir, results)
    metrics: dict[str, PerformanceMetric] = {}
    pass_steps = {step for step, record in records.items() if _status_for_record(record) == 'PASS'}

    for step, record in records.items():
        status = _status_for_record(record)
        scene = step
        metric = 'step总耗时'
        unit = 's'
        duration = float(record.get('duration_seconds', 0.0))
        key = _metric_key(scene, metric, unit)
        note = '' if status == 'PASS' else 'step 未通过，不参与性能对比'
        metrics[key] = PerformanceMetric(
            key=key,
            scene=scene,
            metric=metric,
            value=duration,
            unit=unit,
            source_step=step,
            status=status,
            note=note,
            lower_is_better=True,
        )

    parsed_values = _parse_performance_log(log_dir, pass_steps)
    for key, values in parsed_values.items():
        scene = values[0][0]
        unit = values[0][2]
        lower_is_better = values[0][3]
        metric = key.split('|')[1]
        average = sum(value for _, value, _, _ in values) / len(values)
        source_step = scene.split(': ', 1)[0]
        metrics[key] = PerformanceMetric(
            key=key,
            scene=scene,
            metric=metric,
            value=average,
            unit=unit,
            source_step=source_step,
            status='PASS',
            note=f'从 {len(values)} 次输出取平均值',
            lower_is_better=lower_is_better,
        )

    return metrics


def compare_performance_metrics(
    baseline_metrics: dict[str, PerformanceMetric],
    target_metrics: dict[str, PerformanceMetric],
    *,
    threshold_percent: float,
) -> list[PerformanceComparison]:
    comparisons: list[PerformanceComparison] = []
    keys = sorted(set(baseline_metrics) | set(target_metrics))
    for key in keys:
        baseline = baseline_metrics.get(key)
        target = target_metrics.get(key)
        reference = target or baseline
        assert reference is not None

        baseline_value = baseline.value if baseline else None
        target_value = target.value if target else None
        note_parts = [part for part in [baseline.note if baseline else '', target.note if target else ''] if part]
        note = '; '.join(dict.fromkeys(note_parts))

        if target and target.status != 'PASS':
            comparisons.append(
                PerformanceComparison(
                    key=key,
                    scene=reference.scene,
                    metric=reference.metric,
                    unit=reference.unit,
                    baseline_value=baseline_value,
                    target_value=target_value,
                    change_percent=None,
                    exceeds_threshold=None,
                    conclusion='执行失败',
                    note=target.note or 'target step 未通过',
                )
            )
            continue

        if baseline_value is None or target_value is None or baseline_value == 0:
            comparisons.append(
                PerformanceComparison(
                    key=key,
                    scene=reference.scene,
                    metric=reference.metric,
                    unit=reference.unit,
                    baseline_value=baseline_value,
                    target_value=target_value,
                    change_percent=None,
                    exceeds_threshold=None,
                    conclusion='数据不足',
                    note=note or '缺少 baseline 或 target 指标',
                )
            )
            continue

        change_percent = ((target_value - baseline_value) / baseline_value) * 100
        if reference.lower_is_better:
            exceeds_threshold = change_percent > threshold_percent
        else:
            exceeds_threshold = change_percent < -threshold_percent
        conclusion = '疑似回归' if exceeds_threshold else '通过'
        comparisons.append(
            PerformanceComparison(
                key=key,
                scene=reference.scene,
                metric=reference.metric,
                unit=reference.unit,
                baseline_value=baseline_value,
                target_value=target_value,
                change_percent=change_percent,
                exceeds_threshold=exceeds_threshold,
                conclusion=conclusion,
                note=note,
            )
        )
    return comparisons


def build_performance_report_summary(
    log_dir: Path,
    *,
    results: Sequence[StepResult] | None = None,
    baseline_log_dir: Path | None,
    threshold_percent: float,
) -> PerformanceReportSummary:
    baseline_metrics = load_performance_metrics(baseline_log_dir) if baseline_log_dir else {}
    target_metrics = load_performance_metrics(log_dir, results=results)
    comparisons = compare_performance_metrics(
        baseline_metrics,
        target_metrics,
        threshold_percent=threshold_percent,
    )
    return _build_performance_summary(
        comparisons,
        baseline_log_dir=baseline_log_dir,
        target_log_dir=log_dir,
        threshold_percent=threshold_percent,
    )


def write_performance_report(
    log_dir: Path,
    *,
    baseline_log_dir: Path | None,
    baseline_label: str = '',
    target_label: str = '',
    threshold_percent: float,
    results: Sequence[StepResult] | None = None,
) -> PerformanceReportSummary:
    del baseline_label, target_label
    return build_performance_report_summary(
        log_dir,
        results=results,
        baseline_log_dir=baseline_log_dir,
        threshold_percent=threshold_percent,
    )


def _build_performance_summary(
    comparisons: Sequence[PerformanceComparison],
    *,
    baseline_log_dir: Path | None,
    target_log_dir: Path,
    threshold_percent: float,
) -> PerformanceReportSummary:
    regressions = sum(1 for comparison in comparisons if comparison.conclusion == '疑似回归')
    insufficient = sum(1 for comparison in comparisons if comparison.conclusion == '数据不足')
    failed = sum(1 for comparison in comparisons if comparison.conclusion == '执行失败')
    passed = sum(1 for comparison in comparisons if comparison.conclusion == '通过')
    if regressions:
        conclusion = '疑似性能回归'
    elif failed:
        conclusion = '性能数据存在执行失败'
    elif insufficient:
        conclusion = '性能数据不足'
    elif comparisons:
        conclusion = '通过'
    else:
        conclusion = '未执行，无法判断'
    return PerformanceReportSummary(
        baseline_log_dir=str(baseline_log_dir or ''),
        target_log_dir=str(target_log_dir),
        threshold_percent=threshold_percent,
        total=len(comparisons),
        passed=passed,
        regressions=regressions,
        insufficient=insufficient,
        failed=failed,
        conclusion=conclusion,
        comparisons=list(comparisons),
    )


def write_benchmark_summary(
    log_dir: Path,
    modules: Sequence[str],
    *,
    repeat: int,
    results: Sequence[StepResult],
) -> None:
    del log_dir, modules, repeat, results


def _build_lattisense_cpu_steps(
    lattisense_root: Path,
    targets: Sequence[str],
    timeout_seconds: int | None,
) -> list[CommandStep]:
    return [
        CommandStep(
            name='benchmark-cpu-configure',
            command='cmake -S . -B build-bench-cpu -DCMAKE_BUILD_TYPE=Release -DLATTISENSE_BUILD_EXAMPLES=ON',
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        ),
        CommandStep(
            name='benchmark-cpu-build',
            command=f'cmake --build build-bench-cpu -j$(nproc) --target {" ".join(targets)}',
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        ),
    ]


def _build_lattisense_gpu_steps(
    lattisense_root: Path,
    targets: Sequence[str],
    cuda_arch: str,
    timeout_seconds: int | None,
) -> list[CommandStep]:
    return [
        CommandStep(
            name='benchmark-gpu-configure',
            command=(
                'cmake -S . -B build-bench-gpu '
                '-DCMAKE_BUILD_TYPE=Release '
                '-DLATTISENSE_ENABLE_GPU=ON '
                f'-DLATTISENSE_CUDA_ARCH={cuda_arch} '
                '-DLATTISENSE_BUILD_EXAMPLES=ON'
            ),
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        ),
        CommandStep(
            name='benchmark-gpu-build',
            command=f'cmake --build build-bench-gpu -j$(nproc) --target {" ".join(targets)}',
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        ),
    ]


def _convolution_benchmark_steps(
    backend: str,
    conv_dir: Path,
    repeat: int,
    timeout_seconds: int | None,
) -> list[CommandStep]:
    binary = f'../../build-bench-{backend}/examples/benchmark_convolution/benchmark_convolution_{backend}'
    steps: list[CommandStep] = []
    for input_size, kernel_size, in_channels, out_channels in CONVOLUTION_BENCHMARK_CONFIGS:
        name = (
            f'benchmark-conv-{backend}-{input_size}x{input_size}-k{kernel_size}-in{in_channels}-out{out_channels}-run'
        )
        command = f'{binary} {input_size} {kernel_size} {in_channels} {out_channels}'
        steps.append(
            CommandStep(
                name=name,
                command=_repeat_command(command, repeat),
                cwd=conv_dir,
                log_name=PERFORMANCE_LOG,
                timeout_seconds=timeout_seconds,
            )
        )
    return steps


def _cpu_benchmark_steps(
    lattisense_root: Path,
    config: RegressionConfig,
    repeat: int,
    timeout_seconds: int | None,
    python_env: dict[str, str],
) -> list[CommandStep]:
    run_configs = [
        ('benchmark-cpu-bfv-mult-relin-run', '0'),
        ('benchmark-cpu-ckks-mult-relin-run', '1'),
        ('benchmark-cpu-bfv-rotate-col-run', '2'),
    ]
    steps = [
        CommandStep(
            name='benchmark-cpu-generate-tasks',
            command=f'{config.python_bin} examples/benchmark_cpu/benchmark_cpu.py',
            cwd=lattisense_root,
            env=python_env,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        )
    ]
    steps.extend(
        CommandStep(
            name=name,
            command=_repeat_command(f'./build-bench-cpu/examples/benchmark_cpu/benchmark_cpu {arg}', repeat),
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        )
        for name, arg in run_configs
    )
    return steps


def _gpu_benchmark_steps(
    lattisense_root: Path,
    config: RegressionConfig,
    repeat: int,
    timeout_seconds: int | None,
    python_env: dict[str, str],
) -> list[CommandStep]:
    run_configs = [
        ('benchmark-gpu-bfv-mult-relin-run', '0'),
        ('benchmark-gpu-ckks-mult-relin-run', '1'),
        ('benchmark-gpu-bfv-rotate-col-run', '2'),
    ]
    steps = [
        CommandStep(
            name='benchmark-gpu-generate-tasks',
            command=f'{config.python_bin} examples/benchmark_gpu/benchmark_gpu.py',
            cwd=lattisense_root,
            env=python_env,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        )
    ]
    steps.extend(
        CommandStep(
            name=name,
            command=_repeat_command(f'./build-bench-gpu/examples/benchmark_gpu/benchmark_gpu {arg}', repeat),
            cwd=lattisense_root,
            log_name=PERFORMANCE_LOG,
            timeout_seconds=timeout_seconds,
        )
        for name, arg in run_configs
    )
    return steps


def _example_inference_command(repo_root: Path, example: str, *, gpu: bool) -> str:
    task_dir = repo_root / 'examples' / f'test_{example}' / 'task'
    command_parts = [
        './examples/inference',
        '--task-dir',
        str(task_dir),
        '--input',
        str(task_dir / 'client' / 'img.csv'),
        '--verify',
    ]
    if gpu:
        command_parts.append('--gpu')
    return ' '.join(shlex.quote(part) for part in command_parts)


def _repeat_command(command: str, repeat: int) -> str:
    return (
        f'for i in $(seq 1 {repeat}); do echo "BENCHMARK_REPEAT=$i"; '
        f'/usr/bin/time -f "wall_seconds=%e" {command} || exit $?; done'
    )


def _monitored_repeat_command(
    command: str,
    repeat: int,
    *,
    step_name: str,
    config: RegressionConfig,
    gpu: bool,
) -> str:
    monitor_script = config.repo_root / 'tests' / 'resource_monitor.py'
    monitor_command = ' '.join(
        [
            shlex.quote(config.python_bin),
            shlex.quote(str(monitor_script)),
            '--step',
            shlex.quote(step_name),
            '--repeat',
            '"$i"',
            '--output',
            f'"$REGRESSION_LOG_DIR/{RESOURCE_METRICS_JSONL}"',
            '--gpu',
            'true' if gpu else 'false',
            '--',
            *[shlex.quote(part) for part in shlex.split(command)],
        ]
    )
    return (
        f'for i in $(seq 1 {repeat}); do echo "BENCHMARK_REPEAT=$i"; '
        f'{monitor_command} || exit $?; done'
    )


def _module_for_step(step_name: str) -> str:
    if step_name.startswith('benchmark-conv-cpu'):
        return 'conv-cpu'
    if step_name.startswith('benchmark-conv-gpu'):
        return 'conv-gpu'
    if step_name.startswith('benchmark-examples-cpu'):
        return 'examples-cpu'
    if step_name.startswith('benchmark-examples-gpu'):
        return 'examples-gpu'
    if step_name.startswith('benchmark-gpu'):
        return 'gpu'
    if step_name.startswith('benchmark-cpu'):
        return 'cpu'
    if step_name == 'benchmark-conv-generate-tasks':
        return 'conv'
    return 'unknown'
