from __future__ import annotations

import getpass
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tests.performance_benchmark import (
    PERFORMANCE_LOG,
    PerformanceReportSummary,
    build_performance_report_summary,
    is_performance_step,
    load_resource_delta_summaries,
    module_for_step,
    remove_legacy_performance_artifacts,
)
from tests.regression_runner import StepResult


QUALITY_CHECK_LABELS = {
    'clang-format-pre-commit': 'C/C++ 格式检查',
    'ruff-check': 'Python lint',
    'ruff-format-check': 'Python 格式检查',
    'cppcheck': 'C++ 静态分析',
}

GPU_CHECK_LABELS = {
    'nvidia-smi-compute-capability': 'CUDA 能力采集',
    'clean-heongpu-build': 'HEonGPU 构建目录清理',
    'heongpu-configure': 'HEonGPU 配置',
    'heongpu-build': 'HEonGPU 构建',
    'heongpu-install': 'HEonGPU 安装',
    'clean-build-gpu': 'latti-ai GPU 构建目录清理',
    'cmake-gpu-configure': 'latti-ai GPU 配置',
    'cmake-gpu-build': 'latti-ai GPU 构建',
    'python-compiler-tests-gpu': 'GPU compiler tests',
    'sync-e2e-output-to-gpu-build': 'GPU E2E 输出同步',
    'test-e2e-gpu': 'GPU E2E',
    'gen-mega-ag-mnist': 'MNIST mega ag 生成',
    'gen-mega-ag-cifar10': 'CIFAR-10 mega ag 生成',
}

LATTISENSE_CHECK_LABELS = {
    'clean-build-lattisense': 'lattisense 构建目录清理',
    'lattisense-configure': 'lattisense standalone 配置',
    'lattisense-build': 'lattisense standalone 构建',
    'lattisense-ctest': 'lattisense CTest',
}


def _suggested_conclusion(results: Sequence[StepResult]) -> str:
    if any(result.status == 'FAIL' for result in results):
        return '不通过，暂停发布'
    if any(result.status == 'NON_BLOCKING_FAIL' for result in results):
        return '有条件通过，带风险继续'
    if results:
        return '通过，可继续发布'
    return '未执行，无法判断'


def write_summary(log_dir: Path, results: Sequence[StepResult]) -> Path:
    counts = Counter(result.status for result in results)
    summary_path = log_dir / 'summary.md'

    lines = [
        '# Post-merge Regression Summary',
        '',
        '## Result Counts',
        '',
        f'- PASS: {counts.get("PASS", 0)}',
        f'- FAIL: {counts.get("FAIL", 0)}',
        f'- NON_BLOCKING_FAIL: {counts.get("NON_BLOCKING_FAIL", 0)}',
        '',
        '## Step Results',
        '',
        '| Step | Status | Exit Code | Duration(s) | Log |',
        '| --- | --- | ---: | ---: | --- |',
    ]

    for result in results:
        log_name = Path(result.log_path).name
        lines.append(
            f'| {result.name} | {result.status} | {result.exit_code} | {result.duration_seconds:.3f} | `{log_name}` |'
        )

    if not results:
        lines.append('| No command results recorded | N/A |  |  |  |')

    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary_path


def _module_versions(environment_info: dict[str, Any]) -> list[dict[str, str]]:
    raw_versions = environment_info.get('module_versions', [])
    if not isinstance(raw_versions, list):
        return []

    versions: list[dict[str, str]] = []
    for item in raw_versions:
        if not isinstance(item, dict):
            continue
        module = str(item.get('module') or '').strip()
        version = str(item.get('version') or '').strip()
        if module:
            versions.append({'module': module, 'version': version or 'N/A'})
    return versions


def _first_started_at(results: Sequence[StepResult]) -> str:
    started = [result.started_at for result in results if result.started_at]
    return min(started) if started else ''


def _last_finished_at(results: Sequence[StepResult]) -> str:
    finished = [result.finished_at for result in results if result.finished_at]
    return max(finished) if finished else ''


def _value_or_na(value: object) -> str:
    text = str(value or '').strip()
    return text or 'N/A'


def _is_quality_only_report(results: Sequence[StepResult]) -> bool:
    if not results:
        return False
    result_names = {result.name for result in results}
    return result_names <= set(QUALITY_CHECK_LABELS)


def _is_gpu_only_report(results: Sequence[StepResult]) -> bool:
    if not results:
        return False
    result_names = {result.name for result in results}
    return result_names <= set(GPU_CHECK_LABELS)


def _is_lattisense_only_report(results: Sequence[StepResult]) -> bool:
    if not results:
        return False
    result_names = {result.name for result in results}
    return result_names <= set(LATTISENSE_CHECK_LABELS)


def _write_quality_report(report_path: Path, results: Sequence[StepResult], *, target_label: str = '') -> Path:
    lines = [
        '# 代码质量检查报告',
        '',
        '## 1. 格式检查和静态分析',
        '',
        '| 项目 | 内容 |',
        '| --- | --- |',
        '| 测试类型 | 格式检查和静态分析 |',
        f'| 合并后 main commit | {_value_or_na(target_label)} |',
        f'| 建议测试结论 | {_suggested_conclusion(results)} |',
        '',
        '| 检查项 | Step | 结果 | 退出码 | 耗时(s) | 日志 |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    for result in results:
        lines.append(
            f'| {QUALITY_CHECK_LABELS[result.name]} | {result.name} | {result.status} | '
            f'{result.exit_code} | {result.duration_seconds:.3f} | `{Path(result.log_path).name}` |'
        )
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def _write_gpu_report(report_path: Path, results: Sequence[StepResult], *, target_label: str = '') -> Path:
    lines = [
        '# GPU 回归测试报告',
        '',
        '## 1. GPU 构建与核心验证',
        '',
        '| 项目 | 内容 |',
        '| --- | --- |',
        '| 测试类型 | GPU 构建与核心验证 |',
        f'| 合并后 main commit | {_value_or_na(target_label)} |',
        f'| 建议测试结论 | {_suggested_conclusion(results)} |',
        '',
        '| 检查项 | Step | 结果 | 退出码 | 耗时(s) | 日志 |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    for result in results:
        lines.append(
            f'| {GPU_CHECK_LABELS[result.name]} | {result.name} | {result.status} | '
            f'{result.exit_code} | {result.duration_seconds:.3f} | `{Path(result.log_path).name}` |'
        )
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def _write_lattisense_report(
    report_path: Path,
    results: Sequence[StepResult],
    *,
    target_label: str = '',
    environment_info: dict[str, Any] | None = None,
) -> Path:
    environment_info = environment_info or {}
    commits = environment_info.get('commits', {}) if isinstance(environment_info.get('commits'), dict) else {}
    lines = [
        '# lattisense standalone 验证报告',
        '',
        '## 1. lattisense standalone 构建与 CTest',
        '',
        '| 项目 | 内容 |',
        '| --- | --- |',
        '| 测试类型 | lattisense standalone 构建与 CTest |',
        f'| 合并后 main commit | {_value_or_na(target_label)} |',
        f'| lattisense commit | {_value_or_na(commits.get("lattisense"))} |',
        f'| 建议测试结论 | {_suggested_conclusion(results)} |',
        '',
        '| 检查项 | Step | 结果 | 退出码 | 耗时(s) | 日志 |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    for result in results:
        lines.append(
            f'| {LATTISENSE_CHECK_LABELS[result.name]} | {result.name} | {result.status} | '
            f'{result.exit_code} | {result.duration_seconds:.3f} | `{Path(result.log_path).name}` |'
        )
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def write_final_report_template(
    log_dir: Path,
    results: Sequence[StepResult],
    performance_summary: PerformanceReportSummary | None = None,
    *,
    target_label: str = '',
    environment_info: dict[str, Any] | None = None,
) -> Path:
    report_path = log_dir / 'final-report.md'
    if _is_quality_only_report(results):
        return _write_quality_report(report_path, results, target_label=target_label)
    if _is_gpu_only_report(results):
        return _write_gpu_report(report_path, results, target_label=target_label)
    if _is_lattisense_only_report(results):
        return _write_lattisense_report(
            report_path,
            results,
            target_label=target_label,
            environment_info=environment_info,
        )

    suggested_conclusion = _suggested_conclusion(results)
    environment_info = environment_info or {}
    commits = environment_info.get('commits', {}) if isinstance(environment_info.get('commits'), dict) else {}
    lines = [
        '# dev 合并 main 后回归测试报告',
        '',
        '## 1. 基本信息',
        '',
        '| 项目 | 内容 |',
        '| --- | --- |',
        '| 测试类型 | dev 合并 main 后回归测试 |',
        '| 目标分支 | main |',
        f'| 合并后 main commit | {_value_or_na(target_label)} |',
        f'| lattisense commit | {_value_or_na(commits.get("lattisense"))} |',
        f'| HEonGPU commit | {_value_or_na(commits.get("HEonGPU"))} |',
        f'| Lattigo commit | {_value_or_na(commits.get("Lattigo"))} |',
        f'| 测试人员 | {_value_or_na(environment_info.get("tester") or getpass.getuser())} |',
        f'| 测试开始时间 | {_value_or_na(_first_started_at(results))} |',
        f'| 测试结束时间 | {_value_or_na(_last_finished_at(results))} |',
        f'| 建议测试结论 | {suggested_conclusion} |',
    ]

    lines.extend(
        [
            '',
            '## 2. 命令执行结果',
            '',
            '| 测试项 | 结果 | 退出码 | 耗时(s) | 日志 | 备注 |',
            '| --- | --- | ---: | ---: | --- | --- |',
        ]
    )

    for result in results:
        lines.append(
            f'| {result.name} | {result.status} | {result.exit_code} | '
            f'{result.duration_seconds:.3f} | `{Path(result.log_path).name}` |  |'
        )

    if not results:
        lines.append('| 未记录命令结果 | N/A |  |  |  | 请确认 pytest 是否已执行目标 scope |')

    if performance_summary is not None:
        lines.extend(_performance_summary_lines(performance_summary, results, log_dir))

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def _performance_summary_lines(
    performance_summary: PerformanceReportSummary,
    results: Sequence[StepResult],
    log_dir: Path,
) -> list[str]:
    lines = [
        '',
        '## 3. 性能回归结果',
        '',
        '| 项目 | 内容 |',
        '| --- | --- |',
        f'| 性能总体结论 | {performance_summary.conclusion} |',
        f'| 基线日志目录 | `{performance_summary.baseline_log_dir or "未提供"}` |',
        f'| 当前日志目录 | `{performance_summary.target_log_dir}` |',
        f'| 原始性能日志 | `{PERFORMANCE_LOG}` |',
        f'| 回归阈值 | {performance_summary.threshold_percent:.2f}% |',
        f'| 指标总数 | {performance_summary.total} |',
        f'| 通过指标数 | {performance_summary.passed} |',
        f'| 超阈值指标数 | {performance_summary.regressions} |',
        f'| 数据不足指标数 | {performance_summary.insufficient} |',
        f'| 执行失败指标数 | {performance_summary.failed} |',
        '',
        '### 3.1 Benchmark step 执行结果',
        '',
        '| 模块 | Step | 结果 | 退出码 | 耗时(s) | 日志 |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]

    benchmark_results = [result for result in results if is_performance_step(result)]
    if benchmark_results:
        for result in benchmark_results:
            lines.append(
                f'| {module_for_step(result.name)} | {result.name} | {result.status} | '
                f'{result.exit_code} | {result.duration_seconds:.3f} | `{Path(result.log_path).name}` |'
            )
    else:
        lines.append('| 未记录 benchmark step | N/A | N/A |  |  | 请确认 performance scope 是否执行 |')

    lines.extend(
        [
            '',
            '### 3.2 性能指标对比',
            '',
            '| 场景 | 指标 | 基线版本 | 合并后 main | 变化比例 | 是否超过阈值 | 结论 | 备注 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
    )
    if performance_summary.comparisons:
        for comparison in performance_summary.comparisons:
            lines.append(
                f'| {comparison.scene} | {comparison.metric} | '
                f'{_format_final_report_value(comparison.baseline_value, comparison.unit)} | '
                f'{_format_final_report_value(comparison.target_value, comparison.unit)} | '
                f'{_format_final_report_change(comparison.change_percent)} | '
                f'{_format_final_report_threshold(comparison.exceeds_threshold)} | '
                f'{comparison.conclusion} | {comparison.note} |'
            )
    else:
        lines.append('| 未解析到性能指标 | N/A | 缺失 | 缺失 | N/A | N/A | 未执行，无法判断 | 未找到 benchmark 结果 |')
    lines.extend(_performance_resource_lines(log_dir))
    lines.append('')
    return lines


def _performance_resource_lines(log_dir: Path) -> list[str]:
    summaries = load_resource_delta_summaries(log_dir)
    if not summaries:
        return []

    lines = [
        '',
        '### 3.3 Examples 资源增量',
        '',
        '| 模块 | Step | repeat次数 | OpenMP线程数 | 内存增量峰值(MiB) | 显存增量峰值(MiB) | 日志 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]

    for step in sorted(summaries):
        summary = summaries[step]
        lines.append(
            f'| {module_for_step(summary.step)} | {summary.step} | {summary.repeat_count} | '
            f'{_format_optional_int(summary.openmp_actual_threads)} | '
            f'{_format_kib_as_mib(summary.memory_delta_peak_kib)} | '
            f'{_format_gpu_memory(summary.gpu_memory_delta_peak_mib, summary.gpu_sampled)} | '
            f'`{PERFORMANCE_LOG}` |'
        )
    return lines


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return 'N/A'
    return str(value)


def _format_kib_as_mib(value: int | None) -> str:
    if value is None:
        return 'N/A'
    return f'{value / 1024:.3f}'


def _format_gpu_memory(value: int | None, gpu_sampled: bool) -> str:
    if value is None:
        return 'N/A' if gpu_sampled else '不适用'
    return f'{value:.3f}'


def _format_final_report_value(value: float | None, unit: str) -> str:
    if value is None:
        return '缺失'
    return f'{value:.3f} {unit}'


def _format_final_report_change(value: float | None) -> str:
    if value is None:
        return 'N/A'
    return f'{value:+.2f}%'


def _format_final_report_threshold(value: bool | None) -> str:
    if value is None:
        return 'N/A'
    return '是' if value else '否'


def write_all_reports(
    log_dir: Path,
    results: Sequence[StepResult],
    *,
    baseline_log_dir: Path | None = None,
    baseline_label: str = '',
    target_label: str = '',
    threshold_percent: float = 10.0,
    environment_info: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    del baseline_label
    remove_legacy_performance_artifacts(log_dir)
    has_performance_results = any(is_performance_step(result) for result in results)
    has_performance_log = (log_dir / PERFORMANCE_LOG).exists()
    performance_summary = None
    if has_performance_results or has_performance_log or baseline_log_dir:
        performance_summary = build_performance_report_summary(
            log_dir,
            results=results,
            baseline_log_dir=baseline_log_dir,
            threshold_percent=threshold_percent,
        )
    return (
        write_summary(log_dir, results),
        write_final_report_template(
            log_dir,
            results,
            performance_summary,
            target_label=target_label,
            environment_info=environment_info,
        ),
    )
