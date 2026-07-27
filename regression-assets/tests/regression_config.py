from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CCCL_SOURCE_DIR = '/home/qianc/latti-ai-deps/cccl-src'


SCOPE_STAGES: dict[str, set[str]] = {
    'environment': {'environment', 'report'},
    'quality': {'environment', 'quality', 'report'},
    'cpu': {'environment', 'quality', 'cpu', 'report'},
    'gpu': {'environment', 'gpu', 'report'},
    'docker': {'environment', 'docker', 'report'},
    'lattisense': {'environment', 'lattisense', 'report'},
    'performance': {'environment', 'performance', 'report'},
    'docs': {'environment', 'docs', 'report'},
    'main-flow': {'environment', 'quality', 'cpu', 'gpu', 'performance', 'report'},
    'full': {
        'environment',
        'quality',
        'cpu',
        'gpu',
        'docker',
        'lattisense',
        'performance',
        'docs',
        'report',
    },
}


@dataclass(frozen=True)
class RegressionConfig:
    repo_root: Path
    log_dir: Path
    scope: str
    enabled_stages: set[str]
    cuda_arch: str | None
    continue_on_error: bool
    python_bin: str
    install_python_deps: bool
    baseline: str | None
    target: str | None
    baseline_log_dir: Path | None
    performance_regression_threshold: float
    skip_performance: bool
    skip_docker: bool
    skip_lattisense: bool
    benchmark_modules: str = 'all'
    benchmark_repeat: int = 3
    benchmark_timeout_seconds: int = 3600
    cccl_source_dir: Path | None = None

    def stage_enabled(self, stage: str) -> bool:
        if stage == 'docker' and self.skip_docker:
            return False
        if stage == 'lattisense' and self.skip_lattisense:
            return False
        if stage == 'performance' and self.skip_performance:
            return False
        return stage in self.enabled_stages


def build_config(
    *,
    repo_root: str,
    log_dir: str | None,
    scope: str,
    cuda_arch: str | None,
    continue_on_error: bool,
    python_bin: str,
    install_python_deps: bool,
    baseline: str | None,
    target: str | None,
    skip_performance: bool,
    skip_docker: bool,
    skip_lattisense: bool,
    baseline_log_dir: str | None = None,
    performance_regression_threshold: float = 10.0,
    benchmark_modules: str = 'all',
    benchmark_repeat: int = 3,
    benchmark_timeout_seconds: int = 3600,
    cccl_source_dir: str | None = None,
) -> RegressionConfig:
    if scope not in SCOPE_STAGES:
        valid = ', '.join(sorted(SCOPE_STAGES))
        raise ValueError(f'Unsupported --scope {scope!r}; expected one of: {valid}')

    root = Path(repo_root).resolve()
    run_started_at = datetime.now()
    run_date = run_started_at.strftime('%Y%m%d')
    run_time = run_started_at.strftime('%H%M%S')
    resolved_log_dir = Path(log_dir).resolve() if log_dir else root / 'logs' / run_date / run_time
    resolved_target = target or _detect_git_head(root)

    if performance_regression_threshold < 0:
        raise ValueError('--performance-regression-threshold must be greater than or equal to 0')

    resolved_baseline_log_dir: Path | None = None
    if baseline_log_dir:
        resolved_baseline_log_dir = Path(baseline_log_dir).resolve()
        if not resolved_baseline_log_dir.exists():
            raise ValueError(f'--baseline-log-dir does not exist: {resolved_baseline_log_dir}')
        if not resolved_baseline_log_dir.is_dir():
            raise ValueError(f'--baseline-log-dir must be a directory: {resolved_baseline_log_dir}')

    resolved_cccl_source_dir = _resolve_cccl_source_dir(cccl_source_dir)

    return RegressionConfig(
        repo_root=root,
        log_dir=resolved_log_dir,
        scope=scope,
        enabled_stages=set(SCOPE_STAGES[scope]),
        cuda_arch=cuda_arch,
        continue_on_error=continue_on_error,
        python_bin=python_bin,
        install_python_deps=install_python_deps,
        baseline=baseline,
        target=resolved_target,
        baseline_log_dir=resolved_baseline_log_dir,
        performance_regression_threshold=performance_regression_threshold,
        skip_performance=skip_performance,
        skip_docker=skip_docker,
        skip_lattisense=skip_lattisense,
        benchmark_modules=benchmark_modules,
        benchmark_repeat=benchmark_repeat,
        benchmark_timeout_seconds=benchmark_timeout_seconds,
        cccl_source_dir=resolved_cccl_source_dir,
    )


def _resolve_cccl_source_dir(cccl_source_dir: str | None) -> Path | None:
    if cccl_source_dir:
        resolved = Path(cccl_source_dir).resolve()
        if not resolved.exists():
            raise ValueError(f'--cccl-source-dir does not exist: {resolved}')
        if not resolved.is_dir():
            raise ValueError(f'--cccl-source-dir must be a directory: {resolved}')
        return resolved

    default_path = Path(DEFAULT_CCCL_SOURCE_DIR).resolve()
    if default_path.exists() and default_path.is_dir():
        return default_path
    return None


def _detect_git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None
