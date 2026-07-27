from __future__ import annotations

from pathlib import Path

import pytest

from tests.regression_config import SCOPE_STAGES, RegressionConfig, build_config
from tests.regression_runner import RegressionRunner
from tests.report_writer import write_all_reports


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('post-merge-regression')
    group.addoption(
        '--scope',
        action='store',
        default='environment',
        choices=sorted(SCOPE_STAGES),
        help='Regression scope to execute. Default is environment to avoid accidental heavy runs.',
    )
    group.addoption(
        '--cuda-arch',
        action='store',
        default=None,
        help='CUDA architecture for GPU scope, for example 89 for compute capability 8.9.',
    )
    group.addoption(
        '--log-dir',
        action='store',
        default=None,
        help='Directory for command logs and Markdown reports.',
    )
    group.addoption(
        '--repo-root',
        action='store',
        default=str(Path(__file__).resolve().parents[1]),
        help='Repository root to run commands from.',
    )
    group.addoption(
        '--continue-on-error',
        action='store_true',
        default=False,
        help='Continue later commands in the same pytest test after a critical command fails.',
    )
    group.addoption(
        '--python-bin',
        action='store',
        default='python3',
        help='Python executable used by compiler-test commands.',
    )
    group.addoption(
        '--install-python-deps',
        action='store_true',
        default=False,
        help='Install training/requirements.txt before compiler tests. Disabled by default to avoid long downloads.',
    )
    group.addoption('--baseline', action='store', default=None, help='Optional baseline commit or tag.')
    group.addoption(
        '--target',
        action='store',
        default=None,
        help='Optional target main commit. Defaults to git rev-parse HEAD under --repo-root when omitted.',
    )
    group.addoption(
        '--baseline-log-dir',
        action='store',
        default=None,
        help='Optional log directory from the baseline benchmark run for automatic performance comparison.',
    )
    group.addoption(
        '--performance-regression-threshold',
        action='store',
        default=10.0,
        type=float,
        help='Allowed percent change before a slower target metric is marked as performance regression. Default: 10.0.',
    )
    group.addoption('--skip-performance', action='store_true', default=False)
    group.addoption('--skip-docker', action='store_true', default=False)
    group.addoption('--skip-lattisense', action='store_true', default=False)
    group.addoption(
        '--benchmark-modules',
        action='store',
        default='all',
        help='Comma-separated benchmark modules to run: all, cpu, gpu, conv-cpu, conv-gpu, examples-cpu, examples-gpu.',
    )
    group.addoption(
        '--benchmark-repeat',
        action='store',
        default=3,
        type=int,
        help='Number of repeated benchmark runs per module.',
    )
    group.addoption(
        '--benchmark-timeout-seconds',
        action='store',
        default=3600,
        type=int,
        help='Timeout in seconds for each benchmark command step.',
    )
    group.addoption(
        '--cccl-source-dir',
        action='store',
        default=None,
        help='Optional local CCCL source directory for HEonGPU FetchContent. Must exist when provided.',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', 'regression_scope(name): required regression stage')
    config.addinivalue_line('markers', 'heavy: test invokes build, Docker, GPU, or other long-running commands')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    regression_config = _config_from_pytest(config)
    for item in items:
        scope_markers = list(item.iter_markers(name='regression_scope'))
        if not scope_markers:
            continue
        required_stages = {marker.args[0] for marker in scope_markers if marker.args}
        if required_stages and not any(regression_config.stage_enabled(stage) for stage in required_stages):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f'requires one of stages {sorted(required_stages)}, '
                        f'but --scope {regression_config.scope!r} enables '
                        f'{sorted(regression_config.enabled_stages)}'
                    )
                )
            )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    regression_config = _config_from_pytest(session.config)
    regression_config.log_dir.mkdir(parents=True, exist_ok=True)
    runner = getattr(session.config, '_post_merge_regression_runner', None)
    if runner is not None:
        runner.collect_commit_report_info()
    results = runner.results if runner is not None else []
    environment_info = runner.environment_info if runner is not None else {}
    write_all_reports(
        regression_config.log_dir,
        results,
        baseline_log_dir=regression_config.baseline_log_dir,
        baseline_label=regression_config.baseline or '',
        target_label=regression_config.target or '',
        threshold_percent=regression_config.performance_regression_threshold,
        environment_info=environment_info,
    )


@pytest.fixture(scope='session')
def regression_config(pytestconfig: pytest.Config) -> RegressionConfig:
    return _config_from_pytest(pytestconfig)


@pytest.fixture(scope='session')
def regression_runner(pytestconfig: pytest.Config, regression_config: RegressionConfig) -> RegressionRunner:
    runner = RegressionRunner(regression_config)
    setattr(pytestconfig, '_post_merge_regression_runner', runner)
    return runner


def _config_from_pytest(config: pytest.Config) -> RegressionConfig:
    cached = getattr(config, '_post_merge_regression_config', None)
    if cached is not None:
        return cached

    built = build_config(
        repo_root=config.getoption('--repo-root'),
        log_dir=config.getoption('--log-dir'),
        scope=config.getoption('--scope'),
        cuda_arch=config.getoption('--cuda-arch'),
        continue_on_error=config.getoption('--continue-on-error'),
        python_bin=config.getoption('--python-bin'),
        install_python_deps=config.getoption('--install-python-deps'),
        baseline=config.getoption('--baseline'),
        target=config.getoption('--target'),
        baseline_log_dir=config.getoption('--baseline-log-dir'),
        performance_regression_threshold=config.getoption('--performance-regression-threshold'),
        skip_performance=config.getoption('--skip-performance'),
        skip_docker=config.getoption('--skip-docker'),
        skip_lattisense=config.getoption('--skip-lattisense'),
        benchmark_modules=config.getoption('--benchmark-modules'),
        benchmark_repeat=config.getoption('--benchmark-repeat'),
        benchmark_timeout_seconds=config.getoption('--benchmark-timeout-seconds'),
        cccl_source_dir=config.getoption('--cccl-source-dir'),
    )
    setattr(config, '_post_merge_regression_config', built)
    return built
