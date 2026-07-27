from __future__ import annotations

import pytest

from tests.performance_benchmark import build_benchmark_steps, parse_benchmark_modules
from tests.regression_config import RegressionConfig
from tests.regression_runner import RegressionRunner


@pytest.mark.heavy
@pytest.mark.regression_scope('performance')
def test_run_performance_benchmarks(
    regression_runner: RegressionRunner,
    regression_config: RegressionConfig,
) -> None:
    try:
        modules = parse_benchmark_modules(regression_config.benchmark_modules)
        steps = build_benchmark_steps(
            regression_config,
            modules,
            repeat=regression_config.benchmark_repeat,
            timeout_seconds=regression_config.benchmark_timeout_seconds,
        )
    except ValueError as exc:
        pytest.fail(str(exc))

    regression_runner.run_many(steps)
