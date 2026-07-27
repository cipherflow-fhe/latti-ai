from __future__ import annotations

import pytest

from tests.regression_runner import CommandStep, RegressionRunner


ENVIRONMENT_LOG = 'environment.log'


@pytest.mark.regression_scope('environment')
def test_collect_git_and_submodule_versions(regression_runner: RegressionRunner) -> None:
    regression_runner.run_many(
        [
            CommandStep(
                name='git-current-head',
                command='git rev-parse HEAD',
                log_name=ENVIRONMENT_LOG,
            ),
            CommandStep(
                name='git-submodule-status',
                command='git submodule status',
                log_name=ENVIRONMENT_LOG,
            ),
        ]
    )


@pytest.mark.regression_scope('environment')
def test_collect_linux_environment(regression_runner: RegressionRunner) -> None:
    regression_runner.collect_environment_report_info()
    regression_runner.run_many(
        [
            CommandStep(name='kernel-version', command='uname -a', log_name=ENVIRONMENT_LOG),
            CommandStep(name='os-release', command='cat /etc/os-release', log_name=ENVIRONMENT_LOG),
            CommandStep(name='cpu-info', command='lscpu', log_name=ENVIRONMENT_LOG),
            CommandStep(name='memory-info', command='free -h', log_name=ENVIRONMENT_LOG),
            CommandStep(
                name='nvidia-smi-gpu-info',
                command='nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader',
                log_name=ENVIRONMENT_LOG,
                critical=False,
            ),
        ]
    )
