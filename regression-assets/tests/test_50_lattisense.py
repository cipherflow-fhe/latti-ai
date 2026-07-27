from __future__ import annotations

import pytest

from tests.regression_config import RegressionConfig
from tests.regression_runner import CommandStep, RegressionRunner


LATTISENSE_LOG = 'lattisense.log'


@pytest.mark.heavy
@pytest.mark.regression_scope('lattisense')
def test_lattisense_standalone_build_and_ctest(
    regression_runner: RegressionRunner,
    regression_config: RegressionConfig,
) -> None:
    regression_runner.collect_lattisense_report_info()
    regression_runner.run_many(
        [
            CommandStep(
                name='clean-build-lattisense',
                command='rm -rf build-lattisense',
                log_name=LATTISENSE_LOG,
            ),
            CommandStep(
                name='lattisense-configure',
                command='cmake -S inference/lattisense -B build-lattisense -DLATTISENSE_BUILD_TESTS=ON',
                log_name=LATTISENSE_LOG,
            ),
            CommandStep(
                name='lattisense-build',
                command='cmake --build build-lattisense -j$(nproc)',
                log_name=LATTISENSE_LOG,
            ),
            CommandStep(
                name='lattisense-generate-bfv-test-data',
                command=f'cd inference/lattisense/unittests && {regression_config.python_bin} test_cpu_bfv.py',
                log_name=LATTISENSE_LOG,
            ),
            CommandStep(
                name='lattisense-generate-ckks-test-data',
                command=f'cd inference/lattisense/unittests && {regression_config.python_bin} test_cpu_ckks.py',
                log_name=LATTISENSE_LOG,
            ),
            CommandStep(
                name='lattisense-ctest',
                command='ctest --output-on-failure',
                cwd=regression_config.repo_root / 'build-lattisense',
                log_name=LATTISENSE_LOG,
            ),
        ]
    )
