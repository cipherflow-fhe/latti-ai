from __future__ import annotations

import pytest

from tests.regression_runner import CommandStep, RegressionRunner


QUALITY_LOG = 'quality.log'


@pytest.mark.heavy
@pytest.mark.regression_scope('quality')
def test_quality_checks_match_release_plan(regression_runner: RegressionRunner) -> None:
    regression_runner.run_many(
        [
            CommandStep(
                name='clang-format-pre-commit',
                command='pre-commit run clang-format --all-files',
                log_name=QUALITY_LOG,
            ),
            CommandStep(
                name='ruff-check',
                command="ruff check . --exclude='inference/lattisense,build,venv,.venv,.git,logs'",
                log_name=QUALITY_LOG,
            ),
            CommandStep(
                name='ruff-format-check',
                command="ruff format --check . --exclude='inference/lattisense,build,venv,.venv,.git,logs'",
                log_name=QUALITY_LOG,
            ),
            CommandStep(
                name='cppcheck',
                command="""
cppcheck \
  --enable=warning,performance,portability \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --suppress=preprocessorErrorDirective:inference/lattisense/lib/nlohmann/json.hpp \
  --suppress=useStlAlgorithm \
  --suppress=noExplicitConstructor \
  --suppress=unusedFunction \
  --error-exitcode=1 \
  --inline-suppr \
  --language=c++ \
  --std=c++20 \
  -i inference/lattisense \
  -i inference/lib \
  -i build \
  -i venv \
  -i .venv \
  -i logs \
  inference/data_structs \
  inference/fhe_layers \
  inference/inference_task \
  inference/util \
  inference/util.cpp \
  inference/common.h \
  examples
""".strip(),
                log_name=QUALITY_LOG,
            ),
        ]
    )
