from __future__ import annotations

import pytest

from tests.regression_runner import CommandStep, RegressionRunner


DOCKER_LOG = 'docker.log'


@pytest.mark.heavy
@pytest.mark.regression_scope('docker')
def test_docker_build_and_cpu_verify(regression_runner: RegressionRunner) -> None:
    regression_runner.run_many(
        [
            CommandStep(
                name='docker-build-post-merge-test-image',
                command='docker build -t latti-ai:post-merge-test .',
                log_name=DOCKER_LOG,
            ),
            CommandStep(
                name='docker-cpu-verify',
                command=r"""docker run --rm latti-ai:post-merge-test bash -c '
  set -e
  export PYTHONPATH=/workspace:${PYTHONPATH:-}
  export LATTI_HETERO_BASE_PATH=/workspace/build/inference/hetero
  export LATTI_E2E_BASE_PATH=/workspace/build/inference/hetero_e2e
  cd /workspace/build
  cmake ..
  make -j$(nproc)
  cd inference/unittests
  python test_gen_layers.py TestLayerExport.test_sq
  if [ ! -d /workspace/build/build/inference/hetero ]; then
    echo "Missing generated hetero layer output: /workspace/build/build/inference/hetero"
    exit 1
  fi
  rm -rf /workspace/build/inference/hetero
  mkdir -p /workspace/build/inference
  cp -a /workspace/build/build/inference/hetero /workspace/build/inference/hetero
  ./test_data_structs
  ./test_fhe_layers_hetero "sq*"
  cd /workspace
  python -m pytest training/model_compiler/unittests/test_compiler.py -v
' """,
                log_name=DOCKER_LOG,
            ),
        ]
    )
