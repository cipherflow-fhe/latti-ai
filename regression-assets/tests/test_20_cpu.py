from __future__ import annotations

import pytest

from tests.regression_config import RegressionConfig
from tests.regression_runner import CommandStep, RegressionRunner


CPU_LOG = 'cpu.log'
PYTHON_DEPENDENCY_IMPORT_CHECK = r"""
python - <<'PY'
import importlib
import sys

modules = [
    'torch',
    'torchvision',
    'numpy',
    'scipy',
    'networkx',
    'tqdm',
    'onnx',
    'onnxruntime',
    'ordered_set',
    'typing_extensions',
    'graphviz',
    'h5py',
    'pytest',
    'matplotlib',
]
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f'{module}: {exc}')

if missing:
    print('Missing or broken Python dependencies required by compiler tests:')
    for item in missing:
        print(f'  - {item}')
    print('Install them manually with:')
    print('  python -m pip install -r training/requirements.txt')
    print('Or rerun this pytest suite with:')
    print('  --install-python-deps')
    sys.exit(1)

print('Python dependency import check passed.')
PY
""".strip()


@pytest.mark.heavy
@pytest.mark.regression_scope('cpu')
def test_cpu_build(regression_runner: RegressionRunner) -> None:
    regression_runner.run_many(
        [
            CommandStep(
                name='submodule-update-recursive',
                command='git submodule update --init --recursive',
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='lattisense-lattigo-submodule-update',
                command='git -C inference/lattisense submodule update --init fhe_ops_lib/lattigo',
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='clean-build-cpu',
                command='rm -rf build-cpu',
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='cmake-cpu-configure',
                command='cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release',
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='cmake-cpu-build',
                command='cmake --build build-cpu -j$(nproc)',
                log_name=CPU_LOG,
            ),
        ]
    )


@pytest.mark.heavy
@pytest.mark.regression_scope('cpu')
def test_cpu_ctest_and_core_paths(
    regression_runner: RegressionRunner,
    regression_config: RegressionConfig,
) -> None:
    python_bin = regression_config.python_bin
    steps = [
        CommandStep(
            name='ctest-cpu-data-structs',
            command='ctest -R "^test_data_structs$" --output-on-failure',
            cwd=regression_config.repo_root / 'build-cpu',
            log_name=CPU_LOG,
        ),
        CommandStep(
            name='gen-hetero-layer-sq',
            command=(
                'export PYTHONPATH=$(pwd):${PYTHONPATH:-}; '
                'export LATTI_HETERO_BASE_PATH=$(pwd)/build-cpu/inference/hetero; '
                'cd build-cpu/inference/unittests; '
                f'{python_bin} test_gen_layers.py TestLayerExport.test_sq'
            ),
            log_name=CPU_LOG,
        ),
        CommandStep(
            name='sync-hetero-layer-output-to-cpu-build',
            command=(
                'if [ ! -d build-cpu/build/inference/hetero ]; then '
                'echo "Missing generated hetero layer output: build-cpu/build/inference/hetero"; '
                'exit 1; '
                'fi; '
                'rm -rf build-cpu/inference/hetero; '
                'mkdir -p build-cpu/inference; '
                'cp -a build-cpu/build/inference/hetero build-cpu/inference/hetero'
            ),
            log_name=CPU_LOG,
        ),
        CommandStep(
            name='test-fhe-layers-hetero-sq',
            command="cd build-cpu/inference/unittests && ./test_fhe_layers_hetero 'sq*'",
            log_name=CPU_LOG,
        ),
    ]

    if regression_config.install_python_deps:
        steps.append(
            CommandStep(
                name='pip-install-training-requirements',
                command=f'{python_bin} -m pip install -r training/requirements.txt',
                log_name=CPU_LOG,
            )
        )

    steps.extend(
        [
            CommandStep(
                name='python-deps-check',
                command=PYTHON_DEPENDENCY_IMPORT_CHECK.replace('python -', f'{python_bin} -'),
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='python-compiler-tests-cpu',
                command=(
                    'export LATTI_E2E_BASE_PATH=$(pwd)/build-cpu/inference/hetero_e2e; '
                    f'{python_bin} -m pytest training/model_compiler/unittests/test_compiler.py -v'
                ),
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='sync-e2e-output-to-cpu-build',
                command=(
                    'if [ ! -d build/inference/hetero_e2e ]; then '
                    'echo "Missing generated E2E output: build/inference/hetero_e2e"; '
                    'exit 1; '
                    'fi; '
                    'rm -rf build-cpu/inference/hetero_e2e; '
                    'mkdir -p build-cpu/inference; '
                    'cp -a build/inference/hetero_e2e build-cpu/inference/hetero_e2e'
                ),
                log_name=CPU_LOG,
            ),
            CommandStep(
                name='test-e2e-cpu',
                command="cd build-cpu/inference/unittests && ./test_e2e '[batch][cpu]'",
                log_name=CPU_LOG,
            ),
        ]
    )

    regression_runner.run_many(steps)
