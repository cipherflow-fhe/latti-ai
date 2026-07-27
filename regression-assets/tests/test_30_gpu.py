from __future__ import annotations

import pytest

from tests.regression_config import RegressionConfig
from tests.regression_runner import CommandStep, RegressionRunner


GPU_LOG = 'gpu.log'


@pytest.mark.heavy
@pytest.mark.regression_scope('gpu')
def test_gpu_build(
    regression_runner: RegressionRunner,
    regression_config: RegressionConfig,
) -> None:
    if not regression_config.cuda_arch:
        pytest.fail('GPU scope requires --cuda-arch, for example --cuda-arch 89')

    cuda_arch = regression_config.cuda_arch
    heongpu_configure_command = (
        'cmake -S . -B build '
        f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch} '
        '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc '
        '-DCMAKE_INSTALL_PREFIX=$(pwd)/install '
        '-DTHRUST_INCLUDE_DIR=/usr/local/cuda/targets/x86_64-linux/include/cccl'
    )
    if regression_config.cccl_source_dir:
        heongpu_configure_command += f' -DFETCHCONTENT_SOURCE_DIR_CCCL={regression_config.cccl_source_dir}'

    regression_runner.run_many(
        [
            CommandStep(
                name='nvidia-smi-compute-capability',
                command='nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='clean-heongpu-build',
                command='rm -rf build',
                cwd=regression_config.repo_root / 'inference' / 'lattisense' / 'HEonGPU',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='heongpu-configure',
                command=heongpu_configure_command,
                cwd=regression_config.repo_root / 'inference' / 'lattisense' / 'HEonGPU',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='heongpu-build',
                command='cmake --build build -j$(nproc)',
                cwd=regression_config.repo_root / 'inference' / 'lattisense' / 'HEonGPU',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='heongpu-install',
                command='cmake --install build',
                cwd=regression_config.repo_root / 'inference' / 'lattisense' / 'HEonGPU',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='clean-build-gpu',
                command='rm -rf build-gpu',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='cmake-gpu-configure',
                command=(
                    'cmake -B build-gpu '
                    '-DINFERENCE_SDK_ENABLE_GPU=ON '
                    f'-DLATTISENSE_CUDA_ARCH={cuda_arch} '
                    '-DCMAKE_BUILD_TYPE=Release'
                ),
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='cmake-gpu-build',
                command='cmake --build build-gpu -j$(nproc)',
                log_name=GPU_LOG,
            ),
        ]
    )


@pytest.mark.heavy
@pytest.mark.regression_scope('gpu')
def test_gpu_tests_and_examples(
    regression_runner: RegressionRunner,
    regression_config: RegressionConfig,
) -> None:
    python_bin = regression_config.python_bin
    regression_runner.run_many(
        [
            CommandStep(
                name='python-compiler-tests-gpu',
                command=(
                    'export LATTI_E2E_BASE_PATH=$(pwd)/build-gpu/inference/hetero_e2e; '
                    'cd build-gpu/inference/unittests; '
                    f'{python_bin} -m pytest ../../../training/model_compiler/unittests/test_compiler.py -v'
                ),
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='sync-e2e-output-to-gpu-build',
                command=(
                    'if [ ! -d build/inference/hetero_e2e ]; then '
                    'echo "Missing generated E2E output: build/inference/hetero_e2e"; '
                    'exit 1; '
                    'fi; '
                    'rm -rf build-gpu/inference/hetero_e2e; '
                    'mkdir -p build-gpu/inference; '
                    'cp -a build/inference/hetero_e2e build-gpu/inference/hetero_e2e'
                ),
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='test-e2e-gpu',
                command="cd build-gpu/inference/unittests && ./test_e2e '[batch][gpu]'",
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='gen-mega-ag-mnist',
                command=f'{python_bin} inference/interface/gen_mega_ag.py --task-dir examples/test_mnist/task',
                log_name=GPU_LOG,
            ),
            CommandStep(
                name='gen-mega-ag-cifar10',
                command=f'{python_bin} inference/interface/gen_mega_ag.py --task-dir examples/test_cifar10/task',
                log_name=GPU_LOG,
            ),
        ]
    )
