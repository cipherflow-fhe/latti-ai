# Inference Module Build Guide

This document contains the complete build and installation guide for the encrypted inference engine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [CPU-Only Build (Without GPU Acceleration)](#cpu-only-build-without-gpu-acceleration)
- [GPU Build (Recommended)](#gpu-build-recommended)
- [Build Options](#build-options)
- [GPU Architecture Settings](#gpu-architecture-settings)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency | Version | Description |
|------------|---------|-------------|
| CMake | >= 3.13 | Build system |
| C++ Compiler | GCC 12  | C++20 support required |
| Go | >= 1.18 | For building Lattigo crypto library |
| Python | 3.12 | For computation graph compiler |

**For GPU Acceleration (Recommended):**

| Dependency | Version | Description |
|------------|---------|-------------|
| CUDA Toolkit | >= 12.0 | GPU compute support |
| HEonGPU | 1.1 | GPU acceleration library (requires pre-build)|

---

## CPU-Only Build (Without GPU Acceleration)

If GPU acceleration is not needed, use this simplified process:

### Build

```bash
git clone https://github.com/cipherflow-fhe/latti-ai.git
cd latti-ai
git submodule update --init
cd inference/lattisense
git submodule update --init fhe_ops_lib/lattigo
cd ../..
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## GPU Build (Recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/cipherflow-fhe/latti-ai.git
cd latti-ai
git submodule update --init --recursive  # This may take ~6 minutes
```

### Step 2: Build HEonGPU (GPU Acceleration Library)

```bash
cd inference/lattisense/HEonGPU
mkdir build && cd build
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> \
  -DCMAKE_CUDA_COMPILER=<path/to/cuda>/bin/nvcc \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install
```

> If the build hangs at `-- CPM: Adding package CCCL@2.5.0`, see [CCCL Package Hangs](#cccl-package-hangs).

### Step 3: Compile and Install HEonGPU

> If compilation fails with a missing `cstdint` header, see [Missing cstdint Header](#missing-cstdint-header).

```bash
make -j$(nproc)
make install
```

### Step 4: Build Project

```bash
cd ../../../..  # Return to project root
mkdir build && cd build
cmake .. -DINFERENCE_SDK_ENABLE_GPU=ON
make -j$(nproc)
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `INFERENCE_SDK_ENABLE_GPU` | OFF | Enable GPU acceleration |


Example:
```bash
cmake .. -DINFERENCE_SDK_ENABLE_GPU=ON
```

---

## GPU Architecture Settings

Set `CMAKE_CUDA_ARCHITECTURES` according to your GPU model (see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)):

| GPU Model | Architecture Code |
|-----------|-------------------|
| RTX 30xx Series | 86 |
| RTX 40xx / RTX 5880 Ada | 89 |
| A100 | 80 |
| H100 | 90 |

---

## Project Structure

```
inference/
├── CMakeLists.txt
├── fhe_layers/
├── inference_task/
├── unittests/
└── lattisense/                     # git submodule
    ├── fhe_ops_lib/
    │   └── lattigo/                # git submodule — Go FHE crypto backend
    ├── HEonGPU/                    # git submodule — GPU-accelerated FHE (optional)
    ├── cxx_sdk_v2/
    ├── mega_ag_runners/
    └── frontend/
```

---

## Troubleshooting

### CCCL Package Hangs

If the build hangs at `-- CPM: Adding package CCCL@2.5.0`:
1. Edit `build/_deps/rapids-cmake-src/rapids-cmake/cpm/versions.json`
2. Change line 16 `git_url` to: `"git_url": "https://gitee.com/Vizl/cccl.git"`
3. Re-run the cmake command

### Missing cstdint Header

Add the following to the beginning of `HEonGPU/thirdparty/GPU-NTT/src/include/common/common.cuh`:
```cpp
#include <cstdint>
```

### CUDA Architecture Mismatch

Adjust `-DCMAKE_CUDA_ARCHITECTURES` according to your GPU model, referring to the GPU architecture table above.

### Slow Submodule Update

Initial submodule cloning may take approximately 6 minutes depending on network speed.
