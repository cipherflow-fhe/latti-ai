# Examples

This directory contains end-to-end encrypted inference examples. Each example requires a `task/` folder containing adapted model weights, compiled encrypted computation graphs, and configuration files. We provide pre-prepared `task/` folders so you can run directly, or generate them from scratch by following the [Quick Start](../README.md#quick-start) guide in the root README.

## Examples Overview

| Example | Model | Dataset | Input Size |  Encryption | Bootstrapping |
|---------|-------|---------|------------|------------|---------------|
| `test_mnist` | Simple CNN | MNIST | 1 x 16 x 16 |  CKKS (N=16384) | No |
| `test_cifar10` | ResNet-20 | CIFAR-10 | 3 x 32 x 32 |  CKKS (N=65536) | Yes |
| `test_imagenet` | MobileNetV2 | ImageNet | 3 x 256 x 256 | CKKS (N=65536) | Yes |

## Directory Structure

```
examples/
├── inference.cpp               # C++ inference entry point using InferenceClient/InferenceServer (--task-dir, --gpu)
├── CMakeLists.txt              # Build configuration (single 'inference' binary)
└── test_<name>/
    └── task/
        ├── client/
        │   ├── img.csv             # Sample input image
        │   ├── ckks_parameter.json # CKKS encryption parameters
        │   └── task_config.json    # Inference task configuration
        └── server/
            ├── model_parameters.h5 # Model weights
            ├── ckks_parameter.json # CKKS encryption parameters
            ├── task_config.json    # Inference task configuration
            └── ergs/
                └── erg0.json       # Compiled encrypted computation graph
```

## Prerequisites

Make sure the inference module has been built. See the [Inference Module Build Guide](../docs/en/build-guide.md).

## Build

Examples are built automatically as part of the main project build (see the root [Build & Install](../README.md#build--install) guide).

## Run

See the [Running Examples](../README.md#running-examples) section in the root README for instructions on running each example.

## Example Details

### test_mnist

The simplest example — a small CNN for MNIST digit classification. Uses standard CKKS without bootstrapping. Good starting point for understanding the framework.

- **Input**: 16x16 grayscale image (1 channel)
- **Output**: 10-class logits (digits 0-9)
- **Poly degree**: N = 16384

### test_cifar10

ResNet-20 on CIFAR-10 with bootstrapping enabled to support the deeper network. Demonstrates the framework's ability to handle residual connections and deeper computation graphs.

- **Input**: 32x32 RGB image (3 channels)
- **Output**: 10-class logits
- **Poly degree**: N = 65536

### test_imagenet

MobileNetV2 on ImageNet — a larger-scale example with 1000-class classification. Shows encrypted inference on production-grade models.

- **Input**: 256x256 RGB image (3 channels)
- **Output**: 1000-class logits
- **Poly degree**: N = 65536
