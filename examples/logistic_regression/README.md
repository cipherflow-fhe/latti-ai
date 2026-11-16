# Encrypted logistic regression example

This example expands full-batch logistic-regression training into an explicit CKKS computation graph. Each update contains encrypted feature/weight products, a polynomial sigmoid, encrypted-label subtraction, gradient reduction, and weight updates. The graph keeps every operation visible so level consumption and bootstrap placement can be inspected by the compiler.

The plaintext implementation and polynomial fitting utilities are in `experiment.py`. `prepare_ckks.py` creates a deterministic Breast Cancer Wisconsin task with matching plaintext references. `ckks_step_gpu.cpp` executes the generated graph with the GPU CKKS backend.

## Plaintext and graph generation

```bash
python examples/logistic_regression/prepare_ckks.py \
  --steps 20 --degree 3 --approximation least-squares --radius 10 \
  --learning-rate 0.05 --output /tmp/logistic-ckks
```

Supported polynomial degrees are 3, 5, 7, 9, and 15. `least-squares` and discretized `minimax` fitting are available. The generated directory contains the task manifest, data, DAG, and plaintext reference weights.

## GPU execution

Configure and build the example with GPU support:

```bash
cmake -S . -B /tmp/latti-ai-build \
  -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/latti-ai-build --target ckks_step_gpu -j8

CUDA_VISIBLE_DEVICES=0 python examples/logistic_regression/run_gpu_benchmark.py \
  /tmp/latti-ai-build/examples/ckks_step_gpu /tmp/logistic-ckks
python examples/logistic_regression/verify_ckks.py /tmp/logistic-ckks
```

The GPU executable performs key generation, encryption, one complete unrolled graph execution, and final decryption. No server-side plaintext training loop is used. `verify_ckks.py` compares decrypted weights and test predictions with the generated plaintext reference; use `--allow-failure` only for diagnostic runs.

The default task is intentionally small enough for a functional check. Longer runs and serialized key material can require tens of gigabytes of host or GPU memory.

## Validation and scope

The demonstrated 3-degree, 20-update configuration previously achieved 93.7063% plaintext and encrypted test accuracy, with maximum weight error 8.63e-6 (single GPU run). Longer or higher-degree runs are experimental: degree-15 minimax training has shown severe numerical instability. This example is not a guarantee of arbitrary-round convergence or precision.

Bootstrap placement is currently explicit in the primitive DAG, not automatically scheduled through `LayerAbstractGraph`. The executable serializes secret keys for storage measurement; keep the task directory private and do not distribute these generated artifacts.

Core graph tests:

```bash
python -m unittest discover -s tests -p 'test_logistic_primitive_graph.py' -v
```
