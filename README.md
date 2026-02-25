# LattiAI

[![Build & Test](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml)
[![Format Check](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml)
[![Python Lint](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**LattiAI** is a development platform for privacy-preserving AI model inference, built on top of the [LattiSense](https://github.com/cipherflow-fhe/lattisense/tree/main) Fully Homomorphic Encryption (FHE) framework developed by [CipherFlow](https://cipherflow.ai/).

LattiAI covers the complete pipeline from plaintext models trained with frameworks like PyTorch to encrypted inference deployment. Through model adaptation, model compilation, and a high-performance HE operator library, it automatically converts standard AI models into encrypted inference services based on the CKKS fully homomorphic encryption scheme. Throughout the entire inference process, data remains encrypted — the server cannot access the user's raw data, and the user cannot access the model parameters, achieving bidirectional privacy protection for both data and model.

AI developers can complete end-to-end encrypted inference deployment without understanding the underlying cryptographic primitives.

## Main Features

- **Model Adaptation**: Provides plug-and-play polynomial approximation operators that replace non-polynomial activation functions (e.g., ReLU, SiLU) and MaxPooling with FHE-friendly polynomial activations and AvgPool. After fine-tuning or retraining, the adapted model achieves accuracy on par with the original. Validated on ResNet-18, ResNet-44, MobileNetV2, YOLOv5 and more, with additional conversion strategies under active development.

- **Model Compiler**: Takes adapted model files (`.pth`, `.onnx`) and, through operator mapping and computation graph optimization, automatically generates a CKKS-compatible directed acyclic computation graph (DAG) for encrypted inference, with automatic planning of bootstrapping insertion and data packing strategies.

- **HE Operator Library**: Implements encrypted versions of core neural network operators based on the CKKS scheme — convolution, deconvolution, fully connected, AvgPool, and BatchNorm — leveraging SIMD slot encoding for vectorized parallel computation and supporting arbitrary-depth ciphertext operations through bootstrapping.

- **Runtime**: Automatically schedules the complete inference pipeline based on the compiler-generated encrypted computation graph, with support for multi-threaded CPU parallelism and GPU acceleration. Encrypted inference results are nearly identical to plaintext inference.

---

## Quick Start

This guide demonstrates how to transform a standard PyTorch model into an inference service for encrypted queries using the **Latti-AI** framework.

We will use a **ResNet-18** model trained on the **CIFAR-10** dataset as our baseline.

### Prerequisites

Before starting, ensure you have:

- Python dependencies installed:

```bash
pip install -r training/requirements.txt
```

- A trained baseline model checkpoint (e.g., train_baseline.pth). If you don't have one, train it first:

```bash
python examples/test_cifar10/train.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir ./runs/cifar10/model
```

This produces `runs/cifar10/model/train_baseline.pth`.

### Phase 1: Model Adaptation

In this phase, we convert a standard neural network into an **FHE-friendly** version. Fully Homomorphic Encryption (FHE) does not support non-linear activations like ReLU directly, so we swap them for polynomial approximations.

1. Structure Conversion & Fine-Tuning

Run the following command to replace ReLU layers with polynomial functions and fine-tune the parameters to maintain accuracy:

```bash
python examples/test_cifar10/train.py \
  --poly_model_convert \
  --pretrained runs/cifar10/model/train_baseline.pth \
  --epochs 10 \
  --batch-size 36 \
  --lr 0.001 \
  --input-dir runs/cifar10/model \
  --export-dir runs/cifar10/task/server \
  --input-shape 3 32 32
```

2. High-Level FHE Compilation

Next, compile the adapted model into an **FHE Model Graph**. This step performs the optimizations of the following:

- Selecting optimal FHE parameters.
- Determining bootstrapping positions.
- Assigning FHE levels and scales to each layer.

```bash
python training/run_compile.py \           
  --input=runs/cifar10/model/trained_poly.onnx \
  --output=runs/cifar10/ \
  --poly_n=65536 \
  --style=multiplexed
```


### Phase 2: Encrypted Inference

Once the high-level graph is ready, we lower it to hardware-specific instructions for actual execution.

1. Generate Low-Level Instructions

This command generates the necessary code for both CPU and GPU processors:

```bash
python gen_mega_ag.py
```

2. Runtime Execution

In your application, you can now load the fine-tuned parameters and invoke the generated instructions to process encrypted queries.

~~~c++
  ```
      // Load model FHE instructions, load model parameters, and encode parameters.
      InitInferenceProcess init("./task/server/", false);
      init.init_parameters();
      init.load_model_prepare();
      
      // Receive public keys and encrypted query from client.
      ...
  
      // Create an InferenceProcess with the public keys and encrypted input. 
      InferenceProcess fp(&init, true);
      fp.available_keys.push_back("input");
      // Pass the encryption context to the server-side inference engine.
      map<string, unique_ptr<CkksContext>> context_map;
      context_map["param0"] = make_unique<CkksContext>(move(context.shallow_copy_context()));
      fp.ckks_contexts = move(context_map);
      fp.set_feature("input", make_unique<Feature2DEncrypted>(move(input_ct)));
  
      // Select compute device: multi-threaded CPU or GPU hardware acceleration
      fp.compute_device = use_gpu ? ComputeDevice::GPU : ComputeDevice::CPU;
      // Execute the FHE ciphertext inference pipeline
      fp.run_task();
      
      // Send the result to client.
      ...
  ```
~~~


## Build & Install

### CPU-Only Build (Without GPU Acceleration)

If GPU acceleration is not needed, use this simplified process:

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

### GPU Build (Recommended)

#### Step 1: Clone Repository

```bash
git clone https://github.com/cipherflow-fhe/latti-ai.git
cd latti-ai
git submodule update --init --recursive  # This may take ~6 minutes
```

#### Step 2: Build HEonGPU (GPU Acceleration Library)

```bash
cd inference/lattisense/HEonGPU
mkdir build && cd build
cmake .. \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> \
  -DCMAKE_CUDA_COMPILER=<path/to/cuda>/bin/nvcc \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/../install
```

#### Step 3: Compile and Install HEonGPU

```bash
make -j$(nproc)
make install
```

#### Step 4: Build Project

```bash
cd ../../../..  # Return to project root
mkdir build && cd build
cmake .. -DINFERENCE_SDK_ENABLE_GPU=ON
make -j$(nproc)
```

For detailed build prerequisites, troubleshooting, and build options, see the **[Inference Module Build Guide](docs/en/build-guide.md)**.

---

## Quick Start

This guide demonstrates how to transform a standard PyTorch model into an inference service for encrypted queries using the **LattiAI** framework.

We will use a **ResNet-20** model trained on the **CIFAR-10** dataset as an end-to-end example.

### Prerequisites

Before starting, ensure you have:

- Successfully built the project (see [Build & Install](#build--install) above).
- The standard CIFAR-10 dataset files (automatically downloaded on first run).

Install Python dependencies:

```bash
pip install -r training/requirements.txt
```

> **Note:** All commands in this guide are run from the **project root directory** unless otherwise specified.

### Phase 1: Model Adaptation & Compilation

In this phase, we convert a standard neural network into an **FHE-friendly** version and compile it into an encrypted computation graph.

```
Baseline Training  →  Operator Replacement & Fine-tuning  →  Model Compilation
     (Step 1)                   (Step 2)                        (Step 3)
```

#### Step 1: Baseline Training

Train a standard ResNet-20 on CIFAR-10 with ReLU activations:

```bash
python examples/test_cifar10/train.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir ./runs/cifar10/model
```

**Output:** `./runs/cifar10/model/train_baseline.pth`

#### Step 2: Operator Replacement & Fine-Tuning

FHE does not support non-linear activations like ReLU directly. Run the following command to replace ReLU layers with polynomial functions, swap max pooling for average pooling, and fine-tune the parameters to maintain accuracy. The script automatically exports the adapted model to ONNX format and saves model weights in an H5 file.

```bash
python examples/test_cifar10/train.py \
  --poly_model_convert \
  --pretrained ./runs/cifar10/model/train_baseline.pth \
  --epochs 10 \
  --batch-size 36 \
  --lr 0.001 \
  --input-dir ./runs/cifar10/model \
  --export-dir ./runs/cifar10/task/server \
  --input-shape 3 32 32
```

Workflow of `train.py`: when `--poly_model_convert` is enabled, the script replaces FHE-incompatible operators before training and exports the adapted model after training. Without this flag, it performs standard baseline training only.

```python
# 1. Replace FHE-incompatible operators (only when --poly_model_convert is set)
if args.poly_model_convert:
    replace_maxpool_with_avgpool(model)
    replace_activation_with_poly(model, old_cls=nn.ReLU,
                                 upper_bound=args.upper_bound,
                                 degree=args.degree)

# 2. Train (or fine-tune) the model
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

# 3. Export to ONNX & H5 (only when --poly_model_convert is set)
if args.poly_model_convert:
    export_to_onnx(model, save_path=onnx_path, ...)
    fuse_and_export_h5(model, h5_path=h5_path, ...)
```

- `--pretrained`: loads the baseline checkpoint.
- `--input-dir`: directory containing the baseline model (also used as output for `.pth` and `.onnx`).
- `--export-dir`: directory for the H5 weight file, corresponding to the server-side model weights.
- `--upper-bound`: normalization upper bound for RangeNormPoly2d (default: `3.0`). Controls the input range for polynomial approximation.
- `--degree`: degree of the polynomial activation (choices: `2`, `4`, `8`; default: `4`). Higher degree gives better approximation but increases FHE computational depth.

**Output:**

| File | Description |
|------|-------------|
| `./runs/cifar10/model/train_poly.pth` | Adapted model checkpoint with polynomial activations |
| `./runs/cifar10/model/trained_poly.onnx` | Exported adapted model in ONNX format |
| `./runs/cifar10/task/server/model_parameters.h5` | Model weights (BatchNorm absorbed into Conv) |

#### Step 3: High-Level FHE Compilation

Next, compile the adapted model into an **FHE Model Graph**. This step performs the following optimizations:

- Selecting optimal FHE parameters.
- Determining bootstrapping positions.
- Assigning FHE levels and scales to each layer.

```bash
python training/run_compile.py \
  --input=./runs/cifar10/model/trained_poly.onnx \
  --output=./runs/cifar10/ \
  --poly_n=65536 \
  --style=multiplexed
```

- `--input`: the exported adapted model in ONNX format from the previous step.
- `--output`: root output directory; the compiler generates `task/server/` and `task/client/` subdirectories underneath.
- `--poly_n`: polynomial modulus degree for CKKS (determines the number of ciphertext slots and security level). `65536` provides 128-bit security with 32768 slots; `16384` provides 128-bit security with 8192 slots but does not support bootstrapping.
- `--style`: packing style — `multiplexed` (channel-multiplexed packing for higher slot utilization) or `ordinary` (one channel per ciphertext).

**Output:**

| File | Description |
|------|-------------|
| `./runs/cifar10/model/pt.json` | Intermediate computation graph (JSON) |
| `./runs/cifar10/task/server/task_config.json` | Server-side inference task configuration |
| `./runs/cifar10/task/server/ckks_parameter.json` | CKKS encryption parameter configuration |
| `./runs/cifar10/task/server/ergs/erg0.json` | Compiled encrypted computation graph (DAG) |
| `./runs/cifar10/task/client/task_config.json` | Client-side inference task configuration |
| `./runs/cifar10/task/client/ckks_parameter.json` | CKKS encryption parameter configuration |

### Phase 2: Encrypted Inference

Once the high-level graph is ready, we lower it to hardware-specific instructions for actual execution.

#### Step 1: Generate Low-Level Instructions

Navigate to the built example directory and generate low-level instructions:

```bash
cd build/examples/test_cifar10
python gen_mega_ag.py
```

#### Step 2: Runtime Execution

In your application, you can now load the fine-tuned parameters and invoke the generated instructions to process encrypted queries. The following pseudocode illustrates the server-side workflow (see `examples/test_cifar10/inference.cpp` for a complete runnable example):

```cpp
// Load model FHE instructions, load model parameters, and encode parameters.
InitInferenceProcess init("./task/server/", false);
init.init_parameters();
init.load_model_prepare();

// Receive public keys and encrypted query from client.
...

// Create an InferenceProcess with the public keys and encrypted input.
InferenceProcess fp(&init, true);
fp.available_keys.push_back("input");
// Pass the encryption context to the server-side inference engine.
map<string, unique_ptr<CkksContext>> context_map;
context_map["param0"] = make_unique<CkksContext>(move(context.shallow_copy_context()));
fp.ckks_contexts = move(context_map);
fp.set_feature("input", make_unique<Feature2DEncrypted>(move(input_ct)));

// Select compute device: multi-threaded CPU or GPU hardware acceleration
fp.compute_device = use_gpu ? ComputeDevice::GPU : ComputeDevice::CPU;
// Execute the FHE ciphertext inference pipeline
fp.run_task();

// Send the result to client.
...
```

---

## Running Examples

> For a complete end-to-end walkthrough (from model adaptation to encrypted inference), see [Quick Start](#quick-start). The commands below assume pre-built examples with pre-prepared `task/` folders.

### Prerequisites

Make sure the project has been built successfully. See [Build & Install](#build--install) above. Examples are built automatically along with the inference module.

### Run

Each example is built into its own subdirectory under `build/examples/`. Run from there so that the relative path `./task/` resolves correctly:

```bash
cd build/examples

# MNIST
cd test_mnist
python gen_mega_ag.py          # Generate GPU-accelerated computation graph instructions
./test_mnist                   # Run encrypted inference on CPU
./test_mnist --gpu             # Run encrypted inference with GPU acceleration

# CIFAR-10
cd ../test_cifar10
python gen_mega_ag.py
./test_cifar10
./test_cifar10 --gpu

# ImageNet
cd ../test_imagenet
python gen_mega_ag.py
./test_imagenet
./test_imagenet --gpu
```

---

#### Performance

> Testing environment — Server: Intel Xeon Gold 6226R (32 cores) + NVIDIA RTX 5880 Ada (48GB); 128-bit security level.

| Task | Model | Dataset | Baseline Accuracy | FHE Accuracy | CPU Latency (s) | GPU Latency (s) |
|------|-------|---------|-------------------|-------------|-----------------|-----------------|
| Classification | MobileNetV2 | ImageNet | 71.8% | 70.1% | 1210.0 | 82.4 |

For detailed benchmarks and methodology, see the [Technical Whitepaper](docs/en/whitepaper.md#performance-evaluation).

---

## Documentation

- **Technical Whitepaper**: See [docs/en/whitepaper.md](docs/en/whitepaper.md)
- **Inference Module Build Guide**: See [docs/en/build-guide.md](docs/en/build-guide.md)

## Related Links

- **HEonGPU**: [GPU-accelerated Homomorphic Encryption Library](https://github.com/Alisah-Ozcan/HEonGPU)
- **Lattigo**: [Go Homomorphic Encryption Library](https://github.com/tuneinsight/lattigo)

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please reach out:

- Email: info@cipherflow.cn
