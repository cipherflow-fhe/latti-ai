# LattiAI

[![Build & Test](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml)
[![Format Check](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml)
[![Python Lint](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**LattiAI** is a development platform for privacy-preserving AI model inference, built on top of the [LattiSense](https://github.com/cipherflow-fhe/lattisense/tree/main) Fully Homomorphic Encryption (FHE) framework developed by [CipherFlow](https://cipherflow.ai/).

LattiAI covers the complete pipeline from plaintext models trained with frameworks like PyTorch to encrypted inference deployment. Through model adaptation, model compilation, and a high-performance HE operator library, it automatically converts standard AI models into encrypted inference services based on the CKKS fully homomorphic encryption scheme. Throughout the entire inference process, data remains encrypted — the server cannot access the user's raw data, and the user cannot access the model parameters, achieving bidirectional privacy protection for both data and model.

AI developers can complete end-to-end encrypted inference deployment without understanding the underlying cryptographic primitives.

## Main Features

```
Plaintext Model Training (PyTorch, etc.) → Model Adaptation (Op Replacement + Fine-tuning) → Model Compilation (Computation Graph) → Encrypted Inference (CPU/GPU)
```

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

For detailed build prerequisites, troubleshooting, and build options, see the **[Inference Module Build Guide](inference/README.md)**.

---

## Quick Start

This example demonstrates the complete pipeline from model adaptation to encrypted inference, using MNIST handwritten digit recognition.

> For a quick start, we have pre-completed Step 1 in the `examples/test_mnist/` directory, providing adapted model weights and compiled encrypted computation graphs. To directly experience encrypted inference, skip to [Step 2](#step-2-encrypted-inference).

### Step 1: Model Adaptation & Compilation

Convert a standard model into the configuration files and computation graphs required for encrypted inference. For the complete toolchain steps, see [training/README.md](training/README.md).

The following files will be generated after compilation:

| File | Description |
|------|-------------|
| `client/ckks_parameter.json` | CKKS encryption parameter configuration |
| `client/task_config.json` | Client inference task configuration |
| `server/ckks_parameter.json` | CKKS encryption parameter configuration |
| `server/task_config.json` | Server inference task configuration |
| `server/model_parameters.h5` | Model weights |
| `server/ergs/erg0.json` | Compiled encrypted computation graph |

### <a id="step-2-encrypted-inference"></a>Step 2: Encrypted Inference

Load the files generated in Step 1 and perform model inference on encrypted data.

<details>
<summary>Click to expand full code</summary>

**Generate GPU-accelerated computation graph instructions** (`gen_mega_ag.py`):

```python
import json
import os
import sys

# Resolve the directory where this script lives.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find project root by walking up until we find the 'training' directory.
_dir = script_dir
while _dir != os.path.dirname(_dir):
    if os.path.isdir(os.path.join(_dir, 'training')):
        break
    _dir = os.path.dirname(_dir)
project_root = _dir

# Add project root and LattiSense library to the Python path so that
# the frontend and training modules can be imported.
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'inference', 'lattisense'))

from frontend.custom_task import *  # noqa: E402
from training.deploy_cmds import gen_custom_task  # noqa: E402

# Path to the server-side encrypted computation graph (ergs directory).
task_path = os.path.join(script_dir, 'task', 'server', 'ergs')

# Read the server task configuration to determine which computation
# segments (ergs) require GPU-accelerated mega_ag generation.
with open(os.path.join(task_path, '..', 'task_config.json'), 'r', encoding='utf-8') as f:
    config = json.load(f)

# For each erg with FPGA/GPU acceleration enabled, generate the
# corresponding mega_ag instruction sequence.  mega_ag fuses multiple
# HE operations into optimized GPU kernels for faster inference.
for erg_name, erg_config in config['server_task'].items():
    if erg_config['enable_fpga']:
        gen_custom_task(task_path, use_gpu=True)
```

**Run encrypted inference** (`main.cpp`):

```cpp
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include <cxx_sdk_v2/cxx_fhe_task.h>
#include "data_structs/feature.h"
#include "inference_task/inference_process.h"
#include "util.h"

using namespace cxx_sdk_v2;
using namespace std;

int main(int argc, char* argv[]) {
    bool use_gpu = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) use_gpu = true;
    }

    cout << "========== MNIST Encrypted Inference ==========" << endl;
    cout << "Device: " << (use_gpu ? "GPU" : "CPU") << endl;

    // ==================== 1. Read Configuration ====================
    // Load client-side task config and CKKS parameters from JSON files.
    // These configs are generated by the model compilation step.
    auto task_config = read_json("./task/client/task_config.json");
    auto& input_param = task_config["task_input_param"].begin().value();
    auto& output_param = task_config["task_output_param"].begin().value();
    // Ciphertext multiplication depth level for the input
    int level = input_param["level"];
    // Stride between valid data slots in the output ciphertext
    int output_skip = output_param["skip"];
    int channel = input_param["channel"];
    int height = input_param["shape"][0];
    int width = input_param["shape"][1];
    // Packing style: "ordinary" (one channel per ciphertext) or
    // "multiplexed": (generalized interleaved packing strategy)
    string pack_style = task_config["pack_style"];

    auto ckks_config = read_json("./task/client/ckks_parameter.json");
    string ckks_param_id = input_param["ckks_parameter_id"];
    // Number of usable CKKS slots = poly_modulus_degree / 2
    int n_slots = ckks_config[ckks_param_id]["poly_modulus_degree"].get<int>() / 2;

    // ==================== 2. Client Side: Encrypt Input ====================
    // Read the input image from CSV file with shape [channel, height, width].
    auto input_array = csv_to_array<3>("./task/client/img.csv", {(uint64_t)channel, (uint64_t)height, (uint64_t)width});

    // Create CKKS encryption context (without bootstrapping for MNIST).
    CkksParameter param = CkksParameter::create_parameter(16384);
    CkksContext context = CkksContext::create_random_context(param);
    // Generate rotation public keys (for SIMD rotation operations on ciphertext vectors)
    context.gen_rotation_keys();

    // Create an encrypted feature map and pack the input data.
    // The packing strategy depends on the image size relative to the
    // number of available CKKS slots:
    //   - ordinary:     one channel per ciphertext (simple packing)
    //   - multiplexed:  generalized interleaved packing
    //       - if image pixels > n_slots: split into blocks first
    //       - otherwise: parallel multiplexed packing
    Feature2DEncrypted input_ct(&context, level);

    if (pack_style == "ordinary") {
        input_ct.pack(input_array, false, param.get_default_scale());
    } else if (height * width > n_slots) {
        // Image too large for a single ciphertext: split into blocks and
        // pack each block separately.  channel_packing_factor indicates
        // how many blocks tile each spatial dimension.
        Duo block_shape = {task_config["block_shape"][0], task_config["block_shape"][1]};
        Duo channel_packing_factor = {(uint32_t)(height / block_shape[0]),
                                      (uint32_t)(width / block_shape[1])};
        input_ct.split_with_stride_pack(input_array, block_shape, channel_packing_factor, false,
                                        param.get_default_scale());
    } else {
        // Image fits in one ciphertext: use parallel multiplexed packing
        // to encode multiple channels into a single ciphertext.
        input_ct.par_mult_pack(input_array, false, param.get_default_scale());
    }

    // ==================== 3. Server Side: Load Model ====================
    // Load the pre-compiled encrypted computation graph and model weights.
    // No bootstrapping parameters needed for MNIST (smaller model depth).
    InitInferenceProcess init("./task/server/", false);
    init.init_parameters();
    init.is_lazy = false;
    init.load_model_prepare();

    // ==================== 4. Run Encrypted Inference ====================
    // Configure the inference engine with the encryption context and
    // encrypted input, then execute the computation graph.
    InferenceProcess fp(&init, true);
    fp.available_keys.push_back("input");

    // Pass the encryption context to the server-side inference engine.
    map<string, unique_ptr<CkksContext>> context_map;
    context_map["param0"] = make_unique<CkksContext>(move(context.shallow_copy_context()));
    fp.ckks_contexts = move(context_map);
    fp.set_feature("input", make_unique<Feature2DEncrypted>(move(input_ct)));

    Timer timer;
    timer.start();
    // Select compute device: multi-threaded CPU or GPU hardware acceleration
    fp.compute_device = use_gpu ? ComputeDevice::GPU : ComputeDevice::CPU;
    // Execute the full FHE ciphertext inference pipeline
    fp.run_task();
    timer.stop();
    timer.print("Encrypted inference time");

    // ==================== 5. Decrypt and Verify ====================
    // Retrieve the encrypted output and decrypt it.
    // output_skip specifies the stride between valid classification scores
    // in the packed ciphertext slots.
    auto encrypted_output = fp.get_ciphertext_output_feature0D("output");
    encrypted_output.skip = output_skip;
    auto decrypted = encrypted_output.unpack(DecryptType::SPARSE);
    print_double_message(decrypted.to_array_1d().data(), "Encrypted output", 10);

    // Run plaintext inference on the same input as a correctness reference.
    fp.p_feature2d_x["input"] = std::move(input_array);
    fp.run_task_plaintext();
    auto plain_output = fp.p_feature0d_x["output"];
    print_double_message(plain_output.data(), "Plaintext output", 10);

    return 0;
}
```

</details>

---

## Running Examples

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

## Documentation

- **Technical Whitepaper**: See [docs/en/whitepaper.md](docs/en/whitepaper.md)

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
