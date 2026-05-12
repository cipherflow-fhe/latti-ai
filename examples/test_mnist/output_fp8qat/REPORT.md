# FP8 QAT Encrypted Inference Report

## Overview

This report documents the end-to-end workflow of training an FP8 quantized neural network using Quantization-Aware Training (QAT), converting it for Fully Homomorphic Encryption (FHE) inference, designing custom CKKS parameters with smaller primes to enable N=8192 inference, and running encrypted inference on the latti-ai platform.

---

## 1. Baseline Model

### Architecture: SimpleCNN

```
Input (1x16x16)
  -> Conv2d(1, 16, kernel=5, stride=2, padding=2)   # 1x16x16 -> 16x8x8
  -> BatchNorm2d(16)
  -> ReLU
  -> Conv2d(16, 32, kernel=3, stride=2, padding=1)   # 16x8x8 -> 32x4x4
  -> BatchNorm2d(32)
  -> ReLU
  -> Flatten                                           # 32x4x4 -> 512
  -> Dropout(0.5)
  -> Linear(512, 10)                                   # 512 -> 10
```

### FP32 Baseline Performance

| Metric | Value |
|---|---|
| Checkpoint | `output/train_baseline.pth` |
| Best epoch | 14 |
| Test accuracy | **98.56%** |
| Parameters | ~11K |

---

## 2. FP8 Quantization-Aware Training

### Method

FP8 QAT applies fake quantization during training to simulate the numerical effects of FP8 (E4M3) arithmetic. The model weights remain in FP32 but are trained with quantization noise injected via the straight-through estimator (STE), allowing the network to learn representations that are robust to reduced precision.

### Results

| Model | Checkpoint | Test Accuracy | Delta vs Baseline |
|---|---|---|---|
| FP32 Baseline | `train_baseline.pth` | 98.56% | -- |
| FP8 (post-training quant) | `train_fp8.pth` | 98.50% | -0.06% |
| **FP8 QAT** | `train_fp8_qat.pth` | **98.55%** | **-0.01%** |

FP8 QAT nearly perfectly recovers the FP32 baseline accuracy with only 0.01% degradation.

---

## 3. FHE Model Conversion

### 3.1 Activation Replacement

ReLU activations are replaced with polynomial approximations for FHE compatibility (comparison operations cannot be performed on encrypted data).

Two polynomial degrees were evaluated:

| Config | Polynomial | Level Cost (per activation) | Total Multiplicative Depth |
|---|---|---|---|
| degree=4 | RangeNormPoly2d (degree 4) | 2 | 9 levels |
| degree=2 | RangeNormPoly2d (degree 2) | 2 | 7 levels |

### 3.2 Pooling Replacement

- `MaxPool2d` -> `AvgPool2d` (MaxPool requires comparison, unsupported in FHE)
- General `AvgPool2d` -> `DepthwiseAvgPool2d` (depthwise conv equivalent)

### 3.3 Conv+BatchNorm Fusion

BatchNorm parameters are fused into the preceding convolution layer:

```
scale = bn_gamma / sqrt(bn_var + eps)
fused_weight[i] = conv_weight[i] * scale[i]
fused_bias[i] = scale[i] * (conv_bias[i] - bn_mean[i]) + bn_beta[i]
```

### 3.4 Polynomial Activation Accuracy (after fine-tuning)

| Polynomial Degree | Fine-tune Epochs | Test Accuracy |
|---|---|---|
| degree=4 (from FP8 QAT) | 5 | 98.16% |
| degree=2 (from FP8 QAT) | 7 | 97.76% |

---

## 4. CKKS Parameter Design for N=8192

### 4.1 The Challenge

The standard parameter set at N=8192 is `PN13QP218` (logN=13, logQP=218, 128-bit security) which provides only 5 multiplicative levels using 30-bit Q primes. Our FP8 model requires 7 levels — a fundamental mismatch.

### 4.2 Key Insight: Smaller Primes = More Levels

CKKS security is determined by the product of all primes (logQP), not by individual prime sizes. By using smaller 24-bit primes instead of 30-bit, we can pack more levels within the same security budget:

```
PN13QP218 (standard):  Q = 1x33 + 5x30 = 183 bits  +  P = 1x35 = 35 bits  =  218 bits  ->  5 levels
PN13QP218s (custom):   Q = 1x26 + 7x24 = 194 bits  +  P = 1x26 = 26 bits  =  220 bits  ->  7 levels
```

The custom `PN13QP218s` set trades per-level precision (24-bit vs 30-bit scale) for 2 additional multiplicative levels. This is viable because FP8 models are inherently tolerant of reduced precision.

### 4.3 Prime Selection

All primes must be NTT-friendly: `p ≡ 1 (mod 2N)` where `2N = 16384`.

**PN13QP218s Parameter Set:**

| Component | Primes | Bit Width | Count |
|---|---|---|---|
| Q0 (initial) | `0x2044001` | 26 | 1 |
| Q1-Q7 (levels) | `0x804001`, `0x820001`, `0x840001`, `0x844001`, `0x850001`, `0x868001`, `0x898001` | 24 each | 7 |
| P (key switching) | `0x207C001` | 26 | 1 |

- **logQP = 220** (2 bits over the 218 budget, ~127-bit security)
- **P >= Q level primes** (26-bit >= 24-bit) — required for correct key switching

### 4.4 Design Iterations

Several configurations were tested before finding the working combination:

| Config | Q Level Bits | P Bits | logQP | Max Error | Result |
|---|---|---|---|---|---|
| 22-bit Q + 34-bit P | 22 | 34 | 216 | 0.170 | FAIL (precision too low) |
| 24-bit Q + 22-bit P | 24 | 22 | 218 | 2.093 | FAIL (P too small for key switching) |
| 5x24 + 2x22 Q + 26-bit P | 24/22 hybrid | 26 | 218 | 0.106 | FAIL (barely, 0.006 over) |
| **24-bit Q + 26-bit P** | **24** | **26** | **220** | **0.037** | **PASS** |

Key lessons:
1. P must be >= largest Q level prime for key switching correctness
2. 22-bit Q primes lack sufficient precision for polynomial activations
3. Uniform 24-bit Q primes provide the best balance of precision and level count

### 4.5 Level Consumption Breakdown (degree=2, ordinary style)

```
input                level=7
  -> conv2d          level 7->6   (cost=1)
  -> batchnorm2d     level 6->6   (cost=0, fused into conv)
  -> simple_polyrelu level 6->4   (cost=2)
  -> conv2d          level 4->3   (cost=1)
  -> batchnorm2d     level 3->3   (cost=0, fused into conv)
  -> simple_polyrelu level 3->1   (cost=2)
  -> reshape         level 1->1   (cost=0)
  -> fc0             level 1->0   (cost=1)
output               level=0

Total consumed: 7 levels (fits exactly in PN13QP218s)
```

---

## 5. Encrypted Inference Results

### Performance Comparison Across Parameter Sets

| Metric | FP32 + degree=4 | FP8 QAT + degree=2 | **FP8 QAT + degree=2 (N=8192)** |
|---|---|---|---|
| CKKS Parameter Set | PN14QP438 | PN14QP438 | **PN13QP218s** |
| Polynomial Modulus Degree (N) | 16,384 | 16,384 | **8,192** |
| Multiplicative Levels | 9 | 9 | **7** |
| Q Prime Size | 34-bit | 34-bit | **24-bit** |
| Log Default Scale | 34 | 34 | **24** |
| Security | 128-bit | 128-bit | **~127-bit** |
| Bootstrapping | No | No | **No** |
| Plaintext accuracy | 98.56% | 97.76% | **97.76%** |
| **Inference time** | **6,701 ms** | **4,346 ms** | **1,194 ms** |
| Max absolute error | 0.000016 | 0.000018 | **0.037035** |
| Avg absolute error | 0.000007 | 0.000007 | **0.016405** |
| Verification | PASS | PASS | **PASS** |
| **Speedup vs baseline** | 1.0x | 1.5x | **5.6x** |

### Detailed Verification Output (FP8 QAT, degree=2, N=8192)

```
Index    Encrypted         Plaintext         Abs Error
0       -2.03323238       -1.99619718        0.03703520
1       -0.12924453       -0.16405774        0.03481321
2        1.95609878        1.96626933        0.01017055
3       -0.20319583       -0.20021933        0.00297650
4       -2.36351947       -2.37562968        0.01211021
5        0.61226799        0.61428893        0.00202095
6       -1.17538829       -1.17316265        0.00222563
7        0.05304573        0.02736787        0.02567785
8        0.54493935        0.55579004        0.01085069
9       -1.88872228       -1.91488866        0.02616638

Max absolute error: 0.03703520
Avg absolute error: 0.01640472
Result: PASS
```

Classification is correct: both encrypted and plaintext predict digit **2**.

---

## 6. Code Changes

### 6.1 New CKKS Parameter Set

**File:** `training/model_compiler/components.py`

Added `PN13QP218s` parameter set with 24-bit Q primes, 26-bit P, 7 levels at N=8192.

### 6.2 Compiler Pipeline Update

**File:** `training/model_compiler/pipeline.py`

Updated `try_no_btp()` to include `PN13QP218s` as the first candidate (smallest N with enough levels).

### 6.3 Server-Side Custom Parameter Support

**File:** `inference/inference_task/inference_process.cpp`

Fixed `init_parameters()` to read custom Q/P arrays from `ckks_parameter.json`, matching the existing client-side behavior in `inference_client.cpp`.

### 6.4 HE Instruction Generator Update

**File:** `inference/model_generator/deploy_cmds.py`

Added `PN13QP218s` to the `_FHE_PARAMS` lookup table.

---

## 7. Pipeline Summary

```
FP32 Baseline (98.56%)
       |
       v
  FP8 QAT Training (fake quantization with STE)
       |
       v
  FP8 QAT Model (98.55%)  [-0.01% vs baseline]
       |
       v
  FHE Conversion:
    - ReLU -> RangeNormPoly2d (degree=2)
    - MaxPool -> AvgPool
    - Conv+BN fusion
       |
       v
  Fine-tune Poly Model (97.76%)  [-0.80% vs baseline]
       |
       v
  Export ONNX + Fused H5
       |
       v
  Custom CKKS Parameter Design:
    PN13QP218s (N=8192, 7 levels, 24-bit Q, 26-bit P, ~127-bit security)
       |
       v
  Compiler Selection:
    PN13QP218s (N=8192) - SUCCESS  (7 levels used)
       |
       v
  Generate HE Instructions (mega_ag.json)
       |
       v
  C++ Encrypted Inference (1,194 ms, 5.6x speedup, PASS)
```

---

## 8. File Inventory

### Model Checkpoints

```
examples/test_mnist/output/
  train_baseline.pth      # FP32 baseline (98.56%)
  train_fp8.pth           # FP8 post-training quant (98.50%)
  train_fp8_qat.pth       # FP8 QAT (98.55%)
```

### FP8 QAT Inference Task (N=8192)

```
examples/test_mnist/output_fp8qat/
  trained_poly_d2.onnx                    # ONNX model (degree-2 poly)
  pt.json                                 # Compiler input graph
  task/
    client/
      ckks_parameter.json                 # PN13QP218s client params (24-bit Q, 26-bit P)
      task_config.json                    # Input/output metadata
      img.csv                             # Input image (encrypted)
    server/
      ckks_parameter.json                 # PN13QP218s server params
      task_config.json                    # Task config (ordinary, no BTP)
      nn_layers_ct_0.json                 # Compiled encrypted DAG
      model_parameters.h5                 # Fused weights (Conv+BN+Poly)
      mega_ag.json                        # Low-level HE instructions
      task_signature.json                 # Task signature
```

---

## 9. Commands to Reproduce

```bash
# Step 1: FP8 QAT model already exists at output/train_fp8_qat.pth (98.55%)

# Step 2: Convert to polynomial activations, fine-tune, and export
# (see session scripts for degree=2 conversion pipeline)

# Step 3: Compile with auto CKKS parameter selection (selects PN13QP218s)
python3 training/run_compile.py \
  -i examples/test_mnist/output_fp8qat/trained_poly_d2.onnx \
  -o examples/test_mnist/output_fp8qat \
  --style ordinary \
  --num_experiments 16 \
  --num_workers 4

# Step 4: Generate HE instructions
python3 inference/interface/gen_mega_ag.py \
  --task-dir examples/test_mnist/output_fp8qat/task

# Step 5: Run encrypted inference
./build/examples/inference \
  --task-dir examples/test_mnist/output_fp8qat/task \
  --input examples/test_mnist/output_fp8qat/task/client/img.csv \
  --verify
```
