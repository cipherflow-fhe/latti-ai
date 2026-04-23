# ECG Abnormal Detection Example

This directory contains an end-to-end encrypted inference example based on the MIT-BIH Arrhythmia Database, which fully demonstrates how to adapt a lightweight CNN-based ECG classification model into a privacy-preserving encrypted inference service using the LattiAI Fully Homomorphic Encryption (FHE) framework.

| Example Name |   Core Model   |           Dataset           | Input Size  | Encryption Scheme | Bootstrapping |
| :----------: | :------------: | :-------------------------: | :---------: | :---------------: | :-----------: |
| **test_ecg** | TinyECGTwoConv | MIT-BIH Arrhythmia Database | 1 x 16 x 16 |  CKKS (N=16384)   |      No       |

## 1. Model Description

The **TinyECGTwoConv** adopted in this example is an ultra-lightweight CNN model customized for the computational characteristics of FHE. Its core design and advantages are as follows:

- **Network Architecture**: 2 convolutional layers (1→4→8 channels) + 1 ReLU activation function + Global Average Pooling layer + 1 fully connected classification head
- **Parameter Scale**: Only 340 trainable parameters, with extremely low computational depth, which greatly reduces noise accumulation and time consumption during homomorphic computation
- **Classification Performance**: Achieves 93.25% classification accuracy on the MIT-BIH test set, with 84.45% recall for abnormal heartbeats

## 2. Data Preparation

This example uses the **MIT-BIH Arrhythmia Database (version 1.0.0)** as the experimental dataset, which is an authoritative public dataset in the field of arrhythmia analysis.

- **Official Data Source**: https://physionet.org/content/mitdb/1.0.0/

- Preprocessing Pipeline

  1. Based on the R-wave peak annotations, extract single heartbeat segments of 256 sampling points centered on the R-peak
  2. Perform Z-score normalization on the heartbeat segments to eliminate amplitude differences
  3. Reshape the 1D 256-dimensional signal into a 16×16 2D matrix to fit the model input format
  4. Binary label encoding: normal heartbeats are labeled as 0, and abnormal heartbeats are labeled as 1

  

- **Dataset Balancing**: The original dataset has a positive-negative sample ratio of approximately 9:1. Only the training set is processed with oversampling to make the positive-negative sample ratio 1:1; the validation set and test set retain the original distribution to ensure the objectivity of evaluation results.

- **Preprocessing Output**: Generate the `processed_over_1to1/` directory, which contains npy format data of the training/validation/test sets and dataset metadata files.

## 3. Running Instructions

### Prerequisites

1. You have completed the compilation and build of the LattiAI main project, and can normally call the `./build/examples/inference` inference binary file
2. You have completed the download and preprocessing of the MIT-BIH dataset (can be done with one click via `data_prepare_win.py`)
3. You have completed model training, FHE operator adaptation and compilation, and generated the `task/` directory required for encrypted inference

### Quick Start (Inference Only)

If you have prepared the pre-compiled `task/` directory, you can directly execute the following commands to complete encrypted inference:

```
# Generate low-level FHE execution instructions
python inference/interface/gen_mega_ag.py --task-dir ./examples/my_ecg001/runs/exp_over009/task

# Run single-sample encrypted inference (CPU mode, with result verification)
./build/examples/inference --task-dir ./examples/my_ecg001/runs/exp_over009/task --input ./examples/my_ecg001/runs/exp_over009/task/client/ecg_input.csv --verify

# Run single-sample encrypted inference (GPU mode, with result verification, GPU build required)
./build/examples/inference --task-dir ./examples/my_ecg001/runs/exp_over009/task --input ./examples/my_ecg001/runs/exp_over009/task/client/ecg_input.csv --verify --gpu
```

### Full Pipeline (From Training to Inference)

If you need to execute the complete pipeline from scratch: **data preprocessing → model training → FHE adaptation → model compilation → encrypted inference**, follow the steps below:

#### 3.1 Plaintext Baseline Model Training

```
python examples/my_ecg001/train.py \
  --model-name two_conv \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --num-workers 4 \
  --torch-num-threads 4 \
  --num-classes 2 \
  --processed-dir ./examples/my_ecg001/processed_over_1to1 \
  --output-dir ./examples/my_ecg001/runs/exp_over009/model \
  --input-shape 1 16 16
```

#### 3.2 FHE Operator Replacement and Model Fine-tuning

```
python examples/my_ecg001/train.py \
  --poly_model_convert \
  --model-name two_conv \
  --pretrained ./examples/my_ecg001/runs/exp_over009/model/train_baseline.pth \
  --epochs 3 \
  --batch-size 16 \
  --lr 0.0005 \
  --num-workers 4 \
  --torch-num-threads 4 \
  --num-classes 2 \
  --processed-dir ./examples/my_ecg001/processed_over_1to1 \
  --output-dir ./examples/my_ecg001/runs/exp_over009/model \
  --export-dir ./examples/my_ecg001/runs/exp_over009/task/server \
  --input-shape 1 16 16 \
  --degree 4 \
  --upper-bound 3.0 \
  --poly-module RangeNormPoly2d
```

#### 3.3 Model Compilation

```
python training/run_compile.py \
  --input ./examples/my_ecg001/runs/exp_over009/model/trained_poly.onnx \
  --output ./examples/my_ecg001/runs/exp_over009 \
  --style multiplexed
```

#### 3.4 Generate Low-level FHE Execution Instructions

```
python inference/interface/gen_mega_ag.py --task-dir ./examples/my_ecg001/runs/exp_over009/task
```

#### 3.5 Generate Batch Test Samples

```
python examples/my_ecg001/prepare_ten_samples.py
```

**Parameter Configuration Description**

|   Parameter Name   | Type |                        Default Value                         |                         Description                          |
| :----------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| `--processed-dir`  | str  |          `./examples/my_ecg001/processed_over_1to1`          |          Directory path of the preprocessed dataset          |
| `--baseline-ckpt`  | str  | `./examples/my_ecg001/runs/exp_over009/model/train_baseline.pth` |      File path of the plaintext baseline model weights       |
|    `--task-dir`    | str  |         `./examples/my_ecg001/runs/exp_over009/task`         |             Root directory path of the FHE task              |
|   `--model-name`   | str  |                          `two_conv`                          | Model name (options: `tiny_cnn`/`tiny_cnn8`/`two_conv`/`mlp_head`) |
| `--dataset-split`  | str  |                            `test`                            |  Dataset split for sample selection (options: `val`/`test`)  |
|  `--normal-count`  | int  |                             `5`                              |             Number of normal samples to extract              |
| `--abnormal-count` | int  |                             `5`                              |            Number of abnormal samples to extract             |
| `--output-subdir`  | str  |                       `client_batch10`                       | Name of the output subdirectory (created under the task-dir) |

#### 3.6 Execute Batch Encrypted Inference and Result Summary

```
python examples/my_ecg001/run_batch_fhe.py
```

**Parameter Configuration Description**

|  Parameter Name  | Type |                Default Value                 |                         Description                          |
| :--------------: | :--: | :------------------------------------------: | :----------------------------------------------------------: |
|   `--task-dir`   | str  | `./examples/my_ecg001/runs/exp_over009/task` |             Root directory path of the FHE task              |
|    `--binary`    | str  |         `./build/examples/inference`         |          File path of the LattiAI inference binary           |
|   `--threads`    | int  |                     `1`                      | Number of OpenMP threads for inference (recommended: CPU physical core count) |
| `--input-subdir` | str  |               `client_batch10`               | Name of the input sample subdirectory (must match the output directory of prepare_ten_samples.py) |

## 4. Directory Structure

```
my_ecg001/
├── __init__.py                 # Package initialization file, module import declaration
├── augment.py                  # Customized data augmentation module for ECG signals
├── dataset.py                  # PyTorch dataset loading and encapsulation module
├── losses.py                   # Loss function module adapted for class imbalance
├── model.py                    # Lightweight CNN model definition and construction module
├── train.py                    # Main program for model training, FHE operator adaptation and export
├── pick_plaintext_sample.py    # Inference sample selection and plaintext benchmark generation module
├── prepare_ten_samples.py      # Batch test sample generation module
├── run_fhe_once.py             # Single-sample encrypted inference wrapper script
├── run_batch_fhe.py            # Batch sample encrypted inference and statistics script
├── data_prepare_win.py         # Data preprocessing script
└── utils_exp.py                # General utility function module for experiments
```

## 5. Citation

When using this dataset, please cite the original publication:

```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). PMID: 11446209.
```

And the standard citation for PhysioNet:

```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank
```