# ECG Anomaly Detection and FHE Encrypted Inference System Documentation

# 1. Environment Configuration

## 1.1 Hardware Requirements

| Item             | Requirement                                                  |
| :--------------- | :----------------------------------------------------------- |
| CPU              | ≥ 4 cores                                                    |
| Memory           | ≥ 8GB (16GB recommended)                                     |
| GPU              | Not required                                                 |
| Operating System | Linux (ciphertext inference with operator replacement) / Windows (data processing) |

------



## 1.2 Software Environment

| Software     | Version Requirement |
| :----------- | :------------------ |
| Python       | ≥ 3.8               |
| PyTorch      | ≥ 1.10              |
| NumPy        | ≥ 1.20              |
| scikit-learn | ≥ 1.0               |
| wfdb         | ≥ 4.0               |
| tqdm         | Any stable version  |

Install Python dependencies with:

```
pip install -r examples/test_ecg/requirements.txt
```

------



## 1.3 FHE Environment

latti-ai library, official repository: [cipherflow-fhe/latti-ai: A framework for performing AI model inference on encrypted data.](https://github.com/cipherflow-fhe/latti-ai)

# 2. Dependencies

## 2.1 Data Dependencies

| Item          | Content                                    |
| :------------ | :----------------------------------------- |
| Dataset       | MIT-BIH Arrhythmia Database                |
| URL           | https://physionet.org/content/mitdb/1.0.0/ |
| Sampling Rate | 360Hz                                      |
| Data Type     | ECG + annotations                          |

------

## 2.2 Data Processing Logic

| Step           | Description                      |
| :------------- | :------------------------------- |
| R-peak slicing | 256 points (99 left + 156 right) |
| Normalization  | Z-score                          |
| Reshape        | 1×256 → 16×16                    |
| Input format   | (1,16,16)                        |

------

## 2.3 Label Definition

| Class    | Label | Symbol                       |
| :------- | :---- | :--------------------------- |
| Normal   | 0     | N, L, R, e, j                |
| Abnormal | 1     | A, a, J, S, V, E, F, /, f, Q |

------

## 2.4 Data Imbalance Description

| Dataset | Normal | Abnormal | Ratio   |
| :------ | :----- | :------- | :------ |
| Train   | 72068  | 8485     | 8.5 : 1 |
| Val     | 9009   | 1060     | 8.5 : 1 |
| Test    | 9009   | 1061     | 8.5 : 1 |

------

## 2.5 Handling Strategy

- Training set: **Oversampling → 1:1**
- Validation / Test: **Keep original distribution**

Reason: Prevent model bias toward the normal class while ensuring realistic evaluation.

Model definition path: `examples/test_ecg/model/model.py`.

# 3. Running Steps

## Overall Pipeline

```
Data Processing → Plaintext Training → FHE Adaptation → Model Compilation → Encrypted Inference
```

------

## 3.1 Data Processing

```
python examples/test_ecg/data_prepare.py \
  --data-dir /path/to/mit-bih-arrhythmia-database-1.0.0 \
  --output-root ./examples/test_ecg/data
```

Output:

```
examples/test_ecg/data/processed_over_1to1/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
├── y_test.npy
```

------

Note: Data processing can run on Windows or Linux. Pass local paths through command-line arguments and ensure paths match.

Note: Due to GitHub file upload limits, the datasets used in this experiment have been uploaded to email. It is recommended to obtain the `processed_over_1to1.zip` file, extract it, and place it directly under `./examples/test_ecg/data/processed_over_1to1`.。

## 3.2 Plaintext Model Training

```
python examples/test_ecg/train.py \
  --model-name two_conv \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --num-workers 4 \
  --torch-num-threads 4 \
  --num-classes 2 \
  --processed-dir ./examples/test_ecg/data/processed_over_1to1 \
  --output-dir ./examples/test_ecg/runs/exp_over/model \
  --input-shape 1 16 16
```

------

## 3.3 FHE Model Adaptation (Poly Replacement)

```
python examples/test_ecg/train.py \
  --poly_model_convert \
  --model-name two_conv \
  --pretrained ./examples/test_ecg/runs/exp_over/model/train_baseline.pth \
  --epochs 3 \
  --batch-size 16 \
  --lr 0.0005 \
  --num-workers 4 \
  --torch-num-threads 4 \
  --num-classes 2 \
  --processed-dir ./examples/test_ecg/data/processed_over_1to1 \
  --output-dir ./examples/test_ecg/runs/exp_over/model \
  --export-dir ./examples/test_ecg/runs/exp_over/task/server \
  --input-shape 1 16 16 \
  --degree 4 \
  --upper-bound 3.0 \
  --poly-module RangeNormPoly2d
```

------

## 3.4 Model Compilation

```
python training/run_compile.py \
  --input ./examples/test_ecg/runs/exp_over/model/trained_poly.onnx \
  --output ./examples/test_ecg/runs/exp_over \
  --style multiplexed
```

------

## 3.5 Generate Execution Configuration

```
python inference/interface/gen_mega_ag.py --task-dir ./examples/test_ecg/runs/exp_over/task
```

------

## 3.6 Sample Plaintext Inference

```
python examples/test_ecg/prepare_plaintext_samples.py \
  --processed-dir ./examples/test_ecg/data/processed_over_1to1 \
  --baseline-ckpt ./examples/test_ecg/runs/exp_over/model/train_baseline.pth \
  --task-dir ./examples/test_ecg/runs/exp_over/task \
  --model-name two_conv \
  --dataset-split test \
  --normal-count 200 \
  --abnormal-count 200 \
  --output-subdir client_batch400
```

**Parameter Configuration Description**

|   Parameter Name   | Type |                        Default Value                         |                         Description                          |
| :----------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| `--processed-dir`  | str  |          `./examples/test_ecg/data/processed_over_1to1`           |          Directory path of the preprocessed dataset          |
| `--baseline-ckpt`  | str  | `./examples/test_ecg/runs/exp_over/model/train_baseline.pth` |      File path of the plaintext baseline model weights       |
|    `--task-dir`    | str  |           `./examples/test_ecg/runs/exp_over/task`           |             Root directory path of the FHE task              |
|   `--model-name`   | str  |                          `two_conv`                          | Model name (options: `tiny_cnn`/`tiny_cnn8`/`two_conv`/`mlp_head`) |
| `--dataset-split`  | str  |                            `test`                            |  Dataset split for sample selection (options: `val`/`test`)  |
|  `--normal-count`  | int  |                            `200`                             |             Number of normal samples to extract              |
| `--abnormal-count` | int  |                            `200`                             |            Number of abnormal samples to extract             |
| `--output-subdir`  | str  |                      `client_batch400`                       | Name of the output subdirectory (created under the task-dir) |

## 3.7 Sample Ciphertext Inference

```
python examples/test_ecg/run_fhe_batch.py
```

**Parameter Configuration Description**

|  Parameter Name  | Type |              Default Value               |                         Description                          |
| :--------------: | :--: | :--------------------------------------: | :----------------------------------------------------------: |
|   `--task-dir`   | str  | `./examples/test_ecg/runs/exp_over/task` |             Root directory path of the FHE task              |
|    `--binary`    | str  |       `./build/examples/inference`       |          File path of the LattiAI inference binary           |
|   `--threads`    | int  |                   `1`                    | Number of OpenMP threads for inference (recommended: CPU physical core count) |
| `--input-subdir` | str  |            `client_batch400`             | Name of the input sample subdirectory (must match the output directory of prepare_plaintext_samples.py) |

# 4. Results Description

------

## 4.1 Plaintext Model Performance

**Overall Metrics**

| Metric      | Value  |
| :---------- | :----- |
| Accuracy    | 0.9325 |
| Weighted F1 | 0.8932 |
| ROC-AUC     | 0.9438 |
| PR-AUC      | 0.8133 |

------

### 4.1.1 Per-Class Metrics

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Normal   | 0.9809    | 0.9428 | 0.9615   | 9009    |
| Abnormal | 0.8350    | 0.8445 | 0.8249   | 1061    |

------

### 4.1.2 Confusion Matrix

| Actual\Predicted | Normal | Abnormal |
| ---------------- | ------ | -------- |
| Normal           | 8494   | 515      |
| Abnormal         | 165    | 896      |

------

## 4.2 FHE Inference Results

### 4.2.1 Single Sample

| Metric         | Value  |
| -------------- | ------ |
| Inference Time | ≈ 11s  |
| Max Error      | 0.0023 |
| Result         | PASS   |

------

### 4.2.2 200 Samples

| Category                | Metric                         | Plaintext Model | FHE Encrypted Model |
| :---------------------- | :----------------------------- | :-------------- | :------------------ |
| **Overall Performance** | Accuracy                       | 86.00%          | 82.00%              |
|                         | Macro F1                       | 85.96%          | 81.99%              |
|                         | Weighted F1                    | 85.96%          | 81.99%              |
| **Normal Class**        | Precision                      | 82.73%          | 80.77%              |
|                         | Recall                         | 91.00%          | 84.00%              |
|                         | F1-score                       | 86.67%          | 82.35%              |
|                         | Support                        | 100             | 100                 |
| **Abnormal Class**      | Precision                      | 90.00%          | 83.33%              |
|                         | Recall                         | 81.00%          | 80.00%              |
|                         | F1-score                       | 85.26%          | 81.63%              |
|                         | Support                        | 100             | 100                 |
| **Consistency Metric**  | Plaintext-Ciphertext Agreement | -               | 91.00%              |

### 4.2.3 400 Samples

| Category                | Metric                         | Plaintext Model | FHE Encrypted Model |
| :---------------------- | :----------------------------- | :-------------- | :------------------ |
| **Overall Performance** | Accuracy                       | 88.50%          | 83.50%              |
|                         | Macro F1                       | 88.49%          | 83.50%              |
|                         | Weighted F1                    | 88.49%          | 83.50%              |
| **Normal Class**        | Precision                      | 85.98%          | 83.17%              |
|                         | Recall                         | 92.00%          | 84.00%              |
|                         | F1-score                       | 88.89%          | 83.58%              |
|                         | Support                        | 200             | 200                 |
| **Abnormal Class**      | Precision                      | 91.40%          | 83.84%              |
|                         | Recall                         | 85.00%          | 83.00%              |
|                         | F1-score                       | 88.08%          | 83.42%              |
|                         | Support                        | 200             | 200                 |
| **Consistency Metric**  | Plaintext-Ciphertext Agreement | -               | 91.00%              |
