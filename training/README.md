# Model Preparation and Compilation for Encrypted Inference

This guide describes how to use the scripts in the `training/` directory to generate the configuration files and model weights required by `examples/test_cifar10` for encrypted inference.

All commands below should be executed from the `training/` directory:

```bash
cd training
```

## Prerequisites

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The full pipeline consists of three steps:

```
Baseline Training  →  Operator Replacement & Fine-tuning  →  Model Compilation
     (Step 1)                   (Step 2)                        (Step 3)
```

---

## Step 1: Baseline Training

Train a ResNet-20 model on CIFAR-10 with standard ReLU activations.

```bash
python example/train.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir ../runs/cifar10/model
```

This trains a standard ResNet-20 and saves the best checkpoint to `../runs/cifar10/model/tarin_baseline.pth`.

**Output:**

| File | Description |
|------|-------------|
| `runs/cifar10/model/tarin_baseline.pth` | Best baseline model checkpoint |

---

## Step 2: Operator Replacement & Fine-tuning

Replace ReLU activations and max pooling with FHE-friendly polynomial approximations (`RangeNormPoly2d`) and average pooling, respectively, then fine-tune the model to recover accuracy. After fine-tuning, the script automatically exports the model to ONNX format and saves model weights in a H5 file.

```bash
python example/train.py --poly_model_convert --pretrained ../runs/cifar10/model/tarin_baseline.pth --epochs 10 --batch-size 36 --lr 0.001 --input-dir ../runs/cifar10/model --export-dir ../runs/cifar10/task/server --input-shape 3 32 32
```

- `--poly_model_convert`: enables ReLU → RangeNormPoly2d and max pooling → average pooling replacements.
    ```python
    from nn_tools import replace_activation_with_poly, replace_maxpool_with_avgpool, export_to_onnx, fuse_and_export_h5

    replace_maxpool_with_avgpool(model)
    replace_activation_with_poly(model, old_cls=nn.ReLU,
                                 upper_bound=args.upper_bound,
                                 degree=args.degree)
    onnx_path = os.path.join(args.output_dir, 'trained_poly.onnx')
    export_to_onnx(model, save_path=onnx_path,
                   input_size=tuple([1, *args.input_shape]),
                   dynamic_batch=False)

    h5_path = os.path.join(export_dir, 'model_parameters.h5')
    fuse_and_export_h5(model, h5_path=h5_path,
                       upper_bound=args.upper_bound,
                       degree=args.degree, eps=1e-3)
    ```
- `--pretrained`: loads the baseline checkpoint from Step 1.
- `--input-dir`: directory containing the baseline model (also used as output for `.pth` and `.onnx`).
- `--export-dir`: directory for the H5 weight file, corresponding to the server-side model weights.

**Output:**

| File | Description |
|------|-------------|
| `runs/cifar10/model/train_poly.pth` | Best checkpoint for the adapted model with polynomial activations|
| `runs/cifar10/model/trained_poly.onnx` |  The exported adapted model in ONNX format|
| `runs/cifar10/task/server/model_parameters.h5` | Model weights (BatchNorm absorbed into Conv) |

---

## Step 3: Model Compilation

Convert the ONNX model into the encrypted computation graph and generate all configuration files needed for both client and server.

### Compile encrypted computation graph

```bash
python run_compile.py --input=../runs/cifar10/model/trained_poly.onnx --output=../runs/cifar10/ --poly_n=65536 --style=multiplexed
```

- `--input`: the exported adapted model in ONNX format from Step 2.
- `--output`: root output directory; the compiler generates `task/server/` and `task/client/` subdirectories underneath.
- `--poly_n`: polynomial modulus degree for CKKS (determines the number of ciphertext slots and security level).
- `--style`: packing style — `multiplexed` (channel-multiplexed packing for higher slot utilization) or `ordinary` (one channel per ciphertext).

**Output:**

| File | Description |
|------|-------------|
| `runs/cifar10/model/pt.json` | Intermediate computation graph (JSON) |
| `runs/cifar10/task/server/task_config.json` | Server-side inference task configuration |
| `runs/cifar10/task/server/ckks_parameter.json` | CKKS encryption parameter configuration |
| `runs/cifar10/task/server/ergs/erg0.json` | Compiled encrypted computation graph (DAG) |
| `runs/cifar10/task/client/task_config.json` | Client-side inference task configuration |
| `runs/cifar10/task/client/ckks_parameter.json` | CKKS encryption parameter configuration |

---

## Final Directory Structure

After completing all three steps, the `runs/cifar10/` directory will have the following structure:

```
runs/cifar10/
├── model/
│   ├── tarin_baseline.pth          # Step 1: baseline model checkpoint
│   ├── train_poly.pth              # Step 2: adapted model checkpoint
│   ├── trained_poly.onnx           # Step 2: exported adapted model
│   └── pt.json                     # Step 3: intermediate graph
└── task/
    ├── server/
    │   ├── task_config.json         # Server inference config
    │   ├── ckks_parameter.json      # CKKS parameters
    │   ├── model_parameters.h5      # Model weights
    │   └── ergs/
    │       └── erg0.json            # Encrypted computation graph
    └── client/
        ├── task_config.json         # Client inference config
        └── ckks_parameter.json      # CKKS parameters
```

The `task/server/` and `task/client/` directories share the same structure as `examples/test_cifar10/task/server/` and `examples/test_cifar10/task/client/`, and can be used directly for encrypted inference.
