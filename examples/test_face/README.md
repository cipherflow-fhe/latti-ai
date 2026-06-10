# test_face

`test_face` is a FaceNet-style face-recognition example built on the same MobileNetV2 backbone used by `examples/test_imagenet`.

The training method follows `/home/zhongy/facenet-pytorch-change`:

- sample anchor / positive / negative triplets
- train a 128-d embedding head with triplet loss
- train an identity classifier head with cross entropy
- optionally replace MobileNetV2 `ReLU6` with polynomial activations and export ONNX/H5 for the Latti compile pipeline

## Directory layout

```text
examples/test_face/
├── README.md
├── train.py
├── model/
│   ├── __init__.py
│   ├── mobilenetv2.py              # copied from examples/test_imagenet/model/mobilenetv2.py
│   └── facenet_mobilenetv2.py      # MobileNetV2 + FaceNet embedding/classifier heads
└── utils/
    ├── __init__.py
    ├── dataloader.py               # triplet sampler
    └── losses.py                   # triplet + CE loss
```

## Dataset annotation format

`train.py` expects the same annotation format as `facenet-pytorch-change`:

```text
<class_id>;<absolute_or_relative_image_path>
```

Example:

```text
0;/path/to/person_a/0001.jpg
0;/path/to/person_a/0002.jpg
1;/path/to/person_b/0001.jpg
1;/path/to/person_b/0002.jpg
```

Each class used for anchor/positive sampling must contain at least two images. Negative sampling requires at least two classes.

The default annotation path is:

```text
/home/zhongy/facenet-pytorch-change/cls_train_2.txt
```

Override it with `--annotation-path` if needed.

## 1. Train baseline FaceNet-MobileNetV2

Run from the repo root:

```bash
cd /home/zhongy/test/latti-ai
python examples/test_face/train.py \
  --annotation-path /home/zhongy/facenet-pytorch-change/cls_train_2.txt \
  --epochs 500 \
  --batch-size 24 \
  --input-shape 3 256 256 \
  --output-dir examples/test_face/output \
  --gpu 0
```

Notes:

- `--batch-size` is the effective batch size after concatenating anchor/positive/negative images.
- It must be a multiple of 3.
- `--batch-size 24` means the dataloader samples 8 triplets and collates them into 24 images.
- Checkpoints are saved to:

```text
examples/test_face/output/best.pth
examples/test_face/output/last.pth
```

For CPU smoke testing, use:

```bash
python examples/test_face/train.py \
  --annotation-path /home/zhongy/facenet-pytorch-change/cls_train_2.txt \
  --epochs 1 \
  --batch-size 6 \
  --num-workers 0 \
  --gpu -1 \
  --output-dir examples/test_face/output_cpu
```

## 2. Convert to polynomial activations and fine-tune

The Latti FHE pipeline needs polynomial activations. Use the baseline checkpoint, enable `--poly_model_convert`, fine-tune, and export:

```bash
python examples/test_face/train.py \
  --annotation-path /home/zhongy/facenet-pytorch-change/cls_train_2.txt \
  --pretrained examples/test_face/output/best.pth \
  --poly_model_convert \
  --poly-module RangeNormPoly2d \
  --upper-bound 3.0 \
  --degree 4 \
  --epochs 10 \
  --batch-size 96 \
  --input-shape 3 256 256 \
  --output-dir examples/test_face/output_poly \
  --export \
  --gpu 0
```

This writes:

```text
examples/test_face/output_poly/best.pth
examples/test_face/output_poly/last.pth
examples/test_face/output_poly/trained_poly.onnx
examples/test_face/output_poly/model_parameters.h5
```

The exported ONNX model outputs the 128-d face embedding, not classifier logits.

## 3. Compile ONNX to a Latti task

After `trained_poly.onnx` exists, compile it with the repository compile pipeline:

```bash
python training/run_compile.py \
  -i examples/test_face/output_poly/trained_poly.onnx \
  -o examples/test_face \
  --style multiplexed
```

The compile output should be under:

```text
examples/test_face/task/client/
examples/test_face/task/server/
```

If instruction generation is needed, run:

```bash
python inference/interface/gen_mega_ag.py --task-dir examples/test_face/task
```

Expected key artifacts:

```text
examples/test_face/task/client/task_config.json
examples/test_face/task/client/ckks_parameter.json
examples/test_face/task/server/task_config.json
examples/test_face/task/server/ckks_parameter.json
examples/test_face/task/server/nn_layers_ct_0.json
examples/test_face/task/server/model_parameters.h5
examples/test_face/task/server/mega_ag.json
```

## 4. Prepare inference input CSV

The unified example inference binary expects a CSV input file matching the compiled task input shape.

For this model, the input shape is normally:

```text
3 x 256 x 256
```

Place a normalized input CSV at:

```text
examples/test_face/task/client/img.csv
```

The normalization used by training is:

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

So pixel values should be converted from `[0, 1]` to approximately `[-1, 1]` before writing the CSV.

## 5. Optional encrypted/plaintext verification

After building the C++ examples and preparing `task/client/img.csv`, run:

```bash
./build/examples/inference \
  --task-dir examples/test_face/task \
  --input examples/test_face/task/client/img.csv \
  --verify
```

GPU mode:

```bash
./build/examples/inference \
  --task-dir examples/test_face/task \
  --input examples/test_face/task/client/img.csv \
  --verify \
  --gpu
```

`examples/CMakeLists.txt` is not updated yet. Add `test_face` there only after `examples/test_face/task` and `task/client/img.csv` are generated and ready to be committed.
