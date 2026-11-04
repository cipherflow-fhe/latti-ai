# imagenet GPU runtime_lazy 全流程

命令从仓库任意目录启动，在同一个 shell 中依次执行。所有 task、密钥、日志和输出位于 `runs/imagenet/`；各模式顺序运行。验收比较原始 ONNX 与 CSV 的全部输出，最大绝对误差必须小于 0.2。

## 环境与构建

```bash
REPO="$(git rev-parse --show-toplevel)"
PY=/home/linghm/pytorch_venv/bin/python3
CMAKE=/home/linghm/pytorch_venv/bin/cmake
ROOT="$REPO/runs/imagenet"
MODEL="$ROOT/model/trained_poly.onnx"
export PYTHONPATH="$REPO/inference/lattisense:$REPO"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$CMAKE" -S "$REPO" -B "$REPO/build" -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_DEV=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$REPO/build" -j8
mkdir -p "$ROOT/reference"
cd "$ROOT"
```

CPU 数据为推理进程生命周期的最大 RSS（GNU time）；GPU 数据包括每 0.5 秒采样的进程显存峰值和 SDK 每 0.1 秒采样的池内使用峰值。GPU 进程显存含内存池预留。FHE 时间取 SDK 的 `Run GPU time`，不含编译、参数加密、密钥准备和结果解密。GiB = 2³⁰ bytes。

输入加密使用默认编译配置，不传 `--config`。参数加密使用 `training/config/imagenet_storage_config.json`，其中 seed stride `[8,8]` 和存储评分权重为非默认设置。

## 原始 ONNX 明文参考

```bash
"$PY" "$REPO/tools/evaluate_onnx_csv.py" --onnx "$MODEL" --input-csv "$ROOT/img.csv" --output-csv "$ROOT/reference/onnx.csv" --report "$ROOT/reference/onnx.json"
```

## 输入加密

```bash
OUT="$ROOT/task_input_encrypted"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run2" -- "$REPO/build/examples/inference" --task-dir "$OUT/task" --input "$ROOT/img.csv" --output-csv "$OUT/decrypted.csv" --gpu
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_input_encrypted/run2/inference.metrics.json`、`task_input_encrypted/comparison.json`。

## 参数加密

```bash
OUT="$ROOT/task_parameter_encrypted"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed --config "$REPO/training/config/imagenet_storage_config.json" --deployment-mode server_provisioned_runner
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task" --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
/usr/bin/time -v -o "$OUT/provision.time" "$REPO/build/tools/provision_encrypted_runner_bundle" --task-dir "$OUT/task" --out "$OUT/task/runner"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run2" -- "$REPO/build/examples/inference" --task-dir "$OUT/task/runner" --deployment-mode server_provisioned_runner --input "$ROOT/img.csv" --output-cipher "$OUT/run2/output.ct" --gpu
"$REPO/build/tools/decrypt_runner_output" --task-dir "$OUT/task/provisioner" --cipher "$OUT/run2/output.ct" --output-csv "$OUT/decrypted.csv"
"$PY" "$REPO/tools/summarize_parameter_bundle.py" --task-dir "$OUT/task" --output "$OUT/bundle.json"
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_parameter_encrypted/run2/inference.metrics.json`、`task_parameter_encrypted/comparison.json`、`task_parameter_encrypted/bundle.json`、`task_parameter_encrypted/provision.time`。

加密参数目录包含密文和 manifest，共 2,749,043,721 bytes；纯密文 payload 2,748,503,014 bytes；含图和求值密钥的完整 runner 6,386,843,102 bytes。

bundle provisioning 使用 `/usr/bin/time -v` 记录：wall 2:14.30，user 149.91 s，system 41.57 s，峰值 RSS 28016748 KiB。

## 实测记录

| 模式 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | FHE 秒 | 参数 bundle 加密 wall | 加密参数目录 bytes | 最大绝对误差 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 输入加密 | 53.582 | 42.941 | 14.185 | 52.1947 | — | 不适用 | 0.00355737102 | 通过 |
| 参数加密 | 31.305 | 42.941 | 19.546 | 88.3113 | 2:14.30 | 2,749,043,721 | 0.00395387736 | 通过 |
