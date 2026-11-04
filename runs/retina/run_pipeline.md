# retina GPU runtime_lazy 全流程

命令从仓库任意目录启动，在同一个 shell 中依次执行。所有 task、密钥、日志和输出位于 `runs/retina/`；各模式顺序运行。验收比较原始 ONNX 与 CSV 的全部输出，最大绝对误差必须小于 0.2。

## 环境与构建

```bash
REPO="$(git rev-parse --show-toplevel)"
PY=/home/linghm/pytorch_venv/bin/python3
CMAKE=/home/linghm/pytorch_venv/bin/cmake
ROOT="$REPO/runs/retina"
MODEL="$ROOT/model/epoch041.onnx"
export PYTHONPATH="$REPO/inference/lattisense:$REPO"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$CMAKE" -S "$REPO" -B "$REPO/build" -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_DEV=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$REPO/build" -j8
mkdir -p "$ROOT/reference"
cd "$ROOT"
```

CPU 数据为推理进程生命周期的最大 RSS（GNU time）；GPU 数据包括每 0.5 秒采样的进程显存峰值和 SDK 每 0.1 秒采样的池内使用峰值。GPU 进程显存含内存池预留。FHE 时间取 SDK 的 `Run GPU time`，不含编译、参数加密、密钥准备和结果解密。GiB = 2³⁰ bytes。

输入加密使用默认编译配置，不传 `--config`。参数加密共用 `training/config/storage_aware_config.json`，其中 seed stride `[16,16]` 和存储评分权重为非默认设置。

## 原始 ONNX 明文参考

```bash
"$PY" "$REPO/tools/evaluate_onnx_csv.py" --onnx "$MODEL" --input "input_0=$ROOT/img.csv" --input "input_1=$ROOT/non_inv.csv" --input "input_2=$ROOT/inv.csv" --output-csv "$ROOT/reference/onnx.csv" --report "$ROOT/reference/onnx.json"
```

## 输入加密

```bash
OUT="$ROOT/task_input_encrypted"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task" --input "input_0=$ROOT/img.csv" --input "input_1=$ROOT/non_inv.csv" --input "input_2=$ROOT/inv.csv" --output-csv "$OUT/decrypted.csv" --gpu
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_input_encrypted/run/inference.metrics.json`、`task_input_encrypted/comparison.json`。

## 参数加密

```bash
OUT="$ROOT/task_parameter_encrypted"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed --config "$REPO/training/config/storage_aware_config.json" --deployment-mode server_provisioned_runner
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task" --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
/usr/bin/time -v -o "$OUT/provision.time" "$REPO/build/tools/provision_encrypted_runner_bundle" --task-dir "$OUT/task" --out "$OUT/task/runner"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task/runner" --deployment-mode server_provisioned_runner --input "input_0=$ROOT/img.csv" --input "input_1=$ROOT/non_inv.csv" --input "input_2=$ROOT/inv.csv" --output-cipher "$OUT/output.ct" --gpu
"$REPO/build/tools/decrypt_runner_output" --task-dir "$OUT/task/provisioner" --cipher "$OUT/output.ct" --output-csv "$OUT/decrypted.csv"
"$PY" "$REPO/tools/summarize_parameter_bundle.py" --task-dir "$OUT/task" --output "$OUT/bundle.json"
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_parameter_encrypted/run/inference.metrics.json`、`task_parameter_encrypted/comparison.json`、`task_parameter_encrypted/bundle.json`、`task_parameter_encrypted/provision.time`。

加密参数目录包含密文和 manifest，共 11,604,074,214 bytes；纯密文 payload 11,603,165,289 bytes；含图和求值密钥的完整 runner 19,591,960,539 bytes。

bundle provisioning 使用 `/usr/bin/time -v` 记录：wall 12:44.41，user 671.13 s，system 75.51 s，峰值 RSS 29378848 KiB。

## 实测记录

| 模式 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | FHE 秒 | 参数 bundle 加密 wall | 加密参数目录 bytes | 最大绝对误差 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 输入加密 | 97.295 | 42.941 | 20.096 | 137.687 | — | 不适用 | 0.000178451652 | 通过 |
| 参数加密 | 58.696 | 42.941 | 34.083 | 1092.32 | 12:44.41 | 11,604,074,214 | 0.000242663423 | 通过 |
