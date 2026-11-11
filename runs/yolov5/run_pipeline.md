# yolov5 GPU runtime_lazy 全流程

命令从仓库任意目录启动，在同一个 shell 中依次执行。模型与输入保留在 `runs/yolov5/`；所有生成的 task、密钥、日志和性能采样位于 `/tmp/v6_six_task_validation/yolov5/`；各模式顺序运行。验收比较原始 ONNX 与 CSV 的全部输出，最大绝对误差必须小于 0.2。

## 环境与构建

```bash
REPO="$(git rev-parse --show-toplevel)"
PY=/home/linghm/latti_venv/bin/python3
CMAKE="${CMAKE:-$(command -v cmake)}"
DATA="$REPO/runs/yolov5"
ROOT="/tmp/v6_six_task_validation/yolov5"
MODEL="$DATA/model/best_conv_outputs.onnx"
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
"$PY" "$REPO/tools/evaluate_onnx_csv.py" --onnx "$MODEL" --input-csv "$DATA/img.csv" --output-csv "$ROOT/reference/onnx.csv" --report "$ROOT/reference/onnx.json"
```

## 输入加密

```bash
OUT="$ROOT/task_input_encrypted"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task" --input "$DATA/img.csv" --output-csv "$OUT/decrypted.csv" --gpu
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json" --output-map '_model_24_m_0_Conv_output_0=/model.24/m.0/Conv_output_0' --output-map '_model_24_m_1_Conv_output_0=/model.24/m.1/Conv_output_0' --output-map '_model_24_m_2_Conv_output_0=/model.24/m.2/Conv_output_0'
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
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task/runner" --deployment-mode server_provisioned_runner --input "$DATA/img.csv" --output-cipher "$OUT/output.ct" --gpu
"$REPO/build/tools/decrypt_runner_output" --task-dir "$OUT/task/provisioner" --cipher "$OUT/output.ct" --output-csv "$OUT/decrypted.csv"
"$PY" "$REPO/tools/summarize_parameter_bundle.py" --task-dir "$OUT/task" --output "$OUT/bundle.json"
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.csv" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json" --output-map '_model_24_m_0_Conv_output_0=/model.24/m.0/Conv_output_0' --output-map '_model_24_m_1_Conv_output_0=/model.24/m.1/Conv_output_0' --output-map '_model_24_m_2_Conv_output_0=/model.24/m.2/Conv_output_0'
```

证据：`task_parameter_encrypted/run/inference.metrics.json`、`task_parameter_encrypted/comparison.json`、`task_parameter_encrypted/bundle.json`、`task_parameter_encrypted/provision.time`。

## 本轮 v6 实测记录

证据根目录：`/tmp/v6_six_task_validation/yolov5/`。比较全部输出，阈值 `max_abs < 0.2`。GPU 为 RTX 5880 Ada（48 GiB），每个 GPU 同时只执行一个推理任务；图生成、密钥准备和 provision 可能与另一张卡的推理重叠，wall 受主机 I/O 竞争影响。

| 模式 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | FHE 秒 | 参数 bundle 加密 wall | 加密参数目录 bytes | 最大绝对误差 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 输入加密 GPU lazy | 77.261 | 42.941 | 16.619 | 44.2581 | — | 不适用 | 6.29466403e-05 | 通过 |
| 参数加密 GPU runtime_lazy | 38.204 | 42.941 | 26.190 | 265.6730 | 5:56.42 | 3,512,418,068 | 9.06926237e-05 | 通过 |

`task_parameter_encrypted/bundle.json`：参数密文 1,343 个；纯密文 payload 3,511,692,305 bytes；包含 manifest 的参数目录 3,512,418,068 bytes；完整 runner 17,599,888,313 bytes。完整 runner 含图和求值密钥，不含 secret key。

`task_parameter_encrypted/provision.time`：wall 5:56.42，user 173.85 s，system 130.65 s，峰值 RSS 68039512 KiB。
