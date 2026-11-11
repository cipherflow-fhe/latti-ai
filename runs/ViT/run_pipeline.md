# ViT GPU runtime_lazy 全流程

命令从仓库任意目录启动，在同一个 shell 中依次执行。模型与输入保留在 `runs/ViT/`；所有生成的 task、密钥、日志和性能采样位于 `/tmp/v6_six_task_validation/ViT/`；各模式顺序运行。验收比较原始 ONNX 与 CSV 的全部输出，最大绝对误差必须小于 0.2。

## 环境与构建

```bash
REPO="$(git rev-parse --show-toplevel)"
PY=/home/linghm/latti_venv/bin/python3
CMAKE="${CMAKE:-$(command -v cmake)}"
DATA="$REPO/runs/ViT"
ROOT="/tmp/v6_six_task_validation/ViT"
MODEL="$DATA/model/trained_poly.onnx"
export PYTHONPATH="$REPO/inference/lattisense:$REPO"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$CMAKE" -S "$REPO" -B "$REPO/build" -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_DEV=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$REPO/build" -j8
mkdir -p "$ROOT/reference"
cd "$ROOT"
```

CPU 数据为推理进程生命周期的最大 RSS（GNU time）；GPU 数据包括每 0.5 秒采样的进程显存峰值和 SDK 每 0.1 秒采样的池内使用峰值。GPU 进程显存含内存池预留。FHE 时间取 SDK 的 `Run GPU time`，不含编译、参数加密、密钥准备和结果解密。GiB = 2³⁰ bytes。

配置中的打包格式、矩阵块大小和 bootstrap 缩放使用非默认设置，保留配置文件。BERT 自定义算子语义从 ONNX/JSON 图识别，配置无需 `model_type`。

## 原始 ONNX 明文参考

```bash
"$PY" "$REPO/tools/evaluate_transformer_onnx_csv.py" --onnx "$MODEL" --input-csv "$DATA/patch_input.csv" --output "$ROOT/reference/onnx.npz"
```

参考直接按 ONNX 自定义算子的属性、权重与多项式定义以 NumPy float64 执行。ViT 比较全部 1970 个输出。

## PCM 输入加密

```bash
OUT="$ROOT/task_input_encrypted/pcm"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed --config "$REPO/training/config/ViT_pcm_config.json"
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task" --input "$DATA/patch_input.csv" --output-csv "$OUT/decrypted.csv" --gpu
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.npz" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_input_encrypted/pcm/run/inference.metrics.json`、`task_input_encrypted/pcm/comparison.json`。

## PCM 参数加密

```bash
OUT="$ROOT/task_parameter_encrypted/pcm"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed --config "$REPO/training/config/ViT_pcm_config.json" --deployment-mode server_provisioned_runner
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task" --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
/usr/bin/time -v -o "$OUT/provision.time" "$REPO/build/tools/provision_encrypted_runner_bundle" --task-dir "$OUT/task" --out "$OUT/task/runner"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task/runner" --deployment-mode server_provisioned_runner --input "$DATA/patch_input.csv" --output-cipher "$OUT/run/output.ct" --gpu
"$REPO/build/tools/decrypt_runner_output" --task-dir "$OUT/task/provisioner" --cipher "$OUT/run/output.ct" --output-csv "$OUT/decrypted.csv"
"$PY" "$REPO/tools/summarize_parameter_bundle.py" --task-dir "$OUT/task" --output "$OUT/bundle.json"
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.npz" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_parameter_encrypted/pcm/run/inference.metrics.json`、`task_parameter_encrypted/pcm/comparison.json`、`task_parameter_encrypted/pcm/bundle.json`、`task_parameter_encrypted/pcm/provision.time`。



## PDM 输入加密

```bash
OUT="$ROOT/task_input_encrypted/pdm"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$MODEL" --output "$OUT" --style multiplexed --config "$REPO/training/config/ViT_tiny_config.json"
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task"
"$PY" "$REPO/tools/measure_gpu_command.py" --output-dir "$OUT/run" -- "$REPO/build/examples/inference" --task-dir "$OUT/task" --input "$DATA/patch_input.csv" --output-csv "$OUT/decrypted.csv" --gpu
"$PY" "$REPO/tools/compare_inference_csv.py" --reference "$ROOT/reference/onnx.npz" --actual "$OUT/decrypted.csv" --report "$OUT/comparison.json"
```

证据：`task_input_encrypted/pdm/run/inference.metrics.json`、`task_input_encrypted/pdm/comparison.json`。

## 本轮 v6 实测记录

证据根目录：`/tmp/v6_six_task_validation/ViT/`。比较全部输出，阈值 `max_abs < 0.2`。GPU 为 RTX 5880 Ada（48 GiB），每个 GPU 同时只执行一个推理任务；图生成、密钥准备和 provision 可能与另一张卡的推理重叠，wall 受主机 I/O 竞争影响。

| 模式 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | FHE 秒 | 参数 bundle 加密 wall | 加密参数目录 bytes | 最大绝对误差 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PCM 输入加密 GPU lazy | 74.612 | 42.941 | 16.473 | 138.2260 | — | 不适用 | 0.0287025047 | 通过 |
| PCM 参数加密 GPU runtime_lazy | 29.436 | 42.941 | 20.737 | 330.0600 | 6:10.37 | 11,901,331,270 | 0.0401525409 | 通过 |
| PDM 输入加密 GPU lazy | 77.343 | 42.941 | 16.806 | 116.8200 | — | 不适用 | 0.000584884326 | 通过 |

`task_parameter_encrypted/pcm/bundle.json`：参数密文 2,859 个；纯密文 payload 11,898,942,693 bytes；包含 manifest 的参数目录 11,901,331,270 bytes；完整 runner 25,557,616,508 bytes。完整 runner 含图和求值密钥，不含 secret key。

`task_parameter_encrypted/pcm/provision.time`：wall 6:10.37，user 262.25 s，system 206.15 s，峰值 RSS 66217244 KiB。
