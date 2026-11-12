# BERT ONNX prefix/suffix GPU 全流程

默认比例为 `0.9`：prefix 明文执行，suffix 分别执行输入加密 GPU lazy 和参数加密 GPU `runtime_lazy`。从仓库任意目录启动，在同一个 Bash shell 中依次执行；生成物位于 `/tmp/onnx_split_validation/BERT/`。两种模式均比较原始完整 ONNX 接收原始 CSV 的全部输出，要求 **最大绝对误差严格小于 0.1**。

## 环境与构建

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
PY=/home/linghm/latti_venv/bin/python3
DATA="$REPO/runs/BERT"
ROOT="/tmp/onnx_split_validation/BERT"
MODEL="$DATA/model/trained_poly.onnx"
export PYTHONPATH="$REPO/inference/lattisense:$REPO"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cmake -S "$REPO" -B "$REPO/build" -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_DEV=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build "$REPO/build" -j8
mkdir -p "$ROOT"
cd "$ROOT"
```

## ONNX 切分与前缀明文执行

输入 manifest 显式绑定名称、完整逻辑 shape 和轴语义；边界 CSV 的 Feature 类型由原图语义传播和后缀编译契约确定，无需人工为每个 CSV 选择打包。重复准备时使用新的输出目录，脚本拒绝覆盖已发布的 split manifest。

```bash
"$PY" - "$DATA" "$ROOT/inputs.json" <<'PYINPUT'
import json
from pathlib import Path
import sys
base = Path(sys.argv[1])
inputs = [{'name': 'inputs_embeds', 'csv': 'input.csv', 'shape': [1, 44, 768], 'axes': ['N', 'T', 'D']}]
for entry in inputs:
    entry['csv'] = str(base / entry['csv'])
Path(sys.argv[2]).write_text(json.dumps({'inputs': inputs, 'model_kind': 'bert'}, indent=2))
PYINPUT
"$PY" "$REPO/tools/prepare_split_onnx.py" --onnx "$MODEL" \
  --inputs-manifest "$ROOT/inputs.json" --output-dir "$ROOT/split" > "$ROOT/prepare.log" 2>&1
SPLIT="$ROOT/split"
```

准备阶段保存原始完整模型全部明文输出、prefix/suffix ONNX、全部边界逻辑 CSV 和 hash，并验证 CSV 往返及切分明文等价。BERT/ViT 使用项目既有 NumPy float64 明文参考实现；其他任务使用 ONNX Runtime 与既有多项式适配。

## 输入加密 GPU lazy

```bash
OUT="$ROOT/input"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$SPLIT/suffix.onnx" \
  --split-manifest "$SPLIT/split_manifest.json" --output "$OUT" --style multiplexed --config "$REPO/training/config/BERT_pcm_config.json" \
  > "$ROOT/compile_input.log" 2>&1
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task" > "$ROOT/graph_input.log" 2>&1
"$PY" "$REPO/tools/run_split_inference.py" --task-dir "$OUT/task" --output-dir "$OUT/run" \
  > "$ROOT/infer_input.log" 2>&1
```

## 参数加密 GPU runtime_lazy

```bash
OUT="$ROOT/parameter"
mkdir -p "$OUT"
cd "$OUT"
"$PY" "$REPO/training/run_compile.py" --input "$SPLIT/suffix.onnx" \
  --split-manifest "$SPLIT/split_manifest.json" --output "$OUT" --style multiplexed --config "$REPO/training/config/BERT_pcm_config.json" \
  --deployment-mode server_provisioned_runner > "$ROOT/compile_parameter.log" 2>&1
"$PY" "$REPO/inference/interface/gen_mega_ag.py" --task-dir "$OUT/task" \
  --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy \
  > "$ROOT/graph_parameter.log" 2>&1
/usr/bin/time -v -o "$OUT/provision.time" "$REPO/build/tools/provision_encrypted_runner_bundle" \
  --task-dir "$OUT/task" --out "$OUT/task/runner" > "$ROOT/provision.log" 2>&1
"$PY" "$REPO/tools/run_split_inference.py" --task-dir "$OUT/task" --output-dir "$OUT/run" \
  > "$ROOT/infer_parameter.log" 2>&1
"$PY" "$REPO/tools/summarize_parameter_bundle.py" --task-dir "$OUT/task" --output "$OUT/bundle.json"
```

两种模式共用同一 split manifest、suffix 和逻辑 CSV，并分别核对实际编译契约与 Client/Runner、图签名及 CKKS 上下文。切分后缀保持 PN15Default，命令无需额外指定参数配置。本页验收使用 PCM；PDM 沿用原任务配置，不属于本次验收矩阵。

`run_split_inference.py` 按契约传入全部命名 CSV，调用现有 Client/Runner，完成解密和输出汇合，并比较原始完整模型全部输出。证据为 `split/split_manifest.json`、各模式的 `task/input_contract.json`、`run/inference.log`、`run/inference.metrics.json`、`run/decrypted.csv`、`run/merged_outputs.npz` 和 `run/comparison.json`。比较失败返回非零状态；不使用 task 内部明文输出替代原始 ONNX 基准。

CPU 峰值取 GNU time RSS，GPU 进程显存每 0.5 秒采样，GPU 池内峰值每 0.1 秒采样；FHE 时间取 `Run GPU time`，不含编译、密钥准备、provision 和解密。GiB = 2³⁰ bytes。GPU lazy 验收结果见 [切分验收报告](../../docs/report/onnx_split_gpu_lazy_acceptance.md)。

## 历史完整模型 v6 实测记录

证据根目录：`/tmp/v6_six_task_validation/BERT/`。比较全部输出，阈值 `max_abs < 0.2`。GPU 为 RTX 5880 Ada（48 GiB），每个 GPU 同时只执行一个推理任务；图生成、密钥准备和 provision 可能与另一张卡的推理重叠，wall 受主机 I/O 竞争影响。

| 模式 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | FHE 秒 | 参数 bundle 加密 wall | 加密参数目录 bytes | 最大绝对误差 | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PCM 输入加密 GPU lazy | 79.344 | 42.941 | 17.116 | 264.1700 | — | 不适用 | 0.0330424909 | 通过 |
| PCM 参数加密 GPU runtime_lazy | 41.694 | 42.941 | 16.507 | 809.6290 | 4:29.84 | 9,261,683,525 | 0.000292990589 | 通过 |
| PDM 输入加密 GPU lazy | 82.126 | 42.941 | 16.501 | 309.4750 | — | 不适用 | 0.00418093536 | 通过 |

`task_parameter_encrypted/pcm/bundle.json`：参数密文 3,991 个；纯密文 payload 9,254,714,425 bytes；包含 manifest 的参数目录 9,261,683,525 bytes；完整 runner 26,817,435,375 bytes。完整 runner 含图和求值密钥，不含 secret key。

`task_parameter_encrypted/pcm/provision.time`：wall 4:29.84，user 249.54 s，system 163.32 s，峰值 RSS 47295332 KiB。
