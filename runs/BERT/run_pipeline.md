# BERT GPU lazy pipeline

模型：`runs/BERT/model/trained_poly.onnx`，输入：`runs/BERT/input.csv`。输入加密和参数加密 task 分目录保存；PCM 使用 `par_block_col_major`，PDM 仅用于输入加密（当前仓库不支持 PDM 参数加密）。

## PCM 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/BERT/model/trained_poly.onnx --output ./runs/BERT/task_input_encrypted/pcm --style multiplexed --config ./training/config/BERT_pcm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/BERT/task_input_encrypted/pcm/task
./build/examples/inference --task-dir ./runs/BERT/task_input_encrypted/pcm/task --input ./runs/BERT/input.csv --verify --gpu
```

## PCM 参数加密 runtime_lazy

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/BERT/model/trained_poly.onnx --output ./runs/BERT/task_parameter_encrypted/pcm --style multiplexed --deployment-mode server_provisioned_runner --config ./training/config/BERT_pcm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/BERT/task_parameter_encrypted/pcm/task --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle --task-dir ./runs/BERT/task_parameter_encrypted/pcm/task --out ./runs/BERT/task_parameter_encrypted/pcm/task/runner
./build/examples/inference --task-dir ./runs/BERT/task_parameter_encrypted/pcm/task/runner --deployment-mode server_provisioned_runner --input ./runs/BERT/input.csv --output-cipher ./runs/BERT/task_parameter_encrypted/pcm/output.ct --gpu
./build/tools/decrypt_runner_output --task-dir ./runs/BERT/task_parameter_encrypted/pcm/task/provisioner --cipher ./runs/BERT/task_parameter_encrypted/pcm/output.ct --output-csv ./runs/BERT/task_parameter_encrypted/pcm/decrypted.csv
```

## PDM 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/BERT/model/trained_poly.onnx --output ./runs/BERT/task_input_encrypted/pdm --style multiplexed --config ./training/config/BERT_pdm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/BERT/task_input_encrypted/pdm/task
./build/examples/inference --task-dir ./runs/BERT/task_input_encrypted/pdm/task --input ./runs/BERT/input.csv --verify --gpu
```

## 实测记录

| 模式 | CPU RSS 峰值 | GPU 显存峰值 | FHE 时间 | 参数 bundle | 相对 ONNX+CSV 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| PCM 输入加密 | 143.24 GiB | 42.94 GiB | 255.353 s | 不适用 | 0.04279189 |
| PCM 参数加密 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| PDM 输入加密 | 待运行 | 待运行 | 待运行 | 不适用 | 待运行 |

已完成的 PCM 输入加密记录：`task_input_encrypted/pcm/run/inference.metrics.json`、`task_input_encrypted/pcm/comparison.json`。CPU 为进程生命周期 RSS 峰值；GPU 为每 0.5 秒采样的进程显存峰值（含内存池预留），池内实际使用峰值为 19.875 GiB。误差比较原始 ONNX 的全部 2 个 logits，参考文件为 `reference/onnx.npz`。其余路径待完成。
