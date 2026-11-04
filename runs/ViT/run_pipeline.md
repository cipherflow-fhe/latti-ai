# ViT GPU lazy pipeline

模型：`runs/ViT/model/trained_poly.onnx`，输入：`runs/ViT/patch_input.csv`。输入加密和参数加密 task 分目录保存；PCM 使用 `par_block_col_major`，PDM 仅用于输入加密（当前仓库不支持 PDM 参数加密）。

## PCM 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/ViT/model/trained_poly.onnx --output ./runs/ViT/task_input_encrypted/pcm --style multiplexed --config ./training/config/ViT_pcm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/ViT/task_input_encrypted/pcm/task
./build/examples/inference --task-dir ./runs/ViT/task_input_encrypted/pcm/task --input ./runs/ViT/patch_input.csv --verify --gpu
```

## PCM 参数加密 runtime_lazy

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/ViT/model/trained_poly.onnx --output ./runs/ViT/task_parameter_encrypted/pcm --style multiplexed --deployment-mode server_provisioned_runner --config ./training/config/ViT_pcm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/ViT/task_parameter_encrypted/pcm/task --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle --task-dir ./runs/ViT/task_parameter_encrypted/pcm/task --out ./runs/ViT/task_parameter_encrypted/pcm/task/runner
./build/examples/inference --task-dir ./runs/ViT/task_parameter_encrypted/pcm/task/runner --deployment-mode server_provisioned_runner --input ./runs/ViT/patch_input.csv --output-cipher ./runs/ViT/task_parameter_encrypted/pcm/output.ct --gpu
./build/tools/decrypt_runner_output --task-dir ./runs/ViT/task_parameter_encrypted/pcm/task/provisioner --cipher ./runs/ViT/task_parameter_encrypted/pcm/output.ct --output-csv ./runs/ViT/task_parameter_encrypted/pcm/decrypted.csv
```

## PDM 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py --input ./runs/ViT/model/trained_poly.onnx --output ./runs/ViT/task_input_encrypted/pdm --style multiplexed --config ./training/config/ViT_pdm_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py --task-dir ./runs/ViT/task_input_encrypted/pdm/task
./build/examples/inference --task-dir ./runs/ViT/task_input_encrypted/pdm/task --input ./runs/ViT/patch_input.csv --verify --gpu
```

## 实测记录

| 模式 | CPU RSS 峰值 | GPU 显存峰值 | FHE 时间 | 参数 bundle | 相对 ONNX+CSV 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| PCM 输入加密 | 60.61 GiB | 42.94 GiB | 186.444 s | 不适用 | 0.02709967 |
| PCM 参数加密 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| PDM 输入加密 | 91.88 GiB | 42.94 GiB | 213.282 s | 不适用 | 0.00272374 |

已完成 PCM 输入加密：CPU RSS 峰值 60.61 GiB，GPU 进程显存峰值 42.94 GiB，池内使用峰值 17.364 GiB，FHE 时间 186.444 s，全部 1970 个输出最大绝对误差 0.02709967。证据：`task_input_encrypted/pcm/run/inference.metrics.json`、`task_input_encrypted/pcm/comparison.json`。

已完成 PDM 输入加密：CPU RSS 峰值 91.88 GiB，GPU 进程显存峰值 42.94 GiB，池内使用峰值 16.918 GiB，FHE 时间 213.282 s，最大绝对误差 0.00272374。证据：`task_input_encrypted/pdm/run/inference.metrics.json`、`task_input_encrypted/pdm/comparison.json`。
