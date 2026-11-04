# imagenet GPU lazy pipeline

模型：`runs/imagenet/model/trained_poly.onnx`。输入加密使用 `client_encrypted_input`；参数加密使用 `server_provisioned_runner`、`encrypted_offline`、`runtime_lazy`。两类 task 分别保存在 `task_input_encrypted/` 和 `task_parameter_encrypted/`。
编译配置说明：输入加密使用编译器默认值；参数加密只使用共享的 `training/config/storage_aware_config.json`。

## 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/imagenet/model/trained_poly.onnx --output ./runs/imagenet/task_input_encrypted \
  --style multiplexed
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/imagenet/task_input_encrypted/task
./build/examples/inference --task-dir ./runs/imagenet/task_input_encrypted/task \
  --input ./runs/imagenet/img.csv --verify --gpu
```

首轮输入加密执行完成，但数值验收失败：FHE 52.4784 s，CPU RSS 峰值 51.99 GiB，GPU 进程显存峰值 42.94 GiB（池内使用峰值 14.845 GiB）。全部 1000 个输出最大绝对误差 523863.807；正在定位，不能视为通过。证据：`task_input_encrypted/run/inference.metrics.json`、`task_input_encrypted/comparison.json`。

## 参数加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/imagenet/model/trained_poly.onnx --output ./runs/imagenet/task_parameter_encrypted \
  --style multiplexed --deployment-mode server_provisioned_runner \
  --config ./training/config/ImageNet_parameter_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/imagenet/task_parameter_encrypted/task \
  --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle \
  --task-dir ./runs/imagenet/task_parameter_encrypted/task \
  --out ./runs/imagenet/task_parameter_encrypted/task/runner
./build/examples/inference \
  --task-dir ./runs/imagenet/task_parameter_encrypted/task/runner \
  --deployment-mode server_provisioned_runner --input ./runs/imagenet/img.csv \
  --output-cipher ./runs/imagenet/task_parameter_encrypted/output.ct --gpu
./build/tools/decrypt_runner_output \
  --task-dir ./runs/imagenet/task_parameter_encrypted/task/provisioner \
  --cipher ./runs/imagenet/task_parameter_encrypted/output.ct \
  --output-csv ./runs/imagenet/task_parameter_encrypted/decrypted.csv
```

验收记录（参考 ONNX+CSV 输出）：待运行后填写 CPU RSS 峰值、GPU 显存峰值、FHE 时间、参数 bundle 字节数和最大绝对误差。
