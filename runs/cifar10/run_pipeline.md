# cifar10 GPU lazy pipeline

模型：`runs/cifar10/model/trained_poly.onnx`。输入加密使用 `client_encrypted_input`；参数加密使用 `server_provisioned_runner`、`encrypted_offline`、`runtime_lazy`。两类 task 分别保存在 `task_input_encrypted/` 和 `task_parameter_encrypted/`。
编译配置说明：输入加密使用编译器默认值；参数加密只使用共享的 `training/config/storage_aware_config.json`。

## 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/cifar10/model/trained_poly.onnx --output ./runs/cifar10/task_input_encrypted \
  --style multiplexed
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/cifar10/task_input_encrypted/task
./build/examples/inference --task-dir ./runs/cifar10/task_input_encrypted/task \
  --input ./runs/cifar10/img.csv --verify --gpu
```

实测输入加密：CPU RSS 峰值 36.43 GiB，GPU 进程显存峰值 42.94 GiB（含内存池预留；池内使用峰值 11.871 GiB），FHE 时间 6.5531 s；全部 10 个输出相对原始 ONNX+CSV 的最大绝对误差为 0.000239314。证据：`task_input_encrypted/run2/inference.metrics.json` 和 `task_input_encrypted/comparison.json`。

## 参数加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/cifar10/model/trained_poly.onnx --output ./runs/cifar10/task_parameter_encrypted \
  --style multiplexed --deployment-mode server_provisioned_runner \
  --config ./training/config/CIFAR10_parameter_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/cifar10/task_parameter_encrypted/task \
  --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle \
  --task-dir ./runs/cifar10/task_parameter_encrypted/task \
  --out ./runs/cifar10/task_parameter_encrypted/task/runner
./build/examples/inference \
  --task-dir ./runs/cifar10/task_parameter_encrypted/task/runner \
  --deployment-mode server_provisioned_runner --input ./runs/cifar10/img.csv \
  --output-cipher ./runs/cifar10/task_parameter_encrypted/output.ct --gpu
./build/tools/decrypt_runner_output \
  --task-dir ./runs/cifar10/task_parameter_encrypted/task/provisioner \
  --cipher ./runs/cifar10/task_parameter_encrypted/output.ct \
  --output-csv ./runs/cifar10/task_parameter_encrypted/decrypted.csv
```

实测参数加密：CPU RSS 峰值 19.43 GiB，GPU 进程显存峰值 42.94 GiB（池内使用峰值 12.530 GiB），FHE 时间 21.5728 s；加密参数目录共 1448755393 字节（含 manifest）；全部 10 个输出最大绝对误差 0.000142126366。证据：`task_parameter_encrypted/run/inference.metrics.json`、`task_parameter_encrypted/bundle.json`、`task_parameter_encrypted/comparison.json`。
