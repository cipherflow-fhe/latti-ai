# cifar10 GPU lazy pipeline

模型：`runs/cifar10/model/trained_poly.onnx`。输入加密使用 `client_encrypted_input`；参数加密使用 `server_provisioned_runner`、`encrypted_offline`、`runtime_lazy`。两类 task 分别保存在 `task_input_encrypted/` 和 `task_parameter_encrypted/`。

## 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/cifar10/model/trained_poly.onnx --output ./runs/cifar10/task_input_encrypted \
  --style multiplexed --config ./training/config/CIFAR10_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/cifar10/task_input_encrypted/task
./build/examples/inference --task-dir ./runs/cifar10/task_input_encrypted/task \
  --input ./runs/cifar10/img.csv --verify --gpu
```

验收记录（参考 `tools/evaluate_onnx_csv.py` 的 ONNX+CSV 输出）：待运行后填写 CPU RSS 峰值、GPU 显存峰值、FHE 时间和最大绝对误差。

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

验收记录（参考 ONNX+CSV 输出）：待运行后填写 CPU RSS 峰值、GPU 显存峰值、FHE 时间、参数 bundle 字节数和最大绝对误差。
