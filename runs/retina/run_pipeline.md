# retina GPU lazy pipeline

模型：`runs/retina/model/epoch041.onnx`。输入加密使用 `client_encrypted_input`；参数加密使用 `server_provisioned_runner`、`encrypted_offline`、`runtime_lazy`。两类 task 分别保存在 `task_input_encrypted/` 和 `task_parameter_encrypted/`。
编译配置说明：输入加密使用编译器默认值；参数加密只使用共享的 `training/config/storage_aware_config.json`。

## 输入加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/retina/model/epoch041.onnx --output ./runs/retina/task_input_encrypted \
  --style multiplexed
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/retina/task_input_encrypted/task
./build/examples/inference --task-dir ./runs/retina/task_input_encrypted/task \
  --input input_0=./runs/retina/img.csv --input input_1=./runs/retina/non_inv.csv --input input_2=./runs/retina/inv.csv --verify --gpu
```

实测输入加密：CPU RSS 峰值 97.28 GiB，GPU 进程显存峰值 42.94 GiB，池内使用峰值 20.096 GiB，FHE 137.687 s，最大绝对误差 0.000178452。参数加密正在执行，完成后补充。

## 参数加密

```bash
export PYTHONPATH="$PWD/inference/lattisense:$PWD"
/home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input ./runs/retina/model/epoch041.onnx --output ./runs/retina/task_parameter_encrypted \
  --style multiplexed --deployment-mode server_provisioned_runner \
  --config ./training/config/Retina_parameter_config.json
/home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/retina/task_parameter_encrypted/task \
  --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle \
  --task-dir ./runs/retina/task_parameter_encrypted/task \
  --out ./runs/retina/task_parameter_encrypted/task/runner
./build/examples/inference \
  --task-dir ./runs/retina/task_parameter_encrypted/task/runner \
  --deployment-mode server_provisioned_runner --input input_0=./runs/retina/img.csv --input input_1=./runs/retina/non_inv.csv --input input_2=./runs/retina/inv.csv \
  --output-cipher ./runs/retina/task_parameter_encrypted/output.ct --gpu
./build/tools/decrypt_runner_output \
  --task-dir ./runs/retina/task_parameter_encrypted/task/provisioner \
  --cipher ./runs/retina/task_parameter_encrypted/output.ct \
  --output-csv ./runs/retina/task_parameter_encrypted/decrypted.csv
```

验收记录（参考 ONNX+CSV 输出）：待运行后填写 CPU RSS 峰值、GPU 显存峰值、FHE 时间、参数 bundle 字节数和最大绝对误差。
