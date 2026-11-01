# 验收与复查命令

工作目录：`/home/linghm/project/latti-ai`。使用 `/home/linghm/pytorch_venv/bin/python`，
环境 `PYTHONPATH=inference/lattisense:.`。GPU 任务通过 benchmark 设置 `CUDA_VISIBLE_DEVICES=0`。

```bash
cmake --build build --target test_fhe_layers_hetero test_encrypted_parameter_store \
  provision_encrypted_runner_bundle inference decrypt_runner_output -j6

PYTHONPATH=inference/lattisense:. /home/linghm/pytorch_venv/bin/python -m unittest \
  inference.unittests.test_interleaved_seed_weights \
  inference.unittests.test_kernel_seed_weights \
  inference.unittests.test_dense_seed_tiles \
  inference.unittests.test_poly_seed_weights \
  inference.unittests.test_server_provisioned_runner_generator \
  inference.unittests.test_encrypted_param_count \
  inference.unittests.test_mobilenet_encrypted_params \
  inference.unittests.test_parameter_acceptance_tools \
  training.model_compiler.unittests.test_storage_score

./build/inference/unittests/test_encrypted_parameter_store
./build/inference/unittests/test_fhe_layers_hetero \
  '[joint_seed],[dense_tile],[poly_seed],[inverse_canonical],[multiplexed_joint],[kernel_seed]'

PYTHONPATH=inference/lattisense:. /home/linghm/pytorch_venv/bin/python \
  tools/benchmark_server_provisioned_runner.py \
  --output-root /tmp/latti-param-compression-acceptance --delete-tasks

git -c core.whitespace=cr-at-eol diff --check -- . ':(exclude)docs/report/raw/*.log'
```

每个子命令的完整 argv、cwd、returncode、wall time 和 RSS 见同名前缀的
`*-command.json`、`*-time.json`；GPU binary SHA-256 见 `build_artifacts.json`。
build flags 为 `INFERENCE_SDK_ENABLE_GPU=ON`、`LATTISENSE_DEV=ON`。
DEV monitor 临时修复见代码摘要，验收后恢复依赖工作树；不提交这处依赖修改。
原始 `.log` 保留程序输出的行尾空格及末尾空行，空白检查排除这些日志；源码和其他报告文件仍检查。

`*-layout-contract.json` 在 Provisioner 完成后、删除 task 前，原样保留公开的
`nn_layers_ct_0.json`、`task_signature.json`、`manifest.json`、编译存储报告及
`eval_context.bin` 的字节数，不包含权重、密文或密钥。

`python3 docs/report/raw/audit_layout_storage.py` 重建 `layout_storage_audit.json`，独立复核每个 layer、level、param kind 的 manifest 元素数，
与编译报告的 counts 比较；聚合实际 container span 长度，与逐层及总 storage bytes
比较；各 level 的实际元素长度还须与校准表相等，且 `unaccounted` 为空。
`seed_weight` 归入 `weight`，`poly_coeff` 归入 `coefficient`，
`mult_scalar_weight` 归入 `scalar` 后比较。实际 container span 和索引的一致性由
benchmark 的 `summarize_parameter_bundle.py` 另行验证。

早期 smoke 在 `smoke/`，其中 JSON 的绝对 raw 路径指移动前的位置，以本目录文件名
定位即可；其结果不参与正式验收。`attempt-1-missing-scalar-scale/` 保留首次完整
运行在 GPU 前遇到的 signature scale 缺失失败，修复后从隔离 task 全部重跑。

`attempt-2-layout-object-order/` 保留下一次运行的完整编译、导出和 legacy 测量，
及 compressed Runner 在 GPU 前拒绝有效布局的日志。原因是 C++ `ordered_json`
比较把对象的键顺序当成契约差异；已改为 JSON 对象语义比较并补充回归测试。
此修复不改变编译图、参数生成或密文文件；保留原隔离 task，以最终 binary 重新运行
两种模式各一次 warm-up 和三次正式测量：

```bash
PYTHONPATH=inference/lattisense:. /home/linghm/pytorch_venv/bin/python \
  tools/benchmark_server_provisioned_runner.py --model resnet20 \
  --output-root /tmp/latti-param-compression-acceptance --reuse-task --delete-tasks

PYTHONPATH=inference/lattisense:. /home/linghm/pytorch_venv/bin/python \
  tools/benchmark_server_provisioned_runner.py --model mobilenet \
  --output-root /tmp/latti-param-compression-acceptance --delete-tasks
```

ResNet20 最终模型 JSON 的 `commands` 已合并上述归档目录中成功的 compile/MegaAG/Provisioner
记录，并将日志指向归档文件；`preparation_provenance` 记录对应源码 hash。正式 Runner/decrypt
使用最终 binary；MobileNet 从 compile 开始全部使用最终源码与 binary。

最终独立复查（重新读取完整 CSV 并重算误差、检查所有采样和命令、核对 binary hash）：

```bash
/home/linghm/pytorch_venv/bin/python docs/report/raw/verify_acceptance.py
```
