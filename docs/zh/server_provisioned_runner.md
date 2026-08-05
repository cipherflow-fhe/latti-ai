# Server-Provisioned Runner 部署模式

本文档说明 `server_provisioned_runner` 部署模式的角色分工、产物目录、安全边界和基本使用流程。该模式的目标是让 server 侧离线加密模型参数，Runner 现场读取明文输入并执行同态计算，最后由 server 侧解密 Runner 返回的密文结果。

## 角色分工

- `Provisioner`：server 侧离线准备者，持有明文模型参数和 secret key，负责生成 eval context、按 use-site level 加密模型参数，并导出可搬运的 Runner bundle。
- `Runner`：现场执行者，只持有 eval context、MegaAG、task signature 和加密后的模型参数。Runner 读取明文输入，输出密文结果，不持有 secret key。
- `Decryptor`：server 侧解密者，持有 secret key，负责解密 Runner 返回的密文输出。

默认的 `client_encrypted_input` 模式不变；`server_provisioned_runner` 是新增的部署模式，不改变旧 quick start 和旧 API 的默认语义。

## 安全边界

`server_provisioned_runner` 模式只保护模型参数和输出密文不被 Runner 解密，不保护 Runner 现场输入。

- Runner 可以看到输入明文，因为该模式假设输入隐私不是目标。
- Runner bundle 不应包含 `secret_key.bin`、`secret_context.bin`、`model_parameters.h5` 或其他明文模型参数 dump。
- Runner 持有 eval context，包括 relin key、rotation/galois key 和 bootstrapping 所需公钥材料；这些公钥只用于同态计算，不能解密参数密文或输出密文。
- Runner 可读取加密后的模型参数并执行 `PT * CT`、`CT * CT` 等计算，但不能解密参数或结果。
- `Provisioner` / `Decryptor` 位于 server 侧，持有 secret key，因此可以访问明文模型参数并解密输出。
- v1 已将 CIFAR10 runner 图中会进入 signature 的 `convm_` structural masks 也作为 offline ciphertext 参数处理，避免 Runner 需要额外的 plaintext mask 输入。avgpool select tensors、concat masks、repack masks、upsample select tensors 和 routing/mask 类常量如果后续出现在 runner signature 中，也应按相同模式迁移到 offline encrypted parameters，或显式标注为 Runner-local public constants。

## 目录结构

保留旧模式的 `client/` 和 `server/` 目录，并为新模式新增 `provisioner/` 和 `runner/`：

```text
runs/cifar10/task/
  client/
  server/
    task_config.json
    ckks_parameter.json
    nn_layers_ct_0.json
    model_parameters.h5
    mega_ag.json

  provisioner/
    task_config.json
    ckks_parameter.json
    nn_layers_ct_0.json
    model_parameters.h5
    secret_key.bin 或 secret_context.bin
    eval_context.bin

  runner/
    task_config.json
    ckks_parameter.json
    nn_layers_ct_0.json
    mega_ag.json
    task_signature.json
    eval_context.bin
    encrypted_model_parameters/
      manifest.json
      shard_*.bin
```

`runner/` 是可搬运目录。部署前应检查其中没有 secret key、secret context、H5 模型参数或明文参数文件。

## Quick Start

以下以 CIFAR10 风格目录为例。路径可按实际任务调整。

### 1. 编译并生成新模式目录布局

```bash
python training/run_compile.py \
  --input ./runs/cifar10/model/trained_poly.onnx \
  --output ./runs/cifar10 \
  --style multiplexed \
  --deployment-mode server_provisioned_runner
```

该步骤生成常规 `task/server/` 产物，并写出 `task/provisioner/`、`task/runner/` 的基础配置。`--runner-output-dir` 可用于指定 runner 目录，默认是 `<output>/task/runner`。

### 2. 生成 Runner MegaAG

```bash
python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/cifar10/task \
  --deployment-mode server_provisioned_runner
```

该步骤把 `mega_ag.json` 和 `task_signature.json` 写入 `task/runner/`，并将模型参数声明为 offline ciphertext arguments。

### 3. Provisioner 离线加密参数

```bash
./build/tools/provision_encrypted_runner_bundle \
  --task-dir ./runs/cifar10/task \
  --out ./runs/cifar10/task/runner
```

该工具在 server/provisioner 侧执行，读取 `task/provisioner/` 或 `task/server/` 中的配置和 `model_parameters.h5`，生成 secret/eval context，并把模型参数按 task signature 与 manifest 要求加密到 `task/runner/encrypted_model_parameters/`。

### 4. Runner 现场执行

```bash
./build/examples/inference \
  --task-dir ./runs/cifar10/task/runner \
  --deployment-mode server_provisioned_runner \
  --input ./runs/cifar10/task/client/img.csv \
  --output-cipher ./runs/cifar10/output.ct \
  --gpu
```

Runner 读取明文 CSV 输入，将输入编码为 plaintext/ringt argument，加载 offline encrypted parameters，执行 MegaAG，并把输出密文序列化到 `--output-cipher`。

GPU 运行需要先用 `-DINFERENCE_SDK_ENABLE_GPU=ON` 构建，并在运行命令中显式传入 `--gpu`；CPU 运行时省略 `--gpu`。多输入任务可以使用 `--input name=path` 形式显式绑定输入名。

### 5. Server 解密输出

```bash
./build/tools/decrypt_runner_output \
  --task-dir ./runs/cifar10/task/provisioner \
  --cipher ./runs/cifar10/output.ct
```

Decryptor 使用 provisioner 目录中的 secret key 或 secret context 解密 Runner 输出，并打印输出向量和 top-1。

## 参数 level 与 use-site 物化

Python MegaAG 图不会显式插入 `ringt_to_mul()` 节点。原始 `CT * PT` 路径中，C++ executor 在运行到 `mult(ct, pt_ringt)` 或 fused MAC 节点时，会读取实际输入 ciphertext 的 level，然后调用 `ringt_to_mul(pt_ringt, level)`。`add_plain_ringt(ct, pt_ringt)` 不显式传入 plaintext level，但其语义等价于按被加 ciphertext 的当前 level 解释该 ringt。

加密参数模式沿用这条 level/scale 轨迹，不重新做一套 depth 规划：

- 乘法 use-site：Provisioner 先按 layer 原有 `generate_weight_pt_for_*` 逻辑生成 `pt_ringt`，再根据该 use-site 的 ciphertext 操作数 level `L` 物化 plaintext，并加密为对应的参数密文。
- `mult_scalar` use-site：该算子的 scalar 来自模型数值参数，不属于 public structural constant。新模式中会为 `mult_scalar_<layer_id>` 生成 offline ciphertext argument，Provisioner 通过 `MultScalarLayer::generate_weight_pt_for_index()` 生成原始 `pt_ringt`，再按输入 feature 的 level 物化并加密。
- 加法 use-site：例如 dense/conv bias，原图通常是 `partial_sum = rescale(partial_sum)` 后执行 `add_plain_ringt(partial_sum, bias_pt_ringt)`。虽然 `add_plain_ringt()` 自身不需要显式传入 level，但 encrypted bias 必须从 `partial_sum.level` 推导目标 level `L`，再对 `bias_pt_ringt` 物化并加密为 `bias_ct_L`。
- 同一个逻辑参数如果在多个不同 level 的 use-site 被复用，manifest 和 signature 必须保存多个离线密文实例；相同 level 的实例可以在 manifest 层去重。
- offline argument id 应能对应到 use-site level，例如通过 `source_id + use_site + level` 的映射记录到 manifest。

这样做的结果是：执行计算图的 level/scale 管理与原始 `CT * PT` 模式保持一致，额外差异主要来自参数变为 ciphertext 后需要 `CT * CT`、`relin` 和更高的运行成本。

## 注意事项

- `CT * CT` 比 `CT * PT` 更重，参数密文体积也远大于 H5 或 plaintext ringt 参数；`convm_` 这类 structural masks 加密后同样会引入额外 relin/rescale 成本。Runner 侧应依赖 manifest 和 lazy load 降低峰值内存。
- 仅打开 CMake GPU 编译开关不会自动切换运行时 backend；`./build/examples/inference` 需要显式传入 `--gpu` 才会使用 GPU Runner。
- eval context 必须覆盖 MegaAG 中所有 relin、rotation/galois 和 bootstrapping key 需求。
- 不要把每个 term 都改成“先 rescale 再累加”的通用写法，除非原 layer 的 `run_core` 本来就是这个轨迹；否则会改变 level/scale 语义。
- `client_encrypted_input` 默认流程仍然保留，未显式指定 `--deployment-mode server_provisioned_runner` 时使用旧模式。
