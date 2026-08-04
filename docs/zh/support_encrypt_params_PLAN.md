# Server-Provisioned Encrypted-Parameter Runner Mode Plan

> 状态：Phase 1-4 已完成。Phase 4 按当前要求仅新增中文文档
> `docs/zh/server_provisioned_runner.md`，未更新 README。本文档后续仅保留背景、设计决策、
> 风险与验收清单，不再包含新的待执行 Phase。

## 1. 背景与目标

当前 README 的 CIFAR10 quick start 链路是：

1. `examples/test_cifar10/train.py` 导出 FHE-friendly ONNX 和 `task/server/model_parameters.h5`。
2. `training/run_compile.py` 将 ONNX 编译为 `task/server/nn_layers_ct_0.json`，并生成 server/client 侧 task config。
3. `inference/interface/gen_mega_ag.py` 基于 `nn_layers_ct_0.json` 生成 `task/server/mega_ag.json`。
4. `./build/examples/inference` 使用 client 加密输入，server 导入 client eval context，server lazy encode 明文参数并执行密文推理，密文结果回 client 解密。

希望新增的真实部署模式不是“client 加密输入”，而是：

1. server 端持有模型明文参数、加解密密钥和解密能力。
2. server 离线将模型参数按现有 layer packing 逻辑编码，并加密为参数密文。
3. server 将只包含加密参数、evaluate 公钥、计算图和运行配置的硬件/存储 bundle 搬运到 client 现场。
4. client 现场 Runner 读取明文输入，不加密输入，只编码为 plaintext。
5. Runner 按需加载参数密文，执行 plaintext input + encrypted parameter / encrypted activation + encrypted parameter 的 FHE 计算。
6. Runner 输出 ciphertext，不持有 secret key，不能解密结果。
7. ciphertext 结果回到 server 端，由 server 解密。

本计划建议不要交换现有 `client` / `server` 命名，而是引入更精确的角色抽象：

- `Provisioner`: server 侧离线准备者，持有明文模型和 secret key，生成 runner bundle。
- `Runner`: 可搬运到 client 现场的执行环境，只持有 eval 公钥、加密参数和计算图。
- `Decryptor`: server 侧解密者，持有 secret key，解密 Runner 返回的结果。

这样可以同时保留现有 README quick start 的 client/server 语义，并清楚表达新模式的安全边界。

## 2. 新模式语义

新增 deployment mode：

```json
{
  "deployment_mode": "server_provisioned_runner",
  "input_mode": "plaintext",
  "parameter_mode": "encrypted_offline",
  "decryptor": "provisioner"
}
```

默认模式保持当前行为，可显式表示为：

```json
{
  "deployment_mode": "client_encrypted_input",
  "input_mode": "client_ciphertext",
  "parameter_mode": "plaintext_lazy",
  "decryptor": "client"
}
```

安全语义：

- `Runner` 可以看到输入明文，因为输入隐私不是新模式目标。
- `Runner` 不应持有 `model_parameters.h5`、明文模型参数或 secret key。
- `Runner` 持有 eval 公钥，包括 relinearization key、rotation/galois key、bootstrapping 所需公钥。
- `Runner` 可读取参数密文并执行计算，但不能直接解密参数或输出。
- `Provisioner/Decryptor` 持有 secret key，因此 server 仍可解密输出，也天然能访问明文模型。

## 3. 目录与产物布局

现有 quick start 产物继续保留：

```text
runs/cifar10/task/
  client/
  server/
    task_config.json
    ckks_parameter.json
    nn_layers_ct_0.json
    model_parameters.h5
    mega_ag.json
```

新模式建议新增并行目录，不覆盖旧模式：

```text
runs/cifar10/task/
  provisioner/
    task_config.json
    ckks_parameter.json
    secret_key.bin
    eval_context.bin
    encrypted_parameter_manifest.json
    model_parameters.h5              # server 私有，可软链接或复制自 task/server

  runner/
    task_config.json
    ckks_parameter.json
    nn_layers_ct_0.json
    mega_ag.json
    task_signature.json
    eval_context.bin
    encrypted_model_parameters/
      manifest.json
      shard_00000.bin
      shard_00001.bin
      ...
```

`runner/` 是可搬运目录，必须通过测试保证不包含：

- `secret_key.bin`
- `model_parameters.h5`
- 任何明文参数 dump

## 4. CIFAR10 目标链路

### 4.1 编译

保留现有编译能力，并新增 deployment mode 参数：

```bash
python training/run_compile.py \
  --input=./runs/cifar10/model/trained_poly.onnx \
  --output=./runs/cifar10 \
  --style=multiplexed \
  --deployment-mode server_provisioned_runner
```

编译阶段仍生成 `nn_layers_ct_0.json` 和 `model_parameters.h5`。不同点是配置中要声明新模式，并触发 encrypted-parameter 模式下的 use-site level 校验。

### 4.2 生成 Runner MegaAG

```bash
python inference/interface/gen_mega_ag.py \
  --task-dir ./runs/cifar10/task \
  --deployment-mode server_provisioned_runner
```

输出写入 `task/runner/mega_ag.json` 和 `task/runner/task_signature.json`，不要覆盖 `task/server/mega_ag.json`。

### 4.3 Provisioner 离线加密参数

新增工具：

```bash
./build/tools/provision_encrypted_runner_bundle \
  --task-dir ./runs/cifar10/task \
  --out ./runs/cifar10/task/runner
```

该工具负责：

- 生成或加载 server-owned CKKS secret key。
- 导出 eval context 到 `runner/eval_context.bin`。
- 读取 `task/server/model_parameters.h5`。
- 对每个参数层调用 C++ layer 的 `generate_*_pt` / packing 逻辑生成 encoded plaintext。
- 按 MegaAG use-site 的目标 level 物化 encoded plaintext，并加密为 `CkksCiphertext`。
- 根据 manifest 写入 `encrypted_model_parameters/`。

### 4.4 Runner 现场执行

```bash
./build/examples/inference \
  --task-dir ./runs/cifar10/task/runner \
  --deployment-mode server_provisioned_runner \
  --input ./runs/cifar10/task/client/img.csv \
  --output-cipher ./runs/cifar10/output.ct
```

Runner 行为：

- 加载 `runner/eval_context.bin`。
- 读取明文 CSV 输入。
- 按 task input 的 packing/scale/level 编码为 plaintext。
- 按 MegaAG 和 manifest 按需加载参数密文。
- 执行 FHE 计算。
- 序列化密文输出。

### 4.5 Server 解密结果

```bash
./build/tools/decrypt_runner_output \
  --task-dir ./runs/cifar10/task/provisioner \
  --cipher ./runs/cifar10/output.ct
```

Decryptor 使用 server secret key 解密 Runner 结果，并输出分类 logits/top-1。

## 5. 编译器修改计划

### 5.1 `training/run_compile.py`

新增参数：

- `--deployment-mode`
  - 默认：`client_encrypted_input`
  - 新值：`server_provisioned_runner`
- 可选：`--runner-output-dir`
  - 默认：`<output>/task/runner`

职责：

- 继续支持现有 ONNX -> JSON -> H5 导出。
- 将 deployment mode 写入 server/runner task config。
- 在新模式下标记 input 是 plaintext，模型参数是 encrypted offline。
- 不在该阶段执行参数加密，避免 Python 编译和 C++ 加密 runtime 耦合。

### 5.2 use-site level 物化，而不是重做 depth 规划

这里需要先澄清现有 MegaAG 对 `pt_ringt` 的处理方式。

`inference/fhe_layers/` 的 C++ `run_core` 路径中，layer 先用
`generate_weight_pt_for_*` / `generate_bias_pt_for_*` 生成 `CkksPlaintextRingt`。
随后：

- 乘法路径按被乘 ciphertext 的当前 level 调 `ringt_to_mul(pt_ringt, level)`，再执行 `mult_plain_mul`。
- 加法路径直接调用 `add_plain_ringt(ct, pt_ringt)`，该接口不显式接收 pt level，但语义上等价于用被加 ciphertext 的当前 level 解释该 ringt。

`inference/model_generator/layers/` 中由 `run_core` 翻译出来的 `call()` /
`call_custom_compute()` 统一使用 `mult()` / fused MAC 节点，并不会在 Python 图中显式插入
`ringt_to_mul()` 节点。处理发生在 MegaAG runtime：

- `CkksPlaintextRingtNode` 的 signature level 通常是 0。
- Python `mult(ct, pt_ringt)` 允许 `pt_ringt.level == 0`，输出 level 跟随 ciphertext。
- CPU MegaAG executor 遇到 CKKS `ct * pt_ringt` 时，会取实际输入 ciphertext 的
  `get_level()`，再调用 `context->ringt_to_mul(*pt_ringt, level)`。
- fused `ct_pt_mult_accumulate*` executor 也是按每个输入 ciphertext 的 level 调
  `ringt_to_mul()`。
- `add(ct, pt_ringt)` executor 则直接走 `add_plain_ringt()`。

因此 encrypted-parameter 模式的 level 处理不应重新设计一套全新的 compiler depth 规划，而应把
“同一个 ringt plaintext 在某个 use-site 被解释成哪个 level 的 plaintext”显式物化为离线密文。

物化规则：

- 对 `mult(ct_L, pt_ringt)` / fused MAC 中的参数：Provisioner 生成原始 `pt_ringt` 后，按 use-site
  的 `ct_L.level` 转成同 level 的 encryptable plaintext，再加密为 `param_ct_L`。
- 对 `add_plain_ringt(ct_L, bias_pt_ringt)`：`bias_pt_ringt` 自身没有可直接读取的目标 level，
  Provisioner 必须从该 add use-site 的 ciphertext 操作数 `ct_L` 推导目标 level `L`，再将
  `bias_pt_ringt` 物化为 level `L` 的 encryptable plaintext 并加密。
  对 dense/conv bias 这类项，`L` 通常就是原图中 `partial_sum = rescale(partial_sum)` 之后、
  `result = add(partial_sum, bias_pt)` 之前的 `partial_sum.level`。
- 如果同一个逻辑 pt 在多个 level 被复用，manifest 中必须生成多个离线密文实例；相同 level 的实例可以去重。
- signature 和 MegaAG 的 offline input id 应表达 use-site level，例如在原始参数 id 后附加 level/use-site
  后缀，或在 manifest 中记录 `source_id + use_site + level` 的映射。

仍需注意：

- encrypted parameter 参与后续 activation 乘法时仍是 `CT * CT`，因此 runtime cost 会增加，并需要 relin key。
- 但只要 graph 把 `relin` 插在 `CT*CT` 后、把 `rescale` 保持在原 `run_core` 统一 rescale 的位置，level/scale
  轨迹可以与原 `CT*PT` 图保持一致；不需要把 CIFAR10 作为一个独立 Phase 4 去重新规划 bootstrapping/depth。
- 不能采用“每个 term 先 `mult_relin + rescale` 再累加”的通用写法替代原 fused/accumulate 结构；这会改变旧图的
  rescale 位置和 level 轨迹。Phase 3 中应先把 Phase 1 prototype 调整为“先 relin/accumulate，再按旧图位置 rescale”。

## 6. MegaAG 生成修改计划

### 6.1 `inference/interface/gen_mega_ag.py`

新增参数：

```bash
--deployment-mode client_encrypted_input|server_provisioned_runner
```

当前默认仍调用：

```python
gen_custom_task(..., lazy=True)
```

新模式调用：

```python
gen_custom_task(
    task_path=...,
    style=...,
    lazy=False,
    parameter_mode="encrypted_offline",
    input_mode="plaintext",
)
```

### 6.2 `inference/model_generator/deploy_cmds.py`

为 `gen_custom_task` 增加参数：

- `parameter_mode`
  - `plaintext_lazy`
  - `plaintext_eager`
  - `encrypted_offline`
- `input_mode`
  - `ciphertext`
  - `plaintext`
- `output_dir`

新模式中：

- graph input feature 不创建 `CkksCiphertextNode`，而是创建适合乘法的 plaintext/ringt node。
- 参数不创建 `CustomDataNode(type='*_data_source')`。
- 参数按 use-site 创建 `CkksCiphertextNode`，并加入 `offline_input_args`；同一逻辑 pt 如在不同 level 使用，
  需要拆成不同 offline arg。
- output 仍创建 `CkksCiphertextNode`。
- `process_custom_task` 必须写出完整 online + offline signature。

### 6.3 task signature 修正

现有 `check_signatures` 逻辑在 `offline` 非空时只校验 offline 列表，这会阻断新模式。应修正为：

```text
expected_args = online inputs + offline inputs + online outputs
n_in_args = len(online inputs) + len(offline inputs)
n_out_args = len(online outputs)
```

同时要保证 `mega_ag.inputs` 的顺序与 `CxxVectorArgument` 顺序一致。

## 7. Layer 接口修改计划

### 7.1 新增接口

在 `inference/model_generator/layers` 中，仿照 `call` 和 `call_custom_compute` 增加：

```python
def make_param_ct_nodes(self, layer_id: str, levels: dict | None = None):
    ...

def call_param_ct(self, x, weight_ct, bias_ct, input_is_plaintext: bool = False):
    ...
```

命名可以调整，但语义必须固定：

- `make_param_ct_nodes` 只声明参数密文节点，不负责加密。
- `call_param_ct` 使用参数密文构图，不再使用 `encode_pt` custom compute。
- 参数节点 shape 必须与现有 `make_pt_nodes` 完全一致，便于 manifest 和加密工具复用索引。

### 7.2 乘法规则

第一层明文输入：

```text
input_pt * weight_ct -> output_ct
```

实现时建议调用：

```python
mult(weight_ct, input_pt)
```

避免 frontend 中 `mult(pt, ct)` 以 plaintext 的 level 创建输出节点。

后续层：

```text
activation_ct * weight_ct -> ct3 -> relin -> accumulate -> rescale at original run_core boundary
```

等价构图必须尽量复刻原 `run_core` 的 rescale 放置：

```python
term = relin(mult(activation_ct, weight_ct))
partial_sum = add(partial_sum, term)
partial_sum = rescale(partial_sum)  # 只在原 CT*PT 图 rescale 的位置执行
```

多个 term 累加时，必须保证 level 一致。不要默认先完成每个 term 的 `mult_relin + rescale` 再累加；
只有当原 layer C++ `run_core` 就是逐 term rescale 时才允许这样做。

### 7.3 bias 规则

现有 `CT*PT` 路径常见行为是：

```text
partial_sum = ct_pt_mult_accumulate(...)
partial_sum = rescale(partial_sum)
result = add(partial_sum, bias_pt)
```

新模式下 bias 是 ciphertext：

```text
partial_sum = sum(mult_relin(...))
partial_sum = rescale(partial_sum)
result = add(partial_sum, bias_ct)
```

因此 bias_ct 必须按 `partial_sum` rescale 后的 level 物化和加密。原 `add_plain_ringt()` 虽然不显式接收
pt level，但 encrypted bias 需要从被加 ciphertext 的 level 推导出对应 level。

具体到 bias：

- Python generator 在构造 `add(partial_sum, bias_pt)` 时，应记录该 use-site 的目标 level，即
  `partial_sum.level`。
- signature/offline arg 中的 `bias_ct` level 应等于这个 `partial_sum.level`。
- Provisioner 仍通过 C++ layer 的 `generate_bias_*_pt_for_*()` 得到原始 `CkksPlaintextRingt`，然后调用
  `ringt_to_pt(bias_pt_ringt, partial_sum.level)`，再将该 plaintext 加密为 `bias_ct`。
- 如果同一个 bias ringt 在不同 add use-site 被不同 level 的 ciphertext 相加，需要为这些 level 分别生成
  ciphertext；相同 level 才能复用。

### 7.4 CIFAR10 v1 覆盖层

优先覆盖 CIFAR10 ResNet-20 quick start 必需层：

- `Conv2DPackedLayer`
- `Conv2DPackedDepthwiseLayer`
- `MultiplexedConv2DPackedLayer`
- `MultiplexedConv2DPackedLayerDepthwise`
- `InverseMultiplexedConv2DLayer`
- `InverseMultiplexedDepthwiseConv2DLayer`
- `DensePackedLayer`
- `PolyRelu0D/1D/2D` 或对应 `polyact` 路径

structural constants 的处理策略：

- CIFAR10 runner signature 中实际进入参数输入列表的 `convm_` masks 已按 offline ciphertext 参数处理，和 weight/bias 一样由
  Provisioner 生成原始 `pt_ringt`、按 use-site level 物化并加密。
- avgpool select tensors、concat masks、repack masks、upsample select tensors 和 routing/mask 类常量如果后续出现在
  runner signature 中，也应迁移为 offline ciphertext 参数，或明确标注为 Runner-local public constants，由 Runner 本地生成而不是作为外部 CSV 输入读取。

这样可以避免 Runner 在 `server_provisioned_runner` 模式下继续要求额外 plaintext CSV 输入，也避免暴露 packing/routing 结构细节。

## 8. C++ Provisioner 与 Runner 修改计划

### 8.1 Provisioner 工具

新增 C++ tool：`provision_encrypted_runner_bundle`。

输入：

- `task/server/nn_layers_ct_0.json`
- `task/server/task_config.json`
- `task/server/ckks_parameter.json`
- `task/server/model_parameters.h5`
- 可选已有 secret key 路径

输出：

- `task/provisioner/secret_key.bin`
- `task/provisioner/eval_context.bin`
- `task/runner/eval_context.bin`
- `task/runner/encrypted_model_parameters/manifest.json`
- `task/runner/encrypted_model_parameters/shard_*.bin`

实现原则：

- 不在 Python 端重写 packing 逻辑。
- 复用 `inference/fhe_layers` 里每个 layer 的 `generate_weight_pt_for_*`、`generate_bias_pt_for_*` 等方法。
- Provisioner 不直接按“原始参数 id”只加密一次，而是按 MegaAG/signature 中的 offline arg use-site
  加密；use-site 必须包含目标 level。
- 对 multiply use-site：生成 `pt_ringt` 后，按原 executor 会调用 `ringt_to_mul(pt_ringt, ct_level)` 的
  `ct_level` 物化为同 level 的 encryptable plaintext，再调用 server-owned public key 加密。
- 对 add use-site：从被 add 的 ciphertext level 推导目标 level，物化并加密 bias。
- 序列化 ciphertext 时记录 id、source_id、use_site、shape、level、scale、layer_id、param_kind、indices。
- 如果同一 `source_id + indices` 在多个 level 使用，manifest 中保存多个 ciphertext 实例；如果 level 相同，
  可以在 manifest 层做去重引用。

### 8.2 EncryptedParameterStore

新增 runtime 组件：

```cpp
class EncryptedParameterStore {
public:
    explicit EncryptedParameterStore(std::filesystem::path root);
    const std::vector<CkksCiphertext>& load_argument(const std::string& arg_id);
    bool has_argument(const std::string& arg_id) const;
};
```

职责：

- 读取 manifest。
- 按 `CxxVectorArgument` 需要的 arg id 懒加载 ciphertext。
- 缓存已加载参数，避免每次重复反序列化。
- 校验 ciphertext level/size 与 `task_signature.json` 一致。
- 支持多个 offline arg 指向同一 source parameter 的不同 level 实例；cache key 使用 offline arg id，
  manifest 可额外提供 source 去重信息。

### 8.3 InferenceRunner

新增 API：

```cpp
class InferenceRunner {
public:
    explicit InferenceRunner(const std::string& runner_dir, bool use_gpu = false, int gpu_device = 0);
    void load();
    std::map<std::string, Bytes> evaluate_plaintext_input(
        const std::map<std::string, std::string>& input_csvs,
        lattisense::ProgressCallback progress_cb = nullptr);
};
```

行为：

- 加载 runner config、`mega_ag.json`、`task_signature.json`、`eval_context.bin`。
- 将 CSV 明文输入编码为 plaintext/ringt argument。
- 从 `EncryptedParameterStore` 获取 offline ciphertext parameter arguments。
- 创建 ciphertext output buffer。
- 调用 `FheTaskCpu/FheTaskGpu::run`。
- 返回序列化 ciphertext output。

### 8.4 InferenceProvisioner / Decryptor

新增或拆分 API：

```cpp
class InferenceProvisioner {
public:
    explicit InferenceProvisioner(const std::string& task_dir);
    void setup_or_load_keys();
    void export_runner_bundle(const std::string& runner_dir);
    std::map<std::string, std::vector<double>> decrypt_runner_output(
        const std::map<std::string, Bytes>& encrypted_outputs);
};
```

此类只在 server/provisioner 侧使用，不进入 runner bundle。

## 9. MegaAG Runner 底层注意事项

### 9.1 CPU/GPU CxxVectorArgument

`CxxVectorArgument` 已支持 `CkksCiphertext`、`CkksPlaintext`、`CkksPlaintextRingt`。新模式应优先复用它传入：

- online plaintext input
- offline encrypted parameter ciphertext
- online ciphertext output

### 9.2 offline inputs

`process_custom_task` 已能输出 `offline_input_args` 和 `mega_ag.offline_inputs`，但当前业务路径没有把模型参数作为 offline args 使用。需要补齐：

- Python generator 生成 offline parameter args。
- C++ signature check 正确合并 online/offline/output。
- Runner 运行时在传参顺序上与 signature 保持一致。

### 9.3 CT*CT 的 key 要求

新模式中后续参数层会产生 `relin`，因此 eval context 必须包含足够 level 的 relin key。

卷积、packing 和 bootstrapping 仍需要 rotation/galois keys。Provisioner 导出的 eval context 要覆盖 MegaAG 中所有 key signature。

## 10. 测试计划

本节作为回归与验收清单保留，不再作为独立待执行 Phase。

### 10.1 Python generator unit tests

新增测试：

- 单层 conv2d，`input_mode=plaintext`，`parameter_mode=encrypted_offline`。
- 单层 dense。
- CIFAR10 子图。

检查：

- input signature 类型是 plaintext/ringt。
- parameter signature 类型是 ciphertext，phase 是 offline。
- output signature 类型是 ciphertext。
- 后续层参数乘法为 `MULTIPLY + RELINEARIZE`，但 `RESCALE` 位置必须与原 `CT*PT`/fused MAC 图一致。
- 对 dense/conv accumulate 路径，测试应断言不是“每个 term 一个 rescale”，而是“按原 `run_core` 边界统一 rescale”。
- offline 参数签名要覆盖同一逻辑 pt 的多 level use-site；同 level 可去重，不同 level 必须拆分。

### 10.2 C++ unit tests

新增测试：

- Provisioner 生成 encrypted parameter bundle。
- Runner bundle 不包含 secret key 和 H5。
- Runner 单层 conv 输出 ciphertext。
- Provisioner 解密输出后与明文参考误差在 CKKS 容忍范围内。
- Provisioner 对同一 source pt 的不同 target level 生成不同 ciphertext manifest entry。

### 10.3 CIFAR10 end-to-end

完整跑通：

1. ONNX/H5 导出。
2. compile。
3. runner MegaAG 生成。
4. provision encrypted bundle。
5. runner 明文输入现场执行。
6. decryptor 解密输出。
7. 与 `evaluate_plaintext()` 或现有 verify 模式比较。

### 10.4 回归测试

必须确认不破坏：

- README 当前 quick start。
- `InferenceClient` / `InferenceServer` client encrypted input 模式。
- `gen_mega_ag.py --task-dir ...` 默认行为。
- `ctest -R cifar10` 现有用例。

## 11. 分阶段落地状态

以下 Phase 已完成；后续如果继续扩展能力，应另起新的计划项。

### Phase 1: Graph and signature prototype（已完成）

- 增加 mode config。
- 为 ordinary conv/dense 做最小 `call_param_ct`。
- 生成单层 Runner MegaAG。
- 修正 `check_signatures`。
- 不先追求 CIFAR10 全图。

验收标准：

- 单层 dense 或 conv 能生成合法 `task_signature.json` 和 `mega_ag.json`。
- signature 明确区分 plaintext input、offline ciphertext params、ciphertext output。

### Phase 2: Provisioner and Runner skeleton（已完成）

- 实现 `provision_encrypted_runner_bundle`。
- 实现 `EncryptedParameterStore`。
- 实现 `InferenceRunner` 单层执行。
- 实现 decrypt tool。

验收标准：

- 单层 encrypted parameter bundle 可以在无 secret key runner 目录下执行。
- server 可解密结果。

### Phase 3: CIFAR10 layer coverage（已完成）

- 修正 encrypted-parameter graph，使 CT*CT 后只插入 relin，rescale 放置复刻原 CT*PT 图。
- 实现 use-site level 参数物化和 manifest/signature 适配。
- 覆盖 CIFAR10 quick start 中所有参数层。
- 处理 multiplexed/big/depthwise conv。
- 将 CIFAR10 runner signature 中的 `convm_` structural masks 也物化为 offline ciphertext 参数，避免 Runner 需要额外 plaintext mask CSV/input。
- 处理 residual add 路径中的 `mult_scalar` 参数。
- 处理 polyact 参数。
- 对进入 runner signature 的 structural masks/select tensors，默认应按 offline ciphertext 参数处理；确需 plaintext 的常量必须明确改成 Runner-local public constants。

验收标准：

- encrypted-parameter 图的 level/scale 轨迹与原图一致，额外差异只应是 CT*CT 的 relin 和更高运行成本。
- 同一 pt 多 level 复用时，runner bundle 中有正确的多实例密文或明确的同 level 去重引用。
- CIFAR10 ResNet-20 新模式端到端输出可解密。
- 与明文参考误差可接受。

### Phase 4: Documentation and public API polish（已完成）

- 已新增中文文档：`docs/zh/server_provisioned_runner.md`。
- 已给出 server provision、runner execute、server decrypt 的 quick start。
- 已标明新模式安全边界：Runner 看得到输入明文，但没有 secret key 和明文模型参数。
- README 暂未更新，符合当前“只新增中文文档”的要求。

## 12. 关键风险与决策

### 12.1 性能风险

`CT*CT` 比 `CT*PT` 贵很多，并引入 relin。CIFAR10 ResNet-20 的延迟和内存会明显增加。需要在文档中明确新模式不是现有模式的性能等价替换。

### 12.2 use-site level 风险

当前图的 level/bootstrapping 是按原模式规划的。新模式不应重新引入一套独立 depth 规划，而应复用原图的
level/scale 轨迹：参数密文按每个 use-site 的 target level 生成，CT*CT 后 relin，rescale 保持在原位置。

主要风险是 graph generator 把 CT*CT 改写成逐 term rescale，或者 Provisioner 只按 source pt 加密一次，导致
同一个 pt 在不同 level 的 use-site 被错误复用。

### 12.3 参数密文体积风险

把所有 encoded parameters 加密后，体积会远大于 H5 和 plaintext ringt 参数。需要 shard + manifest，并支持 lazy load。

### 12.4 命名风险

不建议把 client/server 对调。代码中应使用 `Provisioner`、`Runner`、`Decryptor` 表达真实职责；旧 API 名称保留给现有 client encrypted input 模式。

### 12.5 结构常量是否加密

v1 已将 CIFAR10 runner signature 中的 `convm_` masks 纳入 offline ciphertext 参数。后续若有 avgpool select tensors、concat masks、repack masks、upsample select tensors 或其他 routing/mask 类常量进入 runner signature，默认也应按相同策略加密；只有明确由 Runner 本地确定、且不作为外部输入出现的 public constants 才保持 plaintext。

## 13. 最小成功定义

以 README CIFAR10 为例，新模式最终应满足：

1. server 侧可生成 runner bundle。
2. runner bundle 中无 secret key、无 `model_parameters.h5`。
3. runner 可在只有 eval context、MegaAG、encrypted params、明文输入 CSV 的情况下执行。
4. runner 产出 ciphertext。
5. server 可解密 ciphertext 并得到与明文参考接近的 logits。
6. 现有 quick start 和旧 API 默认行为不变。

至此 PLAN.md 中的 Phase 1-4 均已落地；本文档不再包含新的待执行 Phase。
