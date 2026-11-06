# Multiplexed Conv2D lazy mask plaintext design

## 1. 目标

在 `server_provisioned_runner` 模式下，不再把
`MultiplexedConv2DPackedLayer` 的 structural mask (`convm_`) 作为离线
ciphertext 写入 Runner bundle。无论 `parameter_loading_mode` 是 `eager` 还是
`runtime_lazy`，Runner 都在执行到卷积 use-site 时按现有层参数生成 mask
plaintext，并以 `CT × PT` 完成相同的 mask-and-rotate 计算。

本设计只处理 multiplexed Conv2D 的 mask。卷积权重、bias、`mult_scalar`、
polynomial coefficient 仍按各自现有的 eager/runtime-lazy encrypted parameter
流程加载。`client_encrypted_input` 模式和其他层不改变。

## 2. 背景与约束

当前 `convm_` mask 只包含 0/1 结构信息，不包含模型权重，但每个 mask 被编码
并加密为 CKKS ciphertext。CIFAR10 ResNet20 中这些 mask 约占 1.24 GB，并且
运行时会引入不必要的 `CT × CT`、relinearization 和 ciphertext 存储成本。

mask 的 plaintext 编码必须保持 `inference/fhe_layers` 中的现有定义，不能在
Python 图生成器中重新推导 scale：

```text
generate_mask_pt_for_indices(...)
    -> ctx.encode_ringt(mask, ctx.get_parameter().get_q(level_ - 1))
```

因此普通 post-skip 路径和 `reduct_rot` 路径的 mask 都必须以
`q(level - 1)` 编码，并通过 `ringt_to_mul(..., level - 1)` 物化后参与乘法。
不得使用 `default_scale`、权重的 `weight_scale` 或 bias scale 替代该 scale。

## 3. 设计

### 3.1 Graph/signature

在 server-provisioned-runner 的 multiplexed Conv2D 图中：

1. `convm_` 不进入 `encrypted_parameters`，也不出现在 eager 的 `offline`
   参数列表或 runtime-lazy 的参数 store 中；
2. 权重和 bias 的 encrypted parameter 节点保持现有 id、shape、level 和 lazy
   load attributes；
3. mask 改为 Runner-local 的 `encode_pt` custom node，节点
   attributes 至少包含：
   - `op_class=MultiplexedConv2DPackedLayer`；
   - `type=mask_pt`；
   - mask 索引（普通路径为 `i`，`reduct_rot` 路径为 `ct_idx` 和 `i`）；
   - 目标 level `level - 1`，或可由当前 layer level 唯一推导出的等价信息。

节点只在实际执行的 output block 上创建，沿用当前 `valid_i` 截断规则，避免
为不存在的输出通道生成无用 mask。

### 3.2 Runner 生成与缓存

Runner 的 `encode_pt` 执行路径复用 C++ layer 的
`generate_mask_pt_for_indices` 和
`generate_mask_pt_for_indices_reduct_rot`。不复制 mask 构造公式到 Runner
或 Python；这样 padding、skip、stride、周期 packing 的定义只有一个来源。

生成过程为：

1. 从 node attributes 找到 layer 和 mask 索引；
2. 在当前 CKKS context 中构造 0/1 mask；
3. 使用 `ctx.encode_ringt(mask, ctx.get_parameter().get_q(level - 1))`；
4. 以 `(layer_id, path, ct_idx, i, level)` 为缓存键缓存 ringt；
5. 在同一 use-site 通过 `ringt_to_mul(mask_pt, level - 1)` 使用。

缓存是 Runner 进程内缓存，不写回 bundle，也不改变 task signature。eager 和
runtime-lazy 只决定权重/bias 的加载方式，不影响 mask 的生成方式。

### 3.3 两条卷积路径

- `run_core_for_post_skip_rotation`：mask 仍在 sum-slot 后、rotate 前使用，
  mask plaintext level 为 `level_ - 1`，乘法后执行原有 rescale。
- `run_core_for_reduct_rot`：每个 `(ct_idx, i)` 使用对应 mask，仍按原有
  `level_ - 1` 编码和 rescale 轨迹执行。

除 mask 的数据类型从 encrypted CT 改为 generated PT 外，旋转顺序、rescale
次数、bias use-site 和输出分组保持不变。

## 4. 最小修改面

实现只应触及以下职责：

1. `inference/model_generator/layers/multiplexed_conv2d_pack_layer.py`：在
   server-provisioned-runner 图中创建 mask `encode_pt` 节点，并保留
   `client_encrypted_input` 和其他旧部署模式的 encrypted mask 行为；
2. `inference/model_generator/deploy_cmds.py`：将该模式开关传给 multiplexed
   Conv2D 的参数节点生成，不改变其他层的参数分类；
3. `inference/fhe_layers/multiplexed_conv2d_pack_layer.cpp/.h`：让 Runner 的
   `encode_pt` 节点调用已有 mask 生成函数，并明确使用 `q(level - 1)`；必要时
   增加最小的 `(path, index)` 分派，不重写 mask 公式；
4. `inference/interface/inference_runner.cpp` 或现有 encode-pt dispatch：只在
   需要时注册该节点属性和缓存，不引入新的 bundle 格式。

Provisioner 不再解析或加密 `convm_`；manifest 中不应出现 `param_kind=mask`
的 argument。不得修改 `client_encrypted_input` 模式的 signature。

## 5. 正确性与 scale 不变量

实现必须满足：

- `convm_` 从 server-provisioned-runner 的 encrypted parameters 和 manifest
  消失（包括 eager 与 runtime-lazy）；
- 生成的 mask 槽值与当前 C++ `select_tensor()` 结果逐槽相同；
- post-skip 和 `reduct_rot` 的 mask plaintext 均使用 `q(level - 1)`；
- mask 的 `ringt_to_mul` level 为 `level - 1`；
- mask 乘法后的 rescale 次数和目标 scale 与当前路径完全一致；
- 权重/bias 的 use-site level、参数 id 和 lazy load 语义不变；
- `client_encrypted_input` 和非 multiplexed 层行为不变；server-provisioned-runner
  的 eager 与 runtime-lazy 都使用 plaintext mask。

## 6. 验证

先用现有 periodic ResNet20/CIFAR10 任务分别生成 eager 和 runtime-lazy bundle，确认：

1. `manifest.json` 不包含 mask 参数；
2. `task_signature.json` 的 `encrypted_parameters` 不包含 `convm_`；
3. 两种 server-provisioned-runner bundle 都至少减少当前 mask ciphertext 的约
   1.24 GB；
4. GPU Runner 推理成功，且 encode-pt 路径没有把 activation 从 GPU 拷回 CPU；
5. 解密输出与现有 periodic 基线在允许的 CKKS 数值误差内一致。

非 periodic bundle 已作为此前完成的 baseline 保留；本设计实现后的验收不再
重新生成或运行非 periodic 任务。

应运行现有 server-provisioned-runner 单元测试，并增加一个最小 graph/signature
断言测试：server-provisioned-runner 的 eager 和 runtime-lazy multiplexed Conv2D
都不产生 `convm_` encrypted argument，而 `client_encrypted_input` 仍保持原行为。

## 7. 非目标与风险

- 不隐藏 Runner 已能从 graph、shape、rotation 和 signature 推断的完整网络结构；
  本设计只移除 mask ciphertext，不提供图结构保密协议。
- 不把 mask 改成 secret-dependent 数据；mask 继续被视为 Runner-local public
  structural constant。
- 不扩展到 concat、avgpool、upsample、depthwise 或其他 routing mask；这些类型
  需要单独确认其 scale 和执行路径后再设计。
- 不改变 CKKS 参数、level budget 或 rescale 规划。若某个路径无法证明其
  `q(level - 1)` 约束，应保留原 encrypted mask 路径并让测试失败，而不是引入
  新的 scale 猜测。
