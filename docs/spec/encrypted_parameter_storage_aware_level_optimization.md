# 加密参数存储感知的 level 优化设计

## 1. 最小修改方案

仅在 `server_provisioned_runner` 编译路径中，为现有 DP 目标增加加密参数存储成本。
沿用当前 periodic 参数布局和 Runner-local mask PT 行为，不改变 bundle 格式。

### 1.1 由 inference layer 提供参数计数

在 `inference/model_generator/layers/` 的相关层中，参照
`get_fhe_op_count(level)` 增加只读接口：

```python
get_encrypted_param_count(level: int) -> dict[int, dict[str, int]]
```

返回值按实际参数密文 level 和参数类型分组。例如，消耗两级的 multiplexed
Conv2D 返回：

```python
{
    level: {"weight": n_weight},
    level - 2: {"bias": n_bias},
}
```

计数和 level 映射必须复用各层 `make_param_ct_nodes()` 的现有字段及规则：

- Conv2D weight 使用输入候选 level `L`；bias 按现有路径使用 `L-1` 或 `L-2`。
- Multiplexed Conv2D 使用包含 periodic 策略的 `weight_input_count`、
  `n_block_per_ct`、`kernel_size` 等现有字段计算实际参数节点数量。
- poly、dense、mult-scalar 等参数化层使用各自现有参数 shape 和 use-site level。
- Runner 在线生成的 mask PT 不计入加密参数存储；其他实际落盘的加密参数须计入。

接口只计算数量，不创建节点、不读取模型 tensor，也不执行编码、加密或序列化。
先覆盖当前 ResNet20 路径涉及的参数化层；未覆盖的参数化层在报告中明确标记为
未计费，不能将其视为零或将结果称为全部加密参数大小。

### 1.2 编译器计算存储分数

在 `training/model_compiler/score.py` 中复用
`_build_layer_and_get_op_count()` 的层构造方式，调用计数接口，再按实际参数
level 将数量换算为字节数。仅在必要时局部提取构造代码，不引入注册系统，
不在编译器重复实现层内参数布局公式。

```text
storage_bytes = sum(count * bytes_per_element(kind, parameter_level))
total_score = runtime_score + refresh_score
            + encrypted_parameter_storage_weight * normalized_storage_score
```

单元素字节数优先使用 SDK 的确定性公式；没有稳定公式时使用固定校准表，并记录
对应的 CKKS 参数集和序列化版本。存储分数使用固定单位归一化，DP 内只计算数量、
查表，不执行实际加密或序列化。

在现有编译配置中增加 `encrypted_parameter_storage_weight`，默认 `0.0`。
权重为零时跳过新增计费路径，保持原有 score 和编译结果。
`client_encrypted_input` 不启用此分数，不新增 layer/task metadata。

在 `get_compute_score()` 中按 DP 当前候选输入 level 追加存储分数，让它参与
候选比较；不能使用编译后的固定 level。沿用现有包含 pred/succ levels 的缓存，
确保计费所依赖的层属性和成本配置在缓存作用域内固定。

输出权重、字节估计依据及 runtime、refresh、storage 分项。最终
`recompute_final_level()` 完成后，按最终图重新计算存储报告；若 level 重写使
DP 计费与最终结果不一致，须明确报错或重新求解，不能沿用旧估计。

## 2. 验收计划

1. **参数计数与 level**：对当前 ResNet20 涉及的参数化层，将新接口结果与
   `make_param_ct_nodes()` 的实际参数 shape、level 逐项比较，覆盖 periodic
   卷积、stride/skip 分支及 bias level 差异；确认在线 mask PT 未计入。
2. **默认兼容**：使用现有 ResNet20/CIFAR10 图，storage 权重为零时逐节点比较
   feature level、BTP 数量和 runtime score，结果须与修改前一致。
3. **优化生效**：开启正权重，输出逐层参数数量、实际参数 level、估计字节数及
   score 分项；比较 baseline 与优化图的 BTP 数量、drop-level 位置、运行时成本
   和加密参数估计大小。所有候选须满足现有 level/scale 约束，最终报告须与最终图
   一致；记录所用权重和运行时成本变化。
4. **GPU 完整链路**：仅验收现有 periodic 模式，生成独立的
   `server_provisioned_runner` bundle 和任务目录，不覆盖已有产物。执行完整
   ResNet20/CIFAR10 Provisioner、Runner 和输出解密，沿用现有正确性判据，检查
   scale、rescale、参数 expected level 及在线 mask PT 行为。
5. **大小核对**：按新 bundle manifest 的真实参数数量和序列化长度核对估计，
   分别报告加密参数 payload、完整 bundle、完整任务目录的大小，以及 GPU 推理
   耗时；与已有 periodic baseline 比较，无需重跑 CPU 或非 periodic 模式。

验收通过条件：默认配置兼容；正权重下加密参数估计和实测大小下降，所有已覆盖
参数计费正确，GPU 完整推理及解密结果正确，并明确记录耗时与 BTP 数量变化。
