# Multiplexed Conv2D 有效通道周期复制：精简设计规格

状态：提案，尚未实现

关联分析：[权重 pt 数量分析](../research/multiplexed_conv2d_weight_pt_analysis.md)

实现决策：周期复制由 `MultiplexedConv2DPackedLayer` 根据已有层形状自动选择，
不新增 task/layer 配置。明文首层沿用预写周期副本的方案；中间密文 activation
在卷积层入口用 rotate-and-add 扩展。

## 1. 目标和边界

目标是减少 multiplexed Conv2D 的权重 plaintext/parameter 数量，同时保持层间
`Feature2DEncrypted` 的 canonical layout 不变。周期模式只改变当前卷积的输入
shift 数和权重索引，bias、mask、stride/repack、输出 ciphertext 数量和下游算子
接口都保持现状。

以下条件同时满足时自动启用周期模式：

```text
Ci <= C
n_packed_in_channel == 1
external_upsample_factor == [1, 1]
D < B
```

其中 `Ci` 是输入通道数，`C = n_channel_per_ct`，`B` 是一个输入 ciphertext
的 block 数，`D` 是有效通道周期所需的 block 数。当前类只处理普通
`groups == 1`、非 `is_big_size` 的 multiplexed Conv2D，因此不再重复添加配置项。

实现还应在内部确认 `C`、`Cp` 能按 `skip` 整除且 `B % D == 0`；不满足时使用旧
路径。这个检查是防止非法槽布局的保护，不暴露为用户配置。

本规格不覆盖 depthwise、grouped、big-size/inverse multiplexed、多个输入
ciphertext 或非整齐周期。没有开启条件时，现有实现和参数 shape 完全不变。

## 2. 周期和收益

记：

```text
S  = skip[0] * skip[1]
Cp = max(S, next_power_of_two(Ci))
D  = Cp / S
O  = ceil(Co / B)
K  = kernel_height * kernel_width
```

周期输入的前 `D` 个 block 保留真实通道和周期内 padding，之后的 block 是它的
副本。权重只枚举 `b = 0..D-1`，因此权重 shape 从普通模式的
`[O][n_packed_in_channel * B][K]` 变为 `[O][D][K]`，权重 pt 数为 `O*D*K`。

对于每个输出 block `t`、skip 子位置 `r` 和输入 shift `b`，使用：

```text
cin  = ((b + t) * S + r) mod Cp
cout = o * B + t
```

`cin >= Ci` 的位置写零，边界 kernel mask 仍按现有逻辑生成。固定 `t` 时，
`(b, r)` 恰好覆盖一个周期内的通道，不重复也不遗漏真实输入。

若输入是 canonical ciphertext，令 `A = H * W * S`、`q = B / D`，在卷积入口
计算：

```text
expanded = x
for copied_blocks = 1; copied_blocks < q; copied_blocks *= 2:
    expanded += rotate(expanded, copied_blocks * D * A)
```

之后只需对 `D-1` 个 block shift 做卷积。旧路径需要 `B-1` 个 block rotation；
周期路径需要 `D-1 + log2(q)` 个 rotation，并增加 `log2(q)` 个 add。`D < B`
时旋转通常更少；明文首层预先写入副本时不需要 expansion rotation，只保留
`D-1` 个 shift。

## 3. C++ 实现范围

涉及：

- `inference/fhe_layers/multiplexed_conv2d_pack_layer.h`
- `inference/fhe_layers/multiplexed_conv2d_pack_layer.cpp`

### 3.1 自动选择

构造函数在计算 `n_packed_in_channel`、`n_block_per_ct` 后，推导 `Cp`、`D` 并
保存一个内部 `use_periodic_pack`/`periodic_block_count` 状态。构造函数签名不变，
也不读取 JSON metadata。周期状态只由公开的 shape、skip、pack 和 upsample 值决定。

### 3.2 权重生成

`prepare_weight_for_post_skip_rotation()`、lazy 版本以及所有实际使用的 weight
shape 循环统一使用：

```text
input_block_count = use_periodic_pack ? D : n_packed_in_channel * B
```

`generate_weight_pt_for_indices()` 在周期模式下把 `j` 解释为 `b`，使用第 2 节
的 `cin` 公式；普通模式保留现有 `packed_in_channel_idx/block_idx` 解释。eager、
lazy 和 Provisioner 的 custom `encode_pt` 都继续调用这个同一生成函数。

### 3.3 执行路径

在 `run_core()`、`run_core_for_post_skip_rotation()` 以及仍可达的
`run_core_for_reduct_rot()` 中：

1. 周期模式先对每个输入 ciphertext 做一次倍增复制；
2. block rotation、kernel rotation 和逐项乘加只遍历 `D` 个输入 block；
3. 输出的 sum-slot、repack、mask、bias 和 `Feature2DEncrypted` 元数据继续沿用
   旧代码。

expanded ciphertext 只是局部临时值，不写回 feature，也不改变
`n_channel_per_ct`。周期模式不新增乘法、rescale 或 level drop。

## 4. Python 构图范围

涉及：

- `inference/model_generator/layers/multiplexed_conv2d_pack_layer.py`

Python 类使用与 C++ 完全相同的自动判定和 `D`。以下入口必须共享同一个
`input_block_count`：

- `make_pt_nodes()`
- `make_param_ct_nodes()`
- `call()`
- `call_custom_compute()`
- `call_param_ct()`

周期模式下，weight node shape 为 `[O][D][K]`；bias 和 mask shape 不变。图中
对 ciphertext input 先构造与 C++ 相同的 rotate-and-add expansion，再生成 `D`
个 block shift。custom compute 的 `i/j/k` 仍是 row-major，`j` 的范围变化由
同一 shape 自动传递给 lazy executor 和参数签名。

明文首层若采用预写周期副本，`call_param_ct_plaintext_input()` 只消费已经复制
好的输入 block，并按 `D` 解释 `j`。它不新增图内 plaintext rotation。

## 5. 明文输入的必要边界

当前 `server_provisioned_runner` 的 `encode_plaintext_2d_input()` 默认只写
canonical 的第 `0..D-1` 个真实 block，后续 block 是零。若要保留本规格的
“明文首层直接写周期副本”方案，输入编码器必须把这 `D` 个 block 复制到全部
`B` 个 block。

因此：

- 只验证密文 activation 或普通 encrypted-input 路径时，C++/Python 两个 layer
  文件即可完成核心算子改动；
- 要求 server-provisioned plaintext 首层端到端可运行时，还需要在
  `inference/interface/inference_runner.cpp` 的现有 multiplexed 输入分支加上
  这段无配置复制逻辑。它不需要新增 metadata、signature 字段或 task 配置。

这是输入槽内容的事实差异，不能仅靠 C++ 权重索引修复；不应把它隐藏在
Provisioner 或运行时回退中。

## 6. 参数和部署影响

不新增 periodic metadata。Python 生成的 signature 已经包含 `[O][D][K]`，C++
Provisioner 通过初始化后的同一个 layer 生成对应 plaintext，因此 eager、lazy、
encrypted-offline 和 runtime-lazy 不需要另一套索引协议。

Provisioner 只需继续校验 signature shape 与 `weight_pt` shape 一致；不应根据
`n_block_per_ct` 自行推导输入维度。bias、mask、其他算子参数和旧 bundle 的
普通模式 shape 不变。

## 7. 正确性和测试

至少加入以下验证：

1. 对 `Ci=3/16/32/64`、`skip=1/2/4`、`1x1/3x3` 比较普通卷积、周期槽映射和
   直接 zero-padded convolution；周期条件不满足时断言走旧路径。
2. 断言每个输出 block 的 `(b,r)` 覆盖 `0..Cp-1` 一次，并检查周期内 padding
   权重为零。
3. 断言周期输入的每个副本与首周期相同，`D=B` 或 `D>B` 时不启用优化。
4. 比较 eager、lazy、encrypted-parameter 三种 graph 的 weight shape、flat index、
   level 和输出；检查扩展只产生 rotate/add，不改变 level/scale 约定。
5. 对 server-provisioned plaintext 输入增加槽向量测试，确认 Runner 预写副本后
   与密文 expansion 路径得到相同卷积输出。
6. 回归所有未满足自动条件的普通 multiplexed、depthwise、big-size 和多输入
   ciphertext 测试，确认参数数量和输出保持不变。

验收时同时记录 weight pt/ct 数量、rotation/add 数量、实际 bundle 字节和端到端
误差；`O*D*K` 只代表权重项下降，不直接代表 bundle 或延迟按同一比例下降。

