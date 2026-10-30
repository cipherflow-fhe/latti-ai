# Multiplexed Conv2D 交错子网格 seed weight 设计规范

状态：设计规范，包含全局编译配置与旋转方向约定。本文覆盖 multiplexed
普通卷积、depthwise 卷积以及 inverse multiplexed 卷积的参数生成、执行和
`runtime_lazy` 参数加载路径，目标是减少空间重复权重的离线密文数量。

## 1. 目标与非目标

目标是把现有卷积权重 PT 中沿空间位置重复的向量，按 `(a,b)` 子网格交错放入
seed weight，再在 use-site 恢复为现有卷积乘法需要的 weight PT。`a`、`b` 必须是正整数，只有在二者都是 2 的幂、`a` 或 `b` 至少一个大于 1，且分别整除当前
逻辑空间尺寸时才启用 seed 路径；否则保持现有实现和参数格式。

默认构造参数为 `seed_weight_subgrid_stride = {1, 1}`。`(1,1)` 永远回退旧路径，
不生成 seed、mask 或额外旋转。非正数、非法或不适用于当前空间布局的值不通过 seed 路径；
为保持旧任务兼容，算子回退旧路径并记录 debug 信息，不改变输出图结构。

本版覆盖 multiplexed 普通 Conv2D 和 depthwise Conv2D 的一般 kernel shape，包括
`1x1`、`3x3` 和其他已有算子支持的 kernel，并覆盖 inverse multiplexed 普通
卷积和 depthwise 卷积。普通 multiplexed 保持输入 block 轴布局；depthwise
和 inverse 沿最内层 kernel 轴打包并保留其他外轴。reduct-rot 和其他未列出的
卷积实现仍回退旧路径。除 seed 恢复所需操作外，不改变 activation 的 level 排程、输出布局、
stride、skip 或 external upsample 语义。

不做跨层/跨输出通道 seed 共享，不引入独立映射表，不在第一版让 DP 搜索任意
`(a,b)`，也不改变 CKKS 参数集。

## 2. 算子入口与最小路由

在 C++ 和 Python 翻译层的普通/depthwise multiplexed 与 inverse multiplexed
Conv2D 构造函数增加可选参数：

```text
seed_weight_subgrid_stride: [a, b] = [1, 1]
```

Python 层将该参数传递给 C++ 层和参数计数接口；未提供时行为与当前版本完全一致。
算子内部只保存该参数和 `use_seed_weight_pack_` 标志，不维护 layer-to-seed
映射表。标志由以下条件共同决定：

```text
is_power_of_two(a) && is_power_of_two(b)
&& (a > 1 || b > 1)
&& seed_shape[0] % a == 0
&& seed_shape[1] % b == 0
```

普通和 depthwise multiplexed 使用 `seed_shape=input_shape`；inverse 使用
`seed_shape=block_shape`，整除检查必须与实际 seed 空间一致。

`skip` 造成的物理空间尺寸使用现有 `input_shape * skip` 和槽位步长规则；seed
子网格判断在逻辑 `(h,w)` 上进行，物理旋转量沿用现有
`[B,H,sh,W,sw]` 布局。`a`、`b` 大于逻辑尺寸、不能整除尺寸或不是 2 的幂时，
直接走旧的 `prepare_*` / `run_*` / `generate_*` 路径。

### 2.1 全局编译配置与分发

通过 `training/run_compile.py --config <json>` 指定全局 `(a,b)`，字段名称沿用
算子入口。仓库的 `training/config/storage_aware_config.json` 示例为：

```json
{
  "encrypted_parameter_storage_weight": 1.0,
  "encrypted_parameter_storage_normalization_bytes": 1073741824,
  "seed_weight_subgrid_stride": [4, 4]
}
```

配置文件必须由 `--config` 显式传入，程序不自动查找该文件。可将 `[4,4]` 改成
其他合法的 `[a,b]`；代码默认仍为 `[1,1]`。省略字段或设为 `[1,1]` 时不添加
逐层 seed 字段，保持旧图和参数格式。非法类型、非正数或非 2 幂的全局值回退
`[1,1]` 并记录 debug 信息；不能整除某层逻辑空间尺寸的值只在该层回退。

该全局入口用于 `server_provisioned_runner` 的 multiplexed 普通 Conv2D、
depthwise Conv2D 和 inverse multiplexed Conv2D；ConvTranspose 和其他卷积实现
继续使用原路径。编译器在布局和
层类型确定后、DP 评分之前，把该值写入适用卷积节点的
`seed_weight_subgrid_stride` 属性；全局值不作为 DP 搜索变量，也不改变 activation
的 `level_cost` 或主图最高 feature level。

评分和导出必须使用同一份节点属性：`score.py` 构造卷积算子时传入该值，同时
调用该算子的 op count 与 encrypted parameter count；导出的
`nn_layers_ct_0.json` 在对应卷积层记录该字段，并复制到 Provisioner/Runner
任务目录。Python MegaAG 生成器与 C++ 参数生成器从逐层字段读取值，Runner 从
MegaAG 中的恢复操作及公开 mask 属性获得布局。这样模型产物完整记录 `(a,b)`，
运行阶段不依赖 training 配置文件。

只在导出 JSON 后补写 `(a,b)` 会使 DP 仍按旧布局计费，不符合此分发约定。

## 3. seed 布局与索引

对一个现有逻辑 weight PT，位置 `(h,w)` 归属：

```text
r = h mod a
c = w mod b
```

普通 multiplexed 的一个 seed 包含最多 `a*b` 个同一 kernel 位置的逻辑 weight
向量；depthwise/inverse 按第 3.1 节沿 kernel 轴合并。未使用的尾部子网格填零。seed
的向量值不包含卷积边界置零；它只保存对应 kernel/channel 位置的系数。由
`ct_idx`、输入 block `j`、kernel index `k` 推导 seed 序号、`(r,c)` 和尾部有效性，
不创建持久映射表。seed 数量按当前 weight PT 组织规则计算：

```text
seed_count_per_outer_row = ceil(packed_axis_count / (a*b))
```

总数须乘回保留的外轴，不能先将所有参数展平再取整。普通 multiplexed 的
`packed_axis_count` 是旧输入 block 数；depthwise/inverse 是逻辑 kernel 数。
seed 只减少空间重复，不改变通道打包。

### 3.1 depthwise 与 inverse 的 kernel 轴布局

depthwise 和 inverse 不把不同输入 block 合并到同一 seed，而是保留外轴后仅沿
kernel 轴合并。设逻辑 kernel 顺序为 `k=0..K-1`，则：

```text
seed_k = floor(k / (a*b))
subgrid = k mod (a*b)
seed_count = ceil(K / (a*b))
```

depthwise 的外轴是输入/输出 ciphertext，inverse 普通卷积的外轴是
`(out_channel, in_channel)`，inverse depthwise 的外轴是 `out_channel`。inverse
的 `K = kh*kw*prod(output_step)`，kernel 顺序必须复用 inverse 原有枚举顺序；
边界 step 由该枚举的实际 inverse 分解计算，不能套用普通卷积的 kernel shift。
子网格 mask 使用 `shape=block_shape`、`skip=[1,1]` 和一个 block；depthwise
mask 使用其输入 shape、skip 和 `n_block_per_ct`。尾部子网格为零填充。

### 3.2 seed 重复请求与唯一参数实体

生成接口接受 canonical seed index：普通 multiplexed 的
`generate_weight_pt_for_seed_indices(ct_idx, seed_j, k)` 中 `seed_j` 已经是压缩
后的输入 block 索引；depthwise/inverse 的最后一个索引是 `seed_k`。运行时的
逻辑请求先按分组公式映射到该索引和 subgrid，不能把 legacy 索引直接传给生成接口。
不同逻辑请求可共享同一 seed，其身份为：

```text
SeedKey = (layer_id, outer_axes, seed_index, seed_level, a, b)
```

其中 `outer_axes` 是普通卷积的 output/input block 或 depthwise/inverse 的
channel 外轴，`seed_index` 由对应轴的分组规则计算；`(r,c)` 子网格是该 seed
内的逻辑偏移，不再创建独立映射表。对于
同一 `SeedKey`：

- eager 生成只调用一次 seed weight PT 生成函数，并在内存中复用结果；
- Provisioner 只加密一次、manifest 只写一个 seed argument 和一组 elements；
- 加密参数 runtime_lazy 以 canonical 参数节点及其 argument/flat index 加载
  seed，多个逻辑请求共享同一加载节点；
- 如果同一逻辑 seed 在不同 use-site level、参数集或 layer instance 使用，必须
  使用不同 key 和不同密文实例，不能跨 level 复用。

manifest 记录 seed 的 canonical argument、元素顺序、level 和 `param_kind`；
逐层 JSON 保存 `(a,b)`，MegaAG 固化恢复操作及公开 mask 属性。Runner 直接执行
该图，不在运行时重新解释训练配置，也不维护独立的请求到 seed 映射表。

## 4. 生成接口与执行路径

优先在同一个卷积类内增加最小的路由和 helper，而不是复制完整执行类。建议接口
如下，名称可按现有代码风格调整：

```cpp
CkksPlaintextRingt generate_weight_pt_for_seed_indices(
    CkksContext&, int ct_idx, int j, int k) const;
CkksPlaintextRingt generate_mask_pt_for_seed_indices(
    CkksContext&, int ct_idx, int j, int k, SeedMaskKind) const;
```

普通 multiplexed 的 seed index 位于输入 block 轴；depthwise 的 seed index
位于 kernel 轴；inverse 同时保留 output/input channel 外轴，并将
`kernel_shape * output_step` 展平后沿 kernel 轴分组。

这两个接口既服务 eager 离线 PT 生成，也服务 runtime_lazy 的按需生成。现有
`generate_weight_pt_for_indices()` 和 `generate_mask_pt_for_indices()` 保留为旧
路径；seed 路径只在 `use_seed_weight_pack_` 为真时调用新 helper。无需强制新增
公开接口：如果现有生成函数增加一个最小的 `seed` 路由参数能够保持代码更短，
可以复用原函数，但旧路径的输出和函数调用顺序必须保持不变。

对以下函数增加 seed 分支或等价的局部路由：

- `prepare_weight_for_post_skip_rotation()`
- `prepare_weight_for_post_skip_rotation_lazy()`
- `run_core_for_post_skip_rotation()`
- `run_for_post_skip_rotation()`

depthwise/inverse 对应的 `prepare_weight()`、`run_core()` 和 Python
`call_param_ct()` 使用同一 seed 恢复路由。`server_provisioned_runner` 的加密
参数图通过 `load_encrypted_param_ct` 取得 seed CT，再以公开 mask 和旋转恢复
逻辑 weight；不为每个 `call_param_ct*()` 另设 custom compute executor。

其他 `reduct_rot` 路径在本版回退旧逻辑，避免同时改变两套旋转布局。seed 分支
最终必须产生与旧路径相同形状、相同 level、相同 scale 的 weight PT，后面的
`mult_plain_mul`、累加、rescale、bias 和 output select 逻辑不改。

## 5. 一般 kernel 的恢复顺序

对请求的 seed weight，执行顺序固定为：

1. 读取或生成 seed weight CT/PT，其中 seed 只包含未做边界置零的周期系数。
2. 乘第一张公开子网格 mask，提取 `(r,c)`，并 rescale 一次。
3. 按行、列分别使用倍增旋转加法，将子网格复制到完整逻辑空间；每个方向的
   旋转步长使用现有槽布局，方向由 `(r,c)` 的低位决定。
4. 当 kernel shape 为 `1x1` 时，步骤 3 的结果就是旧接口所需 weight PT。
5. 当 kernel 任一维大于 1 时，再乘现有 kernel boundary mask 并 rescale 一次，
   只将 padding/stride 边界置零；结果才进入现有卷积乘法。kernel mask 必须由
   现有 `kernel_masks_` 或同一规则生成，不能在 seed 生成阶段提前置零。

第二张 mask 只负责 kernel 边界，不负责子网格选择；二者不能合并成会破坏周期
复制的单张 mask。对于 stride、skip 和 external upsample，边界 mask 使用当前
`shifted_i/shifted_j`、`zero_inserted_skip` 和物理槽位规则计算，保持与旧
`generate_weight_pt_for_indices()` 相同的有效 kernel 系数。

### 5.1 旋转符号与跨 block 污染

当前 `rotate_cols(x, s)` 的正数表示左旋，即
`rotate_cols(x,s)[t] = x[(t+s) mod slot_count]`。按原始子网格坐标
`r = subgrid // b`、`c = subgrid % b`，从低位到高位执行复制：

```text
行方向第 i 位：delta_h = 2^i * skip[0] * (input_shape[1] * skip[1])
                step_h = -delta_h  if ((r >> i) & 1) == 0 else +delta_h

列方向第 i 位：delta_w = 2^i * skip[1]
                step_w = -delta_w  if ((c >> i) & 1) == 0 else +delta_w

每一步：weight = weight + rotate_cols(weight, step)
```

原始坐标对应位为 0 时向右复制，为 1 时向左复制，确保复制落在同一个空间
tile 内。例如 `a=2,r=0` 的偶数行应通过负旋转复制到其后的奇数行。反向旋转
会跨过行或 channel block 边界，把相邻通道的系数带进来；后续 kernel boundary
mask 无法修复这种系数污染。该规则同样适用于 `1x1`，不能依赖 padding mask
掩盖错误。旋转步长必须保留物理 `skip`，不能直接使用逻辑像素步长。

回归测试应为不同 block、不同 skip 内位置和不同子网格设置不同系数，逐槽比较
恢复结果；全 1 权重不足以暴露跨 block 污染。

## 6. level 与 scale 规范

设旧路径在 use-site 需要目标 weight PT `W_L`，其 level 为 `L`、scale 为
`S_target`。seed weight PT 直接以 `S_target` 编码；seed CT 的初始 level 由
kernel mask 次数决定。公开 mask 使用对应 level 的 q 作为 scale：

| 路径 | seed 初始 level | mask 顺序及 mask scale | 每次 mask 后 | 最终结果 |
| --- | ---: | --- | --- | --- |
| `1x1` | `L+1` | 第一次 mask：`q_{L+1}` | rescale 到 `L` | `W_L`, `S_target` |
| 一般 kernel | `L+2` | 第一次：`q_{L+2}`；第二次 kernel boundary mask：`q_{L+1}` | 分别到 `L+1`、`L` | `W_L`, `S_target` |

inverse 的 `1x1` 若因显式 padding 产生非零边界 step，也使用两次 mask 和
`L+2`；判定依据是实际 inverse kernel 枚举。

在现有 CKKS rescale 约定下，`S_target × q_{L+2}` 经第一次 rescale 回到
`S_target`，第二次同理；`1x1` 只有一次 `S_target × q_{L+1}`。seed PT 不应
预先乘 q 或使用另一套固定 scale。seed CT 加密时保留该 `S_target`，只提高
初始 level；mask 的 q scale 由 `param_.get_q()` 在实际乘法 level 生成。

seed level 不得超过底层 CKKS modulus chain；主图最高 feature level 不因 seed
路径提高。实现必须在 eager、runtime_lazy、Provisioner 和 Runner 中共享这套
level/scale 计算，并验证每次 `mult + rescale` 后的 level/scale；禁止用新的固定
scale 常量覆盖现有 `param_.get_q(level)` 或 layer 的 `weight_scale`。

实现应提供一个局部的 scale/level 计算函数，或直接复用现有 `ringt_to_mul`、
`encode`、`rescale` 的参数规则；该函数必须同时被 eager PT 生成、runtime_lazy
生成和参数密文生成使用。seed CT 加密时使用上述 seed 初始 level/scale，不能先
按目标 `L` 加密后在 Runner 中补 level。

公开 mask PT 不计入模型参数 bundle，但其序列化/编码 scale 必须记录在算子逻辑
中；mask 运行时操作必须使用现有 eval context 所需的乘法和旋转密钥。

## 7. eager、runtime_lazy 与参数加密

- C++ 明文参数 eager：`prepare_weight_for_post_skip_rotation()` 或对应层的
  `prepare_weight()` 保存 seed PT，`run_core*()` 在 host 解码、恢复并重新编码
  当前使用的逻辑 weight；该路径不执行 PT×PT 的 FHE 操作。
- C++ 明文参数 lazy：`prepare_weight*_lazy()` 初始化布局和 scale，
  `run_core*()` 按需调用 seed 生成与恢复 helper。
- 加密参数模式：Provisioner 使用同一 generate 接口生成 seed PT，再按 seed
  初始 level 加密为 seed CT。manifest 的 parameter kind 标记为 seed weight，
  并记录 layer、argument、初始 level 和元素布局；subgrid stride、kernel shape
  由逐层 JSON 和 MegaAG 属性记录。不把展开后的权重重复写入 bundle。
- Runner 只接收 seed CT 和公开 mask 生成所需的非敏感元数据；mask 不加密、不
  写入模型参数 bundle。mask 的结构信息属于当前卷积公开布局，不作为模型参数
  保密边界的一部分。

加密参数的 Python 图入口为 `call_param_ct()`；multiplexed 的明文输入转入
`call_param_ct_plaintext_input()`，inverse 在 `call_param_ct(input_is_plaintext=True)`
中转动参数 CT 和乘积 CT 来实现相同计算。`runtime_lazy` 通过
`materialize_encrypted_param()` 为参数节点插入 `load_encrypted_param_ct` custom
操作，在 activation 就绪后按需加载 seed CT，同一参数节点只插入一次加载。
`call_custom_compute()` 服务于明文参数的 lazy 编码；加密参数的按需加载由上述
公共 helper 完成，无需给两个 `call_param_ct*()` 接口分别新增 custom compute 入口。

manifest 可以保存 seed 的顺序和文件 offset，但不得保存一份独立的完整
`(seed_id,r,c)` 映射表；该索引由算子按稳定公式重建。旧 manifest 或没有 seed
metadata 时必须回退旧参数读取路径。

修改 `(a,b)` 或 seed 布局实现后，必须重新生成 MegaAG，并重新运行 Provisioner
生成匹配的参数包。参数的 shape、数量或 level 相同不代表 seed 内容兼容，不能
据此复用旧参数包。

## 8. 计数、score 与接口兼容

Python layer 增加与现有 `get_encrypted_param_count*()` 风格一致的 seed 计数：

```text
seed_count = product(outer_axes) * ceil(packed_axis_count / (a*b))
```

bias、poly、dense、scalar 和未启用 seed 的卷积保持原计数。存储计费按 seed
初始 level 计算，不能按恢复后的目标 level 计算；公开 mask、rotate、add 和
额外 rescale 不计入加密参数存储，但必须加入该层的 FHE op count / runtime score。
`(1,1)` 的 count、score 和 manifest 必须与现有实现逐项一致。

两次公开 mask 乘法均计为 CT×PT 的 `mult_plain`。第一次 mask 乘法和 rescale
计在 seed 初始 level，复制旋转与加法计在第一次 rescale 后的 level；一般 kernel
的第二次 mask 乘法和 rescale 也计在该 level。参数及 seed 恢复的计费 level 必须
与插入 drop-level 后的实际 use-site 一致。存储校准覆盖所需的 seed 初始 level；
例如 N16QP1546H192H32 的 feature 最高 level 为 9，一般 kernel 的 seed 可以在
现有 modulus chain 上使用 level 10、11，对应密文序列化大小必须单独实测记录。

本规范不要求把 `(a,b)` 加入 DP 搜索。第一版由第 2.1 节的全局编译配置显式传递，
未指定时使用 `[1,1]`；如果后续增加候选，只能复用同一个算子接口和 op count，
不在 compiler 重写布局公式。

## 9. 验收计划

1. **参数生成**：对 `1x1`、`3x3` 和至少一种其他 kernel，比较旧 weight PT 与
   seed 展开后的 PT；在有效空间位置数值相同，padding 边界与旧路径逐槽一致。
2. **路由回退**：验证 `(1,1)`、非 2 幂、不能整除空间尺寸和超出尺寸时完全走
   旧路径，参数数量、level、scale 和输出 byte-for-byte 不变。
3. **level/scale**：覆盖目标 level 0、1、较高 level；验证 `1x1` 使用 `L+1`、
   一般 kernel 使用 `L+2`，两类路径最终都回到旧目标 `L/S_target`，且不改变
   activation level cost。
4. **eager/lazy/加密参数**：分别验证 eager PT、runtime_lazy 按需生成和
   Provisioner seed CT；manifest seed 数量等于计数接口，mask 不进入 bundle。
5. **FHE 算子计数**：检查新增 mask、rescale、rotate/add 计数；`(1,1)` 与
   旧计数一致，启用 seed 时计数包含实际恢复操作。
6. **完整链路**：使用现有 periodic `server_provisioned_runner` GPU 流程，至少
   验证 ResNet/CIFAR10 的 `1x1` 与 `3x3`，检查解密输出 top-1、scale、expected
   level 和 manifest；不覆盖已有 baseline 产物。
7. **存储收益与成本**：报告 seed payload、完整 bundle、FHE 推理时间、额外
   BTP/level 变化；以 `(1,1)` 为 baseline，确认启用 seed 确实减少参数 payload，
   并记录 mask/旋转成本上升。
8. **全局配置分发**：覆盖省略字段、`[1,1]`、`[2,2]`、其他合法组合及非法值；
   检查支持层的导出字段、DP count/score 和 MegaAG 参数数量一致，不适用层保持旧
   布局；验证 level 10、11 的 seed 存储计费不受 feature 最高 level 限制。
9. **旋转方向**：覆盖非均匀 channel block/skip 系数、多种子网格和 `1x1` kernel，
   逐槽检查复制不跨 block 污染。

成功标准是：旧路径完全兼容；一般 kernel 的 seed 展开与旧 PT 数值一致；seed
加密参数可被 eager/lazy Runner 正确读取；最终 FHE 输出正确；bundle 下降和新增
运行时开销均有实测记录。
