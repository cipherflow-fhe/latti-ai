# Multiplexed Conv2D 权重 pt 数量与参数密文膨胀分析

本文针对 `server_provisioned_runner` 的 ResNet-20 / CIFAR-10，分析
`inference/fhe_layers/multiplexed_conv2d_pack_layer.cpp` 中卷积权重编码出的
`CkksPlaintextRingt` 数量，以及这些 pt 逐个加密后对 Runner bundle 的影响。
分析基于当前源码和仓库内任务配置；不修改卷积实现。

## 1. 结论与适用范围

**当前数量不能称为全局最优，也不能无条件称为接近最优。**
需要区分固定现有计算图的局部优化、调整 activation 打包后的算子优化，
以及只压缩离线参数存储的优化。

- 当前配置的 21 个卷积，包括两个 shortcut 的 1×1 卷积，共生成
  **7,296 个权重 pt**，对应 **270,256 个原始卷积权重**。
- 其中 **180 个 pt 因通道 padding 而结构性全零**，可在构图和导出时共同裁剪；
  剩余 7,116 个是现有旋转、逐槽乘法、累加方案需要的非零权重对角项。
- `runtime_lazy` 改变读取时机，不改变这组权重 pt 的数量，也不会自动减少落盘密文。
- 固定输入槽布局和现有逐项乘加方式时，单纯改编码循环、增加线程或使用 lazy
  encoding 无法带来数量级下降。允许改变输入通道的周期复制及权重槽映射时，
  则有明显的进一步优化空间。
- 对本文配置，将 activation 的空闲通道区域填成有效通道的周期副本，
  同时改权重映射，可构造出 **1,940 个权重 pt** 的方案，比当前少 **73.41%**。
  已用实际权重对全部 21 层做明文槽运算验证；这不是已实现的 CKKS 优化，
  也不是全局最优数量的证明。

本地 `runs/cifar10/task/server/` 目前是普通参数模式，没有实际 Runner 参数
manifest；因此上述数字是按同一网络配置及当前 encrypted-parameter 构图规则
推导的权重实例数，不是对用户所述约 10 GB bundle 的实测归因。

## 2. 当前编码数量的精确公式

记：

| 符号 | 含义 | 对应代码 |
| --- | --- | --- |
| `Ci`, `Co` | 输入、输出通道数 | `n_in_channel_`, `n_out_channel_` |
| `H`, `W` | 输入 feature 空间尺寸 | `input_shape_` |
| `S = sh × sw` | 子槽交错数量 | `prod(skip_)` |
| `C` | 每个输入密文可容纳的通道数 | `n_channel_per_ct` |
| `B = ceil(C / S)` | 每个密文的空间 block 数 | `n_block_per_ct` |
| `I = ceil(Ci / C)` | 输入密文数 | `n_packed_in_channel` |
| `O = ceil(Co / B)` | 权重的输出组数 | `n_weight_pt` |
| `K = kh × kw` | 卷积核位置数 | `prod(kernel_shape_)` |
| `N`, `T = N / 2` | 多项式阶数、编码向量槽数 | `get_n()`, `vector<double>(N / 2)` |

`prepare_weight_for_post_skip_rotation()` 中三个循环给出：

```text
weight_pt[O][I × B][K]
Pcurrent = ceil(Co / B) × ceil(Ci / C) × B × K
```

`n_weight_pt` 只是第一维长度，不是整个卷积层的 pt 总数。
输出 feature 密文数 `n_packed_out_channel` 也不能直接替代 `O`：
现有算法先在每个空间 block 计算一个输出通道，再经过 skip 求和和 mask/repack
将结果装入最终输出密文。

`prepare_weight_for_post_skip_rotation_lazy()` 只缓存生成所需的信息。
`run_core_for_post_skip_rotation()` 的 lazy 分支仍遍历相同的三维索引。
加密参数模式下，Python 的 `make_param_ct_nodes()` 采用相同的维度，
Provisioner 再对 signature 中每个元素调用
`generate_weight_pt_for_indices()`、`ringt_to_pt(..., level)` 和
`encrypt_asymmetric()`，因此默认是一个权重 pt 对应一个参数密文。

## 3. 为什么一个小模型会展开出很多大对象

每个权重 pt 的槽向量表示一次「旋转后的输入 × 卷积权重」操作，
不是紧凑存储一批原始权重。其主要开销来自空间位置重复、通道空洞、
输出组拆分，以及参数密文化后的多项式与模数链存储。

### 3.1 一个 pt 实际装了什么

令 `A = H × W × S`，`j = g × B + b`。对槽位置：

```text
ell = t × A + h × (W × sw) + w
r   = (h mod sh) × sw + (w mod sw)
cin = g × C + ((b + t) × S + r) mod C
cout = o × B + t
```

`generate_weight_pt_for_indices(ctx, o, j, k)` 在此槽写入
`weight[cout, cin, ki, kj]`；通道越界或 kernel 平移后空间位置越界则写 0。
同一个卷积权重随空间位置重复，边界位置还包含 zero-padding 的 mask。
`run_core_for_post_skip_rotation()` 将输入先旋转 `b × A`，
再按 kernel 位置旋转，与对应权重逐槽相乘后累加。

因此，pt 是卷积线性变换的一组带空间重复的「对角项」，
不是将原始模型参数连续塞进 CKKS slots 的存储块。
在本文配置中 `N = 65,536`、`T = 32,768`，每层均有 `B = 32`、`I = 1`，
但实际有效输入通道远少于 `C`：

- 初始层：3 个有效通道，容量 32；
- stage 1：16 个有效通道，容量 32；
- stage 2：32 个有效通道，容量 128；
- stage 3：64 个有效通道，容量 512。

最后一种情况下，虽然只有 4 个有效的 16 通道 block，
当前仍枚举全部 32 个 block shift。对于每个输出 block，
不同 shift 才能将这 4 个有效输入 block 送到它所在的位置，
所以不能只因「输入密文只有一个」便将 shift 数量降成 4。

### 3.2 从参数量到存储量的两次放大

当前 270,256 个卷积权重共展开为 `7,296 × 32,768` 个实数槽位置，
平均每个 pt 对应约 **37.04 个不同的原始权重**，
槽位置数与原始权重数之比约为 **884.63 倍**。
该数字包括空间重复及零槽，**不是文件字节膨胀率**。

参数加密后，每个槽向量还会变成两个多项式及多个 RNS 模数分量。
即使编码前只有少数非零槽，正常加密后的多项式也不能只保存这些非零槽；
通用压缩工具同样不能恢复这部分结构性冗余。
另外，`eval_context.bin` 包含计算所需公钥材料，
其大小与 `encrypted_model_parameters/` 应分开统计。

## 4. ResNet-20 / CIFAR-10 的逐组统计

统计来源是仓库内的
[`runs/cifar10/task/server/nn_layers_ct_0.json`](../../runs/cifar10/task/server/nn_layers_ct_0.json)。
[`examples/test_cifar10/task/server/nn_layers_ct_0.json`](../../examples/test_cifar10/task/server/nn_layers_ct_0.json)
具有相同的卷积形状及 pt 总数，但 level 排程不同。
两个 stage 的 shortcut 各包含一个 1×1 卷积，因此这里是 21 层卷积，
不是只计网络名称中的 20 层。所有层 `groups = 1`、`external_upsample_factor = (1,1)`。

表中 pt、全零 pt 和候选方案 pt 均为**单层**数量；最后一列方案见第 6 节。

| 层组 | 层数 | `Ci → Co` | `H × W` | skip | stride | `C` | `B` | `K` | 当前 pt/层 | 全零 pt/层 | 周期复制 pt/层 |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初始 conv | 1 | 3 → 16 | 32 × 32 | 1 × 1 | 1 × 1 | 32 | 32 | 9 | 288 | 126 | 36 |
| stage 1 全部 conv | 6 | 16 → 16 | 32 × 32 | 1 × 1 | 1 × 1 | 32 | 32 | 9 | 288 | 9 | 144 |
| stage 2 首个主路 conv | 1 | 16 → 32 | 32 × 32 | 1 × 1 | 2 × 2 | 32 | 32 | 9 | 288 | 0 | 144 |
| stage 2 shortcut | 1 | 16 → 32 | 32 × 32 | 1 × 1 | 2 × 2 | 32 | 32 | 1 | 32 | 0 | 16 |
| stage 2 其余 conv | 5 | 32 → 32 | 16 × 16 | 2 × 2 | 1 × 1 | 128 | 32 | 9 | 288 | 0 | 72 |
| stage 3 首个主路 conv | 1 | 32 → 64 | 16 × 16 | 2 × 2 | 2 × 2 | 128 | 32 | 9 | 576 | 0 | 144 |
| stage 3 shortcut | 1 | 32 → 64 | 16 × 16 | 2 × 2 | 2 × 2 | 128 | 32 | 1 | 64 | 0 | 16 |
| stage 3 其余 conv | 5 | 64 → 64 | 8 × 8 | 4 × 4 | 1 × 1 | 512 | 32 | 9 | 576 | 0 | 72 |
| **合计** | **21** | | | | | | | | **7,296** | **180** | **1,940** |

### 4.1 哪些 pt 可以直接删掉

忽略空间边界，给定 `(o,g,b)` 的权重向量非零的必要条件是至少存在：

```text
0 <= t < min(B, Co - o × B),  0 <= r < S
g × C + (((b + t) × S + r) mod C) < Ci
```

本文各 kernel 位置均存在有效的内部空间点，所以对实际非零权重，
此条件也足以判断其是否为结构性全零 pt。

- 初始层只有 18 个 block shift 能将 3 个输入通道对齐到 16 个有效输出 block，
  另 14 个 shift 全零，故可删 `14 × 9 = 126` 个 pt。
- stage 1 每层只有 `b = 16` 全零，故各删 9 个，6 层共删 54 个。
- 后续层每个 shift 至少服务一个有效输出 block，虽然单个 pt 内仍有很多零槽，
  但整个 pt 不为零。

两个任务的 H5 中全部 270,256 个卷积权重均非精确零值，
不能再依靠普通权重稀疏性删除整个对角项。
全零裁剪将 **7,296 → 7,116**，只减少 **2.47%** 的权重密文数量。
由于全零项集中在较高 level 的前层，字节节省比例可以比数量比例更高，
但仍不足以解释或解决数量级膨胀。

删除必须同步作用于 Python 图中的乘法项、signature 元素、
Provisioner 的索引映射和运行时读取；
只在 C++ 编码时跳过而保留原索引/shape，会导致缺参或错位。
这些是由公开通道形状决定的结构零，裁剪不依赖私有权重数值。

## 5. 能否称为最优：必须说明约束

### 5.1 固定当前逐项计算图：局部上已接近

固定输入为当前零填充布局、输出中间 block 布局不变，
并且每个 `(o,g,b,k)` 使用一个独立权重向量执行现有逐槽乘加，
则每个非零对角项都需要被表示。对一般稠密权重，
不同输出通道、输入 shift、kernel 位置的系数不能随意共用。
上述 7,116 项就是**这个固定分解**去掉结构零后的数量。
因此，如果只允许在现有编码循环内裁零，当前数量确实很接近此局部结果。

这不构成任意同态卷积实现的乘法次数下界，也不证明每项必须独立落盘：
例如可以从更紧凑的参数密文在运行时通过旋转、mask、加法重建这些项，
用额外计算换取离线存储。允许改变 activation 布局，结论也会改变。

### 5.2 仅看存储容量：差距很大，但容量下界不是算子下界

假设每个实数权重占一个实数槽，完全忽略 level/scale 差异及运行时展开成本，
紧凑存储所有卷积权重的容量下界为：

```text
ceil(270,256 / 32,768) = 9 个槽向量
```

若要求每层单独存储，则 `sum_layer ceil(Ci × Co × K / T) = 26` 个。
这些仅是特定表示假设下的容量下界，
不是「用 9 或 26 次同态乘法就能完成现有卷积」的结论，
也不包含 CKKS 复数槽利用、量化等其他表示的可能性。
参数打包成一个密文后，并不能免费抽取、广播成当前所需的数百个对角项。

### 5.3 对允许重设计的算子：当前明显不是近似最优

第 6 节给出保持空间卷积及输出 repack 思路、
但改变 activation 通道周期的具体构造，权重 pt 可减少约 3.76 倍。
因此不能将当前的 7,296 称作全局最优或全局接近最优。
不过本文也不宣称新数量 1,940 最优；改变输出通道分配、
复数槽用法或线性变换分解仍可能带来进一步变化。

## 6. 可行方向：有效通道周期复制

### 6.1 核心构造

本任务所有卷积均满足 `Ci <= C`，只需一个输入密文。
令 `Cp = max(S, next_power_of_two(Ci))`、`D = Cp / S`，
这里 `Cp` 整除 `C`、`D` 整除 `B`。
把前 `Cp` 个通道组成一个周期：真实通道保留，`Ci..Cp-1` 补零；
再将这段周期复制至整个密文的通道容量。保持每个通道的空间位置和 skip 不变。

```text
当前： [有效通道, 零, 零, ...]
改后： [有效通道及周期内 padding, 同样一份周期, 同样一份周期, ...]

例如 stage 3：Ci=64, S=16, C=512
      4 个有效 block 复制 8 份，填满 32 个物理 block
```

然后只生成 `b = 0..D-1` 的旋转项，将权重映射改为：

```text
cin  = ((b + t) × S + r) mod Cp
cout = o × B + t
```

空间 boundary mask、输出通道检查和 kernel 位置保持原语义，
`cin >= Ci` 的周期内 padding 仍取零。
因为输入在 block 方向具有 `D × A` 的周期，
旋转 `b × A` 后到达输出 block `t` 的正是上式所指定的输入通道。
对任意固定的 `t`，`b = 0..D-1` 与 `r = 0..S-1` 的组合
恰好遍历一次 `0..Cp-1`，因此通道求和仍覆盖全部真实输入且不重复计数。
现有的 skip 求和和输出 repack 可继续用于恢复原输出布局。

在这些条件下：

```text
Pperiodic = ceil(Co / B) × D × K
```

本任务汇总为 **1,940**。例如 64→64、3×3 层
从 `2 × 32 × 9 = 576` 降为 `2 × 4 × 9 = 72`。
初始层 3 通道先 pad 到 `Cp = 4`，所以是 36 个，
而不是假设任意长度周期都能无缝填满槽环后得到 27 个。

### 6.2 为什么不能只改 `n_block_per_ct`

`B = 32` 还决定一次覆盖多少个输出通道；
直接令 `C = Ci` 或令 `B = D` 会同时缩小输出并行度，
不能得到上式的收益。这里保持物理 block 数及 `O` 不变，
只减少独立的输入 block shift 数，并改为按 `Cp` 取模。
当前 multiplexed 输入编码对无效通道填零，
所以仅缩短 `j` 循环而不复制输入，会遗漏输出通道的贡献。

对当前 `MultiplexedConv2DPackedLayer`，这项优化可以由层根据已有 shape 自动选择，
不必增加 task 或 layer metadata。设 `D = Cp / S`，仅在 `Ci <= C`、
`n_packed_in_channel == 1`、`external_upsample_factor == (1,1)` 且 `D < B` 时启用，
并在内部检查 `C`、`Cp` 和 `B / D` 的整除关系；其他情况沿用旧路径。

当 `B / D` 是 2 的幂时，密文输入的 block 旋转数从 `B - 1` 变为
`D - 1 + log2(B / D)`，另加 `log2(B / D)` 次加法。明文首层若在输入编码时直接
写入周期副本，则只需 `D - 1` 个 shift。因此该条件下周期复制通常减少旋转，新增
的加法不会改变 level。

实际实现至少涉及：

1. 输入编码器和中间 activation 的周期扩展，明确 `Cp` 及周期内 padding；
2. C++ 权重槽映射及生成索引，以及 Python 参数节点 shape 和旋转/乘法循环；
3. Provisioner 的三维索引解释、signature、manifest 及 runtime-lazy 读取；
4. 分支相加、非线性、bias、bootstrap、stride/repack 前后布局不变量。

一种保守做法是在每个卷积前选出有效的基础周期，再通过
`log2(B / D)` 次 rotate-and-add 倍增复制。
从零填充布局出发，各副本位置不重叠，不会改变有效通道数值。
但上游多项式激活的常数项可能使原无效槽不再为零，
不能未经 mask 便直接复制、相加。
点运算只有在 bias/系数也遵守相同复制规则时才能自然保持周期；
残差两路也必须使用兼容布局。

初始输入是明文，可在编码时直接复制；
当前 `call_param_ct_plaintext_input()` 通过旋转权重及乘积
实现等价的输入旋转，也必须同步调整其 shift 循环。
仅修改 C++ 普通 CT×PT 路径不会改变 Runner 的 Python 构图数量。

### 6.3 已完成的验证与未覆盖项

本分析以 `runs/cifar10/task/server/model_parameters.h5` 的实际权重，
固定随机种子 `20261027` 生成每层独立的 float64 输入，
对全部 21 层进行了以下对照：

- 按 C++ 公式建立现有权重槽向量，模拟环形旋转、逐槽乘加、skip 求和、mask 和 repack；
- 建立周期复制输入，使用新的通道映射及较短 shift 循环，执行同样的输出流程；
- 两者均与直接的零填充、stride 1/2 普通卷积比较，
  比较整个输出槽向量，包含应为空的槽；bias 不参与对照，比较的是卷积线性部分。

现有和候选方案的最大绝对误差均为约 **3.11×10^-15**，
以 `atol = rtol = 1e-10` 检查均通过。
覆盖了 skip 1/2/4、1×1/3×3 kernel、两次 stride-2 降采样及各输出组。
这是明文槽代数验证，未执行 CKKS 加密、噪声测量、GPU 测速或整个残差网络推理。

周期扩展可能引入额外 mask、旋转、参数及 rescale。
在当前 structural mask 加密策略下，还可能引入额外 CT×CT；
若改成公开常量，需要明确调整部署约定。
所以 **73.41% 仅指卷积权重 pt 数量下降**，
不能直接解释为 bundle、显存、耗时或网络时延也下降相同比例。

## 7. 参数密文的字节量：如何与约 10 GB 对照

### 7.1 当前落盘格式不是每个 RNS 系数固定 8 字节

`write_encrypted_argument()` 调用 `CkksCiphertext::serialize(param)`，
经 Go SDK 进入 `ckks.Ciphertext.ToBytes()`，
再进入 `rlwe.CiphertextToBytes(..., 0, 0, ...)`。
此格式按各 `q[j]` 的实际 bit length 打包系数，未丢弃精度位。
对当前正常加密得到的 degree-1 密文、`N` 为 8 的倍数时，
单个密文 payload 的大小是：

```text
BytesCT(L) = (L + 21) + (2 × N / 8) × sum(bit_length(q[j]), j=0..L)
```

`L+21` 来自 scale、N、poly 数、level、模数位宽及每个 poly 的标志。
此外每个参数文件有 8 字节 count，各元素有 `sizeof(size_t)` 字节 length
（通常是 8），manifest 另计。
不能直接把内存中 `2 × N × (L+1) × 8` 字节的 RNS 数组大小当作落盘大小。

本地配置下单个权重密文约为：

| use-site level `L` | 权重密文个数 | 单个 payload / MiB |
| ---: | ---: | ---: |
| 1 | 864 | 1.594 |
| 2 | 3,520 | 2.219 |
| 5 | 1,472 | 4.109 |
| 6 | 576 | 4.750 |
| 7 | 576 | 5.375 |
| 9 | 288 | 6.625 |

例如配置中的 `q[0]` 实际 bit length 为 61，
不能直接用参数名或预期的 60/40 bit 代替实际值做精确估算。

### 7.2 本地配置预测与实际 bundle 必须分开

若将本地 `runs/cifar10` 的 feature level 原样用于加密参数构图，
当前卷积权重 payload 预测总量为 **24,092,260,736 字节，约 22.44 GiB**。
只去结构零后约 **21.47 GiB**；
周期复制方案若暂假设每层参数 level 不变，则该项约 **5.81 GiB**。
后两项均是同一 level 排程下的算术估算，未计新方案新增操作导致的 level 改变。

以上均不含 bias、structural masks、激活多项式系数、FC、公钥文件及其他参数，
更不是用户所述约 10 GB 的实测拆分。
本地没有对应 Runner bundle，且当前环境生成 MegaAG 时因缺少 `networkx` 停止，
没有重新生成或导出加密产物；数量分析直接依据源码维度及现有配置完成。

这组预测与约 10 GB 并不一致，说明不能把当前本地配置的体积结论
直接套到已有 bundle。应先核对实际 bundle 的 `N`、`q`、
每层 use-site level、signature/manifest 元素数、
统计范围是否包含 `eval_context.bin`，以及使用 GB 还是 GiB。
如果同一个逻辑参数在多个 level 下落盘，还要按实际实例数计算。

`runtime_lazy` 仅按元素延迟读取/反序列化，
已加载项仍可留在 store cache；它既不压缩文件，也不保证缓存自动释放。

## 8. 优化优先级与不成立的捷径

| 方向 | 对当前权重 pt 数量的影响 | 代价及适用性 |
| --- | --- | --- |
| 结构零裁剪 | 7,296 → 7,116 | 低风险小收益；同步修改图和参数索引 |
| 有效通道周期复制 | 本配置可构造为 1,940 | 需要 activation 布局、权重映射及部署链路共同调整 |
| 紧凑参数离线存储，在线展开 | 可减少落盘 CT，在线对角项仍需生成 | 需要抽取/广播/重排；额外操作、mask、深度和缓存策略待评估 |
| seed 压缩参数密文 | 不减少 pt，可能减少每个 CT 字节 | 库有 symmetric compressed API；当前 Provisioner 使用 asymmetric 普通 CT，需改加密与读写协议 |
| lazy、线程数、GPU、NTT 缓存 | 不减少 pt | 影响生成/加载/运行成本，不能消除展开项 |

建议先用真实 manifest 建立「每层数量 × level × 字节」账目，
再裁剪结构零并以一层 64→64、skip=4 的卷积试做周期复制。
该层有明确的 576→72 潜力，最适合测量 mask/复制成本与节省的乘法成本。
验证单层 CKKS 后再处理残差、激活和 bootstrap 的全网络周期保持。
如果主要目标是搬运 bundle 大小，也可并行评估 seed 压缩及紧凑参数存储，
不要将离线传输优化与在线乘法项优化混为一谈。

以下改动不能直接推出更少的权重 pt：

- 把 3×3 的 9 个位置简单合并成一个权重向量：
  各位置乘的是不同旋转后的输入，任意权重下不能用一次逐槽乘法替代。
- 只启用 BSGS 或 `reduct_rot`：
  重排旋转不等于压缩独立权重系数，且当前部署分支并未选择 `reduct_rot` 路径。
- 看到输出最终只有一个 CT 就只编码一个权重 pt：
  最终输出的压紧不能替代中间层对输入通道及 kernel 位置的求和。
- 将参数以明文/H5 带到 Runner 后现场重建：
  不符合本模式的模型参数保密目标。

## 9. 数量及体积的复算

以下脚本只用 Python 标准库，在仓库根目录运行。
它按当前源码公式计算每层数量、结构零、周期复制候选数量及 CT payload 字节，
不生成密钥、不导出 bundle，也不代替第 6.3 节的槽运算验证。

```bash
python3 - <<'PY'
import json
from math import ceil, prod
from pathlib import Path

p = Path('runs/cifar10/task/server/nn_layers_ct_0.json')
cfg = json.loads(p.read_text())
totals = [0, 0, 0, 0, 0]
for name, layer in cfg['layer'].items():
    if layer['type'] != 'conv2d':
        continue
    f = cfg['feature'][layer['feature_input'][0]]
    Ci, Co = layer['channel_input'], layer['channel_output']
    C, S, K = f['pack_num'], prod(f['skip']), prod(layer['kernel_shape'])
    assert layer['groups'] == 1 and Ci <= C and C % S == 0
    assert layer.get('upsample_factor_in', [1, 1]) == [1, 1]
    B, I = ceil(C / S), ceil(Ci / C)
    O = ceil(Co / B)
    current = O * I * B * K
    zero = 0
    for o in range(O):
        for g in range(I):
            for b in range(B):
                valid = any(
                    g * C + ((b + t) * S + r) % C < Ci
                    for t in range(min(B, Co - o * B))
                    for r in range(S)
                )
                zero += K * (not valid)
    Cp = max(S, 1 << (Ci - 1).bit_length())
    assert C % Cp == 0 and Cp % S == 0
    periodic = O * (Cp // S) * K
    param = cfg['ckks_parameter'][f['ckks_parameter_id']]
    N, L = param['poly_modulus_degree'], f['level']
    ct_bytes = L + 21 + (2 * N // 8) * sum(
        int(q).bit_length() for q in param['q'][:L + 1]
    )
    values = [current, zero, periodic, Ci * Co * K, current * ct_bytes]
    totals = [a + b for a, b in zip(totals, values)]
    print(name, 'current=', current, 'zero=', zero,
          'periodic=', periodic, 'level=', L)
print('current, zero, periodic, raw_weights, payload_bytes =', totals)
assert totals == [7296, 180, 1940, 270256, 24092260736]
PY
```

上述全零判断只针对通道形状；应用到极小空间尺寸等其他任务时，
还应加入 kernel 边界的全零判断，并从 H5 判断是否有数值上全零的权重向量。
若换任务或 level 排程，应去掉最后的固定数字断言后重新计算。

## 10. 源码索引

- [C++ 卷积实现](../../inference/fhe_layers/multiplexed_conv2d_pack_layer.cpp)：
  构造函数、`prepare_weight_for_post_skip_rotation()`、
  `generate_weight_pt_for_indices()`、`run_core_for_post_skip_rotation()`。
- [Python 卷积构图](../../inference/model_generator/layers/multiplexed_conv2d_pack_layer.py)：
  `make_param_ct_nodes()`、`call_param_ct()`、`call_param_ct_plaintext_input()`。
- [部署路径选择](../../inference/model_generator/deploy_cmds.py)：
  `parameter_mode = encrypted_offline` 的 multiplexed conv 分支。
- [Provisioner](../../inference/interface/inference_provisioner.cpp)：
  参数索引展开、`encrypt_parameter_argument()`、`write_encrypted_argument()`。
- [Runner 输入编码](../../inference/interface/inference_runner.cpp) 和
  [Feature2D 编码](../../inference/data_structs/feature2d.cpp)：multiplexed 无效通道填零。
- [C++ 序列化入口](../../inference/lattisense/fhe_ops_lib/fhe_lib_v2.cpp)、
  [Go SDK](../../inference/lattisense/fhe_ops_lib/lattigo/go_sdk/main.go)、
  [CKKS ToBytes](../../inference/lattisense/fhe_ops_lib/lattigo/ckks/ciphertext.go)、
  [RLWE 位打包](../../inference/lattisense/fhe_ops_lib/lattigo/rlwe/bit_stream.go)：文件体积公式。
- [现有部署文档](../zh/server_provisioned_runner.md)：角色、安全边界、use-site level 和 runtime-lazy 约定。

分析日期：2026-10-27。本次仅新增分析文档，未修改算子、构图或部署协议。
