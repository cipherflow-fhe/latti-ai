# PCM 打包格式的 BERT 完整支持设计规范

状态：已补齐实现；本机 PCM BERT 两次完整 GPU lazy 实测均满足原始 ONNX 最大绝对误差 ≤ 0.2。原生 `--verify` 的独立 0.1 检查两次均未通过，不能宣称全部验证通过；详细证据与限制见第 10 节。

目标：以最小修改补齐 PCM 的 BERT 非线性 layer 和编译、导出、调度接口，使
`runs/BERT/model/trained_poly.onnx` 与 `runs/BERT/input.csv` 能在 **PCM、输入加密、GPU lazy 执行**路径完成推理，全部原始 ONNX 输出的最大绝对误差 **≤ 0.2**。

本文遵循本次明确调用的 [karpathy-guidelines](/home/linghm/.codex/skills/karpathy-guidelines/SKILL.md) 的最小实现、局部修改、显式假设和可验证验收原则。

## 1. 范围与固定决策

### 1.1 名称和运行模式

- 本文 PCM 指 `FeatureMatEncrypted::par_block_col_major_pack` 对应的布局，现有配置值是 **`mat_pack_style=par_block_col_major`**。不新增 `pcm` 或 `par_col_major_pack` 配置别名。
- PDM 指配置值 `par_diagonal_pack`；本次保留它的已有实现和 BERT 数学展开。
- 验收路径是 `InferenceClient` 加密 CSV 输入 → `InferenceServer::evaluate` → `InferenceProcess::run_task_lazy` → GPU executor → 客户端解密。模型权重在此路径为明文参数，通过 `encode_pt` 按需生成 PT。
- 当前 CLI 的 `parameter_loading_mode=runtime_lazy` 专属于 `server_provisioned_runner + encrypted_offline` 参数加密路径，不能直接用于本次输入加密验收。本文所称 GPU runtime lazy，是上述 `run_task_lazy` 路径；验收必须给出执行证据，不能仅凭一个配置字段判定。
- 本次不实现 PCM 参数加密、seed weight 恢复和参数 bundle 压缩。相关后续工作见 [PCM 参数压缩研究](../research/transformer_pcm_seed_parameter_compression.md)。已有参数加密入口的 PCM 拒绝条件继续有效。

### 1.2 实现边界

复用现有 PCM 的 CPMM、CCMM、transpose、`pcm_add_pt`、gamma、二/四次 PolyActRN 和分阶段 LayerNorm；补齐 BERT 需要的通用多项式、softmax 原语和 GELU 末端乘法。BERT softmax、LayerNorm、GELU、tanh 的数学展开沿用当前 PDM 编译算法和 ONNX 属性。

不以解密中间结果、CPU 明文计算非线性、切换为 PDM、重新训练、替换多项式或减少迭代次数来完成 PCM 验收。不建立新的全局 layer registry、通用 packing 框架或新的部署模式。新增类、文件、JSON 类型、Python 镜像沿用相邻 PCM layer 的组织方式。

“完整支持”的含义是该 BERT 模型从导入、编译、明文权重导出、任务生成、客户端打包到 GPU 密文执行、输出解包的完整链路；不扩大为任意 Transformer 或动态 shape 支持。

## 2. 当前支持情况与真实缺口

下表记录本次补齐前的能力与对应改动，当前实现和验证状态见第 10 节。

| 路径 | 当前代码 | 本次动作 |
| --- | --- | --- |
| PCM 矩阵运算 | `parcpmm`、`parccmm`、`partranspose` | 复用；核对 BERT 权重方向、Q 缩放、FFN K 和分类头尾部 |
| PCM LayerNorm | `pcmstats`、`pcmcenter`、`pcminit`、`pcmgs`、`pcmaffine` | 复用；完整传递 BERT 属性并验证真实 scale/level |
| PCM PolyActRN | `pcmgamma`；`pcmpoly` 只支持二/四次 | 保留；不能用 `pcmpoly` 冒充 BERT 通用多项式 |
| BERT attention 展开 | `_expand_bert_multi_head_attention` 明确拒绝非 PDM | 增加明确的 PCM 分支，复用现有矩阵 layer |
| BERT softmax | 通用 poly、head-column sum、inverse、乘法等硬编码为 PDM | 增加下面定义的 PCM layer 和类型选择 |
| CustomGELU / CustomTanh | `_make_upper_poly`、GELU 末端类型仍为 PDM | 复用展开，选择 PCM 实现 |
| 编译至 runtime | 多处显式类型表、构造、计分及 `encode_pt` 分发 | 逐处注册真实 PCM 接口 |

因此，“已有 PCM LayerNorm、ViT 已跑通”不足以推断 PCM BERT 已完整支持。主要缺口不仅是 softmax 的一个 C++ 类，还包括展开后的所有类型在全链路中的支持。

## 3. PCM 数据布局契约

### 3.1 唯一布局来源

以 [feature_mat.cpp](../../inference/data_structs/feature_mat.cpp) 中的 pack/unpack 为准。下式用 `u` 表示一个 slot 组的 head lane 数，避免与默认 CKKS scale `S` 混淆：

```text
slots = N / 2
R = shape[0]                       # token 行数
W = shape[1]                       # 全部逻辑列数
Chead = head_shape[1]              # 每个 megablock 中，每个 head 的列数
H = n_heads
d = matmul_block_size
K = ceil(W / (H * Chead))
Hp = next_pow2(H)

if slots >= Hp*d*d:
    u = Hp; chunk_size = Hp*d*d; G = 1
else:
    u = slots // (d*d); chunk_size = slots
    if u == 1: Hp = H
    G = Hp // u

repeat = slots // chunk_size
Br = ceil(R/d); Bc = ceil(Chead/d)
cts_per_mb = Br*Bc*G
ct_idx = mb*cts_per_mb + (bi + Br*bj)*G + g
slot = chunk*chunk_size + (row + d*col)*u + h_local
h = g*u + h_local
global_row = bi*d + row
local_col = bj*d + col
global_col = mb*H*Chead + h*Chead + local_col
```

合法值要求同时满足 `h < H`、`global_row < R`、`local_col < Chead`、`global_col < W`。每个重复 chunk 写入相同值。所有新 PT 的 weight、constant、mask 生成器使用这一坐标关系，非法槽的 PT 值为零。

构造时检查 `d` 为正的二次幂、`slots >= d*d`，以及现有 pack 所需整除关系。不得在 PCM 中使用 PDM 的 diagonal segment/boundary 路由或 PDM 的转置后 shape。

### 3.2 BERT 必须覆盖的形状

- 输入 CSV 为 `44×768`。保留已有 `L_prepad=44` 与 ONNX attention mask 的有效 token 处理，不把 44 个有效 token 擅自扩充成 128 个参与 softmax 的 token。
- hidden：`H=12`、`Chead=64`、`d=64`、`K=1`。例如 `slots=32768` 时 `u=8, G=2`，第二组只有 4 个有效 head；不能只实现 `G=1`。
- attention score：逻辑 shape 为 `[44, 12*44]`，`head_shape=[44,44]`，每个 head 按 key token 列归约。
- FFN 扩张后的 3072 列通常为 `K=4, Chead=64`，不是直接改成 `Chead=256, K=1`。两者 CT 顺序和路由不同。
- 两分类 logits 保留真实列数 2；执行可有公开 padding，metadata 和解包必须恢复真实 shape。不能对分类头一律断言 `W % H == 0`。

`head_shape`、`matmul_block_size`、`K`、真实 `shape` 必须从编译 JSON 一直传到 Python generator、C++ layer、客户端和输出签名。对齐不能仅依据总列数推断，更不能借用 PDM head_shape 归一化函数。二元算子的两个输入必须拥有相同布局、CT 数及对应 slot 语义。

## 4. 最少新增的 layer

下表中的类型和类名作为本次实现约定；现有类型不重命名。softmax 的几个类放在同一对 C++ 文件及同一 Python 文件中，与 PDM 组织方式一致。

| JSON type | C++ / Python 类名 | 有效槽数学语义 | level cost |
| --- | --- | --- | ---: |
| `pcm_add_pt`，已有 | `ParBlockColMajorAddPt` | `x + B`；softmax 中 B 为填充有效位置的常数矩阵 | 0 |
| `pcmgamma`，已有 | `ParBlockColMajorPolyActRNGamma` | `gamma*x`，含标量 gamma | 1 |
| `pcmgeneralpoly`，新增 | `ParBlockColMajorPoly` | `sum(c[k,j]*x^k)`，`1 <= order < 64` | `MatPolyBase.compute_stockmeyer_level_cost(order)` |
| `pcmmulsquare`，新增 | `ParBlockColMajorMultipleSquare` | 单次 `x*x`，含有效槽 mask 和 scale 校正 | 2 |
| `pcmheadcolsum`，新增 | `ParBlockColMajorHeadColSum` | 每个 query/head 对有效 key 列求和并广播 | **2** |
| `pcminvinit`，新增 | `ParBlockColMajorInverseInit` | `2-b` | 0 |
| `pcminviter`，新增 | `ParBlockColMajorInverseIter` | `2*a-b*a*a`，输入顺序 `(a,b)` | 2 |
| `pcmctmul`，新增 | `ParBlockColMajorCtMul` | 逐元素 `a*b`，含有效槽 mask 和 scale 校正 | 2 |
| `pcmpolymultct`，新增 | `ParBlockColMajorPolyMultCt` | `x*(0.5+t)`，输入顺序 `(t,x)` | 2 |

新增文件建议为 `par_block_col_major_softmax.{h,cpp}`、`par_block_col_major_poly.{h,cpp}`、`par_block_col_major_poly_mult_ct.{h,cpp}` 及 `inference/model_generator/layers/` 下对应 `.py` 文件。继承方式、命名空间、并行循环、`run_plaintext`、`prepare_weight` / `prepare_weight_lazy`、`generate_*_pt` 接口参照相邻类。

注意：PDM 的 `pdmctmul` 虽由名为 `ParUpperDiagonalGELU` 的类实现，语义只是 `a*b`。它与 `pdmupperpolymultct` 的 `x*(0.5+t)` 不同，不能将两者合并为同一个错误公式。

### 4.1 标量加法复用

BERT softmax 的中点平移和 denominator guard 使用已有 `pcm_add_pt`：导出与输入逻辑 shape 相同、有效位置填 `value` 的 B，按已有接口生成 PT。当前 `_export_add_pt` 只读取 ONNX initializer，因此需增加公开 `value` 的常数矩阵生成分支，并由展开/序列化提供该层独立的 `weight_path` 和 `value`；原有 initializer 分支保持原行为。无需向原始 ONNX 塞入伪权重。

首版这些 attention 输入为 `K=1`，直接复用现有 AddPt 布局；不为标量另造一套 runtime 类。若最终图实际产生 `K>1` 的该类型，才针对该 consumer 补充现有 AddPt 的 megablock 索引，不能以 `W/H` 重解释原布局。

### 4.2 通用多项式

`ParBlockColMajorPoly` 继承已有 `MatPolyBase`，复用其幂次分解、Stockmeyer level cost、power scale 和 coefficient scale 计算。首版只接通 BERT 当前使用的 Stockmeyer 路径；不增加 Horner 策略开关。

`MatPolyBase` 当前不包含完整的密文求值执行器；PCM 按 `ParUpperDiagonalPoly::run_core_stockmeyer` 的同一 DAG 实现本地执行部分，替换系数向量生成和 CT 索引。此次不为减少少量重复而搬迁整个 PDM 执行器或重构全部多项式类。Python mirror 同样复用已有 `mat_poly_base.py` 的计划计算。

系数导出约定为 `(order+1, W)`，即 `c[k,j]` 对应 PCM 逻辑第 j 列；BERT 标量系数沿逻辑列复制。读取 PT 时通过上节的 `global_col` 取系数，覆盖所有 K、head 组、重复 chunk，padding 为零。不能照抄 PDM exporter 使用转置后 `shape[0]` 的轴。

保留多项式的常数项、系数顺序和实际 order；低阶、奇数阶、零系数项均沿用既有计划规则。输出恢复 `(L-cost,S)`。常数项在其实际相加节点的 scale 编码，不能一概使用 S。

### 4.3 HeadColSum 的归约与 padding

这里的归约是每个 query 行、每个 head 独立求 key 列和，不是 LayerNorm 的跨 hidden/head 归约。不得把不同 `g`、`mb` 或 `bi` 相加。

首版固定采用 **输入 mask → 列归约 → 输出 mask**，level cost 为 2：

1. 对每个输入 CT 在 level L 乘有效槽 mask，PT scale 为 `q_L`，rescale 后为 `(L-1,S)`。这一步消除 padded key 列中的非零值，避免把它们加进合法 denominator。
2. 对固定 `(mb,bi,g,bj)`，用步长 `d*u*2^k`，`k=0..log2(d)-1` 做 rotate-add，得到块内所有 d 列的和。重复 chunk 保证列循环归约的周期；不得跨 head lane 归约。
3. 对固定 `(mb,bi,g)` 将所有 `bj` 的局部和相加，再广播给该组每个输出 `bj`。不跨 token block 行或 head 组求和。
4. 每个输出在 level `L-1` 乘对应有效槽 mask，PT scale 为 `q_(L-1)`，rescale 后为 `(L-2,S)`。

输入和输出的 mask 都必须包含真实 token/key 边界、head 尾部及 megablock 边界。只有归约后的 mask 无法撤销已经混入 denominator 的 padded key 值；bootstrap 或常数项也可能使 padding 非零，不能无条件假定输入 padding 恒为精确零。

PDM 现有 HeadColSum 是 cost 1；本次 PCM 采用 cost 2 的原因是以少量 mask 实现明确的 padding 契约，避免新增跨 producer 的隐式约束。编译器展开中的临时 cost、最终 `set_level_costs`、计分器、Python DAG 和 C++ 必须全部使用 PCM 的 cost 2；让现有调度器据此安排 bootstrap，不能为少插一个 bootstrap 虚报 cost 1。mask 的 scale 公式仍直接复用已有 PDM/PCM 单层 mask 规则。

### 4.4 LayerNorm 复用与语义隔离

继续使用 PCM 现有五阶段 LayerNorm。复用 `_bert_layernorm_runtime_attrs` 对实际 ONNX `epsilon`、`min_var`、`max_var`、`wbuffer`、`input_scale`、`c0/c1/c2`、迭代数等属性的解析和传递；不能套用 ViT 默认迭代数或重算另一套近似。

其最终 level cost 为 Stats 4、Centered 2、MinimaxInit 2、Goldschmidt 3、Affine 2；以真实实现和 `set_level_costs` 为准，不能依赖展开时 Centered 的旧临时 cost 1。Stats 跨 hidden/head 的求和与 softmax HeadColSum 的分 head 求和分别保留。

Stats 的结果使用现有紧凑统计量表示：逻辑 shape/head_shape 为 `[R,1]`，每个 token block 只有一个 CT，统计量已沿后续需要的位置广播。它不是普通 hidden 矩阵的 `G` 组 CT。该 shape 必须通过 `pcminit/pcmgs`、drop、bootstrap 前后的 gamma 传递；统计量 gamma 对全槽乘同一标量，不能按 hidden head mask 截断广播。普通 hidden/attention 的 gamma 则使用显式 `head_shape` 和原有有效槽掩码。Python mirror、CT 计数及 C++ 必须采用同一约定。

## 5. Scale 与 level 规范

### 5.1 强制约束与反推方法

记默认 scale 为 `S=param.get_default_scale()`，当前 level L 的实际 rescale 模数为 `q_L=double(param.get_q(L))`。不以名义 bit 数或 S 近似 q。

- 所有 layer 的输入和输出 CT scale 均为 S；layer 内部可以有其他 scale。
- CT/CT、CT/PT 的 add/sub 必须在同一 level、同一 scale 上进行；Ringt PT 转为运算表示时使用实际操作节点的 level。
- mult 的两个操作数必须处于相同 level，scale 可以不同。CT×CT 后按现有规范 relinearize。
- `mult + rescale` 从 level L 得到 level `L-1`，真实 scale 为 **`(S1 / q_L) * S2`**。
- `rescale(ct, S)` 的参数不能强制真实结果 scale 为 S。`drop_level` 不负责校正 scale；不得通过直接改 scale metadata 伪造匹配。

因此对于输入 scale A、目标输出 scale T 的 CT×PT 分支，PT scale 必须反推为：

```text
P = (T / A) * q_L
result_scale = (A / q_L) * P = T
```

先从 layer 输出 `(L-cost,S)` 逆推每个内部合流节点的目标 scale，再反推 coefficient、weight、mask、one、half 等 PT 的 scale。PT 的**数值**和 PT 的**编码 scale**是独立概念；值为 1 的 mask 并不表示 PT scale 为 1。

所有 scale 计算必须先除后乘，例如 `(S/q_L)*S`、`(T/A)*q_L`，不用 `S*S/q_L` 或 `T*q_L/A`。涉及多级时逐级作比值后乘，避免先累乘大数造成溢出。检查 scale 为有限正数，拒绝负 target level 或需要升 level 的计划。

### 5.2 可直接沿用的 PT scale 表

下表假设 layer 输入均为 `(L,S)`。`q1=q_(L-1)`；mask/constant 的值只填合法 PCM 槽。公式沿用 PDM 算术，改变的只是 PT slot 向量。

| 操作或 PT | PT 编码 scale | 对应结果 |
| --- | --- | --- |
| `x+B`、`2-b` 的常数 PT | S | 同 level、同 scale 相加，cost 0 |
| gamma、单次有效槽 mask | `q_L` | CT×PT + rescale → `(L-1,S)` |
| square / CT×CT 后的修正 mask | `(q_L/S)*q1` | 先得 `T=(S/q_L)*S`；在 L-1 乘 mask 再 rescale → `(L-2,S)` |
| `inviter` 的 `one_pt`，值为 1 | `(q_L/S)*(q_L/S)*q1` | `a*one_pt` + rescale → `(L-1,U)`，`U=(q_L/S)*q1` |
| `inviter` 的 `two_pt`，值为 2 | `q_L` | `2*a` + rescale → `(L-1,S)`，再 drop 到 L-2 |
| `polymultct` 的 `one_pt`，值为 1 | `(q1/S)*q_L` | `x*one_pt` + rescale → `(L-1,q1)` |
| `polymultct` 的 `half_pt`，值为 0.5 | S | 在 L-1 与 scale S 的 t 相加 |
| HeadColSum 输入 / 输出 mask | 分别 `q_L` / `q1` | 分别保持 S，最终 L-2 |
| 通用 poly 的 coefficient PT | 采用 `MatPolyBase` 缓存的分支计划 | 常数项在实际 add scale 编码，其余反推至最终 S |

具体推导：

```text
square / ct-mul:
    t = rescale(a*b)                 # L-1, T=(S/q_L)*S
    y = rescale(t*mask)              # L-2, (T/q1)*((q_L/S)*q1)=S

inviter(a,b) = 2*a-b*a*a:
    ba = rescale(b*a)                # L-1, T=(S/q_L)*S
    one_a = rescale(a*one_pt)        # L-1, U=(q_L/S)*q1
    product = rescale(one_a*ba)      # L-2, (U/q1)*T=S
    two_a = rescale(a*two_pt)        # L-1, S
    y = drop(two_a, L-2)-product     # 两边均 L-2, S

polymultct(t,x) = x*(0.5+t):
    x_scaled = rescale(x*one_pt)     # L-1, q1
    t_half = drop(t,L-1)+half_pt     # L-1, S
    y = rescale(x_scaled*t_half)     # L-2, (q1/q1)*S=S
```

drop 后保留原 scale；二元 layer 的两个输入先由现有图调度降到同一 L。不要在 `inviter` 中把 `ba` 与 `one_a` 强行统一成 S，它们相乘只要求同 level；最终 sub 才要求两侧均为 S。

### 5.3 多项式和现有 LayerNorm

Stockmeyer 的每个 power 节点用实际 q 递推 scale；coefficient scale 使用 `MatPolyBase::compute_coefficient_scales_stockmeyer` 的缓存计划。对在 level l、power scale A 上乘系数，若下一目标为 T，则 PT scale 为 `(T/A)*q_l`；存在后续 CT 乘法时还要从后续目标继续反推，不能把每项都提前归一成 S。合流的所有项必须达到同一目标 level 和 scale。

现有 PCM LayerNorm 的 PT 配方保持原实现，并逐分支审计；可复用 PDM 相同算术的配方。例如：

- Centered 的 head mask、`1/n` PT 分别使用当前乘法级的 q；不把 `1/n` 的数学系数误当成编码 scale。
- MinimaxInit 的 `c0`、`c1`、`c2` 分别按 S、`q_L`、`(q_L/S)*q1` 编码。
- Goldschmidt 的 `three_pt` scale 为 `((S/q_L)*S/q1)*S`；`half_norm_pt` 为 `(((q_L/S)*(q_L/S)*q1)/S)*q_(L-2)`。同现有 DAG 一起复用，不单独交换乘法顺序。
- Affine 的 gamma 相关 PT 按 `(q_L/S)*q1` 编码，beta 在输出 add 处按 S 编码。
- Stats 的各分支使用现有 `generate_*_pt` 的真实 scale 计划；其 `x` 与 `x²` 归约分支的内部 scale 不相同，不能统一填 S。

如果审计发现已有 PCM 公式不满足真实 q 下的输入/输出契约，仅修正该分支及镜像 DAG、计分和必要验证；不得因已有注释声称 scale 为 S 就跳过检查。所有新增 layer 在设计审查时应附每条边的 `(level,scale)` 表，至少覆盖上述所有 mult/rescale 与 add 合流。

## 6. 编译与权重导出

### 6.1 展开逻辑的最小修改

主要文件为 [transforms.py](../../training/model_compiler/transforms.py)：

1. 在 `_expand_bert_multi_head_attention` 的当前拒绝点增加明确的 PCM 分支：矩阵类型选择 `parcpmm / partranspose / parccmm`；PDM 保持 `pdmpcmm / pdmtranspose / pdmccmm`。其余格式继续拒绝。
2. 在已有类型选择 helper 中补充本规范的 PCM/PDM 对应关系，使 `_make_upper_poly`、`_append_bert_softmax`、`_append_softmax_normalize`、`_expand_custom_gelu`、`_expand_custom_tanh` 选择正确类型。沿用小函数和显式类型表，不引入新注册系统。
3. 保持 `_add_ccmm_inputs` 的 packing 差异：PCM 为普通 lhs/rhs 顺序，PDM 有现有的输入交换。二元 inverse、GELU 的 `input_index` 也必须显式保留。
4. 补充 shape 透传类型、`set_pcm_K` 和最终 `set_level_costs`。通用 poly 与 GELU 必须保留 FFN 多 megablock；HeadColSum 在所有阶段都标 cost 2。
5. 保持 `pipeline.py` 现有 L_prepad、attention/LN/custom poly 展开、shape/K、level cost 与 absorb-scale 的依赖顺序；仅补必要类型映射。

BERT attention 的 `scaling=1/sqrt(head_dim)` 继续吸收进 Q 的 weight **和 bias**，只应用一次。不能再额外插入 gamma 重复缩放。softmax 保留 exp coefficients、range、exp_divisor、delta_1/delta_2、denominator scales/guards 和 inverse iteration 规则；GELU、tanh 保留原有分段多项式组合及输入缩放。

当前编译器 BERT softmax 的稳定化处理相对原始 ONNX 已有近似偏差；必须分别报告编译近似误差和 FHE 数值误差。不能把原始 ONNX 的迭代控制 epsilon/alpha 直接解释为新的分母加数，也不能为本次验收扩大 guard 或改基准。

### 6.2 序列化、计分与导出

| 文件 | 必须接通的接口 |
| --- | --- |
| `training/model_compiler/components.py` | 新类型的 JSON 读写；order、coefficients、value、参数路径等字段；有序二元输入 |
| `training/model_compiler/graph_partition_dp.py` | `supported_fhe_score_layers` 增加新类型，允许现有划分器处理 |
| `training/model_compiler/score.py` | 调用对应 PCM Python mirror 的 `get_fhe_op_count`，按真实 K/G/重复布局计数；包含两个 HeadColSum mask |
| `training/nn_tools/feature_mat_h5.py` | 新通用 poly 系数导出；复用 gamma 和常数矩阵导出；核对 BERT CPMM 的方向和 multiplier |

`feature_mat_h5.py` 的 `_resolve_parcpmm_weight/_bias` 已支持显式 multiplier，不应另写重复的 Q 缩放函数。现有 ViT 融合 QKV 拆分分支还会自带 Q 缩放；BERT 独立 Q 权重走自己的路径，必须用导出值测试证明没有遗漏或双重缩放。

PCM CPMM 读取逻辑 `[in_features,out_features]` 的矩阵；BERT `CustomMultiHeadAttention` 的独立 Q/K/V/proj 原始权重则按 `[out_features,in_features]` 使用，参考计算为 `x @ W.T`。当前 `_export_parcpmm` 不能仅靠 shape 检查识别这一点，必须在 PCM/BERT 对应的导出分支显式转置这些权重，并保持 Q multiplier 仅应用一次。不能改动 ViT 融合 QKV 已有的 `[in,out]` 导出方向。

FFN、pooler、分类头按各自导入节点的实际 ONNX 语义核对方向，必要时仅修正对应分支。禁止用 reshape 冒充转置，方阵 shape 相等也不能证明方向正确；BERT 的 PDM 导出分支保持原行为。

## 7. Generator、runtime 和客户端接口

### 7.1 在现有拒绝/分发处补 PCM 实现

应当在现在仅允许 PDM 的语义入口增加 PCM 分支，这样错误能在最接近真实能力边界的位置消除；同时完成该类型的全链路注册。仅删除 `only par_diagonal_pack` 检查、把 PCM 类型改名成 PDM 或对错误做 catch 后跳过，均不符合本规范。

以下位置必须逐项核查，而不是假定新增头文件即可自动支持：

| 位置 | 修改要求 |
| --- | --- |
| `inference/model_generator/deploy_cmds.py` | 导入 Python mirror；扩展 `_PCM_LAYER_TYPES` 与 feature_mat 类型、输入 CT 计数、构造和 dispatch；使用真实 head_shape/K；接通 data source 和 custom compute |
| `inference/model_generator/layers/` | 为新类实现普通 `call`、`call_custom_compute`、level 推导和 `get_fhe_op_count`；实际 DAG 与 C++ 公式一致 |
| `inference/fhe_layers/CMakeLists.txt` | 添加新 C++ 源文件 |
| `inference/inference_task/inference_process.h/.cpp` | 包含新类；`is_pcm_layer_type`；layer 初始化及 map；明文与 native CT 执行分发；lazy data source 注册；`encode_pt` executor 的类和 PT 类型分发 |
| `inference/interface/inference_client.cpp` 与任务 metadata 导出 | 保持 PCM 输入不转置；解包使用真实 head_shape、block size、shape；两分类头和 CLS 输出完整且一致 |

Python 的 `op_class`、`type`、`ct_idx` 或 `(mb,bi,bj,g)` 等 PT 索引必须与 C++ `generate_*_pt` 一一对应。参照现有 PCM 类的显式 dispatch；不要使新 PCM 名称误入 `ParUpperDiagonal*` 回调或使用未经类型校验的 PDM cast。

### 7.2 Lazy 生命周期

- lazy 初始化只保存原始参数、布局及必要的 scale/level 计划，不预编码全模型 PT。
- 每个 PT 在现有 `encode_pt` 节点按需生成，传给正确的实际 level；普通 eager 和 `run_plaintext` 接口同样保持可用，便于局部对照。
- 不改变既有 data source 所有权、PT 缓存和 GPU 释放机制；同一 CT 的分支不能因某分支原地降 level 而影响其他 consumer。
- 输出 CT 个数由布局公式确定。新类型既要被 generator 识别，也要被 runtime 识别，否则可能生成成功而执行缺少 data source。

`_normalize_pdm_head_shape` 等函数中属于 PDM 几何约束的拒绝检查无需放宽；应另走现有 PCM 几何接口。参数加密专用 `_ENCRYPTED_OFFLINE_UNSUPPORTED_LAYER_TYPES` 继续阻止新旧 PCM 类型进入尚未实现的参数加密路径。

客户端已有 `select_first_token`，继续用其公开语义得到全部两个 CLS logits。对 PCM 输出 `[44,2]`，客户端按真实 token 行恢复后选首行；不能先按 PDM 转置布局导出，再在验收脚本中随意截断或重排。

## 8. 实施顺序与验证

本次实施分三步，每步完成对应验证再进入下一步：

1. 补齐 PCM 原语、PT scale 与 Python mirror → 以小矩阵验证布局、数学语义及真实 level/scale。
2. 接通展开、导出、计分、generator、runtime → 编译真实 BERT，检查最终 JSON、DAG、参数和签名；不应残留 Custom 非线性或 PDM 运算。
3. 跑完整 PCM BERT GPU lazy → 独立原始 ONNX 对照误差 ≤ 0.2，并完成既有 PDM BERT、PCM ViT 的回归核对。

### 8.1 必要的小规模验证

沿用 `inference/unittests` 与现有 Python 验证方式，不新建测试框架。所有测试临时文件、构建、日志和数值 dump 放在 `/tmp/`。

- **布局与归约**：非整块 token 长度、`H=12,G=2` 的尾部 head、`G=1` 且重复 chunk、多 `bi/bj`、FFN `K>1`。每个 head/row 使用不同数值以发现串 head/串行；HeadColSum 输入 padding 人为放非零干扰值，合法 denominator 必须不受其影响。
- **多项式**：覆盖 BERT 实际各个 order、零常数项/零系数、至少一个低阶；与按逻辑矩阵直接求值的结果比较，覆盖 K 和 head 尾部。
- **非对称二元算子**：`inviter(a,b)` 与 `polymultct(t,x)` 使用不同输入，发现 input_index 颠倒；GELU 检查完整组合，不能只测逐元素乘法。
- **scale/level**：使用实际不等于 S 的 q，验证每个 add/sub 前 level/scale 匹配、mult 前 level 匹配、输出 `L-cost,S`。测试应读取真实 CT scale，不能只断言注释或手工设置的 metadata。
- **lazy 对照**：同一个小 layer 的 eager 和 lazy 输出一致，确认 lazy PT 确实经 `encode_pt` 回调生成。普通输入恰好使 `(S/q_L)*S≈S` 的测试不能代替真实 q 验证。
- **导出和分类头**：用不对称权重校验 BERT Q/FFN 的方向和缩放；覆盖 3072 列 FFN 和真实两列分类输出。JSON round trip 保留 coefficients、迭代属性和二元输入顺序。

### 8.2 端到端配置与命令骨架

复用 `training/config/BERT_base_config.json` 中的 BERT 参数，仅在 `/tmp/` 的配置副本设置 `mat_pack_style=par_block_col_major`。新增 layer 成本由代码决定，不为减少 bootstrap 而更改模型属性或 CKKS scale。以下命令在实现完成后执行：

```bash
set -euo pipefail
repo=/home/linghm/project/latti-ai
python=/home/linghm/pytorch_venv/bin/python
experiment_dir=$(mktemp -d /tmp/bert-pcm-input-gpu-XXXXXX)
export PYTHONPATH="$repo:$repo/inference/lattisense"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$experiment_dir/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
cd "$experiment_dir"

"$python" -B - "$repo" "$experiment_dir" <<'PY'
import json, pathlib, sys
repo, out = map(pathlib.Path, sys.argv[1:])
cfg = json.loads((repo/'training/config/BERT_base_config.json').read_text())
cfg['mat_pack_style'] = 'par_block_col_major'
(out/'config-pcm.json').write_text(json.dumps(cfg, indent=2)+'\n')
PY

"$python" -B "$repo/training/run_compile.py" \
  --input "$repo/runs/BERT/model/trained_poly.onnx" \
  --output "$experiment_dir" --style multiplexed \
  --config "$experiment_dir/config-pcm.json" --num_workers 8 > compile.log 2>&1
"$python" -B "$repo/inference/interface/gen_mega_ag.py" \
  --task-dir "$experiment_dir/task" > generate.log 2>&1
"$python" -B "$repo/tools/evaluate_transformer_onnx_csv.py" \
  --onnx "$repo/runs/BERT/model/trained_poly.onnx" \
  --input-csv "$repo/runs/BERT/input.csv" \
  --output "$experiment_dir/reference.npz" > reference.log 2>&1
```

构建在 `/tmp/` 使用 `cmake -S "$repo" -B "$experiment_dir/build" -DINFERENCE_SDK_ENABLE_GPU=ON` 及当前环境所需的既有依赖选项，随后 `cmake --build "$experiment_dir/build" --target inference -j 8`，并把 configure/build log 写到该目录。使用**链接本次实现后 SDK** 的 `$experiment_dir/build/examples/inference`（源文件 `examples/inference.cpp`）或审计 harness 执行：

```text
<本次构建的 inference / audit 程序>
  --task-dir <experiment_dir>/task
  --input /home/linghm/project/latti-ai/runs/BERT/input.csv
  --gpu --verify
```

审计 harness 可基于现有示例在 `/tmp/` 增加输入/输出密文和解密 CSV 导出；不得复用缺少新 PCM 类的旧静态链接二进制。源码、二进制、命令和哈希一起存档。上述骨架不是已完成运行的证明，独立输出审计也不能由 `--verify` 代替。

### 8.3 唯一的数值通过条件

独立基准使用 `tools/evaluate_transformer_onnx_csv.py` 对**原始 ONNX**及同一 CSV 的计算。它对自定义多项式算子的参考求值必须保持原始属性和有效 token 语义；必要时对照导出模型对应的明文实现。

对每个原始输出名称，检查名称集合、shape/元素数量、索引顺序与任务输出一致，参考值与解密值全部有限。由 `select_first_token=true` 表达 BERT 的首 token 分类语义，得到完整两个 logits 后逐项比较：

```python
assert actual.shape == reference.shape
assert np.isfinite(actual).all() and np.isfinite(reference).all()
max_abs_error = float(np.max(np.abs(actual - reference)))
assert max_abs_error <= 0.2
```

本次要求是 **≤ 0.2，等于 0.2 通过**；不沿用旧报告中的严格 `< 0.2` 文案。每个输出都要通过，不能只看平均误差、类别相同或挑选某个 logit。

同时保留原生 `--verify` 对编译后明文图的误差及其既有容差，以区分“编译近似偏差”和“FHE 数值误差”。编译后明文与原始 ONNX 的对照也要记录；只通过编译后明文核验不构成本次成功。

最终证据还必须满足：

- 最终 task 配置确为 `par_block_col_major`，包含 `parcpmm/parccmm` 及新 PCM 非线性类型，没有遗留的 `pdm*` 运算或未展开的 BERT Custom 算子。
- 输入签名为 CT，确有客户端加密输入与 GPU 执行；执行经过 `run_task_lazy` 和新类的 `encode_pt`。不允许对新非线性绕回明文或 CPU 路径。
- 真实输入/输出密文、全部解密输出、reference、逐项差值、layer 类型统计、退出码、GPU 时间、资源记录、代码版本/工作区差异和运行命令存于同一 `/tmp/` 实验目录。
- 既有 PDM BERT 和 PCM ViT 的布局、模型数学语义和输出核验继续通过；只有完成 PCM BERT 全程解密核验，才能将本文状态改为“已实现并验收”。

## 9. 相关记录与已知基线

此前输入加密的 PDM BERT 实测记录在 `/tmp/bert-input-gpu-46xnbh7z/`，两项原始 ONNX logits 为约 `[-3.16095213, 6.08051821]`，PDM GPU 对原始 ONNX 最大绝对误差约 `0.111388695`；PCM ViT 输入加密记录在 `/tmp/vit-pcm-input-gpu-v85qd3zi/`。这些临时目录仅作为本机历史参考，不能替代本次 PCM BERT 的新产物或成为长期测试依赖。

[既有 Transformer 验收报告](../report/transformer_runtime_lazy_acceptance.md) 说明了编译稳定化造成的参考偏差，但其中参数加密运行模式、性能和 `<0.2` 门槛不直接套用于本次输入加密验收。

不承诺 PCM BERT 的精度、bootstrap 数量或性能一定优于 PDM；本规范以完整 PCM 路径、真实 scale/level 契约以及原始 ONNX 误差 ≤ 0.2 作为可判定结果。

## 10. 本次实现与实测记录

实验目录：`/tmp/pcm-bert-implementation-kSrQMu/`。构建、配置副本、任务、密文、数值 dump、日志和辅助脚本全部位于 `/tmp/`；源代码和本规范留在仓库。

已实现第 4 节七种新类型，以及编译展开、权重方向与系数导出、计分、Python generator、C++ native/plaintext、lazy data source 与 `encode_pt` 分发。复用既有 LayerNorm 算术，并修复其紧凑统计量在 gamma 与 shape 传播中的接口。参数加密专用 PCM 拒绝检查保留。

完整 BERT 实测使用 RTX 5880 Ada、N=65536、block size=64、12 heads。调用 `InferenceClient` 加密实际 CSV，经 `InferenceServer::evaluate` 的 GPU `run_task_lazy` 推理，再由客户端解密完整两个 CLS logits。GPU 密文推理日志耗时 241.422 秒（不含上下文和模型装载）。

| 项目 | logit 0 | logit 1 |
| --- | ---: | ---: |
| 原始 ONNX，独立 NumPy float64 参考 | -3.160952131436212 | 6.080518208917383 |
| PCM GPU 解密 | -3.122854461609292 | 6.053773179527310 |
| 编译后明文图，取任务定义的首 token | -3.27195618 | 6.17442646 |

对原始 ONNX 的最大绝对误差为 **0.038097670，满足 ≤ 0.2**。但对编译后明文图的最大误差为 **0.14910172**，原生 `--verify` 使用未修改的 0.1 容差，因此打印 `FAIL`，进程退出码为 1。

使用相同模型、输入、任务与审计二进制进行第二次完整 GPU 推理，产物在上述目录的 `repeat-current/`，结果如下：

| 执行 | GPU 解密 logits | 对原始 ONNX 最大绝对误差 | 对编译后明文图最大绝对误差 | 原生 `--verify` / 退出码 | GPU 密文推理时间 |
| --- | --- | ---: | ---: | --- | ---: |
| 首次 | `[-3.122854461609292, 6.053773179527310]` | 0.038097670 | 0.149101719 | FAIL / 1 | 241.422 s |
| 复跑 | `[-3.134193281558994, 6.065329208420550]` | 0.026758850 | 0.137762899 | FAIL / 1 | 217.432 s |

两次均通过原始 ONNX 的 ≤ 0.2 门槛，均未通过原生 0.1 检查。编译后明文图相对原始 ONNX 的最大绝对误差为 0.111004049。两种偏差存在部分抵消，不能据两次结果推断跨随机密钥的稳定精度，误差来源也尚未定位。验收报告必须同时呈现两项结果，不能把退出码改写为 0。

现有小规模 CKKS 检查覆盖新原语的真实输出 scale/level、不同 head 组、重复 chunk、跨 block 与多 megablock，以及多项式 order 1/2/3/7/15（含非零常数项、非零三次项和零系数）。HeadColSum 人为注入值为 7 的 padding 干扰后仍通过。使用真实 `q_L=17180393473 != S≈1.71799e10`，输出 `abs(scale/S-1)≤1e-9`，初步独立检查数值最大误差约 `7.1e-6`。

仓库内的 `inference/unittests/gen_pcm_bert_layers.py` 和 `test_pcm_bert_layers.cpp` 提供可复现检查，覆盖上述布局、多项式、完整三段多项式加末端乘法的 GELU 组合，以及已有 LayerNorm 五阶段（含紧凑统计量和跨 head 归约）。CPU native/eager 与 lazy 对照通过 **1,836 项断言**，lazy 每条 add/sub/mult 前读取真实元数据，验证同 level、add/sub 同 scale。GPU lazy 与 CPU native/eager 逐项对照通过 **1,802 项断言**，验证真实输出 scale/level、数值及 PT 按需生成。对应最终日志为 `check-layers-cpu-with-ln.log`、`check-layers-gpu-with-ln.log`。测试 harness 的 LayerNorm PT 分发与生产 runtime 的独立分支一致；曾遗漏此分支导致测试停滞，补齐后通过，未为此改动生产 LayerNorm 算术。

`training/model_compiler/unittests/test_pcm_bert.py` 的 **4 项测试通过**，覆盖非对称 Q 权重、weight/bias 单次缩放、实际 3072 列 FFN、两列分类头、ViT 融合 QKV、PCM 系数轴和标量 AddPt，以及七种新增类型的 JSON 往返和二元输入顺序。实际 ONNX 到最终任务的独立属性核对也通过：12 个 attention 的 exp 系数、inverse/refinement 次数与 Q 缩放，24 个 LayerNorm 的原始参数和各自 7 次迭代，12 个 GELU 和 1 个 tanh 的全部多项式系数。日志和结果分别为 `check-compiler-exports.log`、`check-model-attributes.json`。

分层测试可按以下命令复现；生成器强制要求输出位于 `/tmp/`。GPU 图须额外用 `--without-contract-checks` 生成至另一个 `/tmp/` 目录，再设置 `PCM_BERT_TEST_GPU=1` 运行同一二进制：

```bash
PYTHONPATH="$repo:$repo/inference/lattisense" "$python" -B \
  "$repo/inference/unittests/gen_pcm_bert_layers.py" "$experiment_dir/layer-regression"
cmake --build "$experiment_dir/build" --target test_pcm_bert_layers -j 8
PCM_BERT_TASK_ROOT="$experiment_dir/layer-regression" OMP_NUM_THREADS=2 \
  "$experiment_dir/build/inference/unittests/test_pcm_bert_layers"
PYTHONPATH="$repo:$repo/inference/lattisense" "$python" -B -m unittest \
  training.model_compiler.unittests.test_pcm_bert
```

GPU 使用生成器的 `--without-contract-checks` 生成生产形态的图；每条边的测试专用 CT 回传探针在 GPU 出现调度停滞，90 秒超时退出 124，日志单独保存在 `check-layers-gpu-contract-probes.log`。因此每条边的内部元数据探针通过证据来自 CPU；GPU 通过证据是相同 Python 算术 DAG 的输出与 lazy/eager 对照，不能宣称 GPU 的逐边探针也通过。不为这些测试回调修改底层 SDK。

同一本次构建的 SDK/审计程序还执行了已有任务的 GPU 回归（任务来自第 9 节历史目录，结果另存本实验目录）：

| 回归 | 对原始 ONNX 最大绝对误差 | 原生 `--verify`（0.1） | 退出码 |
| --- | ---: | --- | ---: |
| PCM ViT，全部 1970 个输出 | 0.005648153 | PASS | 0 |
| PDM BERT，完整两个 CLS logits | 0.108617942 | PASS | 0 |

此外，用当前 Python 编译器重新编译并生成 ViT 任务，layer 类型数量及 CKKS 参数与历史基线一致；重新生成的任务也完成 GPU 输入加密推理，全部 1970 个输出（shape 为 `[1,197,10]`）对原始 ONNX 的最大绝对误差为 **0.027367480**，原生 `--verify`（0.1）通过，退出码为 0，总运行时间为 242.305 秒。产物与报告位于 `compile-regression-vit/`。上述回归不消除本次 PCM BERT 的原生 0.1 检查失败。

机器可读报告为实验目录内的 `report.json` 与 `repeat-current/report.json`；两次原始日志 `runner.log`、`run.json`、`decrypted.csv`、`plaintext.csv` 和 `runner-time.txt` 保留原样，独立参考保存在根目录 `reference.npz`。第 8 节列出的验证要求是完整检查清单，不能因上述两次端到端门槛通过就将未执行项目标为通过。
