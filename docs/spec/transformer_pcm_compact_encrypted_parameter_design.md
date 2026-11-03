# PCM Transformer 压缩参数密文与 GPU runtime_lazy 设计规范

状态：实现与压缩参数模式端到端验收通过，实测记录见第 8 节。采用“在行位置和重复 chunk 中合并 seed”的实数布局，定义代码修改和验收要求。本文取代原 PCM 参数压缩调研。

## 1. 执行配置与数据契约

两模型统一使用以下实际配置，沿用现有 Provisioner / Runner 接口：

```json
{
  "deployment_mode": "server_provisioned_runner",
  "input_mode": "plaintext",
  "parameter_mode": "encrypted_offline",
  "parameter_loading_mode": "runtime_lazy",
  "mat_pack_style": "par_block_col_major"
}
```

Provisioner 读取模型参数并离线生成紧凑参数密文；Runner 接收公开 CSV，按需加载和恢复参数，经 GPU FHE 推理输出密文，再由 Provisioner 侧解密。这里的 `runtime_lazy` 是参数 store 的实际延迟加载模式。线性层使用现有 JSON 类型 `parcpmm`、类 `ParBlockColMajorCPMM`，参数加密分支直接消费本文的 seed。

PCM 对应代码中的 `par_block_col_major_pack`，配置值为 `par_block_col_major`。`parpcmm/parcpmm` 的参数加密接口以本文压缩布局为存储契约；当前两模型的线性层沿用已有 `parcpmm` 入口。第 7 节的验收矩阵固定为 BERT、ViT 各一条压缩参数密文 GPU runtime_lazy 路径。

基于 [PCM BERT 完整支持规范](transformer_pcm_bert_full_support_design.md) 已接通的 PCM 算术实现。编译器继续传递真实 `shape`、`head_shape`、`K`、block size、head 数、二元输入顺序、CLS 选择及权重方向。BERT 独立 Q/K/V/proj 和 Linear 的 `[out,in]` 源权重转换为 PCM `[in,out]`；Q 的 weight/bias multiplier 继续只应用一次。

代码修改遵循 [karpathy-guidelines](/home/linghm/.codex/skills/karpathy-guidelines/SKILL.md)：复用现有 layer、显式类型分发、参数 builder、store v2 与 GPU 图操作；新增逻辑限定在参数布局、来源分类及对应消费接口。

## 2. 参数来源与公开 PT

### 2.1 来源记录

在导入、展开、参数融合和 H5 导出过程中，为每类参数保留以下信息，并传到 generator：

| 字段 | 约定 |
| --- | --- |
| `visibility` | `public` 或 `secret`，按参数项而非整个 layer 分类 |
| `source_kind` | `shape`、`math_constant`、`approximation_config`、`onnx_initializer`、`h5_parameter`、`calibration` 或 `derived` |
| `source_refs` | 私有编译侧的原始来源和派生关系 |
| `public_recipe` | 公开项的确定性生成规则及必要配置 |
| `use_site_level`、`scale` | 最终算术使用位置的实际 level 和编码 scale |

来源分类用已有显式分发中的小函数实现。`derived` 的秘密性沿依赖传播：含任何秘密输入的融合结果作为秘密参数。缺少来源记录时按秘密处理，并在审计记录中标明 layer、参数类型和缺失字段；公开项缺少可重建 recipe 时拒绝导出。

在现有 layer JSON 上增加 `parameter_sources`，按生成器的 `type`，以及适用的 `pt_idx`、`degree` 定位参数项；同一参数的不同 use-site 可引用同一来源记录。`components.py` 负责往返序列化，展开与融合函数显式复制或合并记录，`feature_mat_h5.py` 在导出最终值时复核来源。`deploy_cmds.py::_append_matrix_source` 的 builder 通过同一个局部 resolver 查询该记录，决定生成 PT 或注册 CT，替换仅按 `op_class` 判断秘密性的分支。结构 mask 使用显式公开规则；其余项通过来源解析得到分类。

公开项必须能由公开 shape、已发布的近似多项式配置、softmax/LN 数学常数唯一重建。BERT 通用多项式中作为公开近似配置发布的系数可以本地生成；ONNX/H5 中存放的训练、融合或标定系数沿其真实来源继续加密。H5 只是中间载体；公开配置派生的 H5 副本仍按公开 recipe 重建。

公开近似配置随 Runner bundle 发布其版本、内容和 hash，recipe 标明公式及参数；由这些配置重建的值在 Provisioner 与 Runner 两侧逐项一致。标定范围参与形成的 softmax shift、LN `inv_var/inv_std`、gamma 或 Poly 系数沿依赖链标为秘密。公开常数与秘密值融合后的整个执行参数继续加密，公开常数只在独立使用位置本地编码。

### 2.2 各参数族的处理

| 参数族 | 存储与消费 |
| --- | --- |
| CPMM 线性权重 | 按第 3 节生成行/chunk seed CT，按请求恢复后做 CT×CT 乘加 |
| CPMM bias、LN gamma/beta | 按现有 `generate_*_pt` 的执行槽位和 scale 生成独立 CT，保留 token-tail、head 和列 padding |
| 标定 running-max、标定范围、由其派生的系数 | 私有侧完成融合后生成 CT；传给 Runner 的是参数 ID 和布局 |
| `pcmgamma`、`pcmpoly` 的 fused gamma/Poly 系数 | 逐参数、degree、use-site level 和实际 scale 生成 CT |
| `pcm_add_pt`、位置嵌入或任意秘密矩阵 | 按真实逐行逐列值及现有有效槽 mask 生成 CT |
| `pcmgeneralpoly` 系数、softmax 的 shift/gamma 等 | 依 2.1 分类；公开配置生成 PT，秘密派生值生成 CT |
| LN 的 `inv_var`、`inv_std`、初始化系数和 epsilon 组合 | 按各项来源分类；涉及标定值的组合在 Provisioner 内形成秘密 CT |
| transpose/CCMM、head sum、LN、seed selection 的结构 mask | Runner 按公开几何和 use-site scale 生成 PT |
| 公开 softmax/LN/激活数学常数 | Runner 按既有公式生成 PT，包括公开的 0.5、1、2、3、归约维数倒数及归一化常数 |

混合 layer 同时接收公开 PT 和秘密 CT。例如 LN Stats 的 head mask、`1/n` 可公开，`inv_var` 若来自标定则对应 PT 值须加密；LNAffine 的 `inv_std*gamma` 在私有侧生成 CT，beta 在实际 add 位置生成 CT。秘密参数即使数值为 0 或恰好相等，也按同一公开布局写入。

Runner 的 JSON、graph 和公开 PT recipe 仅保留公开配置、shape、索引和参数 ID。Provisioner 导出时剥离秘密 `coefficients`、gamma、标定范围及其数值派生项；源码路径等私有来源信息保存在私有侧审计文件。Runner 的公开 layer 初始化仅使用公开 recipe 和几何，公开 PT 的构造与私有权重数组解耦。

## 3. CPMM 行/chunk seed 布局

### 3.1 几何和 BSGS 值序

统一记 `T=N/2` 为槽位数，`S` 为 default scale，`H` 为实际 head 数，`d` 为 block size。以下 `u` 是每个 chunk 的 head lane 数，与 scale 区分。

```text
Hp = next_pow2(H)
if T >= Hp*d*d:
    u = Hp; C = Hp*d*d; G = 1
else:
    u = T // (d*d); C = T
    if u == 1: Hp = H
    G = Hp // u
rho = T // C
```

沿用 PCM 构造约束并检查 `d` 为正二次幂、`T >= d*d`、整除关系，以及 CPMM 的 `n_per_head=d`。两验收模型逐层核对这个条件；输入/输出尾部通过真实 weight shape 补零。多 megablock 的 per-head 宽度取实际 `head_shape`，完整 W shape 取编译 metadata。

对 PCM `A*W`，令 `n=H*d`、`Kr=ceil(Wrows/n)`、`Kc=ceil(Wcols/n)`、`K=max(Kr,Kc)`，满足 `min(Kr,Kc)=1`。square 取原点；expand 的子矩阵沿 W 列取 `mb*n` 偏移，reduce 沿 W 行取 `mb*n` 偏移。每个 `Wmb` 补至 `(Hp*d,Hp*d)`，真实 shape 外置零。

请求 `r=(mb,g,bp,k)` 的范围为 `mb<K, g<G, bp<H, k<d`。令 `B=ceil(sqrt(d))`、`a=k//B`、`b=k%B`，复用 `build_block_diagonal()` 的实际 BSGS 预移位：

```text
V_r[j,h] = Wmb[(g*u+h)*d + (j+b)%d,
              bp*d + (j-a*B)%d]
D_r[ci*C + (i+d*j)*u+h] = V_r[j,h]
```

其中 `i,j<d, h<u, ci<rho`，模运算取非负余数。`D_r` 是消费位置需要的执行参数值，沿 token 行位置 `i` 及重复 chunk 完全相同，与 token 行块 `bi` 无关。C++ seed generator 与现有 diagonal generator 共用这个取值函数，以现有 `generate_diag_pt()` 解码值作为逐槽校验参考。

### 3.2 请求编号与 seed

同一 layer instance、CKKS 参数集、use-site level、scale 下统一展平所有 megablock 的请求：

```text
ridx = (((mb*G+g)*H+bp)*d+k)
Q = K*G*H*d
P = d*rho
sid = ridx // P
t = ridx % P
ci = t // d
rowpos = t % d

Z_sid[ci*C + (rowpos+d*j)*u+h] = V_r[j,h]
Nseed = ceil(Q/P) = ceil(K*G*H/rho)
```

seed 其余位置补零。请求跨同层 megablock 合并，最后仅统一向上取整一次。存储参数只写 `Nseed` 份 seed，真实输入 head padding 保留在上述槽位中。

seed ID 的去重键包含 layer instance、context、存储 level、scale 和 `sid`。同一 seed 的所有恢复请求共享一个参数 element；实际 load 节点按批次依赖构造，使缓存可以在消费间隙释放或重载。布局及分配数量仅由公开几何决定。

在 `N=65536,d=64` 下，以下数目用作布局检查：

| 线性层 | `u,G,rho,K` | `Q` | `Nseed` |
| --- | --- | ---: | ---: |
| ViT `[192,192]` | `4,1,2,1` | 192 | 2 |
| ViT `[192,768]` / `[768,192]` | `4,1,2,4` | 768 | 6 |
| BERT `[768,768]` | `8,2,1,1` | 1,536 | 24 |
| BERT `[768,3072]` / `[3072,768]` | `8,2,1,4` | 6,144 | 96 |

### 3.3 一次选择与带符号广播

公开 `M(ci,rowpos)` 在选定 chunk 的选定行、全部 `(j,h)` 上为 1，其余为 0。以 SDK 约定 `Rot(X,s)[v]=X[(v+s)%T]` 执行：

```python
E = mult(Z_sid, M(ci, rowpos))
for bit in range(log2(d)):
    step = (1 << bit) * u
    E = add(E, Rot(E, step if (rowpos >> bit) & 1 else -step))
for bit in range(log2(rho)):
    step = (1 << bit) * C
    E = add(E, Rot(E, step if (ci >> bit) & 1 else -step))
D_r = rescale(E)
```

逐低位展开时，方向由原始 `rowpos`、`ci` 的对应位决定，复制范围恰好覆盖全部行和 chunk。每条请求需要一次 CT×PT、`log2(d*rho)` 次旋转/加法和一次 rescale。旋转 key 按 `±u*2^bit`、`±C*2^bit` 收集并检查 seed level 可用性；相同上下文可共享 key。

## 4. Level、scale 与密文参数消费

记 `q_L=param.get_q(L)`，对应 Python `fhe_param.q[L]`。mult 双方 level 相同；add/sub 双方 level 和实际 scale 相同。mult 后 rescale 的结果为 `(L-1,(S1/q_L)*S2)`。所有 scale 公式先除再乘，逐边记录真实 scale，按 layer 输出 S 反推 PT scale。rescale 接口的 default scale 参数不能指定实际结果 scale，参数编码必须使上述公式自然得到目标值。

CPMM 输入为 `(L,S)`，执行权重原 scale 为 `sW=q_L`。恢复计划固定为：

| 对象 | Level | Scale |
| --- | ---: | ---: |
| seed CT | `L+1` | `q_L` |
| selection PT | `L+1` | `q_(L+1)` |
| 选择并广播后 CT | `L+1` | `q_L*q_(L+1)`（仅表示数学乘积） |
| rescale 后 `D_r` | `L` | `(q_L/q_(L+1))*q_(L+1)=q_L` |
| feature 与 `D_r` 乘加并 rescale | `L-1` | `(S/q_L)*q_L=S` |
| head-sum 后乘公开 head0 mask 并 rescale | `L-2` | mask scale 为 `q_(L-1)`，结果为 S |
| bias CT，与最终输出相加 | `L-2` | S |

恢复只增加参数分支的一层；编译器检查 context、序列化、GPU 和 key 都支持 `L+1`，并将其纳入实际参数开销。BSGS 的 baby/giant rotations、head reduction、output head 旋转和 expand/reduce 求和继续使用现有 PCM 顺序。权重消费改用 `mult_relin` 和密文累加，giant rotation 前完成必要 relinearization；各乘积相加时保持同 level/scale。

其余秘密参数按原执行位置的 level/scale 直接加密，CT×PT 位置对应改为 CT×CT + relinearization，参数 add 对应 CT add，保留原 rescale 位置。公开项继续使用 CT×PT。局部复用 `matrix_param_ops.py` 的 deferred builder、`encrypted_param_ops.py` 的 materialize/乘加 helper；CPMM 为每个 seed ID 建独立缓存项，再按请求恢复。

典型 scale 配方如下：`pcmgamma` 为 `q_L`；LN affine 的 gamma 相关 CT 为 `(q_L/S)*q_(L-1)`、beta CT 为 S；通用 Poly 复用 Stockmeyer 实际 coefficient-scale 计划。LN Stats、MinimaxInit、Goldschmidt 及 softmax 的 PT 使用现有实际 q 公式；各项变为 CT 时只改变操作类型，保持数值和 scale 计划。若多项式 degree 间目标不同，分别生成参数。

例如二次 Poly 的 `c0,c1,c2` scale 分别为 `S,q_L,(q_L/S)*q_(L-1)`；四次分解中高阶项复用当前 `generate_coeff_pt()` 的 `c3=((q_L/S)*q_L/S)*q_(L-2)`、`c4=(((q_L/S)*q_L/S)*q_(L-1)/S)*q_(L-2)`。softmax 的 square/CT-mul 在第一次 rescale 后为 `(S/q_L)*S`，其公开有效槽 mask scale 为 `(q_L/S)*q_(L-1)`，第二次 rescale 后回到 S。参数签名、私有编码器和 runtime 使用同一 scale 计划，实际 CT 载入后校验 level 与 scale。

公开 CSV 按 S 编码，与同 level、同 S 的离线 `Enc(0)` 相加进入现有旋转型矩阵算法。相同 context/level/scale 的 `InputZero` 可只存一份并只读共享，实际数量和字节单独统计。

## 5. 代码修改接口

### 5.1 编译、导出与评分

| 位置 | 具体修改 |
| --- | --- |
| `training/model_export/`、`training/model_compiler/transforms.py` | 建立并透传参数来源；展开、absorb/fuse 后保留派生关系；公开配置与秘密参数分开序列化 |
| `training/model_compiler/components.py` | 序列化参数来源及 `pcm_seed_layout`；保存最终 use-site level、scale 和 seed level |
| `training/nn_tools/feature_mat_h5.py` | 私有侧导出权重和秘密参数；保留 BERT 转置、Q 单次缩放及 ViT fused QKV 语义；将可公开重建项关联 recipe |
| `training/model_compiler/score.py`、`graph_partition_dp.py` | 为 PCM 计入 seed 恢复、CT 参数乘加及秘密项存储；CPMM 读取 per-megablock `head_shape` 和完整 W shape，修正以总宽度除 H 估计 reduce 输入的分支 |
| `training/model_compiler/pipeline.py` | 最终 drop/bootstrap/absorb 完成后重新核对 level、来源、seed 数及存储计费，与最终导出图一致 |

来源传播重点覆盖 `transforms.py` 的 `_make_scalar_gamma`、`_make_upper_poly`、BERT softmax 展开、`_bert_layernorm_runtime_attrs`、`expand_poly_act_rn` 和 `expand_parcpmm_add_pt`，以及 `pipeline.py` 的 gamma/scale absorb。将 bias 拆为 `pcm_add_pt`、将公开 scalar 融合到秘密权重、将 running-max 融合到 Poly 等操作都在新参数项上写入合并后的来源。

`pcm_seed_layout` 固定包含 `kind="pcm_row_chunk_v1"`、`weight_shape`、`head_shape`、`d,H,Hp,u,C,G,rho,K`、`request_count=Q`、`requests_per_seed=P`、`seed_count`、`use_site_level=L`、`seed_level=L+1`、`scale=q_L` 及确定性请求编号版本。布局字段由公开 metadata 推导并在导出时复核。

存储评分使用既有 `encrypted_parameter_bytes` 校准：

```text
parameter_payload_bytes = sum_over_layers(Nseed * Bct(L+1))
                        + sum_over_other_secret_items(Bct(item.level))
                        + input_zero_bytes
```

公开 PT 的密文参数数目为 0。最终报告逐层、逐来源、逐 level 汇总，`unaccounted` 为空；导出后以实际容器 element 长度校验字节。seed 数量比例是布局指标，完整 bundle 大小取真实落盘值。

### 5.2 Python layer 与任务生成

1. `par_block_col_major_cpmm.py` 增加紧凑参数 builder 和恢复图，直接输出 seed signature；`_block_mult_cpmm` 使用密文参数乘加。bias 走 CT add，head0 mask 走公开 PT。
2. `par_block_col_major_{polyactrn,poly,layernorm,softmax,poly_mult_ct,add_pt}.py` 对应实际文件中的参数生成/消费接入 deferred builder；按具体 `op_class,type,pt_idx/degree` 解析来源。新增代码沿用相邻类型的 `call_custom_compute`、count 和显式 dispatch 风格。
3. `deploy_cmds.py::_append_matrix_source` 为 PCM 设置 builder，公开项生成 `encode_pt`，秘密项加入 `encrypted_parameters`，相同 seed 仅注册一次。PCM 各层统一使用该入口，包括当前直接传 `CustomDataNode` 的分支。
4. `parcpmm` 的首次输入注册统一走 `_register_feature_nodes`，使 plaintext 输入正确加入 `InputZero`；统一处理其余 PCM 第一 consumer 的同一情形。
5. PCM 类型的 encrypted-offline 能力检查与实现同时接通；任务 signature 的秘密参数为 CT，公开 PT 通过 Runner-local recipe 生成。每个二元输入保留原 `input_index`。

编译、任务生成、Provisioner、Runner 中原先只允许 PDM 的能力检查，在对应 PCM 实现可用后加入明确的 PCM 分发；每处报错保留 layer ID、类型和缺失接口信息。新增分支沿用现有类型表和构造函数。

### 5.3 C++、Provisioner 与 Runner

下表路径相对 `inference/`，解密工具路径相对仓库根目录。

| 位置 | 具体修改 |
| --- | --- |
| `fhe_layers/par_block_col_major_cpmm.{h,cpp}` | 共享 BSGS 值索引；新增 seed 与 selection PT generator、布局校验和 seed 计数；结构 mask generator 仅依赖公开几何 |
| PCM 各参数化 C++ layer | 保留现有 `generate_*_pt` 的数值与 scale 作为 Provisioner 编码源；公开 generator 与秘密数组分离，支持按 recipe 构造 |
| `inference_task/inference_process.{h,cpp}` | `generate_matrix_parameter` 注册 PCM seed、秘密参数及公开 PT；`load_public_matrix_layers` 接入 PCM 公开构造分支 |
| `interface/inference_provisioner.cpp` | 依据来源和布局按 seed/秘密参数逐项加密、写入 store v2；导出经清理的 Runner 配置，私有 H5 与来源文件保留在 Provisioner 侧 |
| `interface/runner_bundle_io.{h,cpp}` | signature、layout、manifest、实际 level/scale、element 数/偏移/长度一致性检查；seed 的 `use_site_level=L` 与存储 `level=L+1` 分别记录 |
| `interface/inference_runner.cpp` | PCM 公开 layer 初始化和 `encode_pt` 分发；PCM plaintext CSV 打包；输出 layout 与解密工具联动；记录参数加载和恢复指标 |
| `data_structs/feature_mat.{h,cpp}` | 提取复用的 PCM 纯 slot 打包函数，供 Runner 编码 PT；原加密 pack 复用该函数 |
| `tools/decrypt_runner_output.cpp` | 按任务 `par_block_col_major` 和真实 shape/head_shape 解包；使用已有 CLS 选择语义导出全部 logits |

Runner 输入保持 PCM 行列方向，使用真实 `matmul_block_size`、head shape、megablock 顺序。BERT 输入为 `[44,768]`；ViT 的 CSV 为 patch embedding 前的 `[197,768]`，首层投影后为 `[197,192]`。这些输入和 classifier 两列/十列尾部均需通过相同 metadata 恢复。

## 6. runtime_lazy 调度与内存

复用 store v2 `load_element`、真实输入依赖及 executor 引用释放机制：

```text
store + actual_trigger
    -> load_encrypted_param_ct(sid, expected_level=L+1)
    -> selection PT multiply
    -> signed rotations/adds
    -> rescale to (L,q_L)
    -> feature mult_relin + BSGS accumulation
    -> subsequent batch trigger
```

每层按请求顺序小批消费，首版以一个 BSGS baby group（至多 `B=ceil(sqrt(d))` 条恢复权重）为批次。同一批的恢复结果复用于所有需要的 token 行块；消费完成后释放恢复结果，再触发下一批。一个 seed 的请求消费结束再推进下一个 seed。跨层加载依赖实际 feature 就绪；批间 load/recovery 依赖前批实际累加结果。公开 PT 编码也连接相应消费前驱，避免一次准备全图 mask。

共享 seed 只读，选择/广播产生独立 CT。store 缓存沿用 64 MiB 上限并记录实际设置；恢复权重的 DAG 存活上限单独按批次检查。feature 的 baby rotations、输出 partial sums 和 GPU allocator 预留池单独计量。

增加 `seed_loads`、`recovered_requests`、`peak_live_recovered_cts`、store hit/eviction、解码/恢复耗时和按层统计。`seed_loads` 允许受缓存容量和执行顺序影响而大于 unique seed 数；manifest 中每个 seed 只写一份。GPU 图显式暴露 load、恢复和消费节点，以实际依赖证明分批调度。

## 7. 验收方案

### 7.1 局部与结构验收

将相关 case 加入现有 generator、layer、store 和 storage-score 测试，构建与生成目录统一放在 `/tmp/`。

1. **整数槽位检查**：按非对称权重标记覆盖全部请求、全部执行槽；对照现有 diagonal generator 的值序。覆盖 square/expand/reduce、跨 megablock 合并、末份 seed、`G>1`、重复 chunk、`u=1`、head padding、真实权重尾部、classifier、`d=32` 的非整数平方根 BSGS。检查独立值映射、正负旋转方向、seed 数和真实零 padding。
2. **实际 CKKS CPU/GPU 检查**：生成 seed，执行同一恢复 DAG，再与 `generate_diag_pt()` 对应值逐槽比较；检查虚部残差、use-site level、实际 scale 及 CPMM 输出。使用 `q_L != S` 的实际模数，输出相对目标 scale 偏差 ≤`1e-9`；局部权重恢复最大复数误差 `<1e-4`，CPMM 局部数值误差 `<1e-3`。
3. **参数契约**：覆盖 bias/affine、gamma、二/四次 Poly、Stockmeyer 多项式、softmax 和 LN 各阶段的秘密/公开混合路径；实际 add/sub 检查双方同 level/scale，mult 检查同 level，layer 输入输出均为 S。CPU 逐边 probe 与 GPU 生产 DAG 输出核验分别保留证据。
4. **来源和导出检查**：对实际模型列出逐项来源、可见性、recipe/参数 ID；修改一个秘密 bias、gamma 或 fused 系数后，相应离线参数发生变化，公开 recipe 保持不变。公开数学常数的 signature 计数为 0；所有秘密项可追溯到加密 element；Runner 可在独立目录中仅凭导出 bundle 和 CSV 启动。
5. **store 与调度检查**：manifest seed 数为 `ceil(Q/P)`，signature/manifest 的布局、level、scale、元素偏移/长度逐项一致；批次存活计数满足第 6 节约定。用小缓存触发 eviction 验证共享 seed 和重载正确性，保留原始 GPU 调度及内存日志。

### 7.2 模型、参考与数值门槛

| 模型 | ONNX | CSV | 全部待比较输出 |
| --- | --- | --- | --- |
| BERT | `runs/BERT/model/trained_poly.onnx` | `runs/BERT/input.csv` | `logits [1,2]`，全部两个 CLS logits |
| ViT | `runs/ViT/model/trained_poly.onnx` | `runs/ViT/patch_input.csv` | `output [1,197,10]`，全部 1,970 个值 |

配置分别以 `training/config/BERT_base_config.json` 和 `ViT_tiny_config.json` 的副本为基础，设置第 1 节运行模式，输出到 `/tmp/<experiment>/<model>/`。保留实际完整编译、导出、provision、GPU 运行、解密命令及源码、模型、CSV、配置、二进制 hash。

参考使用 `tools/evaluate_transformer_onnx_csv.py` 直接读取原始 ONNX 与 CSV，执行其自定义算子语义得到 NumPy float64 输出；BERT 保留 attention mask 的有效 44 token 与原始有限迭代定义。参考计算在私有验收侧读取完整模型参数。

在 `/tmp/<experiment>/<model>/` 下生成参考，并保留参考脚本 hash。以下以 `/tmp/pcm-seed-acceptance/BERT/` 为例，使用安装了 NumPy、ONNX 的 Python 环境：

```bash
mkdir -p /tmp/pcm-seed-acceptance/BERT
PYTHONDONTWRITEBYTECODE=1 python3 /home/linghm/project/latti-ai/tools/evaluate_transformer_onnx_csv.py \
  --onnx /home/linghm/project/latti-ai/runs/BERT/model/trained_poly.onnx \
  --input-csv /home/linghm/project/latti-ai/runs/BERT/input.csv \
  --output /tmp/pcm-seed-acceptance/BERT/reference.npz
```

ViT 使用同一命令替换模型、CSV 与输出路径。比较脚本按模型输出名与逻辑索引读取解密结果：

```python
assert actual.shape == reference.shape
assert np.isfinite(actual).all() and np.isfinite(reference).all()
max_abs_error = float(np.max(np.abs(actual - reference)))
assert max_abs_error < 0.2
```

本规范门槛为严格 **`<0.2`**。两模型均须通过完整压缩参数密文的 GPU runtime_lazy 路径，并保留主验收运行的正常退出码、解密输出及逐项误差。另记录编译后明文相对原始 ONNX 的误差、FHE 相对编译后明文的误差；原生 verifier 的容差、结论与退出码作为独立诊断记录，数值验收统一使用本节的原始 ONNX 门槛。

### 7.3 时间、内存与 bundle 计量

每个模型至少完成一次独立完整 GPU 执行，记录冷启动、GPU 型号、CUDA/SDK 版本、设备映射、线程数、CKKS 参数及 bootstrap 数。单独计量以下项目：

| 指标 | 采集口径 |
| --- | --- |
| FHE GPU 推理时间 | 进入 GPU 执行入口前同步设备，执行结束再次同步，记录两者之间的墙钟时间，包含 runtime_lazy 加载、解码、seed 恢复、PT 生成和 bootstrap；记清框架日志边界 |
| Runner 墙钟时间 | 从进程启动到输出密文写完，包含 context、图和参数 store 初始化 |
| Provision 时间/CPU 内存 | 离线加密与导出墙钟；该阶段进程树峰值 RSS 单独记录 |
| Runner CPU 内存 | 进程树 RSS 定期采样峰值，并记录 `/usr/bin/time -v` 最大 RSS、采样周期和适用范围 |
| GPU 内存 | SDK/RMM 实际分配与 reserved pool 采样峰值；同时采样目标 PID/设备的 `nvidia-smi` used memory，标注上下文、预留池、其他进程和采样间隔 |
| 参数密文 payload | 所有密文 element 的实际序列化长度之和，按线性 seed、其他秘密项、InputZero 分类 |
| 加密参数 bundle | `encrypted_model_parameters/` 全部文件的逻辑字节，包含容器与 manifest；记录文件数、element 数及 payload/容器开销 |
| 完整 Runner bundle | 参数、graph、公开配置、evaluation/rotation keys 的实际目录字节，并分类列出 key 和图大小 |
| 动态参数占用 | store 缓存字节、load/hit/eviction、恢复次数、活跃恢复 CT 峰值和恢复耗时 |

统一报告 bytes 与 GiB（`bytes/1024**3`），报告采样峰值时写明周期。FHE 图时间、Runner 墙钟和 Provision 时间分别呈现。布局压缩效果可用同一 PCM 最终 use-site level 的理论逐请求 payload `sum(Q*Bct(L))` 作账面分母，并清楚标为公式估算；验收以压缩模式真实落盘和真实 GPU 运行为依据。

实验根目录包含 `report.json`、命令与 hash、原始运行日志、参考 NPZ、输出密文和解密 CSV、逐项差值、来源审计、逐层布局/level/scale/字节统计、manifest 审计、CPU/GPU 内存采样及调度统计。编译缓存、Python bytecode、临时配置、构建目录和全部实测产物统一落在 `/tmp/`。`report.json` 为每个模型记录 `max_abs_error`、`fhe_wall_s`、`runner_wall_s`、`cpu_peak_rss_bytes`、`gpu_peak_used_bytes`、`gpu_peak_allocated_bytes`、`gpu_peak_reserved_bytes`、`parameter_payload_bytes` 和 `parameter_bundle_bytes`，缺失指标标为待补齐。两模型均完成运行、严格满足 `<0.2` 且计量完整后通过验收；实测前保持“待验收”状态。

## 8. 实现验收记录（2026-11-03）

证据根目录为 `/tmp/pcm-compact-acceptance/`，汇总文件为 `report.json`。两模型均从独立的 `isolated/<model>/` Runner bundle 读取公开 CSV，执行压缩参数密文的 GPU runtime_lazy 路径并正常退出。该目录只包含 Runner 导出文件；私有 H5、秘密 context 和来源审计位于 Provisioner 侧。主运行、解密与完整逐项比较保存在 `observed/<model>/`，编译、离线加密、manifest/计费核对和逐层恢复统计保存在 `final/<model>/`。

使用 RTX 5880 Ada 48 GiB、driver 535.230.02、`N=65536`、default scale `2^40`、`d=64`，ViT 为 3 heads、BERT 为 12 heads。两运行分别映射 GPU 0/1，OMP/OpenBLAS 均为 2 线程。完整 q/p 链、配置、源码和二进制 hash 见汇总文件。

| 指标 | ViT | BERT |
| --- | ---: | ---: |
| 与原始 ONNX/CSV 比较的输出数 | 1,970 | 2 |
| 最大绝对误差 | 0.027200430 | 0.041254927 |
| FHE GPU 入口墙钟（秒） | 389.841 | 1,232.08 |
| Runner 进程墙钟（秒） | 437.001 | 1,339.192 |
| Runner CPU 进程树峰值 RSS（GiB） | 31.275608 | 42.108307 |
| GPU pool 实际分配峰值（GiB） | 16.760507 | 13.868905 |
| GPU reserved pool 峰值（GiB） | 42.845875 | 42.842945 |
| PID GPU used 峰值（GiB） | 42.941406 | 42.941406 |
| 参数密文 payload（bytes） | 11,944,894,485 | 12,678,494,883 |
| 加密参数 bundle（bytes） | 11,947,281,530 | 12,683,890,542 |
| 完整 Runner bundle（bytes） | 15,991,634,657 | 18,212,295,053 |
| 线性 seed CT 数 | 248 | 3,504 |
| 其他秘密参数 CT 数 | 2,610 | 486 |
| InputZero CT 数 | 1 | 1 |
| 动态恢复请求数 | 28,608 | 224,256 |
| bootstrap layer 数 / CT 操作数 | 123 / 780 | 404 / 976 |
| 单层恢复 CT 峰值 | 8 | 8 |
| 全图恢复 CT 峰值 | 14 | 8 |
| 结束时活跃恢复 CT 数 | 0 | 0 |

CPU/PID GPU 采样周期为 0.5 秒，SDK pool 采样周期为 100 ms；pool CSV 的 GB 字段实际单位为 GiB，打印精度为小数点后六位。PID 占用包含预留池。FHE 时间覆盖加载、恢复、公开 PT 编码和 bootstrap；Runner 时间另包含初始化和输出序列化。Provision 时间分别为 162.346 秒和 158.587 秒，CPU 峰值分别为 33,398,349,824 和 34,679,824,384 bytes。

恢复存活和 CUDA event 计量采用 `/tmp/` 中的临时 SDK observer；完整补丁、恢复节点映射和补充环境记录均保留，SDK 源码已恢复。累计恢复 GPU event 时间分别为 105.168 和 557.586 秒，各 stream 可重叠；它们是操作时长之和。64 MiB store 缓存实测峰值均为 67,093,010 bytes，seed 实际 load 次数分别为 248 和 3,504；每份 seed 在 manifest 中只存一次。逐 level payload 与最终编译器计费逐字节一致，`unaccounted` 为空，公开参数 store 数为 0。

局部检查包括 8,585 条 seed 槽位/CKKS 恢复断言、CPU 混合参数算术契约 1,836 条断言、GPU 混合参数检查 1,802 条断言、store 重载与 metadata 检查，以及提交内容上的 16 项 Python 回归。来源扰动检查修改 ViT 的 bias、LN gamma 和 running-max，确认对应离线 H5 参数及融合 Poly 值变化，198 项公开 recipe 保持一致。

独立数值诊断：ViT 编译后明文相对原始 ONNX 最大差异为 `1.81924e-7`，FHE 相对编译后明文为 `0.027200444`。BERT 两者分别为 `0.111004049` 和 `0.152258976`。Runner CLI 会绕过 `--verify`，因此使用原生明文执行及 `InferenceClient` 解密的辅助程序，应用同一 `0.1` 容差规则；ViT 通过（退出码 0），BERT 未通过（退出码 1）。按第 7.2 节规定的原始 ONNX 参考，两模型均严格满足 `<0.2`，完成验收。
