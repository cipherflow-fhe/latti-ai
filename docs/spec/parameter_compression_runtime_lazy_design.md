# 参数密文压缩与 runtime-lazy GPU 验收设计规范

状态：此前两模型 GPU 验收见[验收记录](../report/acceptance_summary.md)；
`[16,16]` 配置、0D Dense tile 和大图 Poly 扩展已通过编译及局部密文测试；
Retina 参数加密导出成功，整网 GPU 推理因内存不足未完成，尚未通过验收。
本文基于此前的 depthwise / inverse seed 和非卷积参数调研，定义实现和验收要求。

本文的目标是减少 `server_provisioned_runner`、
`encrypted_offline`、`runtime_lazy` 模式的参数密文 bundle，同时保持现有
CKKS 参数、主图 level、明文输入接口和输出语义。它不要求所有合法 `(a,b)`
都达到 `a*b` 倍压缩，也不把理论 slot 下界当作实现结果。每个层先计算实际
字节收益，只有收益为正且能复用已有恢复计算时才启用候选布局。

## 1. 设计决策

### 1.1 选定实现

全局配置仍由 `training/run_compile.py --config` 读取：

```json
{
  "encrypted_parameter_storage_weight": 1.0,
  "encrypted_parameter_storage_normalization_bytes": 1073741824,
  "seed_weight_subgrid_stride": [16, 16]
}
```

`seed_weight_subgrid_stride=[a,b]` 只提供空间恢复的最大容量
`G=a*b`。编译器不得用它直接声称每层压缩 `G` 倍，也不得让 DP 搜索任意
`(a,b)`。以下布局就是规范要求的实现：

| 参数 | 实现 |
| --- | --- |
| 普通 multiplexed 卷积 | 输入 block、输出组、kernel 联合编号 |
| multiplexed depthwise | 通道组、kernel 联合编号 |
| inverse 卷积 | canonical kernel 去重，保留每个 execution index 的 boundary step |
| 结构性 mask | Runner 生成 public PT |
| Dense 权重 | 利用 invalid-fill 空洞的 sparse tile 打包 |
| Dense bias | 复用 Dense 的 tile 网格，在 fold 前恢复稀疏支撑 |
| 多项式系数 | 同一 degree、几何、level 和 scale 内按 channel group 打包 |
| mult-scalar | 保持现有密文参数布局 |

编译和导出仍须按实际 seed level 计算存储量：

```text
compressed_bytes = Σ seed_count * Bct(seed_level)
legacy_bytes     = Σ legacy_count * Bct(use_site_level)
```

`Bct` 使用当前 CKKS 参数和序列化版本的校准值。metadata、公开 PT 编码、
旋转 key、运行时 CT×PT、rescale 和缓存峰值单独统计；不能把这些成本藏在
密文数量中，也不能因密文数量减少就默认运行时间减少。某层的空间尺寸、支撑集
或 modulus chain 不满足该布局的结构约束时，保留该层现有参数布局；这只处理
几何不适用的输入，不改变其他层已经选定的实现。

### 1.2 不跨秘密参数共享，不改变激活布局

一个 seed 只能承载同一 layer instance、CKKS 参数集、目标 scale、目标 use-site
level 和兼容 slot geometry 的完整参数向量。不同参数值可以放入不同子网格，
不需要数值相等；但不能因两个私有值碰巧相等而公开共享。

恢复前保存原有完整向量的零位置和 channel phase。权重在恢复后仍进入原来的
乘法、求和、边界 mask、fold 和 bias 路径；不改变 feature packing、输入旋转或
输出顺序。`client_encrypted_input` 和非 server-provisioned 部署保持原行为。

## 2. 卷积 seed 布局

### 2.1 普通 multiplexed：联合请求布局

当前 input-block-only 布局在 `K=1` 或 `J=1` 时留下大量尾部空位。新布局不修改
activation，只改变 canonical weight 请求编号：

```text
u = ((output_group * J) + input_request) * K + kernel_index
q = floor(u / G)
s = u mod G
```

`J` 必须由现有 `legacy_weight_input_count()` 重建，不能把原始输入通道数当作
`J`；`output_group` 使用当前 weight PT 的外层分组，`K` 使用实际 kernel 枚举。
每个 `U_u` 仍由现有完整 weight generator 产生，seed `Z_q` 仅把不同 `U_u`
放入不相交的空间子网格：

```text
Z_q = Σ_s U_(q*G+s) * M_s
request(u) = restore(Z_q, M_s, subgrid=s)
```

边界 mask 在空间复制之后应用。seed mask 和 boundary mask 使用现有
`interleaved_seed` 旋转方向、物理 skip 和 `q(level)` scale；不能按逻辑像素步长
重写旋转。

`K=1` 的普通卷积是第一优先级，因为输出组可以填补 input-block 轴空位；
`K>1` 也只有在实际字节计算为正时启用。若联合方案需要额外 boundary level，
而当前 level chain 无法容纳，则该层保持现有 input-block-only 或 legacy 布局。

### 2.2 depthwise：通道组与 kernel 联合

depthwise 没有独立的输入求和轴，不能人为构造 `Ci*Co` 个组合。令当前权重
CT 通道组为 `g`、kernel 枚举为 `k`：

```text
u = g*K + k
q = floor(u/G)
s = u mod G
```

`g` 的完整通道 phase、物理 block 和 skip 模板都保留在 `U_(g,k)` 中。恢复后
仍只与对应 `x[g]` 相乘。`K=1` 且有多个通道组时可以压缩；只有一个
`(g,k)` 请求时必须关闭 seed，避免为了多一次 level 反而增大 bundle。

Python 和 C++ 均复用现有 `pack_kernels()` 的 mask/复制实现，新增的只是
canonical `g,k -> q,s` 路由和完整向量生成回调。不能把 kernel-only 的
`ceil(K/G)` 再次除以 `G`。

### 2.3 inverse：保守启用

inverse 的 `output_step` 枚举可能多次引用同一个原始 kernel 参数。本设计在
参数生成层把

```text
e -> (co, ci, original_kernel, input_selection, boundary_step)
```

分开：只有 `original_kernel` 完全相同的请求共享 canonical seed；每个 `e`
仍保留自己的输入选择和 boundary mask。暂不启用 inverse 的空闲物理 block
多载体 packing，也不在没有实测收益前增大 `(a,b)`。这样能消除明确的重复参数，
又不改变 inverse 的计算枚举。

### 2.4 level 和 scale

目标 use-site level 记为 `L`，参数目标 scale 记为 `S`：

| 情况 | seed | 选择 PT | 恢复后 |
| --- | --- | --- | --- |
| 只需一次选择和复制 | level `L+1`, scale `S` | `q(L+1)` | level `L`, scale `S` |
| 还需一次 boundary mask | level `L+2`, scale `S` | 依次为 `q(L+2)`、`q(L+1)` | level `L`, scale `S` |

旋转加法不改变 scale。seed 不能以 `default_scale` 替代原 `S`，也不能为了
节省 bundle 抬高 feature 主图 level。最终图 level 重新计算后，若与 DP 计费不符，
编译必须重新评分或报错；不得导出一份按旧 level 计价的 bundle。

## 3. 非卷积参数布局

### 3.1 结构性 mask 公开 PT

在 `server_provisioned_runner` 中，所有由公开 shape、stride、padding、skip、
通道路由和 seed index 决定的 mask 都由 Runner 的 `encode_pt` executor 生成，
不进入 `encrypted_parameters`、offline 参数列表或 encrypted parameter store。
这包括当前 depthwise/inverse 的 `convm_`，以及后续接入的 concat、repack、
upsample 和 padding mask。

depthwise 的 mask 索引必须保持 `(ct_idx, local_idx)`，并复用
`MultiplexedConv2DPackedLayerDepthwise::generate_mask_pt_for_indices()` 的
source-position 结果。原 CT×CT 路径改为 CT×PT，保留原 rescale 和旋转；PT
scale 是 mask 所在 level 的 `q`。Runner 不能用普通卷积 mask 的索引公式替代它。

`client_encrypted_input` 继续使用原 signature；普通 multiplexed 已有的 public
mask 行为保持兼容。Runner-local mask 只缓存 ringt/PT，不写回 bundle，不需要
H5 权重或 secret key。

### 3.2 Dense 权重：利用现有 invalid-fill 空洞

Dense 不假设空间位置权重相同，而是平移完整稀疏向量。编译器从 Dense 参数
metadata 得到每个物理 block 的有效支撑集 `S`，选择不相交的 tile 偏移
`delta_s`：

```text
Z_q = Σ_s Rot(U_(q*C+s), -delta_s)
U_u = Rot(Z_q * indicator(S+delta_s), +delta_s)
```

对当前 MobileNet classifier，`P=128*128=16384`，每 CT 有两个 block，每个
block有效支撑为 `16*16`，一个 block 可放 `8*8=64` 个 tile；因此 `[500,6]`
的 3,000 个 weight CT 候选为 `ceil(3000/64)=47` 个 seed。该布局只搬运不同
权重值，不广播或删除独立参数。`modified_scale`、两个 block 和越界输入的
零值必须保持。

Dense bias 复用同一 tile 网格，恢复后仍在原来的 fold 之前以稀疏支撑相加；
当前 500 个 bias 向量候选为 `ceil(500/64)=8` 个 seed。不能把 bias 广播到
整个 block，否则 14 轮 fold 会把它累加多次。

如果某层支撑集没有足够不相交 tile，或新增选择/rescale/旋转导致字节收益为负，
则保留当前 Dense 参数布局。布局参数和 tile 数必须写入逐层 metadata，不能让
Runner 猜测。

concat 后没有 `special_info` 的 0D Dense，当 `pack_num=1` 且
`skip=slots` 时，也复用此 sparse tile 布局。其有效支撑只有 slot 0，使用
`input_shape=[1,1]`、`skip=invalid_fill=[1,slots]`，容量为 `slots`。
按 `(output,input)` 顺序打包 weight，按 output 顺序打包 bias；恢复后仍执行
原有 0D 求和，不增加空间 fold，也不广播输出到 invalid 位置。
输入 invalid 位置可含无效数字；恢复出的 weight 在这些位置为零。
每个 weight seed 的 scale 保持原 `modified_scale`，level 为 `L+1`；
bias seed 的 scale 为原 `default_scale`，level 为 `L`。选择 PT 分别以
`q(L+1)`、`q(L)` 编码，rescale 后恢复到 weight `L`、bias `L-1`。
评分读取 DAG 的实际 0D skip，与导出图一致，不能使用 `sp_info` 的默认 skip。

### 3.3 多项式系数：只在同 degree 且 scale 相容时打包

固定 `(layer_id, degree, geometry, use_site_level, coefficient_scale)`，将 channel
group 作为 seed 的 logical request。保留 BSGS 的 degree 外轴和每 degree 的
实际 scale；不能把五个 degree 直接展平，也不能仅因 level 相同就假设 scale 相同。

大图 multiplexed 2D Poly 同样按实际物理 block 打包：
`seed_shape=input_shape/block_expansion`，mask、复制旋转与 metadata 都使用此
shape，行步长为 `skip[0]*seed_shape[1]*skip[1]`。每 degree 的请求数为
`ceil(channels/pack_num)*prod(block_expansion)`，保持原有空间 block 和通道枚举。
同一通道的不同空间 block 仍按各自请求恢复，不改变主图布局。
seed level 为该 degree 的 use-site level 加一，scale 复用原 BSGS coefficient
scale；选择 PT 使用 seed level 的 `q`。恢复不增加 feature 主图乘法深度。

`zero_skip`、ordinary packing 或物理 block 不满足 subgrid 整除条件的层继续
使用现有布局。已有 MobileNet 样本中
35 个 order-4 层的候选是 1,245 个 CT 到 200 个 seed 的静态估算，实际导出后
仍须按 manifest 重新核对。

### 3.4 mult-scalar 和其余 bias

`MultScalarLayer` 的 `weight` 是通道数组，不是可公开的单 scalar。本设计保留
其 encrypted CT 和原 scale；只允许后续在同一参数 identity、几何、level/scale
完全相容时做 canonical dedup。普通卷积 bias、inverse bias 和其它独立 scale/shift
同样先保留原布局，待逐层支撑集验证后再接入，不因名称包含 mask 或 scalar 就公开。

## 4. 代码修改要求

### 4.1 编译配置与 Python graph

修改以下职责，保持旧配置默认行为：

| 文件/模块 | 必须修改 |
| --- | --- |
| `training/run_compile.py`、`training/model_compiler/pipeline.py` | 读取并校验全局 `(a,b)`；将其写入适用层 metadata；`[1,1]` 保持旧布局 |
| `training/model_compiler/score.py` | 对每个候选计算实际 canonical seed count、seed level 和 bytes；storage score 与最终 level 一致；不在 DP 中假设固定 `G` 倍压缩 |
| `inference/model_generator/layers/kernel_seed_packing.py` | 提供有限布局的 `u -> (q,s)`、有效支撑和 level cost helper；保留旧 kernel-only 路径 |
| `inference/model_generator/layers/multiplexed_conv2d_pack_layer.py` | 接入联合请求布局，更新 weight nodes、seed masks、`call_param_ct()` 和 `call_param_ct_plaintext_input()` |
| `inference/model_generator/layers/multiplexed_conv2d_pack_layer_depthwise.py` | 接入 `g,k` 联合路由；strided mask 走 public PT；两个 `call_param_ct*()` 保持同一恢复逻辑 |
| `inference/model_generator/layers/inverse_multiplexed*_layer.py` | 保留 execution index，接入 canonical kernel 去重和逐请求 boundary mask |
| `inference/model_generator/layers/dense_packed_layer.py` | 增加 Dense sparse tile 的 node、恢复和计数；bias 使用同一 tile geometry |
| `inference/model_generator/layers/poly_relu*_layer.py` | 同 degree/channel group seed；逐 degree 保留真实 level/scale |
| `inference/model_generator/deploy_cmds.py` | 将逐层布局写入 MegaAG；server runner 的 public mask 不生成 encrypted argument |
| `inference/model_generator/layers/encrypted_param_ops.py` | 所有 canonical CT 通过 `materialize_encrypted_param()` 加载；同一 seed 的多个恢复分支不得原地修改共享 CT |

Python graph 必须在导出时固定 seed argument 的 shape、level、scale、布局几何、
outer axes、group size 和 tile geometry；Runner 不读取训练配置推断这些字段。
普通 multiplexed 的 public seed mask 按实际几何和编码 level 共享：subgrid mask
不依赖 kernel index，boundary mask 不依赖 subgrid。不得按每个私有参数请求重复
创建相同的公开 mask 节点，否则无输入的编码节点会提前累积大量主机内存。

### 4.2 C++ 参数生成、Provisioner 和 Runner

| 文件/模块 | 必须修改 |
| --- | --- |
| `inference/fhe_layers/interleaved_seed.h` | 提供有限 joint request 的公开 subgrid、tile、旋转和 boundary mask helper；复用统一 `Rot` 约定 |
| `inference/fhe_layers/multiplexed_conv2d_pack_layer*.cpp` | 根据 metadata 生成 canonical seed，并恢复成旧完整 weight；depthwise mask 生成保持 `(ct_idx,local_idx)` |
| `inference/fhe_layers/inverse_multiplexed_conv2d_layer*.cpp` | 将 execution index 映射到 canonical kernel，按请求保留 boundary step |
| `inference/fhe_layers/dense_packed_layer.cpp` | 生成/恢复 sparse tile seed，保持 `modified_scale` 和 bias pre-fold 支撑 |
| `inference/fhe_layers/poly_relu*.cpp` | 复用 BSGS coefficient scale，生成同 degree/channel seed |
| `inference/interface/inference_provisioner.cpp` | 按逐层布局几何和 `param_kind` 解释 canonical shape；每个 seed 只加密和写入一次；public mask 不写 manifest |
| `inference/interface/inference_runner.cpp` | `encode_pt` 支持 depthwise/inverse ordinary mask、seed/tile mask；CT×PT 只替换结构 mask，不拷回 GPU |
| `inference/interface/encrypted_parameter_store.*` | runtime-lazy 按 canonical argument/element 加载；共享 seed 只读；按 graph release/refcount 或有界 LRU 释放无后继消费者的 CT，并记录 cache hit/miss |
| `inference/interface/runner_bundle_io.*` | 检查 bundle/manifest 的布局 metadata；不把 public PT 序列化为密文 |

Provisioner 必须拒绝 shape、level 或 scale 不匹配的 seed，而不是静默按错误索引
解释。没有新布局字段的旧 manifest 按已有布局读取。
Runner 在 public mask 生成失败、缺少必要旋转 key 或 modulus chain 不足时应报错或
按编译器已经记录的现有布局执行，不能用错误 mask 继续计算。

### 4.3 完整输出与验收工具

新增或修改以下工具：

1. `tools/benchmark_server_provisioned_runner.py`：生成独立 task，执行 compile、
   MegaAG、Provisioner、GPU Runner 和 decrypt；默认 `runtime_lazy`，记录命令、
   git revision、CUDA/device、CKKS 参数、wall clock 和 return code。
2. `tools/evaluate_onnx_csv.py`：以原始 `.onnx` 和同一份 `.csv` 产生完整 plaintext
   logits。必须支持项目的自定义 polynomial operator；不能只比较 top-1 或前十个值。
3. `tools/summarize_parameter_bundle.py`：从 `manifest.json` 汇总每种 `param_kind`
   的 CT 数、元素字节数、总 encrypted payload、完整 runner 目录和完整 task 目录。
4. 解密工具需要提供完整 logits 文件（已有 `decrypt_all.cpp` 的 CSV 输出行为可复用），
   并计算 `max_abs`、`mean_abs`、`RMSE`、`max_rel`、cosine similarity 和 top-1。

工具只写 `docs/report/` 和用户指定的临时 task 目录，不覆盖既有 runs。

## 5. runtime-lazy 约束

`call_param_ct()` 和 `call_param_ct_plaintext_input()` 不需要为每个参数新增一个
custom compute。它们应继续调用：

```text
materialize_encrypted_param()
  -> load_encrypted_param_ct(store, canonical_arg_id, flat_index)
  -> 原有 CT×CT 或 PT×CT 计算
```

custom executor 只用于 Runner-local public `encode_pt` mask 和已有的
`load_encrypted_param_ct`。seed 恢复必须在 graph 中显式表示，保证 lazy loader 能
看到每个 canonical seed 的首次消费者；不能在 custom compute 内偷偷读 H5 或展开整包。

共享 seed 的恢复分支各自执行 mask、rescale 和 rotate；禁止一个分支原地 rescale
后把同一个 CT 交给另一个分支。参数 store 至少报告：总 canonical loads、cache hits、
cache evictions、缓存峰值 bytes。eager 与 runtime-lazy 使用同一 layout，只有加载
时机不同。

## 6. `docs/report/` 记录要求

实现开始时创建 `docs/report/` 子目录，并提交以下文件：

```text
docs/report/README.md
docs/report/code_change_summary.md
docs/report/acceptance_summary.md
docs/report/resnet20_runtime_lazy.json
docs/report/mobilenet_runtime_lazy.json
```

`code_change_summary.md` 记录修改的 commit、文件、布局几何、参数计数/level
变化、未采用压缩的层和测试结果；`acceptance_summary.md` 只引用 JSON/CSV 中的实测值，
说明任何失败、重试和未覆盖的参数类型。原始日志用于验收过程，结论确认后可以清理；提交的 JSON/Markdown 报告必须保留可复核的数值、命令结果和来源 hash。

每个模型 JSON 至少包含：

```json
{
  "model": "resnet20",
  "onnx": "runs/cifar10/model/trained_poly.onnx",
  "input_csv": "examples/test_cifar10/task/client/img.csv",
  "deployment_mode": "server_provisioned_runner",
  "parameter_mode": "encrypted_offline",
  "parameter_loading_mode": "runtime_lazy",
  "gpu": true,
  "ckks_parameter": "...",
  "seed_stride": [4, 4],
  "encrypted_parameter_payload_bytes": 0,
  "runner_bundle_bytes": 0,
  "task_directory_bytes": 0,
  "cpu_peak_rss_bytes": 0,
  "gpu_peak_used_bytes": 0,
  "runner_wall_ms": 0,
  "decrypt_wall_ms": 0,
  "e2e_fhe_wall_ms": 0,
  "plaintext_wall_ms": 0,
  "metrics": {
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0,
    "rmse": 0.0,
    "max_rel_error": 0.0,
    "cosine_similarity": 1.0,
    "top1_match": true
  },
  "commands": [],
  "git_revision": "..."
}
```

其中 `encrypted_parameter_payload_bytes` 是 manifest 中所有 CT element 的和，
不含 evaluation keys；`runner_bundle_bytes` 是 `task/runner` 的完整目录大小；
`task_directory_bytes` 是整个独立 task 的大小。三者必须同时报告，不能把其中一个
冒充 bundle 大小。

## 7. GPU runtime-lazy 验收

### 7.1 固定输入和命令

只验收以下两个模型，所有候选与 legacy 对照使用同一个 ONNX、同一个 CSV 和同一
CKKS 配置：

| 模型 | ONNX | CSV | 输出 |
| --- | --- | --- | --- |
| ResNet20 | `runs/cifar10/model/trained_poly.onnx` | `examples/test_cifar10/task/client/img.csv` | 10 logits |
| MobileNetV2 | `runs/imagenet/model/trained_poly.onnx` | `examples/test_imagenet/task/client/img.csv` | 1000 logits |

编译必须显式传入配置：

```bash
PYTHONPATH=inference/lattisense python training/run_compile.py \
  --input <model>.onnx --output <isolated-output> --style multiplexed \
  --deployment-mode server_provisioned_runner \
  --config training/config/storage_aware_config.json
python inference/interface/gen_mega_ag.py \
  --task-dir <isolated-output>/task \
  --deployment-mode server_provisioned_runner \
  --parameter-loading-mode runtime_lazy
./build/tools/provision_encrypted_runner_bundle \
  --task-dir <isolated-output>/task --out <isolated-output>/task/runner
./build/examples/inference \
  --task-dir <isolated-output>/task/runner \
  --deployment-mode server_provisioned_runner \
  --input <input-name>=<model>.csv \
  --output-cipher <isolated-output>/output.ct --gpu
./build/tools/decrypt_runner_output \
  --task-dir <isolated-output>/task/provisioner \
  --cipher <isolated-output>/output.ct
```

实际验收命令由 benchmark 工具展开并写入 report；所有 task 必须使用独立目录，
验收结束后按磁盘策略删除，不提交 task、secret key 或模型参数文件。
GPU build 必须打开 `INFERENCE_SDK_ENABLE_GPU=ON`；为获得 CPU/GPU 采样报告，
验收 build 打开 `LATTISENSE_DEV=ON`，并保留 monitor CSV。

### 7.2 正确性和结构检查

对每个模型分别运行 legacy 与压缩实现，且各模式至少 warm-up 1 次、正式运行
3 次；报告中采用中位数，保留每次原始值。必须通过：

1. `task_config.json` 标记 `server_provisioned_runner`、`plaintext` input、
   `encrypted_offline` parameter 和 `runtime_lazy`；
2. seed manifest 的布局几何、shape、level、scale、stride 与逐层 graph
   一致；public mask 不出现在 encrypted parameter manifest；
3. `load_encrypted_param_ct` 的 canonical argument/element 数量等于导出计数，
   不出现越界或同一 seed 的错误 level；
4. GPU Runner 成功完成，日志没有 CPU fallback，`mem_usage_gpu.csv` 非空，
   `mem_usage_cpu.csv` 或 VmHWM 记录非空；
5. 解密产生完整 logits，所有值有限，压缩实现与 plaintext 的 top-1 相同。

### 7.3 数值误差门槛

对 plaintext logits `p` 和解密 FHE logits `f` 计算：

```text
max_abs = max_i |f_i-p_i|
mean_abs = mean_i |f_i-p_i|
RMSE = sqrt(mean_i (f_i-p_i)^2)
max_rel = max_i |f_i-p_i| / max(|p_i|, 1e-12)
cosine = dot(f,p)/(||f||*||p||)
```

压缩实现必须满足：`max_abs <= 1e-2`、`RMSE <= 1e-3`、`cosine >= 0.9999`、
`top1_match=true`，并且 `max_abs_compressed <= 2*max_abs_legacy + 1e-6`。
若模型的已知 CKKS baseline 不能满足绝对门槛，必须先记录 baseline、参数和
解释，经评审后更新本规范；不能只删除门槛或只比较 top-1。

### 7.4 时间、内存和 bundle 指标

每个模型必须报告：

- `runner_wall_ms`：从 GPU Runner 开始加载输入到输出 ciphertext 写完；
- `decrypt_wall_ms`：解密并写完整 logits；
- `e2e_fhe_wall_ms`：Runner + decrypt 的端到端 wall time；
- `plaintext_wall_ms`：原始 ONNX + 同一 CSV 的完整 plaintext logits 时间；
- `cpu_peak_rss_bytes`：monitor 的 peak RSS/VmHWM；
- `gpu_peak_used_bytes`：GPU memory monitor 的 peak pool used；
- encrypted parameter payload、完整 runner bundle、完整 task 目录大小；
- canonical load、cache hit、cache eviction 和恢复操作计数。

GPU/CPU 时间必须使用同一输入和 warm-up 规则，不能把编译、Provisioner 加密、
CUDA 初始化和重复模型加载混在某一边。若工具无法分离初始化，必须同时报告原始
wall time 和计时范围。

## 8. 单元测试和完成判据

提交前必须增加或更新以下测试：

1. seed layout 单元测试：`K=1/3x3`、`M<G`、`M%G!=0`、不适用布局、不同
   output/input/channel 值、尾部 zero padding、stride/skip 和旋转正负号；
2. Dense tile/bias 测试：两个物理 block、invalid-fill 空洞、不同权重值、尾组、
   bias fold 前恢复；逐槽比较原向量和恢复向量；
3. poly 测试：不同 degree 的 level/scale 不混用，同 degree 相容才共享；
4. graph/provisioner 测试：canonical manifest shape/level、public mask 不落盘、
   runtime-lazy load 节点存在且 eager/lazy 参数语义一致；
5. 现有 `test_interleaved_seed_weights.py`、server runner 生成测试和 storage
   score 测试全部通过。

完成设计实现的必要条件是：ResNet20 和 MobileNet 的 GPU runtime-lazy 两条
压缩任务均通过结构检查、完整 FHE 推理、解密和数值门槛，并在
`docs/report/` 中具备代码修改摘要、实测记录、原始日志和内存 CSV。任何未实现的
参数类型、未采用压缩的层、未覆盖的输入 shape 或失败重试都必须在报告中列出；没有报告或
只有静态估算不能标记验收完成。

## 9. 明确不做的事情

- 不承诺每个层、每个 `(a,b)` 达到 `a*b` 倍；`G` 是容量上限，不是压缩率保证。
- 不把 Dense 的独立空间权重当成卷积常数，不公开 learned weight、bias、poly 或
  mult-scalar。
- 不做跨层 seed、跨 degree 的 scale 对齐、inverse 空闲 block 多载体、
  gzip/zstd 密文压缩、量化或低秩近似。
- 不用静态 manifest 字节估算替代真实 bundle；不使用只打印前十项的输出作为误差
  验证；不以 CPU 或 plaintext FHE 路径代替本次 GPU runtime-lazy 验收。
