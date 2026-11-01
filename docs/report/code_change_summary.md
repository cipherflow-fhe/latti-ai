# 代码修改记录

起始 revision：`3cbcc86`；实现与本报告在同一次 latti-ai 提交中，不 push。
实测源码的逐文件 SHA-256 及依赖 revision 见 `source_provenance.json`。

## 参数布局

| 路径 | 修改文件 | 存储与恢复语义 |
| --- | --- | --- |
| 普通 multiplexed | `model_generator/layers/multiplexed_conv2d_pack_layer.py`、`fhe_layers/multiplexed_conv2d_pack_layer.*` | 联合编号 output group / legacy input request / kernel；全向量按 subgrid 打包，选择、空间复制后应用 boundary mask |
| depthwise | 对应 `multiplexed_conv2d_pack_layer_depthwise.*` | 联合 channel group / kernel；保持通道 phase，不构造输入输出通道笛卡尔积 |
| inverse | Python/C++ `inverse_multiplexed*`、`kernel_seed_packing.py` | canonical 原始 kernel 去重，execution index 的输入选择和 boundary step 分别保留 |
| Dense | Python/C++ `dense_packed_layer.*`、`interleaved_seed.h` | 使用 invalid-fill 的不相交 residue tile，完整搬运不同权重；bias 选择后旋转回原稀疏支撑，在 fold 前加入 |
| polynomial | Python `poly_relu2d.py`、`poly_relu_base.py`，C++ `poly_relu2d.*` | 仅同 degree/channel group 打包，保留 BSGS 各 degree 的真实 scale 和 use-site level |

上表路径均相对 `inference/`。普通卷积的 input request 数取旧 generator 的
`legacy_weight_input_count()`，不是输入通道数。联合布局允许 `K=1` 沿输出组或
通道组压缩；尾组仍补零。inverse 保留 output/input 外轴，不启用空闲物理 block 多载体布局。

global stride 仍由 `training/run_compile.py --config` 读取。
`training/model_compiler/{components,pipeline,score}.py` 持久化选定布局，并按最终
use-site level 和实际序列化 CT 字节计费。候选不能节省 bytes、几何不适用或 seed
level 不可用时保持原布局；`[1,1]` 作为无 seed 的对照。最终 level 重算与 DP
计费不一致时报错。逐层计数、level、候选收益和未采用压缩的层见
`raw/*-layout-contract.json` 与 `layout_storage_audit.json`。

## level、scale 与公开 mask

设参数的 use-site 为 `(L,S)`：选择与复制用 `(L+1,S)` seed，选择 PT 在
`L+1`、scale `q(L+1)`，CT×PT 后 rescale 恢复 `(L,S)`。
需 boundary mask 时 seed 为 `(L+2,S)`；依次使用 scale 为 `q(L+2)` 和
`q(L+1)` 的 PT，各 rescale 一次。旋转加法不改 scale，恢复不增加 activation
算子的 `level_cost`。Dense tile 和同 degree poly 恢复均只增加参数分支的一层。
两组独立编译的 DP 会因 storage score 改变而选择不同的 level/bootstrap 计划；
逐层输入/use-site level 均保存在 audit 中，耗时对照包含这一影响。

weight seed 保留旧权重的 `q(L)` scale；bias 保留 default scale；poly 保留各 degree
经真实模数漂移计算的 scale，不能因 level 相等就混合。导出的 signature、manifest
和实际 ciphertext 交叉校验 shape/level/scale/layout；错误拒绝执行。

`deploy_cmds.py`、`inference_runner.cpp` 将 depthwise structural mask 和 inverse
repack mask 改为 Runner `encode_pt`，沿用原 `(ct_idx, local_idx)` 和 source-position
旋转关系、原 scale 与 rescale。seed/tile mask 同样由公开几何生成，不进入参数密文包。
普通卷积已有的 public mask 继续使用；client 模式的 mask signature 保持兼容。

## runtime-lazy 与工具

恢复操作显式写入 MegaAG，`call_param_ct*()` 继续使用
`materialize_encrypted_param()` → `load_encrypted_param_ct`。每个 canonical seed
只有一个 load 节点；恢复分支产生新的 CT，不能原地修改共享 seed。
`encrypted_parameter_store.*` 对新索引和旧无索引 container 均逐元素读取，lazy
cache 使用 64 MiB 有界 LRU，计费口径为序列化 CT payload；已交给 executor 的
`shared_ptr` 生命周期不受 eviction 影响。日志输出 loads/hits/evictions/peak bytes；
此限额不代表整个进程或 GPU 内存限额。eager 使用同一布局与 level/scale 校验。

`inference_provisioner.cpp` 每个 canonical seed 只加密和写入一次，
`runner_bundle_io.*` 校验布局契约，仍接受缺少新字段的旧 manifest。
`inference_process.cpp` 根据 metadata 初始化相同的 C++ 布局。

新增 `tools/benchmark_server_provisioned_runner.py`、`evaluate_onnx_csv.py` 和
`summarize_parameter_bundle.py`，分别执行隔离验收、原 ONNX 完整明文 logits
评估（含 polynomial operator）、实际 container payload 与目录大小统计。
`decrypt_runner_output.cpp --output-csv` 输出全部 logits；命令、耗时范围、return code、
原始值、监控 CSV 和源码 hash 一并保存。

## 检查与覆盖范围

Python 45 项测试通过，C++ CKKS 7 cases / 1,919 assertions 和参数 store
2 cases / 54 assertions 通过（包括 JSON 对象键顺序不影响布局比较的回归测试）。命令与日志见 `raw/validation_commands.md`、
`raw/python-tests.log`、`raw/ckks-tests.log`、`raw/store-tests.log`。
覆盖不同 output/input/channel 值、K=1/3×3、stride/skip、旋转方向、尾组、两个物理
block、Dense bias pre-fold、degree scale 漂移、eager/lazy、旧 manifest 与 cache eviction。

GPU 实测范围为规范指定的 ResNet20、MobileNetV2 和各一份固定输入，不能据此声称
所有网络/shape 或 client 模式均已通过端到端验证。inverse canonical 另有真实 CKKS
单元测试。本次没有扩展或验收 concat/upsample/padding 等算子的 public mask 路径；
这些路径不在两模型的端到端覆盖范围内。mult-scalar、普通/inverse bias 保留密文原布局；未压缩层和
各 degree 的选择记录见逐层 audit。完整 GPU 验收结论见 `acceptance_summary.md`。

## 临时依赖修改

验收 build 使用 GPU=ON、DEV=ON。lattisense 的 DEV CPU monitor 调用临时修正为
`MemoryMonitor::next_csv_path(".", "mem_usage_cpu")`，仅修复监控构建；lattigo 的两个
Go SDK header 由构建自动生成差异。二者无计算逻辑修改，验收后均恢复，不提交依赖文件
或子模块指针。重建 DEV 监控配置时需要相同的临时一行修复；普通 GPU 构建不依赖它。
