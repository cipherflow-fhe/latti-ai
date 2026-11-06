# TAA v6 功能移植设计总则

日期：2026-11-06。状态：已实现并完成本组功能验收，六任务 14 配置全部通过；发布来源、测量与已知基线测试失败见[验收记录](../research/taa_v6_acceptance_report.md)。

本组规范定义 `feat/taa-adapt` 的功能范围、实现约束与验收条件。源码基点见第 2 节，各专题规范直接引用相关 v6 源码，不依赖独立调研文档；历史性能数字不作为当前验收依据。

## 1. 唯一基准与最小修改原则

1. **Lattigo v6 及 `feat/taa-adapt` 分支当前 SDK 的语义、接口、参数、metadata 和序列化是唯一基准。** 该分支的切出基点见第 2 节。`feat/streaming-key-generation` 只提供用户行为、故障场景和测试意图，不构成接口或字节协议约束。
2. 遵循 `karpathy-guidelines`：每项实现改动必须对应本组规范中的功能或验收；优先扩展现有 v6 类型和局部流程，不引入通用存储框架、通用作业调度器或第二套密码学状态。
3. **不为来源分支编写兼容性 wrapper。** 不恢复 `CkksBtpContext`、advanced context、旧 handle/error ABI、旧压缩密文格式、`ShallowCopy`，不增加旧 preset 到 v6 的伪等价映射，也不为旧 WASM API 维持转发层。必要的 CGO/Worker 边界适配直接服务当前 v6 对象和新文件协议。
4. 最小修改不减少已要求的功能。逐键 keygen、持久化、任务前预生成、浏览器加解密、OPFS/ZIP64、批处理、资源释放和失败传播都必须闭环；可以按依赖分阶段实现。
5. 不改变密码学算法、模型、Q/P、N、scale 或精度阈值来换取通过。不引入旧 HEonGPU/FPGA 子模块，不顺带升级到未被父仓库锁定的远程 main。

若发现本规范与锁定源码语义冲突，应修正规范或明确提出必要的 v6 增量接口；不能通过旧语义兼容层掩盖冲突。

## 2. 分支与源码范围

四个仓库当前均使用 `feat/taa-adapt`，切出基点如下。顶层基点对应 `feat/v6-adapt-encparams`；子模块基点来自其父仓库 gitlink，并非各自独立远程分支的最新 tip。

本组规范中的实施分支均明确指 `feat/taa-adapt`；`feat/v6-adapt-encparams` 是切出基线，`feat/streaming-key-generation` 是功能来源分支。

| 仓库 | 基点 |
| --- | --- |
| latti-ai | `9619f422cfa48ccafb5dc4eaaaff6971db7dea4c` |
| inference/lattisense | `83e58218437eee1afb60f47f786e80a020eede63` |
| inference/lattisense/fhe_ops_lib/lattigo | `22bff324e146968c186106524c06cea607462b05` |
| inference/lattisense/backends/lattisense-gpu-core | `aea1ef873aed3e199aa785a0e4f42d8a42b9351f` |

构建依赖固定 gitlink，CMake configure 不创建/切换分支，不要求存在 `origin/feat/taa-adapt`。开发分支提示不能代替可复现的提交锁定。第三方 GPU-FFT/GPU-NTT/RNGonGPU 无本次功能差额。

## 3. 规范划分与共享契约

| 规范 | 负责的决定 |
| --- | --- |
| [流式密钥与持久化](taa_v6_streaming_context_design.md) | 共享纯 Go 核心、v6 BTP、eval/full、native setup/file import |
| [浏览器客户端与协议](taa_v6_browser_client_design.md) | 完整参数 profile、packing/wire、Worker v2、OPFS 产物发布与鲁棒性 |
| [OPFS ZIP64](taa_v6_opfs_zip64_design.md) | 分块归档读写、64 位大小/偏移、中央目录、EOCD64、磁盘与内存预算 |
| [任务流程与交付](taa_v6_browser_workflow_design.md) | 单/三输入批处理、预后处理、真实 server 互通、WASM 0.2.0 |
| [运行时资源与故障](taa_v6_runtime_resource_design.md) | host/device 生命周期、进程全局 pool、环境预算、CPU/GPU 失败通道 |

以下契约跨文档共用，不由各层另作解释：

- eval 文件沿用 `CKKSCTX1`，包含 PK 和所需评估键，不包含 SK；full 文件沿用现有双 blob 容器，第一段仅参数/PK/标志，第二段独立 SK，不含评估键。
- 参数与所需键按实际值匹配；profile ID、`param0`、目录名和布局相同都不能单独证明可复用。
- 浏览器默认预生成选择 PN15Default CI（LogN15、32768 slots、scale=2^40），具体 Q/P、BTP 模式及键计划按真实配置锁定；standard profile 显式选择。SK 与完整 profile 足以重新生成可用 PK/配套评估键，但生成有随机性；full 仍保存原 PK，恢复不更换其身份。详细边界见[流式密钥规范](taa_v6_streaming_context_design.md)。
- Worker 协议版本为 `2`，包/运行时版本为 `0.2.0`，二者分别比较；总大小/偏移在消息和 JSON 中用十进制字符串，Go 用 `uint64`，JS 用 `BigInt`。
- 跨 Go/JS 及磁盘复制块最大为 4 MiB，单次只有一个未确认块；磁盘写入完成后才推进生成。此上限不代表单个 key/ciphertext 或 encoder 工作区上限。
- OPFS 是浏览器归档的工作区。公开 artifact 与含 SK 的私有 artifact 必须分开标识；只有关闭、校验并记录提交状态的产物才能发布。
- 现有 server 的 eval 导入仍可全量加载；本项目不扩展为服务端 lazy key loader，不以客户端降峰结论描述服务端内存。

## 4. 依赖顺序与完成门槛

| 阶段 | 可审阅实现单元 | 完成门槛 |
| --- | --- | --- |
| G1 | 纯 Go v6 key-only state、参数推导与持久化 | 相同 SK/PK 恢复；无 cgo 构建；不生成 evk |
| G2 | residual 单键 writer、seek 回填 | 当前 v6 reader 直接读取，键集合/level/NthRoot 正确 |
| G3 | BTP 逐键生成 | standard/CI 支持范围和临时 secret 关系正确，bootstrap 精度通过 |
| S1/A1 | 当前 CGO/C++ bridge、默认双文件 setup、server file import、bindings | 发布失败可见，native/Node/Python 链路可用 |
| A2/W1 | 浏览器 v6 wire/profile、Worker v2、OPFS 产物 | 双向 native 互通、背压/回填/恢复和故障终态可观测 |
| W2 | OPFS ZIP32/ZIP64 | 真正大文件、偏移、条目数量和独立工具互通通过 |
| A3/W3 | 任务预后处理、server 批处理、0.2.0 交付 | 解压后的正式包实际运行，ResNet/Retina 联调完成 |
| R1/R2 | GPU core pool、SDK failure、server release | 故障不挂起，多 engine 安全，释放后重载 |
| V | 最终 gitlink 与回归证据 | 下列覆盖完整，未完成项不记为通过 |

测试随实现提交。G2/G3 和 WASM 单键工作区属于先行可行性门槛；不能先完成 UI，再把底层无法流式化留到末尾。资源部分可与浏览器外围独立推进。

## 5. 功能覆盖与验证责任

| 需求 | 负责规范 | 必须出现的验证证据 |
| --- | --- | --- |
| 默认生成阶段降峰 | 流式密钥 | 独立进程 bulk/stream 同参比较，生成轨迹无全集常驻 |
| eval/full、同钥恢复、恢复后重新导出 | 流式密钥 | v6 reader、无 SK 的 eval、无 evk 的 full、旧密文恢复解密 |
| 标准环/CI、跨 N/环、封装键 | 流式密钥 | 键关联、旋转/共轭/乘法/BTP 运算验证 |
| native/纯 Go 共用核心 | 流式密钥、浏览器 | CGO 与 `CGO_ENABLED=0` 构建及同格式测试 |
| 无任务预生成与多任务复用 | 浏览器 | 同参不同布局成功，不兼容参数或缺键绑定前失败 |
| v6 Feature/密文双向互通 | 浏览器 | 0D/1D/2D/Mat、batch metadata、CI 容量与 scale/level |
| 两层输出协议与 OPFS | 浏览器、ZIP64 | 有界输入/输出、write-at、关闭前不发布、刷新恢复 |
| ZIP64 全结构 | ZIP64 | >4 GiB entry、后续 local offset >4 GiB、>65,535 entries |
| 批处理/预后处理/server | 任务流程 | ResNet 单输入、Retina 三输入，真实 logits/标签关联，真实 ZIP64 server |
| WASM 0.2.0 | 任务流程 | package/lock/runtime/manifest 一致，旧 Worker 混装拒绝 |
| GPU release/budget/failure | 运行时 | pool 预算、重载、多 owner 拒绝、故障注入无挂起 |
| 小规模鲁棒性措施 | 浏览器、ZIP64、运行时 | 启动/消息/取消/配额/损坏文件/异步异常故障测试 |

必须保护[现有 encparams v6 规范](v6_adapt_encparams_implementation.md)的回归边界：BERT、cifar10、imagenet、retina、ViT、yolov5 各自的输入加密 GPU lazy 与参数加密 GPU runtime_lazy 主链路，以及 BERT/ViT 原 PDM 输入链路。compact、seed、mask、PCM、EncryptedParameterStore、offline/online signature、level/scale/NTT metadata 和编译器优化不能退化。

所有新任务/测试夹具与日志写临时目录，不覆盖现有 `runs/`。记录实际 SHA、参数、toolchain、输入数量、误差、耗时、JS/WASM/RSS 峰值和 OPFS 占用；旧日志和旧准确率只作历史参考。外部 server、GPU 或浏览器环境缺失时，准确列出未完成项，不能以编译通过替代验收。

## 6. 实施与验收记录

本组规范对应的实现位于 `feat/taa-adapt`。各专题的实现与验收状态标在文首，测量、回归结果、已知基线失败和发布包来源统一记入[验收记录](../research/taa_v6_acceptance_report.md)。专题专项通过不替代第 5 节的整体回归与最终交付门槛。

实现沿用当前 v6 类型与仓库命名风格；文中拟定签名表达职责和语义，以提交源码中的实际签名为准，不为示意接口建立额外兼容层。
