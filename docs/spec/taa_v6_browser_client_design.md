# TAA v6 浏览器客户端、参数与 Worker 协议设计

日期：2026-11-06。状态：已实现，浏览器客户端专项验收通过，证据见[验收记录第 2–4 节](../research/taa_v6_acceptance_report.md)；最终提交发布包核对见任务交付规范。遵循[总则](taa_v6_adaptation_design.md)与[流式密钥规范](taa_v6_streaming_context_design.md)；归档细节见 [OPFS ZIP64](taa_v6_opfs_zip64_design.md)。本模块在 `feat/taa-adapt` 分支新增，`feat/streaming-key-generation` 分支只作行为参考。

## 1. 模块边界

新增 `inference/interface/wasm/`：`core/` 使用共享纯 Go v6 keygen/state，并实现当前 native packing/Feature；`cmd/latti-client-wasm/` 只做 syscall/js 与事件桥接；`src/index.js`/类型声明面向调用方；`src/worker.js` 串行处理会话、磁盘与协议；`src/zip.js` 专门负责 ZIP32/ZIP64。OPFS/产物辅助函数只在确有共用时抽到一个小模块，不建立多后端插件系统。

支持 HTTPS/localhost 的桌面 Chromium、Dedicated Worker、File System Access API 与 OPFS。启动时实际检测所需能力；缺失时报明确错误，不回退为整包内存 ZIP。用户目录是授权的输入/导出目的地，OPFS 是归档与大文件工作区，两者不能混为一个目录。

## 2. 参数 profile 与任务绑定

一个 profile 确定一套可复用密钥能力，至少包含：

**默认预生成选择 PN15Default 的共轭不变子环版本**：RingType=1、LogN=15、N=32768、32768 个实数 slots、默认 scale=2^40。具体 Q/P 与 BTP/键计划从锁定的参数表及实际任务配置生成，完整定义、源码依据和“默认”的适用范围见[流式密钥规范第 2.3 节](taa_v6_streaming_context_design.md#23-默认预生成-profilepn15default-共轭不变子环)。standard profile 须显式选择并单独匹配，不能用同一个名字掩盖环型差异。

| 内容 | 规范 |
| --- | --- |
| residual 参数 | RingType、LogN、完整有序 Q/P、log_default_scale，及 v6 参数构造所用的 secret/error 分布等实际生成选项 |
| 容量/metadata | 从 v6 参数推导 max_slots/log_max_slots；当前 NTT、batched、LogDimensions 等约定，不能写死旧 LogSlots |
| BTP | enable 标志、锁定配置推导标识与展开后的实际 circuit 参数/封装 weight；与 native 共用推导函数 |
| residual 键需求 | RLK、排序去重的 GaloisElement、各键 level/decomposition/压缩选项 |
| boot 键需求 | 跨 N/环模式、dense/sparse、boot RLK/GaloisElement、level/decomposition/压缩选项 |

profile JSON 中 uint64 素数和 GaloisElement 等用十进制字符串。现有任务 `ckks_parameter.json` 的 uint64 数值不能先经普通 JS `JSON.parse` 再 stringify，否则可能丢精度：将原始 UTF-8 JSON 交给 Go 的强类型 uint64 parser，或在构建时由 Go 输出精确的 profile。JS 仅解析已转换为字符串的 profile 元数据。

profile ID 由规范化实际参数及键计划派生，用于定位和快速检查；最终绑定仍比较实际值，不只比较 hash/名字。manifest 保存生成时的具体参数/能力，不以未来可变默认表替代。包内默认 PN15Default CI profile 必须由 `feat/taa-adapt` 分支的真实任务配置生成、带来源 SHA 并经过 native 验证；不得手填未注册的旧 profile 常量，也不宣称一个默认 profile 适用于所有任务。同一 SK 与完整 profile 可以重新生成可用 PK/评估键，但随机化输出不保证字节相同；full 仍保存原 PK，restore 保持原公钥身份，详见[密钥生成关系](taa_v6_streaming_context_design.md#22-skprofile-与其余密钥的生成关系)。

任务前预生成允许只选择密钥目录和已发布 profile，无需 task_config。之后绑定 task 时执行：

1. 精确构造/比较 v6 参数、scale/metadata 和 BTP 推导；任务内 `param0` 仅是局部引用。
2. 验证所有输入/输出/计算节点的密钥需求被已生成能力覆盖。密钥 level 至少满足需求，实际 decomposition 与读入/evaluator 语义相符；仅放宽已证实安全的集合包含关系。
3. 相同参数而 layout/block_shape/输入数量不同的任务可以复用；不同 Q/P、ring、BTP 参数或缺键的任务在加密前拒绝，不自动重生成、不覆盖已有 full。
4. 本轮仍按现有单 context client 执行。任务如包含多套不相同参数则明确拒绝，不把首个输入的参数误用到其他输入；不额外建设多 context 浏览器框架。

实施接口：当前 `gen_mega_ag.py` 从已生成 `mega_ag.json` 的键节点导出客户端 `task_config.json.evaluation_keys`，不从输入/输出布局猜测计算需求。缺少该字段的任务须重新生成，浏览器直接拒绝，不补旧任务兼容层。profile 的 `evaluationKeys` 由共享 Go 核心按实际生成计划展开，参与 profile ID；记录 residual/boot、键类型、精确 GaloisElement、N/NthRoot、Q/P level、base two/RNS 分解及压缩状态。绑定允许已验证的 Q level 向下使用；显式 P level 必须相等，图中 `sp_level=-1` 表示沿用实际 v6 键自身的 P，不推导不同 P 的兼容关系。

## 3. 对外 API 的行为契约

拟定接口保留必要用户流程，不承诺旧 WASM 调用签名：

```ts
type ArtifactRef = {
  artifactId: string;
  role: 'public-eval' | 'private-full' | 'ciphertext' | 'plaintext';
  file: FileSystemFileHandle;
  bytes: string; // uint64 decimal
};
type BatchResult = {
  directory: FileSystemDirectoryHandle;
  manifest: FileSystemFileHandle;
  count: string;
  archive?: ArtifactRef;
};
// openBrowserClient: 可只给 keyDirectory/profile，任务可以之后 bindTask。
// setup: 返回已提交 eval/full 的 ArtifactRef 与 key manifest handle。
// restore: 读取 full + manifest，恢复同一密钥，返回校验后的 profile ID。
// bindTask: 读取 task_config/ckks_parameter 并执行第 2 节匹配。
// encrypt/decrypt: 单文件输入，返回已提交 ArtifactRef。
// encryptBatch/decryptBatch: 目录或 server 模式，返回 BatchResult。
// cancel: 取消指定 requestId；close: 关闭会话并结清 pending。
```

目录批次与 server ZIP 模式属于本次明确功能，均直接执行新流程；不为来源 `encrypt_batch`/旧返回数组增加转发 wrapper。调用方需要逐文件时按磁盘 manifest 流式枚举，不一次返回全部 handle/内容数组。详细命名与 server 模式参数见[任务流程规范](taa_v6_browser_workflow_design.md)。

single encrypt/decrypt 需要已绑定任务和已恢复/生成密钥；setup/restore 不要求绑定任务。close 幂等，只释放内存、writer、锁和未提交临时产物；已提交文件持续可用。私有 full 仅返回给密钥持有方，上传/公共打包路径只从 public 角色清单选取。

## 4. v6 packing 与 Feature wire

唯一基准为 [InferenceClient](../../inference/interface/inference_client.cpp) 及 [Feature0D](../../inference/data_structs/feature0d.cpp)、[Feature1D](../../inference/data_structs/feature1d.cpp)、[Feature2D](../../inference/data_structs/feature2d.cpp)、[FeatureMat](../../inference/data_structs/feature_mat.cpp)。加密使用 `parameter.default_scale()`、任务 level 和真实 max_slots；CI=N，standard=N/2。矩阵的 head/block/transposed 规则按当前 native pack/unpack，不恢复旧 n_heads/形状假设。

当前支持的 native wire ABI 为 64 位 little-endian；公共标量头为 4 字节字段，ciphertext blob 前缀为 8 字节字节数，payload 为 v6 RLWE `WriteTo`/`ReadFrom`（含 metadata）。以逐字段 native 对照 fixture 固化字段宽度，不能直接按 Go struct 内存布局写出。

| Feature | ciphertext 列表之前的字段顺序 | 列表之后 |
| --- | --- | --- |
| 公共前缀 | dim、n_channel、n_channel_per_ct、level | — |
| 0D | skip、skip、n_ct | batch_metadata_present、batch_size、batch_feature_dim、batch_block_size，均 u32 |
| 1D | shape、skip、invalid_fill、n_ct | EOF |
| 2D | shape[0..1]、skip[0..1]、n_ct | EOF |
| Mat | shape[0..1]、matmul_block_size、n_ct | EOF |

每个列表项为 u64 byte length + 一个 v6 ciphertext；没有 `data_compress` 或旧压缩密文数组。Mat 的额外 packing 信息从当前任务配置获取，不凭空加到 wire。

scalar 两个 skip 字段按 native 写入；解析为一个逻辑 scalar skip，内部可用 `[skip,1]`，不能把冗余字段相乘作二维容量。输出始终写当前四个 batch metadata 字段。当前 v6 native reader 允许 0D 无 batch metadata 的 EOF 形式，浏览器 reader 可按同一 v6 规则处理；如果尾部非空则必须完整四字段，不增加旧 streaming 压缩标记识别。

Go 不构造整个 Feature `[]byte`。加密按 header→单 ciphertext→写出释放→后续 ciphertext 的顺序；解密按 header→单 ciphertext→解密输出块逐步消费。CSV 按块解析/编码；需要跨块重排的矩阵/CHW 先放 OPFS 临时数值块，再按 packing 映射读取，不能用整个 ZIP entry 或全部样本数组作为中间层。允许有限 header、当前 ciphertext 和 encoder 必需工作区，分别测量。

所有总长、累计 cursor 使用 uint64，只有当前有界块长转 `int`。校验字段、长度加法、ct 参数/degree/level/metadata、预期项数和 EOF；不把不可信长度直接传给 `make([]byte,total)`。不通过修改 offset 20、删 batch 字段或增加宽松 skip 上限来“修复”互通。

## 5. Go/WASM → Worker：拉取式 I/O 协议

bridge 每次只暴露一个活跃操作。拟定 `start(op,args)→operationId`、`next(operationId,ackOrInput)→event`、`cancel(operationId)`，由单生产者 Go 流程与有界事件通道实现，不积压未来输出。native 直接使用同一个核心 writer/source；JS bridge 不复制密码学状态。

| event | 字段与 Worker 行为 |
| --- | --- |
| begin-output | outputId、role、format、已知长度或 unknown；打开本 job 的 part sink，不分配总长度 buffer |
| write-chunk | outputId、offset、有限 Uint8Array；必须等磁盘写完再 ack/next；追加 offset 必须等于当前逻辑位置 |
| write-at | outputId、offset、有限 Uint8Array；只 patch 已预留范围，保留逻辑追加位置；磁盘确认后 ack |
| need-input | sourceId、offset、maxBytes（≤4 MiB）；Worker 从 OPFS 有界读取后反馈实际长度/eof |
| end-output | outputId、最终 uint64 bytes；只表示生成结束，不表示文件已提交 |
| done/error/cancelled | 单一操作终态；error 含 code/phase/entry/message，不在错误后继续输出 |

offset/总大小为十进制字符串，Go uint64 与 JS BigInt 精确转换。每个操作最多一个待确认 I/O 块；写入/读入失败作为错误反馈，核心停止而非继续生成。serialization buffer 必须在 patch/end 前 flush。不 transfer/detach WASM 线性内存；确需独立副本时仅复制当前有限块，完成后释放引用，不为每块强制 GC。

need-input 源是已校验的 OPFS 文件。外部大 ZIP 先按归档规范落 OPFS；单 Feature/CSV 外部输入也按 stream 导入工作区，避免 File.arrayBuffer 的全输入路径。

## 6. Worker → 主线程：协议 v2

所有消息有 `protocolVersion:2`、`requestId`、`jobId`、`type`。初始化使用保留 requestId 与 session jobId，业务作业用唯一 jobId；回包必须与 pending 操作匹配。命令包括 setup/restore/bindTask/encrypt/decrypt/batch/cancel/close，参数为小型配置及可 structured-clone 的文件/目录 handle，不携带归档 payload。

| type | 数据和终结规则 |
| --- | --- |
| ready | JS/Worker/WASM 各自实际 runtimeVersion=0.2.0、build 标识、protocolVersion、OPFS/ZIP64/raw-DEFLATE 等能力；初始化全部成功才发 |
| progress | phase、case/entry、读写 bytes、耗时、chunkBytes、inFlightChunks、可测的 WASM/JS/磁盘指标；uint64 总数用字符串；节流但终态必发 |
| artifact | 仅已关闭/校验/提交的 ArtifactRef；单文件操作或少量 setup 产物可发，不逐条洪泛大型批次 |
| result | 单文件结果或 BatchResult 的目录/manifest handle 及汇总；是业务成功终态 |
| error | 稳定 code、phase、entry、message、retryable；业务失败终态 |
| cancelled | 操作已停止、当前未提交产物作废；取消终态 |

每个 request 恰有一个 result/error/cancelled；artifact/progress 不是终态。批次全部结束后才提交批次 manifest，失败不能发送此前部分条目的成功清单。handle 使用同源 structured clone，writer/SyncAccessHandle 始终留在 Worker；main 不会收到打开的写锁。

会话状态最少为 initializing→idle↔busy→closing→closed，fatal 可从活动态进入 failed。密钥 ready 和 task bound 分开记录；队列真正执行时复查状态。普通任务失败回到 idle，启动失败、runtime 退出或 Worker 通信损坏进入 failed；close/fatal 拒绝未开始的排队请求并结清 pending。

取消在 chunk/键边界合作执行；已在进行的不可中断密码学 primitive 需等它返回。若调用方需要强制结束，terminate Worker、结清全部 pending 并重新 open/restore；不承诺恢复未完成的随机生成状态。close 先停止接受新作业，再取消/等待当前 I/O 收尾、关闭句柄、释放状态；不能在关锁前报 closed。

## 7. OPFS 产物生命周期

每 job 使用随机唯一工作目录，区分 incoming、part、central spool、临时 entry、公开产物与 private-full。manifest 是磁盘元数据，记录 artifactId、受控相对路径、format、role、十进制 bytes、profileId 和提交状态；不保存 JSON handle/BigInt 或密钥内容。大批次采用小型提交 manifest 指向磁盘 JSONL 条目清单，不能把全清单读为一个 JSON 对象。

统一发布顺序为：写 part→完成全部回填→close/必要 flush→核对实际大小与格式/CRC→写条目清单并关闭→最后关闭 committed manifest→发布可读 handle。消费者在 manifest 关闭后重新取得快照并检查清单/大小；不存在 multi-file atomic rename 假设。失败清理本次未提交文件，启动时只清理能确认无活跃 owner 的遗留目录。

keyDirectory 的最终 full/eval 按同样的双文件规则流式复制到用户目录，并最后发布 key manifest；OPFS 是归档默认工作区，并不意味着必须额外保存所有 key 文件副本。已有 full 不覆盖。同一 keyDirectory 或导出目标使用同源 Web Locks 排他保护检查+创建+提交；锁 key 从共同存储根和规范化相对路径派生，不能使用随机 jobId。无法建立共同目标身份时，使用更粗的同源 key/export 写锁，避免两个 tab 各自认为已加锁。

磁盘用量包括 incoming、当前解压 entry、archive、spool、导出副本。`navigator.storage.estimate()` 只作提示性预估，不是空间预留；实际处理 QuotaExceededError。已提交文件的删除由调用方明确触发，close 不自动删除；未提交清理失败报告残留位置。OPFS 持久化不保证用户永不清除站点数据，私有 full 的用户导出由明确操作完成。

## 8. 小改动、明确收益的鲁棒性措施

本节规定启动、pending 和 I/O 流程所需的局部鲁棒性改动，不新增通用恢复系统。

| 改动 | 必须可观察的行为 |
| --- | --- |
| 启动错误传播 | instantiateStreaming 失败可用同一响应的合法备用读取；404/错误 WASM 不假成功；go.run rejection/提前退出、初始化超时结清 open 与 pending；正常运行中的 go.run Promise 不当作 ready 条件 |
| 消息错误 | 校验协议、type、requestId/jobId 和状态；监听 error/messageerror；重复/未知消息不重复 resolve，不留永久 pending |
| 状态/取消/关闭 | 在执行排队请求时复查，close/fatal 结清队列；取消或崩溃无成功半成品；强制 terminate 后可 reopen/restore |
| 磁盘 I/O | try/finally 关闭 writer/锁；异步写入 await；若采用 sync handle 则处理部分读写、flush/close；配额错误带阶段与清理结果 |
| 多 tab | 唯一 job 目录 + 共同目标锁；清理遗留 job 先确认无 owner，不删除另一 Worker 正在使用的数据 |
| 大文件边界 | uint64 精确往返，超过浏览器安全 number 范围明确拒绝 IO；有限 chunk、CSV 分块、无全包副本 |
| 进度与构建 | 节流 progress，终态不丢；实际版本握手拒绝混装，不信目录名 |

## 9. 验收

- native→WASM 与 WASM→native：全部 Feature 类型、batch metadata、scalar 大 skip、标准/CI、任务实际 scale/level/NTT 和矩阵布局；误差沿用 native 标准，不比较随机 ciphertext 字节。
- 无 task setup，刷新 restore，同参不同布局的两任务复用；Q/P 单素数变化、ring/BTP 不同、缺 GaloisKey 或 level 不足在绑定阶段失败；大素数不经 JS number 舍入。
- 真实 Worker/OPFS 中 keygen、encrypt、decrypt 均使用 chunk/need-input/write-at；单在途块，main 只收小型消息/handle，无全 Feature/ZIP 的 ArrayBuffer 回退。
- 注入 404、错误 MIME/runtime 退出、消息反序列化失败、取消、close 竞争、配额/close/patch 失败、多 tab 冲突；全部 pending 有终态，无 committed 半成品，旧产物可读。
- 分别记录 JS/WASM/可测 RSS、最大单 ciphertext/key 工作区、OPFS 临时/最终占用；目录模式和 ZIP 模式分别测量，不以一个替代另一个。
