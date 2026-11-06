# TAA v6 流式密钥生成与客户端持久化设计

日期：2026-11-06。状态：已实现，流式密钥专项验收通过；整体回归与交付状态见[验收记录](../research/taa_v6_acceptance_report.md)。上位约束见[设计总则](taa_v6_adaptation_design.md)；以其中锁定的 v6 源码为准。

## 1. 目标与边界

默认 setup 在生成阶段逐键写入 eval 文件，结束后只保留 v6 参数、SK/PK、encoder/encryptor/decryptor 和生成所需的小型元数据。客户端仍能加密、解密、保存/恢复和重新生成 eval；不常驻 residual/BTP 全套评估键，不持有 bootstrap evaluator。

现有统一 `CkksContext` 与 v6 RLWE/CKKS/BTP 类型是唯一对象模型。当前 `create_random_context()` 和 `GenEvaluationKeys()` 会批量生成评估键，不能作为默认 streaming setup 的前置步骤。显式 bulk context 可以继续供原有 evaluator 用途使用，但不提供旧 streaming 接口的兼容 facade。

## 2. 最小实现划分

在 Lattigo 增加一个无 cgo 的 `clientcontext` 包，直接使用 `core/rlwe`、`schemes/ckks`、`circuits/ckks/bootstrapping`。职责限于 key-only state、共享参数推导、逐键写出和 client-only 序列化；不另写 evaluator 或密码学算法。

拟新增的内部契约如下，名称可以按当前源码风格微调，语义固定：

| 操作 | 输入/输出与所有权 |
| --- | --- |
| 创建 key-only state | 已验证的 v6 参数、BTP 模式、键需求；只生成一次 SK/PK |
| 恢复 key-only state | client-only context + 独立 SK；重建加解密器，不生成 evk |
| 写 eval | state + 可顺序写/定位回填的 sink；借用 SK/PK，逐键生成并释放，不修改原 state |
| 写 full | state 的 client-only context + SK；不经过 eval 文件，不复制评估键 |
| 释放 | 断开本 state/stream 的引用；现有外部 handle 按当前所有权规则释放 |

sink 只需 `Write(chunk)`、`Patch(offset uint64, chunk)` 及返回错误；累计位置由 writer 用 `uint64` 维护。native sink 写临时文件，WASM sink 经 Worker 事件等待磁盘确认。块最大 4 MiB；回填只允许覆盖已经预留且位于已写范围内的字段。无需实现通用 VFS 或提前计算所有键大小的规划器。

当前 `go_sdk/bootstrap.go` 的 residual→BTP 推导提取到这个纯 Go 可调用的共享位置，原 CGO 入口委托同一实现。保留 degree-16 standard circuit、因子分解深度/scale、LogSlots 和 LogMessageRatio 调整，不维护 native/WASM 两份配置表。

CGO 新入口沿用 `ErrorStatus`、`insertObject/getObject/ReleaseHandle`，C++ 在当前 `CkksContext` 上补 key-only 初始化及流式导出能力。bridge 引用同一 v6 SK/PK 对象，不能通过生成第二对密钥拼装 client 与 eval。流 handle 独占当前生成过程，成功、错误和取消均释放；错误返回不遗留可用输出 handle。生成头按现有流程重建，不能复制来源分支 ABI。

### 2.1 直接复用 Lattigo v6 现有单键生成接口

**本设计直接调用 Lattigo v6 已有的密钥生成接口，不新增或重写单把密钥的密码学生成算法。** `feat/streaming-key-generation` 分支同样采用调用既有接口逐键生成的方式；`feat/taa-adapt` 延续这一实现原则，但具体接口、参数与 secret 关系均以锁定的 v6 为准，不包装旧分支接口。

以下接口均已存在于 [core/rlwe/keygenerator.go](../../inference/lattisense/fhe_ops_lib/lattigo/core/rlwe/keygenerator.go)，不是本次拟新增接口。表中的 kgen 由 `rlwe.NewKeyGenerator(对应的 v6 参数)` 创建；residual、circuit 和 sparse 参数必须分别使用正确的生成器。

| 用途 | 直接调用的现有 v6 接口 |
| --- | --- |
| 客户端 SK/PK | `kgen.GenSecretKeyNew()`、`kgen.GenPublicKeyNew(sk)`；同一 state 只创建一对 |
| residual/boot RLK | `kgen.GenRelinearizationKeyNew(sk, evkParams...)`，每次生成一把重线性化键 |
| residual/boot GaloisKey | `kgen.GenGaloisKeyNew(galEl, sk, evkParams...)`，每次处理一个 GaloisElement |
| standard 跨 N evaluation key | `kgen.GenEvaluationKeyNew(skInput, skOutput, evkParams...)`，两个方向分别调用 |
| CI ring-swap keys | `kgen.GenEvaluationKeysForRingSwapNew(skN2, skN1, evkParams...)`；该既有接口一次返回关联的一对键，允许暂存这一对，不拆写其算法 |
| BTP circuit secret | N 不同时用 `kgen.GenSecretKeyNew()`；N 相同时沿用第 4 节的现有 basis 扩展步骤，不重新采样 |
| BTP sparse secret | `kgenSparse.GenSecretKeyWithHammingWeightNew(weight)`，仅创建一次关联 sparse secret |
| dense→sparse / sparse→dense | `kgenSparse.GenEvaluationKeyNew(skN2, skSparse)`、`kgenDense.GenEvaluationKeyNew(skSparse, skN2)` |

`evkParams` 只传当前 v6 生成路径已有的选项；原调用使用默认值时继续使用默认值，不因流式输出改变 level、decomposition 或压缩策略。v6 同时提供写入已有目标对象的非 `New` 接口，但本次不为推测中的内存收益额外增加对象复用路径。

新增内容限于**键需求遍历、调用顺序、必要 secret 的生命周期、调用现有 `WriteTo` 写出、背压和文件长度回填**。无需修改 `core/rlwe/keygenerator.go` 中上述接口及其算法。BTP 包中拟提取的共用步骤/逐键回调只是对这些现有接口的流程组织；不能将其误解为新增一套 BTP 单键生成原语，也不能先调用批量 `GenEvaluationKeys()`/`GenGaloisKeysNew()` 得到全集再逐个写盘。

### 2.2 SK、profile 与其余密钥的生成关系

**给定已生成的 residual SK、与它匹配的完整 v6 profile 及明确的键需求，可以调用现有 v6 接口重新生成该客户端所需的公钥和评估密钥。** 这一点对共轭不变子环（CI）和标准环都成立，但生成过程使用新的安全随机数，不是确定性派生；相同 SK/profile 不保证产生字节相同的 PK 或评估键。

这里的完整 profile 包括真实环型、N、Q/P、生成选项、BTP 参数和键需求，不能只保留 `PN15Default` 名称。SK 与 profile 必须匹配；不能把一把 CI SK 直接当成 standard SK，或改用另一份 Q/P 后仍视为原客户端。v6 明确支持的跨环/跨 N 操作通过其 evaluation-key/basis 接口完成。

| 密钥类别 | 生成所需输入与恢复边界 |
| --- | --- |
| PK | residual SK + residual 参数 + 新随机数；无需已有 PK 或评估键。重生成的 PK 仍对应同一 SK，但与原 PK 的字节/标识可能不同 |
| residual RLK/GaloisKey | residual SK + residual 参数 + 指定 GaloisElement/生成选项 + 新随机数；无需先生成 PK，也不依赖其他评估键 |
| BTP 评估键整套 | residual SK + 完整 BTP profile/键计划 + 新随机数；按第 4 节生成或扩展 circuit secret，并在启用封装时生成 sparse secret，再调用既有接口生成相关键 |
| 任意两客户端之间的 key-switch key | 需要相应的 input/output secrets；本规范不承诺凭本客户端 SK 和公开参数获得另一客户端的 SK 或生成任意跨客户端转换键 |

因此，**长期保存 SK 与完整 profile/键计划，足以重新生成一套可用的客户端公钥及评估键**；不要求持久化 BTP 临时 secret。BTP 重新生成时，跨 N/环键、dense/sparse 键与 boot RLK/GaloisKey 必须在同一轮内使用配套的临时 secret；不能将前后两轮的部分 BTP 键混用。同一 residual SK 下重新生成的完整配套 BTP 键仍能处理原客户端密文。

本次 full 格式仍同时保存原 PK 与 SK：restore 要恢复原 PK，而不是隐式重生成一个新 PK，以保持现有公钥字节/标识和已发布文件关系。重新导出 eval 使用恢复的原 PK，只重新生成所需评估键；如已有可用 eval，直接继续使用它即可。这里说明 SK 的生成能力，不新增 SK-only 文件格式、确定性随机种子备份或旧接口 wrapper。

### 2.3 默认预生成 profile：PN15Default 共轭不变子环

`feat/taa-adapt` 的浏览器默认预生成 profile 选用当前 **PN15Default 的 CI 版本**，与现有 BTP 编译路径保持一致：

| 项目 | 当前锁定基线的值/来源 |
| --- | --- |
| 定义 | [training/model_compiler/components.py](../../training/model_compiler/components.py) 的 `PN15Default`，读取 [frontend/parameter.json](../../inference/lattisense/frontend/parameter.json) 的 `CKKS["15"]` |
| residual 环 | `RingType.ConjugateInvariant=1`，LogN=15，N=32768，容量为 32768 个实数 slots |
| 默认 scale | log_default_scale=40，即 2^40 |
| 模数链 | 当前参数表含 13 个 Q 素数、3 个 P 素数，参数最大 level=12；具体有序素数值从参数表及生成后的任务配置读取，不在本规范复制常量 |
| BTP | 启用时按当前共享推导构建 standard LogN16 circuit，并生成 CI↔standard ring-swap 等完整配套键；是否启用及键计划必须在发布的 profile 中显式记录 |

这一“默认”指本次浏览器密钥预生成的选择及现有 CI BTP 主路径，不代表所有入口都固定 PN15Default。[pipeline.py](../../training/model_compiler/pipeline.py) 的 `try_btp()` 只列 `PN15Default`，`try_no_btp()` 会尝试 PN12/13/14/15Default；[deploy_cmds.py](../../inference/model_generator/deploy_cmds.py) 的 `gen_custom_task()` 默认参数名仍为 PN14Default。底层 `CkksParam.create_default_param`/C++ 参数工厂的默认 ring_type 为 Standard，调用默认 CI profile 时必须显式传 CI，不能依赖省略参数。

profile 按实际生成的任务配置校验，包括可能受编译选项影响的有效 level。非默认 standard profile 仍按第 2.1–2.2 节的相同接口原则支持，但必须显式选择，不能仅修改 PN15Default 的 ring_type 就宣称与其密钥兼容。默认选择不改变既有任务的参数，也不为浏览器测试重新提高 N、scale 或模数。

## 3. residual 键计划

从实际 `CkksParameter` 推导 `l=log_max_slots()`：CI 容量 N，standard 容量 N/2。默认集合严格对应当前 `CkksContext::gen_rotation_keys()`：正旋转 `2^i, i=0..l-1`，负旋转 `-2^i, i=0..l-2`，标准环另含复共轭，CI 不生成非法共轭键。

用 v6 参数计算 GaloisElement，去重后按 uint64 key 升序输出；不能按 rotation 整数直接编码 map key。先生成并写 residual RLK，再逐个生成/写 GaloisKey。保持当前 v6 level、base/decomposition 和 `EvaluationKeyParameters.Compressed=false` 默认值。显式任务额外键需求必须纳入计划和 profile；不能只增加 manifest 声明而漏生成。

RLK 调用现有 `GenRelinearizationKeyNew`，每把 GaloisKey 调用现有 `GenGaloisKeyNew`；每次生成后直接调用该键的 v6 `WriteTo`，写入确认后丢弃引用再生成下一键。不得创建全集 `MemEvaluationKeySet`、调用全集 `GenGaloisKeysNew`，或为序列化把所有键放回 map。持有小型已排序 GaloisElement 列表是允许的。

## 4. BTP 的 v6 语义与逐键流程

当前支持范围由锁定的 `CreateCkksBtpParameterFromResidualParameter` 决定：standard residual LogN 14/15/16 走 standard LogN16 circuit；CI 要求 circuit LogN=residual LogN+1，因此当前配置只允许 CI residual LogN15。构造参数仍须经过 v6 校验；表内存在并不意味着任意 Q/P 都合法。本项目不采用更新 main 的 native CI circuit。

在现有 `circuits/ckks/bootstrapping/keys.go` 内提取 bulk/stream 共用的生成步骤，并增加逐键回调/写出入口。bulk 可以收集回调得到的键形成原 `EvaluationKeys`，保留原公共接口；stream 消费后释放。允许现有 ring-swap primitive 暂存一对关联键，除此之外只保留当前键。该局部改动的目的是复用 v6 secret/basis 逻辑，不是把整个 BTP 包改造成任务框架。

| 步骤 | 必须保持的 v6 关系 |
| --- | --- |
| 准备 circuit secret | residual N 与 circuit N 不同时只生成一次 `skN2`；相同时按原代码扩展原 secret 的 Q0→Q/P basis |
| 跨 N/环键 | standard 跨 N 生成 N1→N2、N2→N1；CI 用 `GenEvaluationKeysForRingSwapNew(skN2, skN1)`，准确区分返回赋值顺序 |
| 封装键 | weight=0 时两键均缺省；否则仅生成一次 sparse secret，参数为 circuit LogN、Q[:1]、**完整 circuit P**，生成 dense→sparse 与 sparse→dense |
| circuit 核心键 | 用同一 `skN2` 生成 boot RLK；Galois 集合为现有 `p.GaloisElements(paramsN2)` 加 circuit 共轭，去重并按 wire map key 排序逐个生成 |
| 生命周期 | linked keys 全部生成完前保留其共同的 `skN2`/sparse secret；结束或失败释放临时 secret；不把这些临时 secret 写入 eval/full |

生成顺序服务于下面的 v6 序列化顺序。ring-swap primitive 返回的是 complex→real、real→complex，而 wire 顺序相反，不能因函数参数/返回顺序相近而写反。residual 与 boot 的参数、level、NthRoot、键集合分别记录与验证，不合并为单个 map。

## 5. eval 的既有 v6 wire

文件仍由当前 [CkksContext::deserialize](../../inference/lattisense/fhe_ops_lib/schemes/ckks/ckks.cpp) 直接读取；不增加外层新 magic 或旧格式识别分支。`blob(x)=LE uint64 字节数 + x`，parameter/PK/key payload 采用当前 v6 serializer。

```text
8 bytes "CKKSCTX1"
blob(parameter)
u8(use_default_rotation_keys)
u8(enable_bootstrapping)
u8(has_pk=1), blob(PK)
u8(has_residual_evk=1), blob(residual MemEvaluationKeySet)
u8(has_btp_evk), [blob(BTP EvaluationKeys)]
EOF
```

`MemEvaluationKeySet` 完全遵循 `core/rlwe/keys.go`/`utils/structs`：RLK presence + 原生 payload，map presence + uint32 count，按 uint64 map key 排序，每项为 key + 原生 GaloisKey payload。GaloisKey 自带 GaloisElement、NthRoot 等字段，不能省略或写旧 uint16 RTK count。

BTP payload 依次为六项 optional key 的 presence + 原生 payload：N1→N2、N2→N1、RealToCmplx、CmplxToReal、DenseToSparse、SparseToDense；最后为 MemEvaluationKeySet presence + boot key set。未启用 BTP 时外层 `has_btp_evk=0`。

长度处理采用**预留 8 字节、顺序写 payload、按实际累计长度定位回填**。写嵌套 blob 时只需有限的长度位置栈。回填前 flush 当前序列化 buffer，回填不改变逻辑追加位置；sink 完成全部 patch 后才能关闭。不能使用整个 key set 的 `MarshalBinary()`/`BinarySize()`、全文件 `[]byte` 或 C++ `export_raw_data` 来获取长度。单键 `WriteTo` 允许有限 buffer，不能额外复制整个大键。

`use_default_rotation_keys` 仅在确实生成默认集合时为 1；自定义额外集合以真实集合为准，不靠该 flag 代替匹配。PK 始终保留，便于 server/provisioner 使用；任何 eval 路径均不得写 SK。

## 6. full 与同钥恢复

full 沿用 [InferenceClient::export_full_context](../../inference/interface/inference_client.cpp) 的双 blob 容器：

```text
blob(client-only CKKSCTX1 context)
blob(v6 SecretKey)
EOF
```

其中 client-only context 带 parameter、PK、rotation/BTP 模式，但 `has_residual_evk=0`、`has_btp_evk=0`。现有 `ss_write_vector` 长度为 native `size_t`；本次支持的 64 位 little-endian native ABI 下即 8 字节 LE，WASM 必须显式写 uint64，不能使用 wasm32 指针宽度。不为其他 endian/32 位 native 另立格式。

从显式全键 context 导出 full 时，以只读 client-only 序列化选项排除 evk；不能先清空原 evaluator 的键。保留已有 `load_context(context_bytes, secret_key_bytes)` 和 `set_secret_key` 的 v6 恢复路径，校验参数、SK/PK 维度/level 与 EOF。已知不匹配拒绝加载；必要的同钥一致性回归用恢复后加密/解密验证，不添加新密码学证明协议。

恢复状态的 BTP mode 与生成计划来自真实参数和配套 profile；即使没有 BTP key handle，也必须能重新推导同一个 BTP 参数。禁止因 `has_btp_evk=0` 调用批量 `create_bootstrapper()`。恢复后先能解密之前的密文，再能加密新输入、重新导出同 SK 的 eval；保留原 PK，随机化评估键不要求与前次字节相同，BTP 评估键必须按第 2.2 节整套配套生成。

## 7. native 文件与公开接口

默认 `InferenceClient::setup()` 改为生成并发布 `eval_context.bin`、`full_context.bin`，成功时内存只保留 key-only context。增加拟定 `export_eval_context_to_file(path)` 和 server `import_eval_context_file(path)`；恢复后前者按当前计划重新逐键生成，不能假装 full 中已有 evk。

现有 `export_eval_context()` Bytes 接口只接受显式具备完整 evk 的 context；key-only 状态应给出清晰错误并指向 file API，不能静默输出缺键的 eval。`export_full_context()` 保留当前签名但内容改为 client-only 双 blob。默认 setup 的 examples/bindings 测试必须改用已提交 eval 文件；没有另设旧行为开关或旧类 wrapper。

| 发布阶段 | 规则 |
| --- | --- |
| 预检 | 在同一目标目录排他保护下检查两个最终文件；任一已存在均拒绝覆盖，尤其禁止覆盖私钥文件 |
| 写入 | 使用本次专属临时文件，流式写 eval/full；检查短写、回填、flush、close；失败释放新 state |
| 发布 | 两文件关闭并验证后，以不覆盖方式发布，最后提交一个小型 `key_manifest.json`，记录参数/键计划、文件大小和 public/private 角色 |
| 部分失败 | 两文件不是原子事务；无 committed manifest 的一组文件不可作为 setup 成功结果。只清理可证实属于本次的新文件，不动旧文件；清理失败列明残留并停止自动重试 |
| 重试/恢复 | manifest 存在且完整时走 restore；只有残留时先验证 owner 和完整性，再清理重试。不继续未完成的随机 keygen |

manifest 是发布/参数元数据，不改变 eval/full wire；不含 SK、不把完整密钥内容写 JSON。显式 Bytes full 导入可独立工作，但若缺少任务所需的键计划证据，则重新导出前必须重新确认 profile，不能只凭文件名信任。

server file import 做空文件、尺寸/分配范围、短读与当前 v6 格式检查，再复用 Bytes 导入；它仍会加载整套 evk。Node addon 仍委托当前 C++ client，需要重新编译并验证 Promise reject；Python 暴露新增 file/release 入口，不引回旧 SDK bindings。runner/provisioner 可以用 eval 获取 evk、用独立 SK/full 获取客户端能力；保持 offline/online signature 和 metadata 契约。

## 8. 修改落点与排除项

| 层 | 必要落点 |
| --- | --- |
| Lattigo | 新纯 Go `clientcontext/`；现有 BTP keys 的局部共用步骤；`go_sdk/ckks.go`、`bootstrap.go`、当前 handle/error bridge 与生成头 |
| LattiSense | `fhe_ops_lib/schemes/ckks/ckks.*` 的 key-only/流式/只读 client-only 序列化；现有 ABI 构建与测试 |
| latti-ai | `inference/interface/inference_client.*`、`inference_server.*`、Node/Python/examples、文件发布测试 |

不改现有 evaluator 数学语义，不全局重写 `serial.h`，不增加旧 context parser，不让 server 按需从磁盘加载单键。错误长度/尾随/截断检查集中在本次新增和直接涉及的读写边界。

## 9. 验收

1. 新 writer 产物直接交给未改格式的 v6 decoder；验证 PK、RLK、正负旋转、标准环共轭、CI 合法操作、residual/boot level 和 NthRoot。比较 bulk/stream 的运算结果与误差，不比较随机键字节。
2. BTP 覆盖 standard 同 N、standard 跨 N、CI ring-swap、封装启用/关闭；同一临时 secret 关联正确，bootstrap 误差按现有标准判定。非法 CI degree 在生成前失败。
3. full 无 evk、eval 无 SK；恢复同 SK/PK，解旧密文/加新输入/重新导出成功；截断、尾随、错误参数、长度溢出与损坏数据拒绝。
4. 注入生成、短写、patch、flush、close、发布第二文件和 manifest 失败；无双文件成功假象、旧私钥不被覆盖、临时 handle 可释放。
5. 独立进程同参数/键计划/线程/GC 比较 bulk 与 stream，分别记录 keygen/export/full/restore 峰值、最大单键和耗时。观察释放轨迹，证明没有全集常驻；不沿用来源性能数字或用 4 MiB chunk 推断峰值。
6. 在真实 wasm32 上尽早测最大单键与 encoder 工作区。若超线性内存上限，记录为未满足门槛并针对 v6 算法实现优化；不能缩小参数、删键或擅自启用旧压缩来宣称完成。
7. 代码审查与生成轨迹核对第 2.1 节接口表：逐键路径直接调用现有 v6 单键接口，BTP ring-swap 仅允许既有成对返回；当前键完成写入确认后才生成下一键。确认未重写单键算法、未引入旧分支兼容 wrapper、未通过批量生成后逐键导出来冒充流式生成。
8. 固定同一 SK 与完整 profile，分别覆盖 CI 默认 PN15Default 和显式 standard profile：重生成 PK 后加密可由原 SK 解密；不依赖 PK 重生成 RLK/GaloisKey 后完成相应运算；重生成完整配套 BTP 键后处理原密文。restore 单独验证原 PK 字节/标识保持不变；不把随机化密钥的字节相等作为验收条件。
