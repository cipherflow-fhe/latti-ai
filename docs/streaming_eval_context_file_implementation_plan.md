# 密钥流式生成落盘 implementation plan

本计划严格依据 [设计规范](streaming_eval_context_file_design_spec.md)（修订提交 `7b5e690`），把规范要求拆成按模块执行、验证和提交的步骤。M1–M7 已执行并验证；提交对应关系、重编译命令、回归结果和客户端峰值内存实测见 [验收记录](streaming_eval_context_file_validation.md)。

## 1. 分支、基线与提交规则

| 仓库 | 本地分支 | 实现前基线 | 处理方式 |
| :--- | :--- | :--- | :--- |
| Latti-AI | `feat/streaming-key-generation` | `7b5e690`，随后增加本计划提交 | 原 `feat/split-key-gens` 已重命名，保留原提交历史 |
| `inference/lattisense` | `feat/streaming-key-generation` | `982eea49b68c7a1496c567bb531058d64b297866` | 从原 checkout 提交创建本地分支 |
| `inference/lattisense/fhe_ops_lib/lattigo` | `feat/streaming-key-generation` | `d68e5122547b3d992b37b468f303a612132ad5f5` | 从原 checkout 提交创建本地分支 |

直接在上述本地分支修改，不使用 worktree，不切换至其他基线。每个模块验证后单独 commit；先提交 Lattigo，再在 LattiSense 提交其 gitlink 和 C++ 包装，最后在顶层提交 LattiSense gitlink 及调用层改动。分支创建本身不会改变 submodule 指针。

优先采用显著降低客户端密钥生成阶段峰值 RSS 的最小实现：复用现有单键生成和 serializer、一个局部 `bytes.Buffer`、既有 CGO 句柄和错误机制、Node `setup()` Promise。不增加独立 `setup_keypair()` 公共 API、新的 Go module、底层 writer/编码变体、排序步骤、客户端完整文件 Buffer、服务端 mmap 或分块解析。GC 调整只在默认配置实测不达标时局部评估。

## 2. 模块与 commit 顺序

| 步骤 | 仓库/模块 | 对应规范 | 建议 commit message | 依赖 |
| :--- | :--- | :--- | :--- | :--- |
| M1 | Lattigo Go SDK：公共落盘逻辑、普通 CKKS | §三.2、§四.1/2/4、§六.2 | `feat: stream CKKS evaluation keys to file` | 无 |
| M2 | Lattigo Go SDK：BTP | §三.1/3、§四.1/2/3/4、§六.2 | `feat: stream bootstrapping evaluation keys to file` | M1 |
| M3 | LattiSense：C++ 薄包装、Lattigo gitlink | §一.4、§四.5、§七 | `feat: expose streaming key setup in C++ contexts` | M2 |
| M4 | Latti-AI：服务端文件导入、LattiSense gitlink | §五、§七 | `feat: import evaluation context from file` | M3 |
| M5 | Latti-AI：默认客户端 setup、端到端示例 | §一.1/2、§二、§四.5、§六.1、§七 | `feat: make client setup stream evaluation keys by default` | M4 |
| M6 | Latti-AI：Node 调用与验证 | §四.5、§六.2、§七 | `test: verify Node streaming setup workflow` | M5 |
| M7 | Latti-AI：独立客户端内存与完整验收 | §六.2、§八 | `test: validate streaming key generation memory reduction` | M6 |

测试随所属模块提交。M7 只补前序模块尚未覆盖的跨层验收和测量工具，不重复协议单测或引入新测试框架。每次提交只暂存本模块文件和明确需要更新的 gitlink；不提交密钥、临时文件、构建二进制或无关改动。

### M1：普通 CKKS 与公共落盘逻辑

修改 Lattigo 的 `go_sdk/main.go`；相关测试放在同一 SDK 的测试文件中。公共逻辑留在已有 Go 源文件内，复用现有 CMake 的显式源文件列表。

1. 增加供上层默认 setup 调用的流式创建入口，以及已有 context 的文件导出入口；二者调用同一内部逐键 writer。创建入口内部准备参数、SK/PK 和加解密状态，不调用会预生成并保留 RLK 的旧随机 context 构造器，不新增单独的 keypair-only API。
2. 复用现有正负幂次旋转集合，转换为 `galEl` 并去重，加入共轭键；写数量前检查不超过 `uint16` 上限。用同一个 key generator 串行调用 `GenRelinearizationKey(sk, 1)`、`GenSwitchingKeyForGalois(galEl, sk)`，不调用批量 `GenRotationKeys`。
3. 按现有普通 CKKS 格式写参数长度 **BigEndian**、参数、固定 `has_sk=0` 和 `has_pk=0`、RLK、RTKs。RTK 数量 `uint16` 和标识 `uint64` 使用 LittleEndian；参数长度也检查表示范围。
4. 所有密钥 payload 直接复用已有 serializer 和同一个局部 `bytes.Buffer`，编码后 `WriteTo(file)`。不按全套键尺寸 `Grow`；每键写出后结束其作用域，不把键存入 `context.rlk/gk` 或其他持久容器。
5. 在目标目录创建唯一临时文件，逐次检查写入、短写、Close、Rename 错误；关闭成功后发布。失败时清理临时文件，保留旧目标，清理失败附带临时路径。
6. CGO 新入口返回状态码，失败信息走现有 `error_message`/`GetErrorMessage()`；可处理的校验和生成错误不以未捕获 panic 穿过 CGO。创建入口成功后才交出可用 context 句柄，失败不遗留未交付的句柄。已有 context 导出不改变客户端 SK/PK。

验证后提交：用小参数解析完整文件结构，确认 `has_sk=0`、无 SK payload、`has_pk=0`、原 advanced deserializer 可导入且读完文件；比较新旧 key set，用同一 SK/密文验证旋转、共轭和重线性化的功能等价。检查默认创建后加解密可用，评估键未被 context 保留。覆盖非法前置条件、数量/长度校验及写入失败；失败注入只使用内部最小测试接缝，不增加公共 serializer 接口。

### M2：BTP 流式生成

修改 Lattigo 的 `go_sdk/bootstrap.go`，复用 M1 的落盘和错误处理逻辑；测试归属 Go SDK。

1. 增加 BTP 流式创建与已有 context 导出入口。内部只准备客户端 SK/PK、Scheme/BTP 参数和加解密状态；不调用 `CreateRandomCkksBtpContext`、`bootstrapping.GenEvaluationKeys` 或批量旋转键入口，不创建客户端 bootstrapper，也不挂载完整 `context.evk`。
2. 合并通用正负旋转、共轭及 `RotationsForBootstrapping(params)` 所需集合，按 `galEl` 去重。在 marshal BTP 参数前调用 `RotationsForBootstrapping`，保留其对 `LogN/LogSlots` 的更新；键数量由参数推导，不硬编码 50。
3. 严格按 **Scheme → SK 标记 → PK → BTP 参数 → RLK → RTKs → DtS → StD** 写出。Scheme 长度 LittleEndian；`has_sk` 无条件写 `0`，绝不根据 `context.sk != nil` 决定，也不调用 `SecretKeyToBytes`；`has_pk=1`。BTP 参数长度使用 `uint8`，超过 255 报错。
4. RLK 和 RTKs 复用 M1 的单键路径和 buffer。DtS/StD 直接调用 `GenEncapsulationSwitchingKeys(params, sk)`，随后分别序列化、写出和丢弃引用；不为拆开成对生成而改核心代码。
5. `EphemeralSecretWeight == 0` 时两个切换键标志写 `0`。其他情况按原 serializer 用 `Q[:1]+full P` 参数编码 DtS、full 参数编码 StD，检查 sparse 参数构造错误，复用同一个 buffer。

验证后提交：覆盖含 SK 的客户端导出边界、BTP 字段顺序、`uint16` 数量、完整 Galois 并集、sparse/full 编码、零 ephemeral weight，以及原 deserializer 导入和 bootstrap 功能。确认无尾部多余字节。参考参数下复核 `3,329,929,205` 字节推导；文件大小只是辅助检查，不能替代无 SK payload 的结构断言或推广到其他参数。

M1/M2 都复用已有底层实现，不修改 `rlwe/bit_stream.go`、`rlwe/keygenerator.go` 或 `ckks/bootstrapping/bootstrapper.go`。

### M3：LattiSense C++ 包装与构建

修改 LattiSense 的 `fhe_ops_lib/fhe_lib_v2.h/.cpp`，把 `fhe_ops_lib/lattigo` gitlink 更新到 M2 提交；复用 `unittests/test_lattigo.cpp` 增加针对新接口的用例。

1. 为 `CkksContext` 和 `CkksBtpContext` 增加流式创建及 `export_eval_context_to_file(path)` 薄包装，把参数、路径传给 Go，接收 context 句柄并沿用现有 RAII 生命周期。
2. 每个新调用检查状态码，用 `CHECK` 和既有错误消息转换为 C++ 异常；失败不包装无效句柄，不经 `export_raw_data` 或 `Bytes` 传递评估键文件。
3. 原随机构造器、批量密钥生成及内存 serializer 保留，供已有完整 context 和旧路径对照使用；默认流式调用不经过这些入口。
4. 用既有 `go_libs` 构建重新生成 `liblattigo.a/.so`、`liblattigo.h` 和 sanitized 头，再构建 `fhe_ops_lib`。核对后发现两个生成头在实现前基线中已被跟踪，因此提交其重新生成的声明，不手写或增加新的跟踪规则；不要把动态/静态库提交进源码。

验证后提交：两个 C++ context 均能创建、导出、加解密；Go 失败可被 C++ 捕获且文件不错误发布。用原 C++ 导入包装读取新文件，验证旧完整 context 的内存序列化仍可使用。新符号必须由重新生成的头声明并由实际库链接成功，避免只编译头或加载旧库。

### M4：服务端文件导入

修改顶层 `inference/interface/inference_server.h/.cpp`，将 `inference/lattisense` gitlink 更新到 M3；相关读取失败用例与本模块一起提交，必要时在现有 CMake 中登记最小测试目标。

1. 新增 `import_eval_context_file(path)`，使用 `std::ifstream` 的 binary/ate 模式获取长度。
2. 检查打开和 `tellg`、非正长度、`size_t`/`streamsize` 表示范围、`Bytes::max_size()`、seek 和实际读取结果。
3. 完整读入一个 `Bytes` 后调用已有 `import_eval_context`，保留 task config 读取、context 挂载、模型加载逻辑；不新增第二套 deserializer。

验证后提交：普通 CKKS/BTP 新文件都走原导入路径；覆盖不存在、空文件、长度不可表示、读取短于已获取长度。大小边界可通过内部校验用例覆盖，无需分配超大文件。不要把当前 parser 不能可靠拒绝任意损坏 payload 的行为写成验收通过。

### M5：默认客户端 setup 与端到端示例

修改顶层 `inference/interface/inference_client.h/.cpp` 和 `examples/inference.cpp`，相关 C++ 调用测试随本模块提交。

1. 保留 `InferenceClient::setup()` 名称及参数读取逻辑，按 `needs_btp_` 调用对应的流式创建包装。保留普通 CKKS 自定义 Q/P、`log_slots` 等既有参数选择；在调用批量 context 创建或 `gen_rotation_keys()` 之前替换生成路径。
2. 默认写 `client_dir_/eval_context.bin`，完整关闭并发布后才返回。客户端保留 SK/PK 及加解密状态，不保留完整评估键；流式过程同一 context 不可被其他操作并发修改。
3. 新增 `InferenceClient::export_eval_context_to_file(path)`，按 CKKS/BTP 分支调用同一底层 writer，供已有 SK/PK context 或自定义目的路径使用。
4. 保留原 `export_eval_context()` 实现作为完整评估键 context 的兼容/对照入口。新默认 setup 已不持有完整评估键，默认流程不能随后调用旧内存导出；不为兼容该调用额外缓存键、读取全量文件或重建全套键。旧回归通过原 SDK 构造器或加载已有完整 context 准备输入，不增加新的公开批量 setup 模式。
5. 示例改为 `setup()` 后使用 `client/eval_context.bin`，服务端调用 `import_eval_context_file`；删除默认流程里的全量内存导出。文件与密文的跨机器传输仍由调用方负责。

验证后提交：普通 CKKS 与 BTP 的 setup、文件导入、加密、评估、解密完整链路均可用；失败不成功返回，成功时目标文件已完整发布。示例与默认行为在同一 commit 改动，避免示例继续依赖客户端常驻全套键。

重新构建示例后执行规范要求的任务回归：

```bash
ctest --test-dir build -R '^example-cifar10-cpu$' -V
ctest --test-dir build -R '^example-mnist-cpu$' -V
```

检查输出键/数量完整、数值有限、误差满足各任务容限；CIFAR-10 按 `0.1` 判定，不要求等于历史误差。记录新文件导入的证据，不能用未改示例的旧 CTest 结果替代。

### M6：Node 默认调用与验证

检查顶层 `inference/interface/node/binding.cpp`、`test_client.js`，复用现有 `package.json` 和 CMake 构建规则。

1. 保留 `SetupWorker::Execute()` 调用 `client_->setup()`、`OnOK()` resolve `undefined`、`OnError()` reject 的现有 Promise 形态；M5 已使它调用流式入口。若既有绑定无需变化，不为新增方法名或显式模式开关改写 `binding.cpp`。
2. 默认 Node 测试执行 `await client.setup()` 后检查 `client_dir/eval_context.bin` 已发布，再验证加解密；不继续调用 `exportEvalContext()` 把整套评估键变成 Node Buffer。旧内存接口的回归使用前述完整 context 路径，与默认测试区分。
3. 覆盖文件写入失败时 Promise reject；同一 client 的调用顺序等待前一步完成。默认文件路径已经明确，本次不增加可选路径参数或另一套 Node keygen/serializer。
4. 先重新构建 Go 库及 `fhe_ops_lib`，再使用现有 `cmake-js`/npm 构建 Node addon，显式传入新库路径。运行时也使用这次构建的依赖库。

Linux 验证命令示例（从顶层运行；`build` 是本次实际构建目录）：

```bash
cmake --build build --target go_libs fhe_ops_lib inference
npm --prefix inference/interface/node run rebuild -- --CDFHE_OPS_LIB="$PWD/build/inference/lattisense/fhe_ops_lib/libfhe_ops_lib.so"
LD_LIBRARY_PATH="$PWD/build/inference/lattisense/fhe_ops_lib:$PWD/inference/lattisense/fhe_ops_lib/lattigo/go_sdk${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" npm --prefix inference/interface/node test
```

验证后提交：重编译的 addon 保留 `client.setup()` Promise，并只在文件发布后 resolve；普通 CKKS/BTP 默认流程和失败 reject 均有验证记录。需要覆盖 BTP 时增加对应任务 fixture，不把只运行普通 CKKS 的测试写成两个分支都通过。Python 绑定改动不在本计划范围内。

### M7：独立客户端内存测量与最终验收

在顶层增加最小的独立客户端测量程序/运行脚本及验收记录，归入对应示例或测试目录；必要时只登记所需构建目标。测量不能在同时执行服务端导入和推理的进程中进行。

1. 用原 SDK 批量构造和旋转键接口准备旧基线，分别记录旧密钥生成以及后续内存导出阶段；新路径从客户端流式 setup 开始到文件发布结束。旧新各运行独立进程，不在同一进程先后测量，不把新路径前置的旧初始化开销遗漏。
2. 固定普通 CKKS/BTP 参数、Galois 集合、Go 版本、线程数、GC 配置和测量边界；报告硬件/环境、用时、文件大小和原始日志。大参数下旧新输入参数和 key set 一致，不要求随机生成字节一致。
3. 用 `/usr/bin/time -v` 记录整个客户端阶段的峰值 RSS（KiB）；用 `runtime.ReadMemStats` 记录生成/写出、GC 后和发布后的 `HeapAlloc/HeapInuse`，区分 Go 堆、RSS 和 MB/MiB。采样仅用于验收，不要求增加常驻公开遥测接口。
4. 检查没有完整 key set 或整文件序列化副本存活，计入单键 buffer 的实际容量、扩容、key generator 工作区、未回收对象和 DtS/StD 成对生成开销。丢弃引用不等于内存已回收。
5. 先按默认 GC 配置测量。若峰值没有显著降低，定位引用与回收问题，再评估仅在流式函数内的按键回收等最小措施；需要改代码时在所属仓库补充独立修复 commit，并按 M3/M4 顺序更新 gitlink，重跑受影响验收。不得以 Go soft limit 或理论单键大小宣称 RSS 达标。

验证后提交测量工具和真实结果。完成条件是同环境下客户端**密钥生成阶段**峰值 RSS 相比旧批量生成显著降低，同时整个生成/导出阶段的峰值和耗时有记录，全部协议、安全边界、文件失败、Node 和任务精度用例通过；不预设无测量依据的绝对内存阈值或降幅。未满足首要条件时，M7 不标记完成。

## 3. 验收与规范对应

| 验收项 | 负责步骤 | 必须保留的证据 |
| :--- | :--- | :--- |
| 普通 CKKS 协议、单键编码和功能等价 | M1、M3 | BigEndian 参数长度、完整结构、原导入及同 SK 功能结果 |
| BTP 协议、旋转/共轭、DtS/StD、bootstrap | M2、M3 | 字段顺序、数量、完整 key set、sparse/full 及零 ephemeral 用例 |
| 无 SK payload | M1、M2 | 含 SK 的输入、固定零标志、结构断言及导入后 SK 为 nil |
| 短写、磁盘写满、Close/Rename/清理失败 | M1、M2、M3、M6 | 错误传播、旧文件保留、临时文件清理或路径报告 |
| 文件读取失败 | M4 | 长度边界、seek/read 错误用例，不扩大 parser 承诺 |
| 默认 setup 及 Node Promise | M5、M6 | 发布后返回/resolve、失败 reject、保留 SK/PK 并可加解密 |
| 旧内存接口回归 | M3、M5、M6 | 从完整 context 出发的原序列化/导入结果 |
| CIFAR-10 BTP、普通 CKKS 端到端 | M5、M7 | 新文件路径的 CTest 日志、完整且有限的输出、误差容限 |
| 客户端峰值内存首要条件 | M7 | 独立进程、同条件旧新基线、分阶段堆数据、峰值 RSS 和降幅 |

每个模块提交前执行 `git diff --check` 并检查实际暂存文件及所属仓库。最终核对三个仓库都在 `feat/streaming-key-generation`，gitlink 指向已提交实现，工作树没有遗留实现改动；分支名不替代 gitlink。完成状态须以验收记录中的实际证据为准。
