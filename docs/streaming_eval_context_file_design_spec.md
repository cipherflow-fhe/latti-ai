# Latti-AI 密钥流式生成落盘与服务端密文推理设计规范（方案 A）

## 一、范围、背景与设计目标

### 1. 本规范的范围

本文只定义 evaluation context 的计算公钥逐键生成、流式序列化、文件落盘和服务端文件导入。输入 context 必须已有与任务匹配的 SK 和 Scheme 参数；BTP 当前协议固定写入 PK，因此 BTP context 还必须已有 PK 和 BTP 参数；普通 CKKS 文件导出沿用现有客户端不包含 PK 的策略。`setup_keypair()` 的实现和普通 CKKS/BTP context 的初始化不在本文范围内。

低内存导出的前置条件是客户端没有预先常驻完整评估键。已有全套密钥的 context 也可以写文件，但它的常驻内存不会因文件导出接口自动下降。

### 2. 现状痛点

现有 `client.setup()` 会生成并持有全套评估键。`export_eval_context()` 随后在 Go 侧构造完整 `bytes.Buffer`，C++ 的 `export_raw_data` 再复制为 `std::vector<uint8_t>`；客户端同时承受密钥对象和全量序列化副本的内存开销。

以 CIFAR-10、`N=65536`、`N16QP1546H192H32` 为参考参数，evaluation context 的压缩字节流约为 3.10 GiB。原稿中的常驻 RSS `14,408 MB`、峰值 `21,723 MB ～ 28,665 MB` 缺少测量环境和原始日志，只能作为待复核的历史报告值。

### 3. 方案 A 核心思想

1. **客户端逐键处理**：用现有生成接口生成评估键，用现有 serializer 编码到单键 `bytes.Buffer` 后直接写文件，丢弃该键引用，再处理下一个。BTP 的 DtS/StD 复用现有成对生成接口，随后分别写出；不构造完整 evaluation context 的 `bytes.Buffer` 或 C++ `Bytes`。
2. **服务端标准文件读取**：使用 `std::ifstream` 将完整文件读入 `Bytes`，再复用 `import_eval_context`。服务端仍承担完整文件和反序列化对象的内存开销，不引入 `mmap` 或分块反序列化。
3. **保持现有协议**：字段顺序、字节序、数量宽度、压缩编码和密钥集合与对应的现有 advanced serializer/deserializer 一致。
4. **测量收益与精度**：以相同参数和环境下客户端密钥生成及导出阶段的峰值 RSS 相比旧路径显著降低为首要验收条件，报告实测值和降幅，并验证任务误差容限。达到该条件的方案优先选择代码改动最小者，不预设缺少测量依据的绝对内存阈值。

“协议兼容”指格式和解析后的语义兼容。现有旋转键序列化遍历 Go map，顺序不稳定；不同随机生成的密钥也具有不同字节值，因此不要求新旧导出结果无条件逐字节相等。

---

## 二、端到端架构与数据流

```text
Client：已有 SK/PK 和参数的 context（初始化方式不在本文范围内）
  │
  ├─ export_eval_context_to_file(path)
  │    ├─ 校验前置条件并计算 Galois 集合，创建同目录临时文件
  │    ├─ BTP：Scheme → SK 标记 → PK → BTP 参数
  │    ├─ 普通 CKKS：Scheme → has_sk=0 → has_pk=0
  │    ├─ 生成 RLK，用现有 serializer 编码到单键 buffer 并写文件，丢弃引用
  │    ├─ 写 RTKs 标记和 uint16 数量
  │    ├─ 按 galEl 逐键生成、写出并丢弃（含共轭键）
  │    ├─ BTP：调用 GenEncapsulationSwitchingKeys，分别编码并写 DtS/StD
  │    └─ 检查写入、Close 和 Rename 的结果
  │
  ├─ encrypt(input)
  │
  └───────── 完成后的 eval_context.bin ─────────► Server
                                                   │
                                     import_eval_context_file(path)
                                                   │
                                     ifstream 读取为 Bytes
                                                   │
                                     复用 import_eval_context
                                                   │
                                     load_model() / evaluate()
                                                   │
Client.decrypt(encrypted_outputs) ◄─────────────────┘
```

只有导出成功返回后，服务端才可读取目标文件。共享文件路径只是本地示例；部署在不同机器时，调用方负责传输完整文件。

---

## 三、二进制文件格式（`eval_context.bin`）

文件没有额外的容器头、分块长度或尾部校验和；不能增加会破坏现有 deserializer 的字节。除特别说明外，整数均为 LittleEndian，`has_*` 均为一字节 `0/1` 标志。payload 延续 `rlwe/bit_stream.go` 的 seed 压缩和多项式位打包编码。

### 1. BTP CKKS（`use_btp = true`）

| 顺序 | 字段 | 类型/长度 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | `param_len` | `uint32` LittleEndian，4 字节 | Scheme 参数长度 |
| | `param_data` | `bytes[param_len]` | `ckks.Parameters.MarshalBinary()` |
| 2 | `has_sk` | `uint8`，1 字节 | 评估上下文固定为 `0`，没有 SK payload |
| 3 | `has_pk` | `uint8`，1 字节 | 延续现有 BTP 评估上下文，固定为 `1` |
| | `pk_payload` | 变长；参考参数为 `12,746,858` 字节 | `CiphertextQPToBytes` 编码：一个 QP 多项式及重建另一分量的 64 字节 seed |
| 4 | `btp_param_len` | `uint8`，1 字节 | BTP 参数长度，超过 255 字节必须报错 |
| | `btp_param_data` | `bytes[btp_param_len]` | `bootstrapping.Parameters.MarshalBinary()` |
| 5 | `has_rlk` | `uint8`，1 字节 | 固定为 `1` |
| | `rlk_payload` | 变长 | `RelinearizationKeyToByte` 编码：子键数 `uint8`，随后是每个 SwitchingKey |
| 6 | `has_rtks` | `uint8`，1 字节 | 固定为 `1` |
| | `rtks_count` | **`uint16` LittleEndian，2 字节** | 去重后的 Galois 键数量，包含共轭键；超过 65535 必须报错 |
| | `rtks_payload` | 连续 `rtks_count` 个子块 | 每项为 `galEl:uint64` LittleEndian，然后是 SwitchingKey 的 `GadgetCiphertext` |
| 7 | `has_swk_dts` | `uint8`，1 字节 | `EphemeralSecretWeight != 0` 时为 `1`，否则为 `0` |
| | `swk_dts_payload` | 标志为 `1` 时写入 | `GadgetCiphertextToBytes` 编码，使用 `Q[:1] + full P` 的 sparse 参数 |
| 8 | `has_swk_std` | `uint8`，1 字节 | `EphemeralSecretWeight != 0` 时为 `1`，否则为 `0` |
| | `swk_std_payload` | 标志为 `1` 时写入 | `GadgetCiphertextToBytes` 编码，使用完整 Scheme 参数 |

顺序必须是 **Scheme → SK 标记 → PK 标记及 payload → BTP 参数 → RLK → RTKs → DtS → StD**。BTP 参数不能提前写到 SK/PK 标记之前；旋转键数量不能写成 `uint32`。

### 2. 普通 CKKS（`use_btp = false`）

普通 CKKS 的参数长度使用另一种字节序：

1. `param_len:uint32` **BigEndian**；
2. `param_data`；
3. `has_sk=0`；
4. `has_pk` 和可选的 `CiphertextQP` payload；
5. `has_rlk=1` 和 `RelinearizationKey` payload；
6. `has_rtks=1`、`rtks_count:uint16` LittleEndian，再逐项写 `galEl:uint64` LittleEndian 和 SwitchingKey payload。

普通 CKKS 文件导出保持现有 `InferenceClient::export_eval_context()` 语义，写 `has_pk=0`；格式中的可选 PK 字段仍遵循现有 advanced serializer/deserializer，不为本文件接口新增 PK 策略参数。普通 CKKS 没有 BTP 参数或 DtS/StD 字段。

### 3. 参考大小

在当前 `N16QP1546H192H32`、50 个 Galois 键（49 个列旋转键加 1 个共轭键）、现有编码不变的条件下，可推导以下长度：

| 字段 | 字节数 |
| :--- | ---: |
| Scheme 参数 payload | 270 |
| BTP 参数 payload | 123 |
| PK payload | 12,746,858 |
| 一个 full-level SwitchingKey payload | 63,734,292 |
| DtS payload（`Q[:1] + full P`） | 2,998,356 |
| 完整 BTP evaluation context（含所有长度和标志） | 3,329,929,205 |

这些是参考参数的编码长度推导，尚不是流式导出后的文件实测值。参数、key set 或编码改变后必须重新计算；不得把固定 PK 长度或总大小作为通用协议字段。

---

## 四、客户端流式生成与落盘

### 1. 复用现有序列化接口

现有 `CiphertextQPToBytes`、`GadgetCiphertextToBytes`、`SwitchingKeyToBytes`、`RelinearizationKeyToByte` 等函数接受 `*bytes.Buffer`，可直接复用。

最小方案在 Go SDK 导出函数内复用一个 `bytes.Buffer`，每次只编码当前 payload，完成后调用 `WriteTo(file)` 写入临时文件；成功后 buffer 自动清空，供下一次编码使用。以 SwitchingKey payload 为例：

```go
var keyBuf bytes.Buffer // 导出函数局部，跨 payload 复用

rlwe.SwitchingKeyToBytes(swk, &params.Parameters, &keyBuf)
if _, err := keyBuf.WriteTo(file); err != nil {
    return err
}
```

PK、RLK、DtS/StD 分别调用对应的现有 serializer，保持 seed 压缩和多项式位打包编码。这里只暂存当前 payload 的压缩字节；小型 Scheme/BTP 参数继续使用 `MarshalBinary`，不编码或复制完整 context，也不积累旋转键集合。

`bytes.Buffer.WriteTo` 直接把当前内容交给文件写入，返回写入错误或 `io.ErrShortWrite`，不额外复制整段 payload。已有 serializer 向内存 buffer 写入，不涉及磁盘错误；文件写入错误在 Go SDK 层检查并传播即可。无需修改 `rlwe/bit_stream.go`，也无需新增 writer 接口、serializer 变体或额外的 32 MiB `bufio.Writer`。

`WriteTo()` 成功后清空内容，但保留 buffer 容量供下一把键复用；容量取决于最大 payload 及 buffer 扩容行为，不能把序列化长度当成其全部内存开销。每次写出后限制临时键的作用域，下一把键生成前不再保留上一把键的引用。

### 2. 完整旋转/共轭键集合

写数量前，先计算并去重完整的 `galEl` 集合：

- 普通 CKKS：沿用现有正负幂次旋转（正向 `i=0..LogN-2`、负向 `i=0..LogN-3`），加入 `GaloisElementForRowRotation()` 共轭键。
- BTP：在上述集合中再合并 `btp_parameter.RotationsForBootstrapping(params)` 所需的旋转，包括相应的 trace/编码矩阵旋转。

按 `galEl` 去重后写出，项间顺序不影响现有 deserializer，无需新增排序步骤。当前参考 BTP 参数得到 50 个键；该数值必须由参数推导，不能硬编码。只写列旋转无法生成共轭键，而 BTP 的 `CoeffsToSlots` 会执行共轭操作。

`RotationsForBootstrapping` 会更新 BTP 编码矩阵参数中的 `LogN`/`LogSlots`，因此必须在序列化 BTP 参数前调用，使参数内容与旧密钥生成路径一致。

逐项写入流程为：

```text
write has_rtks = 1
write uint16(len(galEls)) in LittleEndian
for galEl in deduplicated galEls:
    write galEl as uint64 LittleEndian
    swk = kgen.GenSwitchingKeyForGalois(galEl, sk)
    encode swk using SwitchingKeyToBytes into keyBuf
    check keyBuf.WriteTo(file)
    discard swk reference
```

每次文件写入都必须检查错误。`GenSwitchingKeyForGalois` 内部完成所需的逆自同构转换，调用方传入的 `galEl` 与文件中的标识一致。复用现有 key generator 完成整个循环，不调用会累计全套键的 `GenRotationKeys` / `GenEvaluationKeys`，已经写出的 key 不加入 `context.gk`、`context.evk` 或其他持久容器。

批量 `GenRotationKeys` 与逐键 `GenSwitchingKeyForGalois` 均在完整 Q/P level 分配键，并调用同一 `genrotKey → genSwitchingKey`，包括相同的逆 Galois 变换与分解参数；逐键路径复用现有采样算法。

默认单 goroutine 顺序生成、编码和写出，复用同一个 `kgen` 与 buffer；其 `buffQP`/`buffQ` 和采样器状态不能被并发访问。本仓库的批量 `GenRotationKeys` 已通过 `WorkerPool` 并行生成，流式路径绕过该入口以减少键对象和并发工作区叠加；耗时与峰值 RSS 一起实测。

### 3. 复用 BTP DtS/StD 成对生成接口

直接调用现有 `btpParams.GenEncapsulationSwitchingKeys(params, sk)`。该接口内部只采样一次 `skSparse`，保证两个方向的切换键匹配，无需在 `ckks/bootstrapping/bootstrapper.go` 新增顺序生成接口或回调：

1. 若 `EphemeralSecretWeight == 0`，现有接口返回两个 nil；两个标志都写 `0`，不解引用切换键。
2. 否则取得 `swkDtS` 和 `swkStD`，它们只作为导出函数局部变量，不挂到 context 上。
3. 按现有 BTP serializer 构造编码用的 `paramsSparse`：`LogN` 不变、`Q=params.Q()[:1]`、`P=params.P()`，检查参数构造错误。写 `has_swk_dts=1`，用 `GadgetCiphertextToBytes` 和 sparse 参数编码 DtS，写文件后丢弃 DtS 引用。
4. 写 `has_swk_std=1`，用 `GadgetCiphertextToBytes` 和完整 Scheme 参数编码 StD，写文件后丢弃 StD 引用。两次编码分别复用同一个 `keyBuf`。

这一阶段会短暂同时持有两把切换键，以及生成函数内部的临时私钥和 key generator 工作区，必须计入峰值测量。它避免了完整评估键集合常驻，也省去为严格“同时只持有一把键”改动 BTP 核心的成本；优先验证该最小方案能否满足显著降低客户端峰值的目标。

### 4. 导出顺序与错误处理

下面为流程伪代码；每个文件写入操作都必须检查返回错误：

```text
validate SK/PK and parameters required by the scheme, and destination
compute galEls before serializing parameters
validate key count
marshal small Scheme/BTP parameters and check errors and lengths
create a unique temporary file in the destination directory
keyBuf = one reusable bytes.Buffer

write Scheme length and payload with the scheme-specific byte order
write has_sk = 0
write has_pk = 1 and PK payload for BTP; has_pk = 0 for ordinary CKKS
if BTP: write BTP length and payload here

write has_rlk = 1
rlk = kgen.GenRelinearizationKey(sk, 1)
encode rlk using RelinearizationKeyToByte into keyBuf
check keyBuf.WriteTo(file)
discard rlk reference

write all RTKs as described above
if BTP: call GenEncapsulationSwitchingKeys, then encode/write DtS and StD separately

check file.Close()
check os.Rename(temporaryPath, destination)
return success only after publication succeeds
```

文件写入直接使用 `os.File`；检查 `binary.Write`、文件 `Write`、buffer `WriteTo`、`Close` 和 `Rename` 的返回结果，任一失败均返回失败。使用同目录临时文件，成功关闭后再发布，避免失败时截断已有目标；不能仅使用 `defer file.Close()` 后无条件返回 `0`。

写入失败时关闭并清理临时文件，已有目标文件保持不变；清理失败时在错误信息中附带临时路径。Linux 使用同目录 `os.Rename` 原子替换，其他平台验证其替换语义，不先删除旧目标。

CGO 导出函数返回 `0` 表示成功、非零表示失败，设置现有错误消息机制供 C++ `CHECK` / `GetErrorMessage()` 读取。校验失败和可处理的生成错误不能以未捕获 panic 穿过 CGO 边界。流式导出期间，同一 context 不得被其他操作并发修改。

### 5. C++ 包装接口

本规范新增以下文件接口；SK/PK context 的准备方式由调用方提供：

```cpp
class InferenceClient {
public:
    // Requires SK and matching scheme parameters; BTP also requires PK and BTP parameters.
    // Generates and streams evaluation keys; throws on export failure.
    void export_eval_context_to_file(const std::string& filepath);
};
```

`CkksContext` 和 `CkksBtpContext` 分别包装对应的 Go stream export。文件导出不在客户端保留新生成的完整评估键集合；旧 `setup()` / `export_eval_context()` 路径仍按原语义运行。

---

## 五、服务端文件导入

新增 `InferenceServer::import_eval_context_file`，读取后调用原 `import_eval_context`。文件大小和读取失败在 C++ 层检查，避免把空文件或非法长度交给 CGO。示意代码如下：

```cpp
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

void InferenceServer::import_eval_context_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open eval context file: " + path);
    }

    const std::streamoff size = file.tellg();
    if (size <= 0 ||
        static_cast<std::uintmax_t>(size) > std::numeric_limits<size_t>::max() ||
        static_cast<std::uintmax_t>(size) > std::numeric_limits<std::streamsize>::max()) {
        throw std::runtime_error("Invalid eval context file size: " + path);
    }
    Bytes buffer;
    if (static_cast<std::uintmax_t>(size) > buffer.max_size()) {
        throw std::runtime_error("Eval context file is too large: " + path);
    }
    buffer.resize(static_cast<size_t>(size));
    if (!file.seekg(0, std::ios::beg) ||
        !file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("Failed to read eval context file: " + path);
    }
    this->import_eval_context(buffer);
}
```

此路径需要服务端内存容纳完整文件和解析后的对象；它保留现有 task config 读取及 context 挂载流程。标准文件 I/O 可移植，不代表整个仓库已经通过所有目标平台的编译与运行验证。

文件读取成功只证明读取了指定字节数。现有 Go deserializer 忽略了部分读取/解析错误，因此不能承诺它会可靠拒绝任意截断或损坏的 payload。本文的正常导入前提是已完成并完整传输的导出文件；需要可靠拒绝损坏文件时，应另行补齐解析错误检查或结构校验，不能靠文件大小替代。

---

## 六、端到端使用与测试

### 1. 调用边界

以下示例从已经准备好的 `client` context 开始；不定义或实现 `setup_keypair()`：

```cpp
std::string eval_context_file = task_dir + "/server/eval_context.bin";
client.export_eval_context_to_file(eval_context_file);
auto encrypted_inputs = client.encrypt(input_csvs);

InferenceServer server(task_dir + "/server", use_gpu, 0);
server.import_eval_context_file(eval_context_file);
server.load_model();
auto encrypted_outputs = server.evaluate(encrypted_inputs);
auto results = client.decrypt(encrypted_outputs);
```

服务端只有在导出及文件传输成功后才启动导入。端到端示例中的同进程峰值包含服务端开销，不能作为“客户端峰值 RSS”的测量值。

### 2. 必需验证

| 用例 | 验证内容 |
| :--- | :--- |
| 普通 CKKS 协议 | 参数长度 BigEndian、`has_sk=0`、`has_pk=0`、RLK/RTKs payload 可由原 advanced deserializer 导入 |
| BTP 协议 | 参数顺序、`uint16` 键数量、DtS/StD sparse/full 参数编码正确，无多余字节 |
| 编码兼容 | 单键 payload 直接复用原 serializer；文件由原 deserializer 导入，检查参数、字段、key set 和功能，不要求旋转键排列相同 |
| Galois 完整性与功能等价 | BTP 专用旋转、通用正负旋转及共轭键组成的并集与旧路径一致；小参数下用同一 SK 和密文，分别以批量键与流式键执行旋转、共轭和重线性化，BTP 另验证 bootstrap，按 CKKS 误差容限比较解密结果 |
| BTP 切换键 | 复用 `GenEncapsulationSwitchingKeys`，验证 DtS/StD 功能及 sparse/full 编码；覆盖 `EphemeralSecretWeight == 0` |
| 写入失败 | 覆盖短写、磁盘写满、Close/Rename 失败，检查非零返回与 C++ 异常；旧目标保留，临时文件清理或报告清理失败 |
| 文件读取失败 | 覆盖不存在、空文件、不可表示的长度和读取短于 `tellg` 的文件；不把当前 parser 的缺陷当成已验证的损坏文件处理能力 |
| 客户端内存 | 用独立客户端进程对比旧路径密钥生成加内存导出与新路径逐键生成落盘，记录相同参数、key set、Go 版本、线程数、GC 配置、Heap 和峰值 RSS；确认新路径不累计键、不分配完整文件 buffer，且峰值显著降低 |
| 端到端回归 | CIFAR-10 BTP 和普通 CKKS 任务均通过各自误差容限；保留旧内存导出路径回归 |

新路径接入示例并重新构建后，运行：

```bash
ctest --test-dir build -R '^example-cifar10-cpu$' -V
ctest --test-dir build -R '^example-mnist-cpu$' -V
```

这些命令仅验证相应的端到端任务，不能替代协议、失败路径和独立客户端内存测试。当前源码的示例仍走旧内存导出路径，现有 CTest 成功不能证明流式接口已验收。

重新生成密钥、评估键和加密噪声后，数值误差可以发生变化。CIFAR-10 当前示例 tolerance 为 `0.1`，应验证输出键/数量完整、数值有限且最大绝对误差不超过容限；不要求每次等于历史记录 `0.00044758`。

---

## 七、工程变动清单（仅流式相关）

| 层级 | 文件 | 变动说明 |
| :--- | :--- | :--- |
| Go SDK BTP | `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/bootstrap.go` | 新增 `StreamExportCkksBtpEvalContextToFile`；复用逐键生成、成对 DtS/StD 生成和现有 serializer，以单键 buffer 写文件 |
| Go SDK CKKS | `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/main.go` | 新增 `StreamExportCkksEvalContextToFile`；复用生成及 serializer 接口，共用小型文件写入/错误处理逻辑，保持 BigEndian 参数长度及不包含 PK 的策略 |
| 生成头文件 | `go_sdk/liblattigo.h`、`go_sdk/liblattigo_sanitized.h` | 由 Go 构建及 CMake 重新生成，使新的 CGO 导出符号可见；不手写生成头 |
| LattiSense C++ | `inference/lattisense/fhe_ops_lib/fhe_lib_v2.h/.cpp` | 为两个 context 增加文件导出包装，检查 CGO 返回值 |
| Client | `inference/interface/inference_client.h/.cpp` | 新增 `export_eval_context_to_file`，分别调用 CKKS/BTP 分支 |
| Server | `inference/interface/inference_server.h/.cpp` | 新增文件导入，检查长度及读错误，复用既有导入逻辑 |
| 示例与测试 | `examples/inference.cpp`、对应协议/失败路径测试 | 接入文件路径并覆盖普通 CKKS、BTP、旧接口和独立内存测量 |

生成算法和底层编码直接复用，不改 `rlwe/bit_stream.go` 或 `ckks/bootstrapping/bootstrapper.go`。上述清单不包含 keypair-only 初始化或其他密钥生命周期 API。本文只承诺 C++ 接口范围；Python/Node 若需暴露文件 API，需另行增加绑定。

---

## 八、内存核算与验收口径

### 1. 密钥存储与进程内存

设 `LQ`/`LP` 为 Q/P 素数数量，`D_RNS`/`D_BIT` 为分解数量，则原始系数存储约为：

```text
一个 QP 多项式：8 * N * (LQ + LP) bytes
SK：一个 QP 多项式
PK：两个 QP 多项式（文件编码只保存一个分量及 seed）
一个 SwitchingKey：2 * D_RNS * D_BIT * 8 * N * (LQ + LP) bytes
```

参考参数 `N=65536, LQ=25, LP=5, D_RNS=5, D_BIT=1` 下，SK 为 15 MiB、PK 为 30 MiB、一个 full-level SwitchingKey 为 150 MiB。SK/PK 的原始下限已是 45 MiB，不能写成进程常驻 RSS 约 36.5 MB。

单个 full-level SwitchingKey 的压缩 payload 约 60.8 MiB。最小方案保留这一单键序列化 buffer，而不是约 3.10 GiB 的完整 context buffer；buffer 容量和扩容期间的临时分配也要计入内存。DtS/StD 成对生成的阶段还需计入两把键及其生成工作区。

这些数值未包含参数/NTT 表、key generator scratch、encoder/encryptor/decryptor 工作区、未回收对象及 Go/C++ 分配器。评审探针复用现有初始化组件，在没有 RLK/GK/bootstrapper 时观察到约 575.5 MiB 的 Go `HeapAlloc`；它不是 RSS，也不是本流式实现的测量结果，但说明不能只按 SK/PK 估算客户端总内存。

`key = nil` 只表示丢弃引用。应限定临时 key 的作用域，确认它未被 context 或其他容器保留，并在每键边界验证 GC 行为；复用单个序列化 buffer 以减少重复分配。先用现有 GC 配置测量，若回收滞后导致峰值不达标，再在导出函数内评估按键回收等局部措施，不把 GC 或 soft memory limit 视为 RSS 硬上限。本文不改变输入 context 的初始化方式或内存预算。

### 2. 指标状态

| 指标 | 基线或推导 | 方案 A 状态/验收 |
| :--- | :--- | :--- |
| 旧路径 RSS | 原稿数字缺测量证据 | 在固定环境下重测，与新路径对比 |
| 客户端流式峰值 RSS | 不由单 key 大小直接决定 | 首要验收：同参数、同环境下相比旧路径显著降低；待实现测量，报告峰值和降幅，不宣称已验收 |
| 评估键内存行为 | key set 大小随参数变化 | 不累计已写出的评估键，仅复用单键序列化 buffer；允许现有 DtS/StD 接口短暂同时持有两把键，不分配完整 context 的序列化副本 |
| BTP 文件大小 | 当前参考条件推导为 `3,329,929,205` 字节 | 待真实输出复核结构和大小；不作为所有参数的固定值 |
| 新旧协议兼容 | 现有 serializer/deserializer 是依据 | 普通 CKKS/BTP 均需通过兼容性测试 |
| CIFAR-10 精度 | 旧路径历史误差 `0.00044758` | 新路径结果待测，按 tolerance 判定，不能复制旧数字作为新路径实测 |

独立客户端测量应分别记录导出前、每键生成/写出、GC 后和文件提交后的 `HeapAlloc`、`HeapInuse`、RSS、峰值 RSS，并注明 MB/MiB 单位。端到端进程的服务端导入和推理开销另行报告。

Linux 下使用 `/usr/bin/time -v` 的 `Maximum resident set size (kbytes)` 记录独立客户端进程峰值 RSS（KiB），用 `runtime.ReadMemStats` 采集各阶段的 Go 堆指标；循环结束时的堆用量不能替代整个生成及导出过程的峰值。
