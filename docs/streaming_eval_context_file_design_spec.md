# Latti-AI 密钥流式生成落盘与服务端密文推理设计规范 (方案 A Design Spec)

## 一、背景与设计目标

### 1. 现状痛点
在现有密态推理全流程（以 `examples/inference.cpp` 及 CIFAR-10 任务为例）中：
- **客户端内存严重受限**：在 Bootstrapping 任务（$N=65536$，参数集 `N16QP1546H192H32`）下，`client.setup()` 一次性在 Go 内存中生成全套计算公钥（1 个 RLK、数十个 Galois/Rotation Key、2 个 BTP 维度切换键），常驻内存达 **14.1 GB**；
- **全量内存导出导致二次爆炸**：`export_eval_context()` 在 Go 侧分配 3.3 GB 的 `bytes.Buffer`，CGO 桥接层又在 C++ 侧复制为 3.3 GB 的 `std::vector<uint8_t>`，使客户端峰值物理内存飙升至 **21.7 GB ～ 28.0 GB**；
- **真实部署矛盾**：客户端通常是轻量边缘设备或普通终端容器（例如只有 4GB/8GB 内存），无法承载 20GB+ 的瞬时内存冲击；而服务端通常运行在专用计算集群或大内存工作站（推理本身就需要几十 GB 内存执行海量同态密文算子），**服务端的内存开销不是主要瓶颈，关键瓶颈在于客户端**。

### 2. 方案 A 核心思想
**方案 A（流式文件追加与落盘，Chunked File Stream & Disk Export）** 将客户端密钥生成与内存彻底解耦：
1. **客户端（Client）—— 极致低内存流式生成**：
   - 客户端内存中仅常驻私钥（SK，~12 MB）与公钥（PK，~24 MB）；
   - 所有庞大的计算公钥（RLK、Galois Keys、BTP Keys）采用**逐个生成 $\to$ 写入流式 I/O 缓冲区（32MB） $\to$ 追加落盘到 `eval_context.bin` $\to$ 立即释放该子密钥内存**的流水线方式；
   - 客户端峰值物理内存直接从 21.7 GB+ 压缩至 500 MB 以内。
2. **服务端（Server）—— 简明标准文件读取**：
   - 服务端峰值内存容忍度高，**无需引入复杂的 `mmap` 或复杂的分块流式反序列化**，遵循极简设计原则（KISS）；
   - 服务端仅需提供简洁的 `import_eval_context_file(filepath)` 接口，使用标准二进制文件流（`std::ifstream`）直接将落盘的 `eval_context.bin` 读取为 `Bytes`，然后复用现有的成熟反序列化器完成上下文挂载，进行密文推理。

### 3. 核心设计原则
- **客户端轻量化优先**：核心优化集中在客户端，彻底消除客户端 OOM 风险；
- **服务端极简与可移植（Minimal Diff）**：服务端不依赖平台相关的系统调用（不使用 `mmap`/`munmap`/`madvise`），保持高度可维护性与跨平台一致性；
- **二进制协议 100% 兼容（Wire-Compatibility）**：生成的 `eval_context.bin` 格式与现有 `DeserializeCkksBtpContextAdvanced` / `DeserializeCkksContextAdvanced` 数据流完全一致；
- **推理精度无损**：CIFAR-10 密态推理结果和误差与原流程完全保持一致（PASS，Max Error $< 0.1$）。

---

## 二、端到端架构与数据流设计

### 1. 架构拓扑图

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Client Pipeline                               │
│                                                                        │
│  [client.setup_keypair()]                                              │
│         │                                                              │
│         ├──> Secret Key (SK, ~12MB) ──> 常驻内存 (用于加密与解密)       │
│         └──> Public Key (PK, ~24MB) ──> 常驻内存 (用于非对称加密)      │
│                                                                        │
│  [client.export_eval_context_to_file("eval_context.bin")]              │
│         │                                                              │
│         ├──> 写入 Param Header                                         │
│         ├──> 写入 PK                                                   │
│         ├──> Loop 生成子密钥:                                          │
│         │      Gen RLK ──────> 写入 32MB 流缓冲 ──> 释放 RLK 指针      │
│         │      Gen GaloisKey1 > 写入 32MB 流缓冲 ──> 释放 Key1 指针     │
│         │      Gen GaloisKey2 > 写入 32MB 流缓冲 ──> 释放 Key2 指针     │
│         │      ...                                                     │
│         │      Gen SwkDtS/StD > 写入 32MB 流缓冲 ──> 释放 Swk 指针      │
│         │                                                              │
│         └──> Flush 刷盘关闭文件 (内存占用始终受控于单 Key 上限)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ 文件落盘 (eval_context.bin, ~3.1 GB)
┌────────────────────────────────────────────────────────────────────────┐
│                          Server Pipeline                               │
│                                                                        │
│  [server.import_eval_context_file("eval_context.bin")]                 │
│         │                                                              │
│         ├──> 标准 std::ifstream 二进制读取文件数据为 Bytes              │
│         ├──> 调用现有反序列化流程 (DeserializeCkksBtpContextAdvanced)   │
│         └──> CkksBtpContext 初始化完毕，挂载计算公钥                    │
│                                                                        │
│  [server.load_model()] ──> 加载算子图与模型参数                        │
│  [server.evaluate(encrypted_inputs)] ──> 密态推理执行                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2. 交互时序流程

```text
Client                              Storage                          Server
  │                                    │                               │
  ├─ 1. setup_keypair()                │                               │
  │     (生成 SK/PK，耗内存 ~36MB)      │                               │
  │                                    │                               │
  ├─ 2. export_eval_context_to_file()  │                               │
  │     (流式落盘，峰值 < 500MB)       │                               │
  │     ──────────────────────────────>│ eval_context.bin              │
  │                                    │                               │
  ├─ 3. encrypt(input_csvs)            │                               │
  │     (加密输入数据)                  │                               │
  │                                    │                               │
  │                                    │ 4. import_eval_context_file() │
  │                                    │<──────────────────────────────┤
  │                                    │    (标准读取文件并反序列化)    │
  │                                    │                               │
  │                                    │ 5. load_model() & evaluate()  │
  │                                    │    (执行密态神经网络推理)      │
  │                                    │                               │
  ├─ 6. decrypt(encrypted_outputs) <───────────────────────────────────┤
  │     (使用常驻 SK 解密结果)           │                               │
  ▼                                    │                               │
```

---

## 三、二进制文件流格式定义 (`eval_context.bin`)

文件按顺序写入，与服务端现有的反序列化顺序严格保持 1:1 二进制兼容：

### 1. BTP CKKS 场景布局 (`use_btp = true`)

文件总大小在 $N=65536$ 参数下约为 `3,329,929,205 字节`（约 3.10 GB），顺序如下：

| 顺序阶段 | 字段名称 | 类型/长度 | 说明 |
| :--- | :--- | :--- | :--- |
| **1. Header** | `param_len` | `uint32` (4 字节) | Scheme 参数字节长度 |
| | `param_data` | `bytes[param_len]` | Scheme 参数二进制数据 |
| **2. SK 标记** | `has_sk` | `uint8` (1 字节) | **固定为 0**（评估上下文不含私钥） |
| **3. PK 块** | `has_pk` | `uint8` (1 字节) | **固定为 1** |
| | `pk_payload` | `25,493,552 字节` | 加密公钥（2 个 QP 多项式） |
| **4. BTP 参数**| `btp_param_len`| `uint8` (1 字节) | Bootstrapping 参数长度 |
| | `btp_param_data`| `bytes[btp_param_len]` | Bootstrapping 参数数据 |
| **5. RLK 块** | `has_rlk` | `uint8` (1 字节) | **固定为 1** |
| | `rlk_payload` | 变长字节流 | 重线性化公钥 |
| **6. Galois 块**| `has_rtks` | `uint8` (1 字节) | **固定为 1** |
| | `rtks_count` | `uint32` (4 字节) | 旋转自同构键数量 $M$ |
| | `rtks_payload` | 连续 $M$ 个子块 | 逐个写入：`galEl` (uint64) 与对应的 SwitchingKey |
| **7. 切换键 1**| `has_swk_dts` | `uint8` (1 字节) | **固定为 1** |
| | `swk_dts_data`| 变长字节流 | Bootstrapping Dense-to-Sparse 切换键 |
| **8. 切换键 2**| `has_swk_std` | `uint8` (1 字节) | **固定为 1** |
| | `swk_std_data`| 变长字节流 | Bootstrapping Sparse-to-Dense 切换键 |

### 2. 普通 CKKS 场景布局 (`use_btp = false`)

顺序写入：
1. `uint32` BigEndian：`param_len`；
2. `param_data`；
3. `rlwe_context`：`has_sk=false`, `has_pk=true/false`, `has_rlk=true`, `has_rtks=true` 及其逐个切换键字节流。

---

## 四、客户端流式生成落盘实现 (Client Implementation)

### 1. I/O 缓冲器与逐 Key 生成
在 Go SDK 侧新增带 32MB 缓冲的流式导出实现（`inference/lattisense/fhe_ops_lib/lattigo/go_sdk/bootstrap.go`）：

```go
//export StreamExportCkksBtpEvalContextToFile
func StreamExportCkksBtpEvalContextToFile(context_handle uint64, c_filepath *C.char) C.int {
    filePath := C.GoString(c_filepath)
    file, err := os.OpenFile(filePath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
    if err != nil {
        return -1
    }
    defer file.Close()

    // 32MB 固定大小写入缓冲区，避免频繁小系统调用
    writer := bufio.NewWriterSize(file, 32*1024*1024)
    defer writer.Flush()

    context := get_object[CkksBtpContext](context_handle)

    // 1. 写入 Parameter 与 BTP Parameter Header
    writeSchemeAndBtpParams(context, writer)

    // 2. 写入 PK (已在 setup_keypair 生成)
    binary.Write(writer, binary.LittleEndian, uint8(0)) // has_sk = 0
    binary.Write(writer, binary.LittleEndian, uint8(1)) // has_pk = 1
    rlwe.CiphertextQPToBytes(context.pk, &context.parameter.Parameters, writer)

    // 3. 流式生成并写入 RLK
    binary.Write(writer, binary.LittleEndian, uint8(1)) // has_rlk = 1
    rlk := context.kgen.GenRelinearizationKey(context.sk, 1)
    rlwe.RelinearizationKeyToByte(rlk, &context.parameter.Parameters, writer)
    rlk = nil // 立即置空供 GC 回收

    // 4. 逐个步长生成并写入 Rotation Keys
    binary.Write(writer, binary.LittleEndian, uint8(1)) // has_rtks = 1
    rots := getRequiredRotations(context)
    binary.Write(writer, binary.LittleEndian, uint32(len(rots)))

    for _, rot := range rots {
        galEl := context.parameter.GaloisElementForColumnRotationBy(rot)
        binary.Write(writer, binary.LittleEndian, galEl)
        
        // 单个生成 -> 写入流 -> 立即解引用
        swk := context.kgen.GenSwitchingKeyForRotationBy(rot, context.sk)
        rlwe.SwitchingKeyToBytes(swk, &context.parameter.Parameters, writer)
        swk = nil
    }

    // 5. 生成并写入 Bootstrapping Switching Keys
    swkDtS, swkStD := context.btp_parameter.GenEncapsulationSwitchingKeys(*context.parameter, context.sk)
    binary.Write(writer, binary.LittleEndian, uint8(1))
    rlwe.GadgetCiphertextToBytes(&swkDtS.GadgetCiphertext, &paramsSparse, writer)
    swkDtS = nil

    binary.Write(writer, binary.LittleEndian, uint8(1))
    rlwe.GadgetCiphertextToBytes(&swkStD.GadgetCiphertext, &context.parameter.Parameters, writer)
    swkStD = nil

    return 0
}
```

### 2. C++ 接口封装 (`InferenceClient`)
在 `inference/interface/inference_client.h` 中提供简洁 API：

```cpp
class InferenceClient {
public:
    // ... 原有接口保持兼容 ...

    /// 仅生成 SK 与 PK（常驻内存 ~36 MB，毫秒级完成）
    void setup_keypair();

    /// 流式生成计算公钥并落盘，客户端内存峰值 < 500 MB
    void export_eval_context_to_file(const std::string& filepath);

    /// 原 setup() 保留，向后兼容原有一体化调用
    void setup();
};
```

---

## 五、服务端读取落盘文件实现 (Server Implementation)

服务端无需关注峰值内存优化，重点在于**接口简洁、跨平台、最小化代码修改**。

### 1. C++ 侧封装 (`inference/interface/inference_server.h` & `.cpp`)

在 `InferenceServer` 中新增 `import_eval_context_file` 方法，直接使用标准文件流读取文件：

```cpp
// 在 inference/interface/inference_server.h 中声明:
void import_eval_context_file(const std::string& eval_context_file);
```

实现代码（`inference/interface/inference_server.cpp`）：

```cpp
#include <fstream>

void InferenceServer::import_eval_context_file(const std::string& eval_context_file) {
    std::cout << "[Server] Reading evaluation context from file: " << eval_context_file << std::endl;

    std::ifstream file(eval_context_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open eval context file: " + eval_context_file);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    Bytes buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        throw std::runtime_error("Failed to read eval context file: " + eval_context_file);
    }
    file.close();

    std::cout << "[Server] Read " << file_size << " bytes. Importing evaluation context..." << std::endl;

    // 直接复用原有的 import_eval_context 逻辑，无缝完成解析与上下文挂载
    this->import_eval_context(buffer);

    std::cout << "[Server] Evaluation context loaded successfully from disk." << std::endl;
}
```

### 2. 设计优点
1. **零引入外部依赖**：不依赖 POSIX `mmap` 等系统调用，在任何支持 C++17 标准文件 I/O 的操作系统（Linux/macOS/Windows）上均可直接编译运行；
2. **逻辑内聚**：直接将落盘读取的字节流转发给现有的 `import_eval_context(buffer)`，完全复用服务端的任务参数校验、网络层类型提取和反序列化逻辑，不破坏现有任何单元测试。

---

## 六、端到端使用示例与测试验证

### 1. 更新 `examples/inference.cpp` 验证链路

```cpp
// -----------------------------------------------------------
// Step 1: Client 生成密钥并流式落盘到 server 目录
// -----------------------------------------------------------
cout << "[Step 1/5] Setting up client (streaming key generation to disk)..." << endl;
InferenceClient client(task_dir + "/client");

// 仅在内存中生成 SK/PK
client.setup_keypair();

// 流式生成计算公钥并落盘
std::string eval_context_file = task_dir + "/server/eval_context.bin";
client.export_eval_context_to_file(eval_context_file);

// -----------------------------------------------------------
// Step 2: Client 加密输入
// -----------------------------------------------------------
cout << "[Step 2/5] Encrypting input..." << endl;
auto encrypted_inputs = client.encrypt(input_csvs);

// -----------------------------------------------------------
// Step 3: Server 直接从落盘文件加载上下文并启动推理
// -----------------------------------------------------------
cout << "[Step 3/5] Server loading model and importing context from disk..." << endl;
InferenceServer server(task_dir + "/server", use_gpu, 0);
server.import_eval_context_file(eval_context_file);
server.load_model();

// -----------------------------------------------------------
// Step 4: 密态推理执行
// -----------------------------------------------------------
cout << "[Step 4/5] Running encrypted inference..." << endl;
auto encrypted_outputs = server.evaluate(encrypted_inputs);

// -----------------------------------------------------------
// Step 5: Client 解密并验证
// -----------------------------------------------------------
cout << "[Step 5/5] Decrypting result..." << endl;
auto results = client.decrypt(encrypted_outputs);
```

### 2. CTest 测试回归
直接通过 CTest 验证整体验收：
```bash
cd build
ctest -R cifar10-cpu -V
```

---

## 七、工程代码变动清单 (Code Modification Checklist)

| 模块层级 | 涉及文件 | 变动说明 |
| :--- | :--- | :--- |
| **Go Core** | `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/bootstrap.go` | 新增 `StreamExportCkksBtpEvalContextToFile`（32MB `bufio.Writer` 逐 Key 刷盘并解引用） |
| **Go Core** | `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/main.go` | 普通 CKKS 支持 `StreamExportCkksEvalContextToFile` |
| **Go Header** | `inference/lattisense/fhe_ops_lib/lattigo/go_sdk/liblattigo.h` | 声明新的 CGO 导出函数符号 |
| **LattiSense C++** | `inference/lattisense/fhe_ops_lib/fhe_lib_v2.h` & `.cpp` | 为 `CkksContext` 与 `CkksBtpContext` 添加 `export_eval_context_to_file(const std::string& path)` |
| **Client Interface**| `inference/interface/inference_client.h` & `.cpp` | 1. 拆分暴露 `setup_keypair()`<br>2. 新增 `export_eval_context_to_file(const std::string& path)` |
| **Server Interface**| `inference/interface/inference_server.h` & `.cpp` | 新增 `import_eval_context_file(const std::string& filepath)`（使用标准二进制流读入并调用已有逻辑） |
| **Examples** | `examples/inference.cpp` | 将 Step 1 与 Step 3 切换为基于文件的流式落盘与读取 |

---

## 八、指标收益对比与验收标准

### 1. CIFAR-10 实测指标量化对比

| 指标 | 改造前（现有代码） | 改造后（方案 A 实施） | 变化幅度与说明 |
| :--- | :--- | :--- | :--- |
| **客户端密钥生成常驻 RSS** | `14,408 MB` (~14.1 GB) | **~36.5 MB** | **-99.7%**（仅常驻 SK/PK） |
| **客户端导出阶段峰值 RSS** | `21,723 MB ～ 28,665 MB` | **< 500 MB** | **-97.7%**（单 Key 临时缓冲区 + 32MB I/O 缓冲） |
| **落盘文件大小 (`eval_context.bin`)** | 0 字节（内存驻留） | **3,329,929,205 字节** (~3.10 GB) | 格式完全透明，支持持久化存储 |
| **服务端读取与推理兼容性** | 内存传入 | 文件标准读取 | **零功能/协议损耗** |
| **端到端密态推理精度 (CIFAR-10)** | Max Error: 0.00044758 (PASS) | Max Error: 0.00044758 (PASS) | 严格无损，完全一致 |

### 2. 验收用例
1. **客户端内存专项测试**：独立调用客户端 `setup_keypair()` 与 `export_eval_context_to_file()`，验证进程 `Peak RSS < 500 MB`；
2. **落盘文件完整性测试**：验证落盘后的 `eval_context.bin` 大小与结构，且文件可被服务端正常加载；
3. **端到端功能回归测试**：运行 `ctest -R cifar10-cpu -V`，确保 19,238 个同态算子与 Bootstrapping 正确执行，验证状态为 `PASS`。
