# Latti-AI 客户端双文件持久化设计规范

## 1. 目标与基线

状态：已实施设计。

在已实现的 [密钥流式生成方案](../streaming_eval_context_file_design_spec.md) 上增加客户端持久化，继续以降低客户端密钥生成阶段的峰值内存和最小化代码修改为原则。

首次生成后保存 `eval_context.bin` 和 `full_context.bin`。客户端刷新页面、关闭浏览器或重启程序后，加载原来的 `full_context.bin`，恢复同一密钥的加解密能力。

本文补充 [纯浏览器 WASM 客户端设计](latti_browser_wasm_client_design_spec.md)。密钥保存、文件读写、接口、恢复和解密流程以本文为准；浏览器沿用 JS/TS 调用层、Dedicated Worker 和 Go WebAssembly 密码学核心。

## 2. 文件内容与布局

```text
client_directory/
  task_config.json
  ckks_parameter.json
  eval_context.bin
  full_context.bin

encrypted/
  <input-name>.ct.bin
decrypted/
  <output-name>.csv
```

| 文件 | 内容 | 用途 |
|---|---|---|
| `eval_context.bin` | 参数、RLK、完整旋转/共轭键集合；BTP 另含 PK 和按参数启用的 DtS/StD | 传给服务端进行同态计算 |
| `full_context.bin` | 参数、解密私钥 SK、加密公钥 PK；BTP 另含 BTP 参数 | 保存在客户端本地，恢复加解密能力 |

两个密钥文件来自同一轮生成，并按任务目录配对保存。密文文件与输入/输出名称的映射由 task config 和调用参数确定。

`full_context` 表示具有完整客户端加解密能力的上下文，其评估键标志全部固定为 `0`。

## 3. 二进制格式

### 3.1 `eval_context.bin`

沿用 [已实现的协议](../streaming_eval_context_file_design_spec.md)：

- 普通 CKKS：参数长度为 BigEndian `uint32`，`has_sk=0`、`has_pk=0`，随后写 RLK 和 RTK 集合。
- BTP：参数长度为 LittleEndian `uint32`，`has_sk=0`、`has_pk=1`，随后写 PK、BTP 参数、RLK、RTK 集合和 DtS/StD。
- RTK 数量为 LittleEndian `uint16`，Galois 标识为 LittleEndian `uint64`。

服务端继续调用 `import_eval_context_file()`。

### 3.2 `full_context.bin`

复用现有 advanced context 序列化布局。

普通 CKKS：

```text
param_len:uint32 BigEndian → CKKS parameter bytes
has_sk=1 → SecretKey payload
has_pk=1 → PublicKey payload
has_rlk=0
has_rtks=0
EOF
```

BTP：

```text
param_len:uint32 LittleEndian → CKKS parameter bytes
has_sk=1 → SecretKey payload
has_pk=1 → PublicKey payload
btp_param_len:uint8 → BTP parameter bytes
has_rlk=0
has_rtks=0
has_swk_dts=0
has_swk_std=0
EOF
```

SK/PK payload 复用 `SecretKeyToBytes`、`CiphertextQPToBytes` 及对应读取/解压接口。BTP 参数采用同轮流式生成更新后的状态，保持与现有 `CkksBtpContext` 导入格式兼容。

## 4. 保存与恢复流程

### 4.1 首次生成

```text
读取任务配置
→ 通过现有流式 setup 创建 SK/PK
→ 逐键生成评估键，写出 eval_context.bin
→ 导出仅含参数/SK/PK 的 full context
→ 写出并关闭 full_context.bin
→ setup 返回成功
```

native 在 `InferenceClient::setup()` 中复用现有流式 `create_crypto_context()`，随后调用 `export_full_context()` 并保存文件。Node 保留 `client.setup()` 的 AsyncWorker/Promise 调用方式。

`full_context.bin` 使用普通文件写入。两个文件完成写入并关闭后，setup 返回成功；读写结果通过现有返回值、异常或 Promise rejection 传播。

新增持久化入口在写入两个文件之前检查已有 `full_context.bin`：已有私钥文件对应恢复操作，新一轮密钥生成使用新目录。

### 4.2 后续恢复

```text
重新选择并授权原客户端目录
→ 读取任务配置和 full_context.bin
→ 调用 load_full_context / loadFullContext 对应的导入能力
→ 恢复参数、SK/PK、encoder、encryptor、decryptor
→ 使用原来的密钥加密新输入、解密服务端输出
```

恢复所需密钥材料全部来自 `full_context.bin`。`close()` 释放内存和 Worker，磁盘文件保留供下次恢复。

客户端恢复入口核对 SK/PK、§3.2 的字段标志以及任务使用的有效参数，包括 scheme、N、Q/P、LogSlots 和 scale。任务配置继续决定输入布局和输出解码方式。

## 5. 最小实现方式与内存约束

默认流式 setup 已仅保留 SK/PK，直接复用现有 full context 导出/导入能力即可完成常规持久化。

`InferenceClient::export_full_context()` 的输出统一符合 §3.2。对于持有全套评估键的旧 context，在 Go SDK 序列化适配层构造仅引用参数、SK、PK 的局部视图，并复用现有 serializer；C++ 增加对应的薄包装。该视图独立设置评估键字段，原 context 保持原状，底层通用 serializer 保持原有用途。

客户端保持以下内存结构：

- 密钥生成沿用单 goroutine、逐键处理和单键序列化缓冲；BTP DtS/StD 复用现有成对生成接口。
- 评估文件导出完成后，结束单键对象与缓冲的引用生命周期，再序列化 full context。
- full context 可在 Worker/Go/C++ 内暂存完整字节，其大小由参数和 SK/PK 决定。
- 恢复后常驻参数、SK/PK 及客户端编码、加密、解密工作区。

按生成、full 导出和恢复三个阶段实测峰值内存，计入 Go GC 的回收时机与跨语言字节复制开销。

## 6. 浏览器接口与文件操作

Web 项目集成 `@latti/client-wasm`，终端用户打开网页后由浏览器加载 JS/TS 和 WASM。第一版使用桌面 Chromium、安全上下文和用户授权的本地目录读写能力。

```ts
export interface BrowserClient {
  setup(): Promise<{
    evalContext: FileSystemFileHandle;
    fullContext: FileSystemFileHandle;
  }>;
  restore(): Promise<void>;
  encrypt(options: {
    inputName: string;
    inputFile: FileSystemFileHandle | File;
    outputDirectory: FileSystemDirectoryHandle;
  }): Promise<FileSystemFileHandle>;
  decrypt(options: {
    outputName: string;
    ciphertextFile: FileSystemFileHandle | File;
    outputDirectory: FileSystemDirectoryHandle;
  }): Promise<FileSystemFileHandle>;
  close(): Promise<void>;
}
```

`openBrowserClient({ clientDirectory })` 读取配置并创建 Worker。首次使用调用 `setup()`；再次使用调用 `restore()`；两者成功后均可调用 `encrypt()` 和 `decrypt()`。同一会话的操作串行执行，页面接收文件句柄和操作状态。

Go 核心按 `GOOS=js GOARCH=wasm CGO_ENABLED=0` 构建，复用 Lattigo 公开算法和编码接口。Worker 负责文件读写、密钥恢复和本地计算。需要跨 native/browser 共享的少量参数、键集合和编码逻辑提取到纯 Go 层。

`eval_context.bin` 通过有界 chunk 和背压流式落盘；`full_context.bin` 在 Worker 内完成导入/导出并保存到用户选择的本地目录。浏览器读取 CSV 输入，沿用 `InferenceClient::encrypt()` 的 packing 和 ciphertext 格式；解密沿用 `InferenceClient::decrypt()` 的输出布局、解码和 unpack 语义，数值结果保存为 CSV。

## 7. 本地密钥管理

`full_context.bin` 按私钥文件管理：它以现有序列化格式保存 SK，由用户在本地保管；native 使用平台可用的受限文件权限，浏览器通过用户授权的文件句柄访问。网页上传流程只选择 `eval_context.bin` 和密文。

恢复使用本程序成功写出的同轮文件。应用按任务目录保持密钥文件配对，并将密文提交给持有对应 evaluation context 的服务端。浏览器沿用可信 JS/WASM 资源、固定版本和文件权限约束。

## 8. 实施与验收

| 模块 | 修改内容 | 复用接口 |
|---|---|---|
| Lattigo Go SDK | 参数/SK/PK 局部视图的导出适配 | 现有 serializer 和 context 导入 |
| LattiSense C++ | 对应导出薄包装 | 现有 context 与加解密接口 |
| InferenceClient / Node | setup 保存 full 文件、收紧 full 导出内容、恢复参数检查 | `setup()`、`export_full_context()` / `exportFullContext()`、`load_full_context()` / `loadFullContext()` |
| 浏览器 WASM/Worker | 双文件写入、restore、加解密和结果落盘 | 纯 Go 密码学接口、浏览器文件 API |
| 服务端 | 接收 eval 文件和密文并推理 | `import_eval_context_file()` 和现有推理接口 |

验收覆盖：

1. 普通 CKKS/BTP 的两个文件符合 §3；full 中 SK/PK 标志为 `1`，评估键标志均为 `0`。
2. 从旧全套 context 导出 full 仍符合 §3.2，原 context 保持可用。
3. 新进程加载 full 后恢复原 SK/PK，并能加解密。
4. 浏览器 setup 后刷新或重启，再授权目录并 restore，能解密此前密文、加密新输入。
5. 浏览器加密 → 现有 CPU/GPU 服务端推理 → 浏览器恢复后解密，精度和输出布局符合任务要求。
6. 文件读写与接口失败遵循现有异常约定，setup 成功对应两个文件均已完成写入。
7. 独立记录流式生成、full 导出和恢复阶段的峰值内存，保持既有流式生成的降峰收益。

实施提交按 Lattigo → LattiSense → Latti-AI 的依赖顺序组织，原生持久化与浏览器 WASM 适配分别验收。
