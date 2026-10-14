# Latti-AI 纯浏览器 WebAssembly 客户端设计规范

> 后续持久化与完整加解密范围以 [客户端双文件持久化设计规范](latti_client_context_persistence_design_spec.md) 为准：保存 `eval_context.bin` 与仅含参数/SK/PK 的 `full_context.bin`，支持刷新后恢复和本地解密。密钥保存、文件读写和接口要求以新规范为准。

## 1. 文档目的

本文定义一个无需 Node.js、Electron、浏览器插件或本地 companion 进程的纯浏览器客户端。用户在普通 Web 页面中选择本地任务目录和输入文件，浏览器直接完成：

1. 读取任务配置；
2. 在 WebAssembly 中生成 CKKS/BTP 客户端密钥；
3. 默认逐密钥生成 evaluation context，并把 `eval_context.bin` 流式写入用户选择的本地目录；
4. 使用同一客户端会话加密本地输入数据；
5. 把密文文件流式写入用户选择的本地目录。

本文只定义浏览器客户端。服务端仍使用现有 `InferenceServer::import_eval_context_file()` 和现有密文格式；LattiSense 的 GPU/CPU 服务端运行时、Node 原生 addon 和 Python 绑定不在本方案的浏览器运行时中。

纯浏览器方案不把现有 `latti_client.node` 改名为 Web 包。浏览器包由纯 Go WebAssembly 核心、JavaScript/TypeScript Worker 适配层和浏览器文件 API 适配层组成。现有原生 Node 包继续作为独立的桌面/Node 交付物。

## 2. 目标、边界与设计原则

### 2.1 目标

- 不安装 Node.js、Electron、agent、插件或独立的本地服务。
- 只在 HTTPS 或 `localhost` 的安全上下文中运行。
- 默认使用当前 `feat/streaming-key-generation` 设计的逐密钥 evaluation-key 生成路径。
- 不把完整 `eval_context.bin` 放入 JavaScript `ArrayBuffer`、`Blob`、IndexedDB 记录或网络请求体。
- 不把私钥传给服务端，也不把私钥作为默认 API 返回值交给页面 JavaScript。
- 复用现有 evaluation-context 二进制协议，使浏览器生成的文件可由现有服务端导入。
- 加密结果保持现有序列化字节格式，服务端无需增加浏览器专用解析器。
- 通过浏览器文件选择和写入 API 把产物保存到用户明确授权的本地目录。

### 2.2 非目标

- 不在浏览器中运行 LattiSense 的模型推理或 CUDA/WebGPU 服务端算子。
- 不提供任意二进制文件的通用 FHE 加密。输入必须符合任务 `task_config.json` 描述的 CSV/数值布局。
- 不在第一版实现浏览器端解密、私钥导出、私钥备份或跨设备同步。
- 不把完整客户端上下文上传到服务器。
- 不改变现有服务端 evaluation-context 文件格式。
- 不为浏览器重新实现一套与 Lattigo 不一致的密钥生成算法。

### 2.3 设计原则

1. **协议单一来源**：流式文件格式、Galois 键集合和 serializer 由 Lattigo 中的共享纯 Go 层定义；原生 CGO SDK 和浏览器 WASM 都调用该层。
2. **磁盘优先**：文件由 Worker 逐块写入 File System Access API 或 OPFS，页面只接收状态和小型元数据。
3. **会话内私钥**：SK 只保存在 WASM 会话状态中，直至 `close()`、失败清理或 Worker 终止。
4. **串行操作**：同一客户端同一时间只允许一个 `setup` 或 `encrypt` 操作，避免密钥状态和文件流并发访问。
5. **能力最小化**：页面拿到的是目录/文件句柄，不是任意绝对路径；WASM 核心不接受路径字符串。
6. **渐进交付**：先验证无 cgo 的纯 Go CKKS 小参数构建，再加入 BTP、浏览器文件落盘和大参数测量。

## 3. 运行时架构

```text
用户 Web 页面（主线程）
  │ 选择目录/文件；显示进度；不接收密钥文件内容
  │ FileSystemDirectoryHandle / FileSystemFileHandle
  ▼
Dedicated Worker
  │ 读取 JSON/CSV，管理文件读写和操作状态
  ▼
纯 Go WebAssembly 核心（CGO_ENABLED=0）
  │ CKKS/BTP 参数、SK/PK、编码、加密、逐键 serializer
  ▼
File System Access API / OPFS
  │ 逐块写入临时文件，完成后发布
  ▼
用户本地文件
  ├── eval_context.bin
  ├── eval_context.manifest.json
  └── encrypted/<input-name>.ct.bin + manifest.json

用户将 eval_context.bin 和密文文件传给现有 FHE 服务端
```

主线程只负责用户手势、句柄选择和 UI。WASM 和文件流放在 Dedicated Worker 中，避免大整数运算阻塞页面。FileSystem 句柄通过 structured clone 传给 Worker；如果目标浏览器不支持 Worker 内的 writable stream，适配层必须退回 OPFS Worker 写入，不得把整个文件搬回主线程。

## 4. 浏览器支持基线

第一版以桌面 Chromium 为基线，要求：

- WebAssembly；
- Dedicated Worker；
- File System Access API 的目录/文件句柄；
- Worker 中可用的 `FileSystemFileHandle.getFile()`、`createWritable()` 或 OPFS 等价能力；
- HTTPS 或 `localhost` 安全上下文；
- 用户手势触发 `showDirectoryPicker()`、`showOpenFilePicker()` 或 `showSaveFilePicker()`。

浏览器不能提供目录写入能力时，只能使用“用户选择单个目标文件”或下载模式；该模式不满足默认的目录落盘体验，必须在 UI 中明确标记。Firefox、Safari 和移动浏览器在没有完成句柄、Worker 和大文件验证前不列入兼容承诺。

浏览器不能提供绝对路径。所有 API 使用 `FileSystemDirectoryHandle`、`FileSystemFileHandle` 或 `File`；不能要求用户输入路径，也不能由页面向 WASM 传递路径字符串。

## 5. 浏览器客户端 API

包名建议为 `@latti/client-wasm`。它是浏览器 JavaScript 包，不是 Node 包；不包含 `.node`、`.so`、`.dll` 或 CUDA 运行库。

### 5.1 TypeScript 接口

```ts
export type ProgressPhase =
  | "load-configuration"
  | "generate-secret-key"
  | "write-public-key"
  | "write-relinearization-key"
  | "write-rotation-key"
  | "write-btp-switching-key"
  | "publish-eval-context"
  | "parse-input"
  | "encrypt-input"
  | "publish-ciphertext";

export interface ProgressEvent {
  phase: ProgressPhase;
  completed?: number;
  total?: number;
  writtenBytes?: number;
  totalBytes?: number;
  elapsedMs: number;
}

export interface BrowserClientOptions {
  clientDirectory: FileSystemDirectoryHandle;
  workerUrl?: URL | string;
}

export interface SetupOptions {
  fileName?: string; // default: "eval_context.bin"
  onProgress?: (event: ProgressEvent) => void;
  signal?: AbortSignal;
}

export interface EncryptOptions {
  inputName: string;
  inputFile: FileSystemFileHandle | File;
  outputDirectory: FileSystemDirectoryHandle;
  outputFileName?: string; // default: `${inputName}.ct.bin`
  onProgress?: (event: ProgressEvent) => void;
  signal?: AbortSignal;
}

export interface ArtifactInfo {
  name: string;
  byteLength: number;
  sha256?: string;
  directory: FileSystemDirectoryHandle;
}

export interface BrowserClient {
  setup(options?: SetupOptions): Promise<ArtifactInfo>;
  encrypt(options: EncryptOptions): Promise<ArtifactInfo>;
  close(): Promise<void>;
}

export async function openBrowserClient(
  options: BrowserClientOptions,
): Promise<BrowserClient>;
```

`openBrowserClient()` 读取 `clientDirectory/task_config.json` 和 `clientDirectory/ckks_parameter.json`，验证任务配置后创建 Worker 会话。`use_btp`、参数 ID、输入名称和输入布局均来自配置文件；API 不提供绕过配置的 scheme、参数或 Galois 集合覆盖选项。

### 5.2 操作语义

- `setup()` 只在本地目录中发布完整的 `eval_context.bin` 和 manifest，成功前不 resolve。
- `setup()` 成功后，SK/PK 留在 Worker 的 WASM 会话中，供同一 `BrowserClient` 调用 `encrypt()`。
- `encrypt()` 只返回输出文件元数据，不返回密文 `Uint8Array`。
- `close()` 终止 Worker 会话并清理可释放的 WASM 对象。终止后不能继续调用 `encrypt()`。
- 同一个客户端上并发调用 `setup()`、`encrypt()` 或 `close()` 必须拒绝并返回明确错误。
- `AbortSignal` 触发后，当前生成或加密操作进入失败清理路径；不能把部分文件报告为完成。

### 5.3 示例调用

```ts
const clientDirectory = await window.showDirectoryPicker({ mode: "readwrite" });
const inputHandle = await window.showOpenFilePicker({
  multiple: false,
  types: [{ description: "CSV input", accept: { "text/csv": [".csv"] } }],
}).then(([handle]) => handle);

const client = await openBrowserClient({ clientDirectory });

const evalContext = await client.setup({
  onProgress: (event) => renderProgress(event),
});

const outputDirectory = await window.showDirectoryPicker({ mode: "readwrite" });
const ciphertext = await client.encrypt({
  inputName: "input",
  inputFile: inputHandle,
  outputDirectory,
  onProgress: (event) => renderProgress(event),
});

console.log(evalContext.name, evalContext.byteLength);
console.log(ciphertext.name, ciphertext.byteLength);
await client.close();
```

示例不读取 `eval_context.bin` 或密文内容到页面内存。若页面需要让用户上传文件，使用用户再次选择的 `FileSystemFileHandle` 或 `File`，而不是构造全文件 `ArrayBuffer`。

## 6. WASM 核心与共享密码学实现

### 6.1 构建约束

浏览器核心必须使用：

```text
GOOS=js
GOARCH=wasm
CGO_ENABLED=0
```

它不能链接 `go_sdk` 当前的 `import "C"` 入口、C++ `fhe_ops_lib`、`liblattigo.so`、`liblattisense.so`、CUDA 或 Node N-API。Go 版本、Lattigo module commit 和 native streaming 分支必须在 manifest 中固定。

### 6.2 共享纯 Go 层

当前流式入口位于 `go_sdk` 的 cgo `package main` 文件中，浏览器不能直接导入。实现时应在 Lattigo fork 的 `feat/streaming-key-generation` 分支中抽出一个不依赖 cgo、文件路径和全局句柄的纯 Go 包，例如：

```text
inference/lattisense/fhe_ops_lib/lattigo/streaming/
  context.go
  ckks.go
  btp.go
  wire.go
  encrypt.go
```

该包负责：

- 创建和持有浏览器客户端所需的参数、SK、PK、encoder、encryptor；
- 计算并去重普通 CKKS/BTP 的完整 Galois 集合；
- 以状态机形式逐步生成 `RLK`、`RTK`、`DtS`、`StD` payload；
- 使用现有 Lattigo serializer 生成完全兼容的字节；
- 对 CSV 数值执行与现有 `InferenceClient::encrypt()` 相同的 packing、编码和加密；
- 在每个 payload 交付后释放该 payload 的临时引用。

packing 和 CSV 布局规则目前位于 C++ `InferenceClient::encrypt()`；浏览器实现必须把这些规则移植到无 cgo 的层，或抽出可被纯 Go 和浏览器共同调用的等价层，并用 native/browser differential tests 约束结果。不能在浏览器 wrapper 中自行定义不同的 shape、skip 或 packing 语义。

原生 `go_sdk` 的 CGO 导出函数和浏览器 WASM wrapper 都调用该包。不得在浏览器 wrapper 中复制一套 key generator、rotation 集合或 wire encoder；否则 native 和 browser 的格式会产生长期漂移。

共享层不得使用 `os.OpenFile`、`filepath`、`runtime/cgo` 或 C 指针。文件写入通过抽象的块读取/块写出状态机完成，主机适配层负责 File System API。

### 6.3 不增加独立 keypair-only API

浏览器公开入口只有 `openBrowserClient().setup()` 和 `encrypt()`。内部可以有 `CreateClientSession`，但它只服务于持有客户端加密状态和流式输出，不对外暴露“只生成密钥对”的另一个公共接口，也不允许绕过 `task_config.json` 创建任意参数会话。

## 7. evaluation-context 流式生成

### 7.1 生成状态机

WASM 核心向 Worker 提供同步的、可重复调用的 chunk 状态机；Go 不直接等待 JavaScript Promise。

```text
beginSetup()
  → 返回协议头和初始状态
nextSetupChunk()
  → 每次最多返回一个受限大小的 Uint8Array
  → 生成并编码下一个 payload 后立即丢弃键引用
  → 返回 {done, phase, index, total, chunk}
...
finishSetup()
  → 返回字节数、digest 输入和状态
abortSetup()
```

Worker 在每次 `nextSetupChunk()` 返回后将 chunk 顺序写入临时文件，再请求下一块。不得提前生成所有 RTK，也不得把所有 chunk 累积到数组。

第一版默认跨 WASM/JavaScript 边界的 chunk 为 4 MiB；单个密钥的 Go 序列化 buffer 可以短暂大于该值，但不得超过单个 payload 所需的上界。后续可把 serializer 改成真正的 bounded `io.Writer`，但不能以扩大 chunk 或缓存整个文件来解决性能问题。

### 7.2 文件协议

浏览器生成的文件必须与 [streaming_eval_context_file_design_spec.md](../streaming_eval_context_file_design_spec.md) 相同：

- 普通 CKKS：参数长度 BigEndian，`has_sk=0`，`has_pk=0`，随后为 RLK、完整 RTK 集合；
- BTP：参数长度 LittleEndian，`has_sk=0` 无条件写入，`has_pk=1` 及 PK payload，随后为 BTP 参数、RLK、完整 RTK 集合、DtS、StD；
- BTP 绝不根据 `context.sk` 是否存在决定 `has_sk`，也绝不写入 SecretKey payload；
- RTK 数量为 LittleEndian `uint16`，Galois 标识为 LittleEndian `uint64`；
- 复用现有 seed 压缩、QP 多项式和 GadgetCiphertext 编码；
- 不增加容器头、分块头、压缩层或浏览器专用尾部。

分块只存在于主机写入过程。最终 `eval_context.bin` 的逻辑字节流必须与原 deserializer 看到的连续字节完全一致。

### 7.3 生成顺序

1. 读取并 marshal Scheme 参数；
2. 无条件写 `has_sk=0`；
3. 按 scheme 写 PK 标志及 payload；
4. BTP 写 BTP 参数；
5. 生成、编码并写出 RLK；
6. 计算完整去重 Galois 集合；
7. 逐个生成、编码并写出 RTK，包括共轭键和 BTP 所需旋转；
8. BTP 调用现有成对生成逻辑写 DtS、StD；
9. 关闭、校验并发布文件。

BTP 的 `RotationsForBootstrapping()` 必须在 BTP 参数序列化前调用，保留对 `LogN`/`LogSlots` 的更新。DtS/StD 仍使用同一个生成操作产生匹配的临时密钥；生成后分别编码和释放。

## 8. 浏览器文件落盘与发布

### 8.1 用户授权

目录和文件句柄只能由用户手势触发的浏览器选择器获得。API 不接受任意路径，不通过 URL 参数拼接文件名，不调用隐式下载目录。

在 `clientDirectory` 中读取的文件白名单为：

```text
task_config.json
ckks_parameter.json
```

输出文件名只允许 `[A-Za-z0-9._-]`，禁止 `/`、`\\`、`..` 和控制字符。输入名称必须存在于 `task_input_param`，不能由用户任意构造新的密文键名。

### 8.2 临时文件和完成标记

浏览器目录 API 的原子 rename 能力在不同实现中不一致。适配层必须采用以下发布协议：

1. 在目标目录创建唯一的 `.<name>.partial-<nonce>`；
2. 顺序写入所有 chunk，计算字节数和 SHA-256；
3. 关闭 writable stream，并重新读取文件元数据确认长度；
4. 若浏览器支持同目录 `move()`，将 partial 文件发布为最终文件；
5. 否则保留版本化文件名，并写入 `eval_context.manifest.json` 作为完成记录，不得把未完整文件标记为 `eval_context.bin`；
6. 失败时关闭并删除 partial 文件，保留已有最终文件。

Manifest 至少包含：

```json
{
  "format": "latti-eval-context-artifact-v1",
  "file": "eval_context.bin",
  "byteLength": 3329929205,
  "sha256": "...",
  "scheme": "CKKS",
  "useBtp": true,
  "complete": true
}
```

Manifest 是浏览器本地完整性和 UI 展示信息，不是 evaluation-context 二进制协议的一部分。服务端仍导入 `eval_context.bin` 本身；上传前客户端必须检查 `complete=true`、文件长度和 digest。

### 8.3 取消和故障

以下任一情况都必须使操作失败：输入读取错误、权限撤销、短写、writable stream reject、WASM 异常、digest/长度不匹配、用户取消或 Worker 终止。失败结果不得 resolve 为可上传 artifact。

## 9. 本地数据加密

### 9.1 输入格式

浏览器版本先支持与现有客户端一致的 CSV 数值输入。Worker 读取 `FileSystemFileHandle.getFile().stream()`，使用增量 CSV parser；禁止无条件调用 `file.text()` 或构造完整文本副本。

parser 根据 `task_config.json` 验证：

- 输入名称存在；
- dim、channel、shape、pack_num 和 skip 合法；
- 数字数量与任务布局匹配；
- NaN、Infinity 和超出明确限制的字段被拒绝。

### 9.2 密文输出

对于每个输入名称，WASM 核心生成现有序列化 ciphertext bytes。Worker 将其写成单独文件：

```text
encrypted/
  input.ct.bin
  manifest.json
```

`manifest.json` 保存输入名称、文件名、长度、SHA-256、task 配置摘要和格式版本。多个输入使用多个 `.ct.bin` 文件，避免把 map 中所有密文拼成一个大 JavaScript 对象。单个密文仍允许在 WASM 内存中暂存其自身序列化结果；若实测尺寸过大，再增加有界 ciphertext writer，不能回退为整批 map 缓存。

服务端调用方根据 manifest 将 `{inputName: bytes}` 传给现有 `InferenceServer::evaluate()`。浏览器包不改变服务端密文协议。

### 9.3 API 与任务生命周期

`encrypt()` 必须在同一个 `BrowserClient` 完成 `setup()` 后调用。浏览器刷新、Worker 崩溃或 `close()` 后，`eval_context.bin` 仍可保留，但其中没有 SK，不能恢复该客户端的加密状态。

第一版不自动保存 full context。若未来要支持跨会话加密/解密，必须单独设计用户口令或 WebCrypto 包装的私钥存储，不能把 `exportFullContext()` 的明文 bytes 放进普通下载、localStorage 或未加密 IndexedDB。

## 10. 内存、性能和可观测性

### 10.1 内存约束

实现禁止以下对象存在：

- 完整 evaluation-context `Uint8Array`、`Blob` 或字符串；
- 所有 RTK 的数组或 map；
- 完整密文文件的页面级缓存；
- 多个并行的 key-generation payload；
- JS 和 Worker 同时各持有一份完整 payload。

允许短暂存在：当前 SK/PK、参数、一个 key generator 工作区、当前 RLK/RTK/DtS/StD payload buffer、一个不超过配置上限的 JS 文件 chunk。

浏览器版本不承诺沿用 native `/usr/bin/time -v` 的 RSS 数值。验收同时记录：

- 浏览器进程/Worker 的峰值 working set（Chrome Task Manager 或 OS 工具）；
- WASM linear memory `byteLength` 峰值；
- JS 在途 chunk 数量和最大字节数；
- 当前 payload buffer 最大容量；
- evaluation-context 文件最终长度和耗时。

当前 native CIFAR-10 约 1.187 GiB 的峰值只能作为参考，不能直接当作浏览器阈值。浏览器大参数测试必须在目标 Chromium 版本单独测量；若浏览器内存上限无法容纳目标参数，必须报告不支持该参数，而不是自动切换到批量或服务端生成。

### 10.2 进度事件

Worker 发送的进度事件只包含阶段、已完成 key 数、总 key 数、已写字节、总字节（若可计算）和耗时。事件不得包含 SK、PK raw bytes、当前 key payload 或输入明文。

## 11. 安全和隐私边界

- `eval_context.bin` 不包含 SK；BTP 的 PK 按现有协议写入，不能误写 SK。
- SK 只存在 WASM Worker 会话；页面 API 不提供导出方法。
- Worker 消息不得传输私钥、完整评估键或完整明文输入。
- 页面必须配置 CSP、SRI/固定 WASM digest，并避免加载未审计的第三方脚本；同源 XSS 可以调用当前会话的加密能力，不能通过 API 读取 SK，但仍属于高风险。
- 目录句柄只在用户授权后获得；Worker 不得枚举或读取授权目录中的非白名单文件。
- 默认不联网上传任何密钥、输入或密文。服务器通信属于上层 Web 应用，必须明确由用户触发。
- 错误信息不能包含 SK、完整输入内容或 raw key payload；日志只记录阶段、长度和错误类别。
- Go/WASM GC 不提供强保证的内存清零。`close()` 应显式清理可控的 byte slice、停止 Worker 并释放句柄，但威胁模型不能承诺浏览器进程内存被完全擦除。

## 12. 源码和打包布局

建议新增浏览器专用目录，不把浏览器代码混入 Node CMake 目标：

```text
inference/interface/wasm/
  go.mod
  cmd/latti-client-wasm/main.go
  worker/latti-client-worker.ts
  src/client.ts
  src/file_sink.ts
  src/csv_stream.ts
  src/types.ts
  tests/

inference/lattisense/fhe_ops_lib/lattigo/streaming/
  ... shared pure-Go streaming and client primitives ...
```

发布包建议为：

```text
@latti/client-wasm/
  dist/latti_client.wasm
  dist/wasm_exec.js
  dist/latti-client-worker.js
  dist/index.js
  dist/index.d.ts
  README.md
  SHA256SUMS
```

构建命令由固定 Go toolchain 生成 `wasm_exec.js`，不能混用不同 Go 版本的 runtime。包内不包含 `third_party/latti-ai-runtime/build/inference/lattisense`、Node addon、动态库或服务器模型资源。

## 13. 测试和验收矩阵

### 13.1 协议和密码学

| 用例 | 验收条件 |
|---|---|
| 普通 CKKS 小参数 | 现有 deserializer 成功导入；`has_sk=0`；无 SK payload；完整 RLK/RTK 可用 |
| BTP 小参数 | 字段顺序、PK、RLK、完整旋转集合、DtS/StD 与现有 deserializer 兼容 |
| BTP SK 边界 | 输入会话持有 SK 时仍固定写 `has_sk=0`，文件中无 SK payload |
| native/browser 对照 | 相同参数下，native 和 WASM 文件结构字段一致；随机 key bytes 不要求一致 |
| 加密兼容 | 浏览器生成 ciphertext 可被现有服务端反序列化；使用 native 客户端解密服务端输出验证误差 |

### 13.2 浏览器文件和故障

| 用例 | 验收条件 |
|---|---|
| 目录授权 | 只读取 `task_config.json`、`ckks_parameter.json`，拒绝越权名称 |
| 流式落盘 | 页面和 Worker 不出现完整 eval context buffer |
| 写入失败 | partial 文件清理，旧完整文件保留，Promise reject |
| 用户取消 | Worker 停止，partial 文件不标记 complete |
| 短写/关闭失败 | 不发布 artifact，错误包含阶段和文件名 |
| 刷新/Worker 崩溃 | 下次启动能识别并清理未完成 partial 文件 |
| 多输入加密 | 每个输入得到独立 ciphertext 文件和 manifest，名称可映射回服务端输入 key |

### 13.3 内存和性能

1. 小参数单元测试验证状态机每次只交付一个 chunk。
2. 普通 CKKS 和 BTP 分别在 Dedicated Worker 中运行，确认 UI 帧不因 `setup()` 长时间停止。
3. 参考 `N16QP1546H192H32` 的 BTP 文件验证最终大小与 native 参考值同量级；不把固定大小当通用协议断言。
4. 大参数测试记录浏览器进程 working set、WASM memory、chunk 峰值、文件大小和耗时。
5. 只有在“没有完整文件缓存、没有完整 RTK 集合、没有并行 keygen”均有证据时，才报告满足流式内存设计。

### 13.4 浏览器端到端

使用 Playwright 或等价 Chromium harness：

```text
选择 clientDirectory
→ setup()
→ 检查 eval_context.manifest.json complete=true
→ 选择 input.csv
→ encrypt()
→ 检查 ciphertext manifest 和文件 digest
→ 将文件交给现有服务端导入
```

测试 harness 不应通过 Node addon 完成被测操作；Node 只可用于启动浏览器、服务端验证和读取测试结果。

## 14. 分阶段实施和提交边界

纯浏览器方案的改动量明显高于复用 Node addon 的方案，不能把它包装成一次打包任务。建议按以下模块提交：

| 阶段 | 模块 | 主要内容 | 提交建议 |
|---|---|---|---|
| W0 | 可行性探针 | `CGO_ENABLED=0` 编译 Lattigo CKKS 小参数；验证 `js/wasm` 可调用 keygen/serializer | `test: probe pure Go browser build` |
| W1 | Lattigo 共享层 | 抽出无 cgo 的 stream state machine、wire encoder、CKKS/BTP 参数和 Galois 集合 | `refactor: share pure Go streaming client core` |
| W2 | WASM client | 创建会话、`begin/next/finish/abort`、SK/PK 生命周期和加密原语 | `feat: add browser wasm client core` |
| W3 | Worker/文件适配 | Worker、File System Access/OPFS、partial/manifest、取消和错误处理 | `feat: stream wasm artifacts to browser storage` |
| W4 | CSV 加密 | 增量 parser、packing、ciphertext 文件和 manifest | `feat: encrypt task inputs in browser wasm` |
| W5 | 验证和发布 | Chromium E2E、协议对照、内存测量、CSP/SRI、包 README | `test: validate browser wasm client` |

W1 必须先让 native CGO SDK 改用共享层并通过已有 Node/CPU/GPU 回归，再继续 W2；这样能避免浏览器版复制出第二套格式。LattiSense 的服务端代码不应因浏览器包增加新的运行时依赖。每个阶段只提交源码、测试和必要的 manifest，不提交生成的 WASM 大文件或密钥样本，除非发布流程明确要求。

## 15. 风险、限制和决策记录

### 15.1 主要风险

- 当前 Go SDK 是 cgo `package main`，抽出共享纯 Go 层需要重排类型和构建入口。
- BTP 的大参数 keygen、编码和浏览器 WASM heap 可能超过部分浏览器可用内存。
- File System Access API 的 Worker 支持、目录 move/rename 和浏览器权限模型存在实现差异。
- 浏览器没有 native RSS 等价指标，峰值测量必须结合浏览器进程工具和 WASM/JS 计数。
- Go/WASM GC 无法提供强密钥擦除保证，浏览器端威胁模型弱于专用 native 进程。

### 15.2 必须提前验证的决策

1. 目标浏览器是否限定为桌面 Chromium；若不是，需要为无目录写入能力的浏览器定义下载/单文件 fallback。
2. BTP 参考参数是否必须在浏览器中支持；若必须支持，W0/W2 必须包含该参数的真实峰值测量。
3. 用户是否需要刷新页面后继续使用同一 SK；若需要，必须增加独立的加密私钥持久化设计。
4. 输入是否始终为 CSV；若要支持图片/JSON/二进制，必须先定义模型输入转换层。

在这些问题得到验证前，不能宣称“所有浏览器、所有模型参数、任意本地文件”均可用。纯浏览器方案的最小可交付范围应是：桌面 Chromium、CSV 输入、CKKS/BTP 配置由任务文件确定、浏览器 Worker 内流式生成和本地文件落盘。

