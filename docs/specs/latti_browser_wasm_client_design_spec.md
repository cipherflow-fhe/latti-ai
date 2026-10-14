# Latti-AI 纯浏览器 WebAssembly 客户端设计规范

> 本文定义浏览器端密钥生成和本地数据加密的最小可交付范围。客户端密钥文件的内容和恢复方式遵循[客户端双文件持久化设计规范](latti_client_context_persistence_design_spec.md)，evaluation context 的字节协议遵循[密钥流式生成设计规范](../streaming_eval_context_file_design_spec.md)。

## 1. 交付范围

第一版面向桌面 Chromium，在 HTTPS 或 `localhost` 安全上下文中运行。用户打开网页后选择一个可读写的客户端目录和 CSV 输入文件，浏览器完成：

1. 读取客户端目录中的任务配置；
2. 在 Dedicated Worker 中启动纯 Go WebAssembly 客户端；
3. 按任务配置选择普通 CKKS 或 BTP，逐个生成 evaluation key，并将 `eval_context.bin` 写入本地目录；
4. 将仅含参数、SK 和 PK 的 `full_context.bin` 写入本地目录，供客户端后续恢复加解密能力；
5. 按任务配置解析 CSV、完成本地加密，并将密文文件写入用户选择的本地目录。

浏览器生成的 `eval_context.bin` 继续由现有服务端 `import_eval_context_file()` 导入；服务端不需要浏览器专用的密文或 evaluation-context 解析器。

## 2. 文件和配置

客户端目录使用以下文件：

```text
client_directory/
  task_config.json
  ckks_parameter.json
  eval_context.bin
  full_context.bin
```

加密输出写入调用方传入的目录：

```text
output_directory/
  <input-name>.ct.bin
```

`task_config.json` 是任务布局和算法模式的来源，至少决定：

- `use_btp`：普通 CKKS 或 BTP；
- `task_input_param`、`task_output_param`：输入名称、维度、形状、通道、层级和 packing 参数；
- `pack_style`、`block_shape` 以及任务使用的其他布局字段。

每个输入参数的 `ckks_parameter_id` 用于从 `ckks_parameter.json` 选择 CKKS 参数。页面 API 不提供 scheme、参数、旋转集合或 packing 的覆盖参数；Worker 只使用任务文件确定的配置，并在配置缺失或不一致时返回错误。

`eval_context.bin` 是发往服务端的公开 evaluation context，不包含 SK。`full_context.bin` 仅保存在客户端本地，包含恢复本地加解密所需的参数、SK 和 PK，不包含评估键。两个文件来自同一轮 `setup()`。

## 3. 运行时架构

```text
网页主线程
  │ 用户选择目录和 CSV；显示进度
  │ FileSystemDirectoryHandle / FileSystemFileHandle / File
  ▼
Dedicated Worker
  │ 读取 task_config.json、ckks_parameter.json 和 CSV
  │ 串行调用 WASM 客户端并写入文件
  ▼
纯 Go WebAssembly（GOOS=js GOARCH=wasm CGO_ENABLED=0）
  │ CKKS/BTP 参数、SK/PK、packing、加密和 evaluation-key serializer
  ▼
浏览器 File System Access API
  │ createWritable() 顺序写入文件
  ▼
用户本地目录
  ├── eval_context.bin
  ├── full_context.bin
  └── <input-name>.ct.bin
```

主线程不接收完整 evaluation context、评估键或私钥内容。Worker 持有 WASM 会话和文件句柄，同一客户端的操作按顺序执行。文件 API 使用句柄或 `File`，不接受绝对路径字符串。

## 4. 已提交的实现基线

以下能力已在当前 `feat/streaming-key-generation` 分支提交，规范只保留其接口和验收边界：

- `inference/interface/wasm/` 提供无 cgo 的 Go/WASM 核心、Worker 适配层和浏览器包构建脚本；
- Worker 已实现 `setup()`、`restore()`、`encrypt()`、`decrypt()` 和 `close()` 的串行调用；
- `setup()` 通过 `begin`/`next` 状态机生成评估文件，每次只处理当前 payload，并以有界 chunk 写入 `eval_context.bin`，随后写入 `full_context.bin`；
- `task_config.json`、`ckks_parameter.json`、CSV 数值解析以及现有 native packing/ciphertext 布局已有纯 Go/JavaScript 实现；
- 普通 CKKS 和基础 BTP evaluation-context 编码路径已有实现，BTP evaluation 文件固定写 `has_sk=0`；
- `full_context.bin` 已支持新 Worker 会话恢复，恢复后可继续本地加密和解密；
- Go 核心、CSV parser 和 Chromium harness 已随代码提交。

这些条目不再作为新的代码设计任务；后续只验证它们在目标任务和目标浏览器中的互操作性及内存行为。

## 5. 浏览器客户端接口

浏览器包的外部接口保持当前实现：

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

export function openBrowserClient(options: {
  clientDirectory: FileSystemDirectoryHandle;
  onProgress?: (progress: Progress) => void;
}): Promise<BrowserClient>;
```

`openBrowserClient()` 打开 Worker 并读取两个配置文件。首次使用调用 `setup()`；已有 `full_context.bin` 时调用 `restore()`。`setup()`、`restore()` 成功后才能调用 `encrypt()` 或 `decrypt()`；`close()` 后会话失效。`setup()`、`encrypt()` 和 `decrypt()` 返回文件句柄，不把完整文件内容返回给页面。

## 6. 流式生成和文件格式

### 6.1 Worker 流式过程

Worker 按以下顺序工作：

```text
读取并校验任务文件
→ WASM begin：生成 SK/PK，建立 evaluation stream
→ WASM next：取得一个 payload，按有界 chunk 写入 eval_context.bin
→ 重复 next，直到 evaluation stream 完成
→ WASM full：导出参数/SK/PK，写入 full_context.bin
→ setup() 返回两个文件句柄
```

默认跨 WASM/JavaScript 边界的 chunk 为 4 MiB。一个 payload 可以在 WASM 内短暂保持其序列化 buffer，但 Worker 不保存完整 evaluation context、全部 RTK 或多个并行 keygen payload。文件写入采用浏览器 `createWritable()` 的顺序写入；文件写入完成并关闭后，操作才返回成功。

### 6.2 evaluation-context 协议

最终字节流必须与现有 native serializer 和服务端 deserializer 一致：

- 普通 CKKS：参数、`has_sk=0`、`has_pk=0`、RLK 和完整 RTK 集合；
- BTP：参数、无条件 `has_sk=0`、`has_pk=1` 及 PK、BTP 参数、RLK、完整 RTK 集合、DtS 和 StD；
- BTP 不根据 Worker 中是否持有 SK 改变 `has_sk`，也不写入 SecretKey payload；
- 复用现有 seed 压缩、QP 多项式、GadgetCiphertext、RTK 数量和 Galois 标识编码。

文件不增加浏览器专用容器头或尾部。分块只属于 Worker 写入过程，服务端看到的是连续的既有 evaluation-context 字节流。

### 6.3 full context 和密文

`full_context.bin` 沿用客户端双文件规范中的 advanced context 布局，评估键标志固定为 `0`。CSV 加密结果沿用现有 ciphertext 序列化格式，服务端按任务输入名称读取 `<input-name>.ct.bin`。

## 7. CSV 和本地加密

Worker 从 `FileSystemFileHandle` 或 `File` 的 stream 读取 CSV，并解析有限格式的数值字段。解析结果必须满足任务输入的名称、维度、形状、通道数量、层级和 packing 约束；NaN、Infinity、空字段和非法数值必须报错。packing、编码、加密和 ciphertext wire layout 与现有 `InferenceClient::encrypt()` 保持一致。

本地加密流程为：

```text
选择 input.csv
→ Worker 增量读取 CSV
→ WASM 按 task_config.json packing 并加密
→ Worker 写入 <input-name>.ct.bin
→ 返回密文文件句柄
```

## 8. 仍需完成的最小验收

当前代码已经提供基础实现，剩余工作集中在目标范围内的验证和交付：

1. **桌面 Chromium 实际目录流程**：使用 `showDirectoryPicker({mode: "readwrite"})` 选择用户目录，验证配置读取、`setup()`、本地文件落盘、刷新后的 `restore()` 和 CSV `encrypt()`；
2. **CKKS 互操作**：将浏览器生成的 `eval_context.bin` 导入现有 CPU 服务端，完成一次加密输入、服务端推理和客户端解密；
3. **BTP 互操作**：使用真实 BTP 任务文件在 Chromium 中完成同样流程，验证 PK、RLK、RTK、DtS/StD 和任务参数均可被现有服务端导入；
4. **任务配置一致性**：覆盖至少一个普通 CKKS 和一个 BTP 任务，确认 `use_btp`、参数 ID、packing 和输入布局完全由任务文件驱动，并拒绝配置不匹配的 context；
5. **流式内存测量**：在目标 CKKS/BTP 参数下记录 Worker/WASM 峰值内存、当前 payload 大小、chunk 大小、文件大小和耗时，确认生成过程中没有完整 evaluation-context 或全量 RTK 常驻；
6. **浏览器包交付检查**：执行 `inference/interface/wasm/build.sh`，在桌面 Chromium 中加载生成的 `dist` 包，确认 WASM runtime、Worker 和包内资源版本一致。

验收通过后，第一版交付物就是浏览器 WASM 包及其 Worker；native Node 包和服务端 GPU/CPU 运行时继续沿用现有交付物。

