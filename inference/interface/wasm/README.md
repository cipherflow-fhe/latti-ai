# Latti 浏览器 WASM 客户端

这个包提供浏览器端的本地密钥生成、上下文持久化、CSV 加密和结果解密能力，适用于通过 HTTPS 或 `http://localhost` 打开的桌面版 Chrome、Edge 等 Chromium 浏览器普通网页。

“desktop Chromium web application” 指在桌面版 Chromium 浏览器中运行的 Web 应用。普通网页就符合使用场景，不需要 Electron、系统 WebView、PWA、CEF、Node.js 或本地代理。加解密在浏览器 Dedicated Worker 中执行，文件通过用户授权的 File System Access API 保存到本地磁盘。刷新或重新打开网页后，重新选择目录并调用 `restore()` 即可恢复。

## 包内容与部署

```text
latti-client-wasm-<version>/
  dist/
    index.js           # 浏览器 API
    index.d.ts         # TypeScript 类型声明
    worker.js          # Dedicated Worker
    csv.js             # CSV 流式解析器
    wasm_exec.js       # Go WASM 运行时
    latti_client.wasm  # FHE 客户端核心
  package.json
  SHA256SUMS
  README.md
```

将 ZIP 解压到项目静态资源目录，保持 `dist/` 内的相对路径，并从网页导入 `dist/index.js`。使用构建工具时，将整个 `dist/` 作为静态资源复制。服务器须以 `application/wasm` 提供 `.wasm`，并允许模块 Worker 加载。正式环境使用 HTTPS，开发环境可使用 localhost。

## 提前生成密钥

用户只需通过目录选择器选择密钥保存目录。客户端自动使用内置的 `N16QP1546H192H32` 完整 BTP 预设，并创建或打开同名子目录；直接选择已有的 `N16QP1546H192H32` 目录时，会使用该目录。首次生成后得到：

```text
<用户选择的目录>/N16QP1546H192H32/
  secret/full_context.bin
  public/eval_context.bin
```

参数 ID 固定为 `param0`，参数由客户端自动指定。`setup()` 在 Worker 中串行生成评估键，以 4 MiB chunk 等待磁盘写入，关闭两个文件后返回文件句柄。该预设的 eval 文件约 3.10 GiB，full 文件约 24.3 MiB；请为生成和浏览器写入暂存预留磁盘空间。

```js
import { openBrowserClient } from './vendor/latti-client-wasm/dist/index.js';

let keyDirectory, clientDirectory, client, inputFile, outputDirectory, resultFile;

// 以下每个 select 函数分别绑定一个按钮的点击事件。
// 浏览器选择器需要即时用户手势，生成完成后再点击下一步按钮。
async function selectKeys() {
  keyDirectory = await window.showDirectoryPicker({ mode: 'readwrite' });
}

async function generateKeys() {
  const keys = await openBrowserClient({
    keyDirectory,
    onProgress: p => console.log(p.phase, p.bytes, p.elapsedMs),
  });
  try {
    const { evalContext, fullContext } = await keys.setup();
    console.log(evalContext.name, fullContext.name);
  } finally {
    await keys.close();
  }
}
```

目录中已有任一密钥文件时，`setup()` 拒绝覆盖。已有完整密钥通过 `restore()` 恢复；需要另一套密钥时，为该用户选择新的密钥保存目录。仅传 `keyDirectory` 的会话支持生成和恢复，CSV 加解密在任务配置加载后调用。

## 收到任务后恢复和加解密

任务目录提供 `task_config.json` 和 CSV。任务须使用 `use_btp=true`，输入输出的 `ckks_parameter_id` 为内置的 `param0`；形状、level、packing 等布局来自任务文件。同一用户在需求兼容的任务之间复用上述两个密钥文件。

接续上面的变量，按按钮顺序执行：

```js
async function selectTask() {
  clientDirectory = await window.showDirectoryPicker({ mode: 'read' });
}

async function restoreClient() {
  client = await openBrowserClient({ clientDirectory, keyDirectory });
  await client.restore();
}

async function selectCSV() {
  [inputFile] = await window.showOpenFilePicker({
    types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }],
  });
}

async function selectOutput() {
  outputDirectory = await window.showDirectoryPicker({ mode: 'readwrite' });
}

async function encryptCSV() {
  return await client.encrypt({ inputName: 'input', inputFile, outputDirectory });
}

async function selectResult() {
  [resultFile] = await window.showOpenFilePicker();
}

async function decryptResult() {
  return await client.decrypt({
    outputName: 'output', ciphertextFile: resultFile, outputDirectory,
  });
}

async function closeClient() {
  await client?.close();
}
```

`inputName` 和 `outputName` 须与任务配置一致。`encrypt()` 写入 `<inputName>.ct.bin`，`decrypt()` 写入 `<outputName>.csv`；文件参数接受 `FileSystemFileHandle` 或 `File`，返回值为文件句柄。一个实例的操作经同一 Worker 串行执行，调用方按顺序 `await`。

网页按服务端业务接口上传 `public/eval_context.bin` 和密文。服务端直接使用现有 `import_eval_context_file()`，加载模型并推理。`secret/full_context.bin` 含 SK 和 PK，仅供本地恢复；上传时明确选择公开文件。同一次生成的两个密钥文件应一同保管。

刷新后重新选择密钥目录和任务目录，调用 `restoreClient()`。切换任务时先关闭旧实例，再以相同 `keyDirectory` 和新的 `clientDirectory` 打开并恢复。

## 原任务目录调用

仅传 `openBrowserClient({ clientDirectory })` 时，沿用原 CKKS/BTP 配置和路径：目录包含 `task_config.json`、`ckks_parameter.json`，`setup()` 将双文件写到该目录根部，`restore()` 读取根部的 `full_context.bin`。

## 构建与验证

在源码目录 `inference/interface/wasm/` 中执行：

```bash
npm ci
npm test
npm run build
npm run test:browser
npm run package -- /tmp
```

`SHA256SUMS` 可在解压后的包根目录通过 `sha256sum -c SHA256SUMS` 校验。`onProgress` 的 `wasmBytes` 表示 WASM 线性内存，`heapBytes` 表示 Go HeapAlloc，均区别于浏览器进程 RSS。

维护者可在有既有 GPU 服务端构建和 CIFAR-10 task 文件的 Linux 服务器运行 `npm run test:pregeneration`。验收使用隔离的 D-Bus/Xvfb 自动选择 `/tmp` 测试目录并点击浏览器授权，生产 Worker 通过真实 File System Access API 写入超过 3 GiB 的 eval 文件；随后刷新、复用密钥、调用现有 GPU 服务端并解密比较。依赖 `dbus-run-session`、Xvfb、系统 Python 的 `dbus`/`gi`、X11/XTest 和桌面 Chrome；测试脚本仅用于自动化验收。JSON 指标及测试密钥保存在打印出的 `/tmp/latti-pregeneration.*` 目录。
