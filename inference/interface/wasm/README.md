# Latti 浏览器 WASM 客户端

当前版本为 `0.1.2`。这个包提供浏览器端的本地密钥生成、上下文持久化、单文件及批量 CSV 加密和结果解密能力，适用于通过 HTTPS 或 `http://localhost` 打开的桌面版 Chrome、Edge 等 Chromium 浏览器普通网页。收到任务后的默认主流程为图像预处理、批量加密、服务端推理、批量解密和准确率统计；配套 CIFAR-10 任务包提供独立图像预处理和分类评估脚本，数据包提供原始验证图像及标注。

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

let keyDirectory;

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

## 图像预处理、批量加解密和准确率

收到任务后，默认按本节流程执行。任务目录提供配置和预处理、后处理脚本，无需包含 CSV；模型输入 CSV 由原始图像预处理生成，并保存到独立目录。任务须使用 `use_btp=true`，输入输出的 `ckks_parameter_id` 为内置的 `param0`；形状、level、packing 等布局来自任务文件。同一用户在需求兼容的任务之间复用上述两个密钥文件。

`latti-client-tasks.zip` 中的 CIFAR-10/ResNet-20 任务结构为：

```text
tasks/resnet20/client/
  task_config.json
  ckks_parameter.json
  preprocess.js
  postprocess.js
```

将任务脚本作为项目受信任静态资源部署；用户选择的本地 `client/` 目录用于读取任务配置。预处理将 `.png`、`.jpg`、`.jpeg` 原图顺序转为 3072 个 CHW 数值的 CSV，复用 `train.py` eval 阶段的 Resize、ToTensor、Normalize 规则。命名采用原图主文件名：`001.png → 001.csv`；批次内主文件名应唯一。

配套的 `latti-dataset-cifar10.zip` 从仓库同级的 `datasets/CIFAR-10/dataset/` 打包，解压后为：

```text
dataset/
  images/001.png
  images/002.png
  …
  images/010.png
  label.csv
```

这 10 张 PNG 来自 CIFAR-10 验证时使用的官方测试集（`torchvision.datasets.CIFAR10(train=False)`），十个类别各一张，保留原始 `32×32` RGB 像素，不进行缩放或数据增强。样本经配套 ResNet-20 任务逐张验证可正确分类；原始测试集索引见仓库同级 `datasets/CIFAR-10/README.md`。使用自己的图像时，在原始数据侧准备对应的标注表。

`label.csv` 为 UTF-8，按原图完整文件名关联 CIFAR-10 类别编号：

```csv
filename,label
001.png,0
002.png,1
```

类别 `0–9` 依次为 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。含逗号的文件名用 CSV 双引号包裹。

在各自按钮事件中用 `showDirectoryPicker()` 选取任务目录 `clientDirectory`、已有密钥目录 `keyDirectory`、原始图像目录 `imageDirectory`（数据包中的 `dataset/images/`）及各阶段输出目录，用 `showOpenFilePicker()` 选取 `dataset/label.csv` 作为 `labelFile`，再按下面的顺序调用。每个阶段使用独立输出目录；目录句柄及 `labelFile` 均已取得用户授权：

```js
import { openBrowserClient } from '/vendor/latti-client-wasm/dist/index.js';
import { preprocess_batch } from '/tasks/resnet20/client/preprocess.js';
import { postprocess_batch } from '/tasks/resnet20/client/postprocess.js';

await preprocess_batch({ inputDirectory: imageDirectory, outputDirectory: csvDirectory });
const client = await openBrowserClient({ clientDirectory, keyDirectory });
try {
  await client.restore();
  const encryptedFiles = await client.encrypt_batch({
    inputName: 'input', inputDirectory: csvDirectory,
    outputDirectory: encryptedInputDirectory,
  });

  // 按项目业务接口顺序推理，将结果保存至 encryptedResultDirectory，
  // 保持样本文件名，例如 001.ct.bin；服务端任务特征名仍为 input/output。

  await client.decrypt_batch({
    outputName: 'output', inputDirectory: encryptedResultDirectory,
    outputDirectory: logitsDirectory,
  });
} finally {
  await client.close();
}
const { predictionsFile, total, correct, accuracy } = await postprocess_batch({
  inputDirectory: logitsDirectory, labelFile, outputDirectory: reportDirectory,
});
console.log(`${correct}/${total}，准确率 ${(accuracy * 100).toFixed(2)}%`);
```

`inputName` 和 `outputName` 须与任务配置一致，表示任务特征名。磁盘输出名从输入文件名自动推导：`001.png → 001.csv → 001.ct.bin → 001.csv`。一个实例的操作经同一 Worker 串行执行，调用方按顺序 `await`。

网页按服务端业务接口上传 `public/eval_context.bin` 和密文。服务端直接使用现有 `import_eval_context_file()`，加载模型并推理。`secret/full_context.bin` 含 SK 和 PK，仅供本地恢复；上传时明确选择公开文件。同一次生成的两个密钥文件应一同保管。

刷新后重新选择密钥目录和任务目录，打开客户端并调用 `client.restore()`，再继续相应的批量阶段。切换任务时先关闭旧实例，再以相同 `keyDirectory` 和新的 `clientDirectory` 打开并恢复。

不传 `mode` 时，`encrypt_batch()`、`decrypt_batch()` 保持按目录处理 `.csv`、`.ct.bin` 的兼容行为，按文件名的 JavaScript 字符串顺序处理并返回文件句柄数组。服务端联调使用 `mode: 'server'`，由 WASM 直接读写 server 契约的 ZIP：

```js
const inputZip = await taskDirectory.getFileHandle('input.zip', {create: true});
await client.encrypt_batch({
  mode: 'server',
  inputDirectories: {input: cifarCsvDirectory},
  outputFile: inputZip,
});

await client.decrypt_batch({
  mode: 'server', outputName: 'output', inputFile: outputZip,
  outputDirectory: logitsDirectory,
});
```

单输入 ZIP 条目为 `<case_id>.ct`，多输入条目为 `<case_id>__<input_name>.ct`；结果 ZIP
条目为 `<case_id>.result.bin`。服务模式会检查各输入目录的 case 集合完全一致，任务 runner
不需要重命名密文或实现 ZIP reader/writer。输入 ZIP 使用无压缩存储以保持密文原字节，结果
ZIP 支持 server 常用的 stored/raw-deflate 条目。

`postprocess_batch()` 检查标注与结果一一对应，逐个读取 10 个有限 logits，取 argmax 为预测类别，并列最大值取最小下标。它写入 `reportDirectory/predictions.csv`（`filename,label,prediction,correct`），返回 `total`、`correct` 和 `[0,1]` 范围的 `accuracy=correct/total`。标注表行序可任意排列；文件名匹配负责样本关联。

## 原任务目录调用

仅传 `openBrowserClient({ clientDirectory })` 时，沿用原 CKKS/BTP 配置和路径：目录包含 `task_config.json`、`ckks_parameter.json`，`setup()` 将双文件写到该目录根部，`restore()` 读取根部的 `full_context.bin`。

## 构建与验证

在源码目录 `inference/interface/wasm/` 中执行：

```bash
npm ci
npm test
npm run build
npm run test:browser
npm run test:workflow
npm run package -- /tmp
```

`SHA256SUMS` 可在解压后的包根目录通过 `sha256sum -c SHA256SUMS` 校验。`onProgress` 的 `wasmBytes` 表示 WASM 线性内存，`heapBytes` 表示 Go HeapAlloc，均区别于浏览器进程 RSS。

`npm run package -- /tmp` 同时输出浏览器运行时 `latti-client-wasm-0.1.2.zip`、独立任务包 `latti-client-tasks.zip` 和验证数据包 `latti-dataset-cifar10.zip`。任务包不含输入 CSV；数据包只包含 `dataset/images/001.png` 至 `010.png` 和 `dataset/label.csv`。任务脚本使用浏览器 API，随任务包原样部署。

`npm run test:workflow` 在隔离 D-Bus/Xvfb 中通过真实 File System Access API 验证预处理、顺序落盘和分类评估；参考夹具由 `python3 tests/preprocessing-reference.py` 从真实训练 transform 生成。有既有 GPU 服务端、CIFAR-10 数据和固定参数密钥时，可运行完整多样本验收：

```bash
LATTI_WORKFLOW_KEYS=/path/to/N16QP1546H192H32 \
LATTI_CIFAR10_DATA=/path/to/data \
npm run test:workflow
```

数据目录包含 `cifar-10-batches-py/`。验收复用已有密钥，浏览器串行处理三个真实测试集原图，现有 GPU 服务端逐样本推理，页面刷新后恢复并批量解密，最后独立核对 logits、argmax、预测报告和准确率。结果写入打印出的 `/tmp/latti-workflow.*`。

维护者可在有既有 GPU 服务端构建和 CIFAR-10 task 文件的 Linux 服务器运行 `npm run test:pregeneration`。验收使用隔离的 D-Bus/Xvfb 自动选择 `/tmp` 测试目录并点击浏览器授权，生产 Worker 通过真实 File System Access API 写入超过 3 GiB 的 eval 文件；随后刷新、复用密钥、调用现有 GPU 服务端并解密比较。依赖 `dbus-run-session`、Xvfb、系统 Python 的 `dbus`/`gi`、X11/XTest 和桌面 Chrome；测试脚本仅用于自动化验收。JSON 指标及测试密钥保存在打印出的 `/tmp/latti-pregeneration.*` 目录。
