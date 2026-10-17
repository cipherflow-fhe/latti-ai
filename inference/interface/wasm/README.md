# Latti 浏览器 WASM 客户端

本文只说明 `0.1.2` 浏览器包如何运行 ResNet20/CIFAR10 和 Retina-DKD
degree8_02 两个任务。交付流程统一使用 server-compatible 批量模式：浏览器
预处理原图、批量加密并生成 server 需要的 `input.zip`，server 推理后返回
`output.zip`，浏览器批量解密并由任务脚本统计准确率。调用方不需要修改密文、
重命名文件或重新打 ZIP。

## 交付包目录

完整交付包 `latti-e2e-0.1.2.zip` 包含：

```text
latti-e2e-0.1.2/
  WASM-0.1.2.zip
  resnet20-task-0.1.2.zip
  resnet20-data-0.1.2.zip
  retina-task-degree8-02-0.1.2.zip
  retina-data-degree8-02-0.1.2.zip
```

任务包中的脚本导出接口一致：

```text
preprocess.js  -> preprocess_batch({inputDirectory, outputDirectory})
postprocess.js -> postprocess_batch({inputDirectory, labelFile, outputDirectory})
```

WASM 包解压后，将 `dist/` 作为静态资源部署。浏览器必须运行在 HTTPS 或
`http://localhost` 安全上下文中，并支持 Dedicated Worker、File System Access
API、`createImageBitmap`、`OffscreenCanvas` 和 `DecompressionStream`。

## 生成并复用密钥

两个任务都使用 `N16QP1546H192H32`、`param0` 和 `use_btp=true`，因此同一用户
只需生成一次密钥。`keyDirectory` 可以是用户选择的父目录；客户端会在其中创建
`N16QP1546H192H32/`，也可以直接选择该参数目录。

```js
import {openBrowserClient} from '/vendor/latti-client-wasm/dist/index.js';

// 必须在按钮点击等即时用户手势中调用。
const keyDirectory = await window.showDirectoryPicker({mode: 'readwrite'});
const keyClient = await openBrowserClient({keyDirectory});
const {evalContext, fullContext} = await keyClient.setup();
await keyClient.close();

// evalContext 对应 public/eval_context.bin，可上传给 server。
// fullContext 对应 secret/full_context.bin，只保存在浏览器本地。
console.log(evalContext.name, fullContext.name);
```

任务切换时不重新生成密钥。关闭旧客户端，选择新的任务 `client/` 目录，使用
同一个 `keyDirectory` 打开并调用 `restore()`：

```js
const client = await openBrowserClient({
  clientDirectory: taskClientDirectory,
  keyDirectory,
});
await client.restore();
```

只有参数预设、`param0` 和 BTP 配置完全一致时才能复用。`full_context.bin` 不上传
server，也不放入任何交付 ZIP。

## 统一批处理流程

每个任务都按以下顺序执行：

```text
选择 task/client/、dataset/、label.csv 和各阶段输出目录
  -> preprocess_batch()
  -> openBrowserClient({clientDirectory, keyDirectory}).restore()
  -> encrypt_batch({mode: 'server', inputDirectories, outputFile})
  -> 提交 input.zip 给 latti-client server
  -> 下载 server 返回的 output.zip
  -> decrypt_batch({mode: 'server', outputName, inputFile, outputDirectory})
  -> postprocess_batch() 生成 predictions.csv 和 accuracy
```

预处理和后处理脚本由任务包导入，WASM 客户端只负责密钥、加密和解密：

```js
const task = await import('/tasks/<task>/client/preprocess.js');
await task.preprocess_batch({
  inputDirectory: datasetDirectory,
  outputDirectory: csvDirectory,
});

const client = await openBrowserClient({clientDirectory, keyDirectory});
try {
  await client.restore();
  // 见下节的 inputDirectories。
} finally {
  await client.close();
}

const result = await (await import('/tasks/<task>/client/postprocess.js')).postprocess_batch({
  inputDirectory: logitsDirectory,
  labelFile,
  outputDirectory: reportDirectory,
});
console.log(`${result.correct}/${result.total} = ${(result.accuracy * 100).toFixed(2)}%`);
```

### ResNet20/CIFAR10

任务目录为 `tasks/resnet20/client/`，数据目录为：

```text
dataset/
  images/001.png ... 010.png
  label.csv                 # filename,label；类别为 0–9
```

`preprocess_batch()` 接收 `dataset/` 或 `dataset/images/`，输出一个 CSV 目录：

```text
csv/
  001.csv ... 010.csv       # 每个文件 3072 个 CHW 输入值
```

加密和解密使用任务特征名 `input`、`output`：

```js
const inputZip = await encryptedDirectory.getFileHandle('input.zip', {create: true});
await client.encrypt_batch({
  mode: 'server',
  inputDirectories: {input: csvDirectory},
  outputFile: inputZip,
});

await client.decrypt_batch({
  mode: 'server',
  outputName: 'output',
  inputFile: outputZip,
  outputDirectory: logitsDirectory,
});
```

每个 case 的输入 ZIP 条目为 `<case_id>.ct`，解密后为 `<case_id>.csv`，每个结果
包含 10 个 logits。`postprocess_batch()` 对 10 个值取 argmax，与 `label.csv` 关联后
写出 `predictions.csv`。

### Retina-DKD degree8_02

任务目录为 `tasks/retina_degree8_02/client/`，数据目录为：

```text
dataset/
  images/<original>.jpg
  disk/<stem>_disk.jpg
  lesion/<stem>_lesion.jpg
  risk_factor_5.xlsx
  label.csv                 # filename,label；标签为 0 或 1
```

`preprocess_batch()` 接收整个 `dataset/`，并在输出目录创建三个输入目录：

```text
csv/
  input_0/<case>.csv         # 原图和两个分割通道，5 x 512 x 512
  input_1/<case>.csv         # 非侵入临床特征，2 个值
  input_2/<case>.csv         # 侵入临床特征，3 个值
```

三路目录必须有完全相同的 case 集合。加密时特征名必须与任务配置一致：

```js
const inputZip = await encryptedDirectory.getFileHandle('input.zip', {create: true});
await client.encrypt_batch({
  mode: 'server',
  inputDirectories: {
    input_0: await csvDirectory.getDirectoryHandle('input_0'),
    input_1: await csvDirectory.getDirectoryHandle('input_1'),
    input_2: await csvDirectory.getDirectoryHandle('input_2'),
  },
  outputFile: inputZip,
});

await client.decrypt_batch({
  mode: 'server',
  outputName: 'output',
  inputFile: outputZip,
  outputDirectory: logitsDirectory,
});
```

多输入 case 的 ZIP 条目为 `<case_id>__input_0.ct`、`<case_id>__input_1.ct` 和
`<case_id>__input_2.ct`。server 返回 `<case_id>.result.bin`，WASM 直接读取该
ZIP 并输出 `<case_id>.csv`；结果包含 `[DN, NDRD]` 两个 logits。
`postprocess_batch()` 取较大值对应的类别，平票取 0，写出预测和准确率。配套的
10 张演示图像历史结果为 9/10（90%）；实际准确率始终以本次解密 logits 计算。

## Server 文件契约

`encrypt_batch({mode: 'server', ...})` 生成的 ZIP 直接提交给现有 latti-client
server：

```text
input.zip
  <case_id>.ct                         # 单输入任务
  <case_id>__<input_name>.ct           # 多输入任务

output.zip
  <case_id>.result.bin
```

WASM 会检查输入目录的 case 集合、重复名称和非法名称；server 负责解压、按任务配置
运行模型并生成结果 ZIP。浏览器端不处理 `skip` 字段，不重命名 `.ct`、`.ct.bin` 或
`.result.bin`，也不实现额外的 ZIP reader/writer。

## 构建和打包

WASM 版本保持为 `0.1.2`。在本目录执行：

```bash
npm ci
npm test
npm run package -- /tmp
```

`npm run package -- /tmp` 生成 `latti-client-wasm-0.1.2.zip`、ResNet20 任务包和
CIFAR10 数据包。再从仓库根目录组装包含两个任务及两个数据包的完整交付包：

```bash
python3 examples/package_e2e_0_1_2.py \
  --wasm-zip /tmp/latti-client-wasm-0.1.2.zip \
  --output-dir /LattiAI-Task-Resources
```

输出为 `/LattiAI-Task-Resources/latti-e2e-0.1.2.zip`，其中的 WASM 版本仍为
`0.1.2`，不包含用户密钥。
