# 浏览器图像预处理、批量加解密与分类评估设计规范

> 状态：已实现并完成验收（2026-10-15）。基于现有[浏览器客户端](latti_browser_wasm_client_design_spec.md)和[固定参数密钥预生成](latti_parameter_key_pregeneration_design_spec.md)能力，补充任务预处理、后处理脚本及 JS/TS 批量调用层。模块提交及验证记录见[实施计划](../plans/latti_browser_preprocessing_batch_implementation_plan.md)。

## 1. 目标与流程

桌面 Chromium 普通网页使用用户授权的 File System Access API，按以下流程处理 CIFAR-10/ResNet-20 推理任务：

```text
原始图像目录 → preprocess_batch() → CSV 目录
→ encrypt_batch() → 输入密文目录
→ 现有服务端逐样本推理 → 结果密文目录
→ decrypt_batch() → logits CSV 目录
→ postprocess_batch() + 独立标注表 → 逐样本预测与批次分类准确率
```

每个阶段串行处理文件，完成当前文件的写入并关闭后，再处理下一文件。目录的“主目录”指用户所选目录的直接子项；各阶段读取和输出的文件均位于该层。

批量接口复用同一个已 `setup()` 或 `restore()` 的 `BrowserClient`，每个文件沿用单样本的 packing、加解密及密文格式。

## 2. 任务交付与目录

当前任务的客户端目录增加独立的浏览器 ES Module：

```text
examples/test_cifar10/task/client/
  task_config.json
  preprocess.js
  postprocess.js
```

`preprocess.js` 从 `examples/test_cifar10/train.py` 的 `get_cifar10_loaders()` 中提取 `test_transform` 语义，即 `evaluate()` 接收输入前的预处理。它独立完成图像到 CSV 的转换，只依赖浏览器内置 API。

`postprocess.js` 按 `evaluate()` 中 `outputs.argmax(1).eq(targets)` 的语义，读取解密后的 logits 和用户准备的标注表，计算逐样本预测及分类准确率。

固定参数模式继续使用内置 `N16QP1546H192H32` 和独立的密钥目录。任务包将两个脚本与 `task_config.json` 一起交付；用户选择原图目录和 CSV 输出目录后生成输入数据。

Web 项目将任务附带的两个脚本作为受信任静态资源部署，通过 HTTPS 或 localhost 的模块 URL 导入。所选本地客户端目录供现有客户端读取任务配置；预处理和后处理代码由项目明确导入。

## 3. 独立预处理脚本

### 3.1 接口

```ts
export function preprocess_batch(options: {
  inputDirectory: FileSystemDirectoryHandle;
  outputDirectory: FileSystemDirectoryHandle;
}): Promise<FileSystemFileHandle[]>;
```

该任务脚本固定输出 `3 × 32 × 32`，与当前 `task_config.json` 的 `task_input_param.input` 对齐：`dim=2`、`channel=3`、`shape=[32,32]`。任务发布时同时核对脚本输出和配置。

输入采用单帧、不透明的 RGB 图像，支持 `.png`、`.jpg`、`.jpeg` 后缀，后缀匹配忽略大小写。脚本只收集根层匹配的文件句柄，以原文件名的 JavaScript 字符串顺序排序，然后逐张解码、转换和写入。

### 3.2 与 eval 一致的转换

| 步骤 | 处理规则 |
| --- | --- |
| 解码 | 取得 RGB 顺序的 8-bit 像素；Canvas 用于按原尺寸读取解码像素 |
| Resize | 对 RGB 像素缩放到高 32、宽 32，采用 torchvision 对 PIL 输入的默认双线性缩放及下采样抗锯齿语义，输出 RGB8 |
| ToTensor | 像素除以 255，以 float32 表示，转换为 CHW 顺序 |
| Normalize | 按 R/G/B 分别使用 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`，计算 `(value-mean)/std` |
| CSV | 按 `c → y → x` 展平，写一行 3072 个逗号分隔的数值，末尾换行 |

Resize 在脚本中以可对照 Pillow 的双线性计算实现。采样采用像素中心坐标，缩小时扩大三角滤波核，边界有效权重重新归一化，并对水平、垂直两次处理的 RGB8 结果取整。同尺寸输入直接使用原像素。缩放一致性由第 9 节的参考夹具验证。

归一化使用 `Math.fround()` 对除以 255、减均值、除标准差的各步结果取 float32，均值和标准差也先取 float32。CSV 保留这些数值的往返精度，由现有 CSV parser 和 Go packing 接收。输出仅包含模型输入数值。

每张图像保留原图主文件名，将最后的图像扩展名替换为 `.csv`，例如 `001.png → 001.csv`。预处理开始前按忽略大小写的输出名检查重名；`001.png` 和 `001.jpg` 等映射到同一 CSV 的源文件需先改为不同主文件名。本张 CSV 的 writable 关闭后，释放图像和像素数组，再处理下一张。

## 4. 单文件接口与自动命名

现有 `inputName`、`outputName` 继续表示任务配置中的特征名。输出文件名由源文件名自动推导，贯穿预处理、加密和解密过程。

单文件接口沿用现有参数结构：

```ts
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
```

Worker 从输入文件取得 `File.name`：加密时将末尾 `.csv` 替换为 `.ct.bin`，解密时将末尾 `.ct.bin` 替换为 `.csv`，后缀匹配忽略大小写。其他名称在完整源文件名后追加目标后缀。单文件和批量调用统一使用该规则。

Worker 写入时使用自动生成的文件名，调用现有 Go 核心时仍传入原来的 `inputName` 或 `outputName`。例如 `001.csv` 以任务特征名 `input` 加密后写为 `001.ct.bin`；服务端结果 `001.ct.bin` 以任务特征名 `output` 解密后写为 `001.csv`。

自动生成的名称按单个文件名校验，允许中文和空格，拒绝 `/`、`\`、NUL，以及 `.`、`..`。任务特征名继续使用现有校验。

## 5. 批量接口与执行规则

在 `BrowserClient` 增加两个 JS 包装方法：

```ts
encrypt_batch(options: {
  inputName: string;
  inputDirectory: FileSystemDirectoryHandle;
  outputDirectory: FileSystemDirectoryHandle;
}): Promise<FileSystemFileHandle[]>;

decrypt_batch(options: {
  outputName: string;
  inputDirectory: FileSystemDirectoryHandle;
  outputDirectory: FileSystemDirectoryHandle;
}): Promise<FileSystemFileHandle[]>;
```

共同执行规则：

1. 收集输入目录直接子项中的文件句柄：加密匹配 `.csv`，解密匹配 `.ct.bin`，后缀匹配忽略大小写。
2. 以原文件名的 JavaScript 字符串顺序排序，例如 `1.csv`、`10.csv`、`2.csv`；完成排序后开始读取文件内容。
3. 加密输出名为去掉最后 `.csv` 的名称加 `.ct.bin`；解密输出名为去掉最后 `.ct.bin` 的名称加 `.csv`。开始处理前校验生成的名称，并按忽略大小写的名称检查本批次输出重名。
4. 用 `for…of` 逐个 `await client.encrypt(...)` 或 `await client.decrypt(...)`，由单文件方法自动生成对应输出名。单文件 Promise 在写入关闭后完成，随后才读取下一文件。
5. 按处理顺序返回输出文件句柄数组。空目录返回空数组；读取、转换或单文件调用异常时，当前批量 Promise 直接 reject，错误关联源文件名，已完成的文件保留。

同名目标文件沿用现有单文件写入的覆盖规则。调用方为各阶段选择独立输出目录，并以 `await` 顺序调用各阶段。现有 `onProgress` 继续报告每次单文件加解密的进度。

批量层只保留文件名、句柄和返回句柄列表，文件内容按单样本生命周期释放；WASM/Worker 沿用现有单文件内存管理和串行请求队列。

命名示例：

| 原图 | 预处理 CSV | 输入密文 | 服务端结果密文 | 解密 CSV |
| --- | --- | --- | --- | --- |
| `001.png` | `001.csv` | `001.ct.bin` | `001.ct.bin` | `001.csv` |
| `002.jpg` | `002.csv` | `002.ct.bin` | `002.ct.bin` | `002.csv` |

输入、结果密文分别存放在各自目录。网页业务层保持样本文件名与服务端请求的对应关系，并将结果保存为相同的样本密文文件名。服务端继续以任务特征名 `input` 接收输入、以 `output` 返回结果；磁盘样本名由业务层管理。

## 6. 独立标注表与分类后处理

### 6.1 标注表

用户提前在原始数据侧准备独立的 `label.csv`，例如与原图目录同级存放：

```text
dataset/
  images/
    001.png
    002.jpg
  label.csv
```

预处理沿用第 3 节接口，只将图像转换为模型输入 CSV。后处理通过用户授权的 `FileSystemFileHandle` 或 `File` 读取标注表。

标注表为 UTF-8 CSV，首行固定为 `filename,label`，每行用原图完整文件名关联真实标签：

```csv
filename,label
001.png,3
002.jpg,5
```

`filename` 是原图的单个文件名，按 CSV 引号规则解析，保留中文和空格。`label` 是整数 `0–9`，沿用 `torchvision.datasets.CIFAR10` 的类别编号，依次对应 airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。

标注表描述本批待评估的全部样本。后处理将原图扩展名替换为 `.csv`，得到对应的 logits 文件名，例如 `001.png → 001.csv`，按名称关联标签和结果。读取后检查重复文件名、映射重名、标签范围以及标注与 logits 样本集合的一致性。

### 6.2 接口与计算

`client/postprocess.js` 提供独立接口：

```ts
export function postprocess_batch(options: {
  inputDirectory: FileSystemDirectoryHandle;
  labelFile: FileSystemFileHandle | File;
  outputDirectory: FileSystemDirectoryHandle;
}): Promise<{
  predictionsFile: FileSystemFileHandle;
  total: number;
  correct: number;
  accuracy: number;
}>;
```

`inputDirectory` 是解密后的 logits 目录，`outputDirectory` 是独立的报告目录。读取根层 `.csv` 文件并按文件名排序，逐个处理。每个文件包含按类别 `0–9` 排列的 10 个有限 logits；`argmax` 返回最大值的类别下标，并列时取最小下标，与 `torch.argmax` 一致。

每个 logits 文件必须恰好对应一条有效标注，标注表中的每个样本也必须有结果。成功统计的样本数为 `total`，预测下标等于真实标签的数量为 `correct`，返回 `accuracy = correct / total`，取值范围为 `[0,1]`；页面以 `accuracy × 100%` 显示准确率。用于计算准确率的样本集合至少包含一个样本。

后处理顺序写入 `outputDirectory/predictions.csv`，关闭后返回文件句柄及统计值。该文件保留逐样本比较结果：

```csv
filename,label,prediction,correct
001.png,3,3,1
002.jpg,5,2,0
```

此例返回 `total=2`、`correct=1`、`accuracy=0.5`。后处理保留标注映射、计数和当前样本的 10 个 logits，逐样本预测记录随处理写入报告。

## 7. Web 项目调用主流程

以下目录句柄均已由用户通过页面按钮授权选取；输入目录申请读取权限，输出目录申请读写权限。

```js
import { openBrowserClient } from '/vendor/latti-client-wasm/dist/index.js';
import { preprocess_batch } from '/tasks/resnet20/client/preprocess.js';
import { postprocess_batch } from '/tasks/resnet20/client/postprocess.js';

await preprocess_batch({
  inputDirectory: imageDirectory,
  outputDirectory: csvDirectory,
});

const client = await openBrowserClient({ clientDirectory, keyDirectory });
try {
  await client.restore();
  const ciphertextFiles = await client.encrypt_batch({
    inputName: 'input',
    inputDirectory: csvDirectory,
    outputDirectory: encryptedInputDirectory,
  });

  // 通过项目已有业务接口逐样本提交推理，
  // 将返回的密文按原样本文件名保存到 encryptedResultDirectory。

  const resultFiles = await client.decrypt_batch({
    outputName: 'output',
    inputDirectory: encryptedResultDirectory,
    outputDirectory: resultCsvDirectory,
  });
} finally {
  await client.close();
}

// labelFile 为用户事先准备并授权读取的独立标注表。
const { predictionsFile, total, correct, accuracy } = await postprocess_batch({
  inputDirectory: resultCsvDirectory,
  labelFile,
  outputDirectory: reportDirectory,
});
console.log(`${correct}/${total}，准确率 ${(accuracy * 100).toFixed(2)}%`);
```

首次密钥生成继续使用现有 `setup()`；后续批处理复用本地 `secret/full_context.bin`。预处理模块只接收图像目录和 CSV 输出目录，后处理模块接收 logits 目录、标注表和报告目录，加解密仍由现有密钥 Worker 完成。

## 8. 模块修改范围

| 位置 | 修改内容 |
| --- | --- |
| `examples/test_cifar10/task/client/preprocess.js` | 新增独立任务预处理模块，提供串行 `preprocess_batch()` |
| `examples/test_cifar10/task/client/postprocess.js` | 新增独立后处理模块，关联标注表、逐样本 argmax、写入预测报告并返回准确率 |
| `inference/interface/wasm/src/index.js` | 包装现有单文件方法，增加目录枚举、排序和两个批量方法 |
| `inference/interface/wasm/src/index.d.ts` | 声明批量方法，保留单文件接口参数 |
| `inference/interface/wasm/src/worker.js` | 按源文件名自动命名并校验，复用既有加解密指令与文件写入函数 |
| `inference/interface/wasm/tests/` | 增加预处理参考夹具、批量顺序、标注关联与分类评估验收 |
| `inference/interface/wasm/README.md` 与任务交付包 | 说明目录流程、接口、命名和独立标注表；任务包包含预处理与后处理脚本 |

Go/WASM 核心、Lattigo、LattiSense、服务端推理接口及二进制协议均直接复用。浏览器构建继续复制现有入口、Worker 和类型声明；任务预处理与后处理模块随任务文件发布。

## 9. 验收

1. **预处理对照**：用 `train.py` 的 `test_transform` 生成参考值。32×32 无损 RGB PNG 的 3072 个 CHW 数值最大绝对误差不超过 `1e-6`；覆盖纯色、通道不同、像素位置不同的夹具。另以放大、缩小及非正方形 RGB 夹具验证 Resize：相同解码像素下，缩放后每通道 RGB8 差不超过 1，归一化差不超过 `1/(255×std[c])+1e-6`。JPEG 单独核对浏览器与 Pillow 的解码差异。
2. **平铺和顺序**：根层至少三张图像生成三个 CSV，子目录中的文件保持原状；CSV、输入密文、结果密文分别按规定名称排序处理。记录每次开始与关闭时间，确认当前文件关闭后才开始下一文件。
3. **文件对应**：原图、CSV、输入密文、结果密文、结果 CSV 保持相同主文件名；预处理及加解密的批次内映射重名在处理前报错。检查输出文件数、名称和返回句柄顺序，包含中文、空格文件名和空目录。
4. **复用与兼容**：同一恢复会话依次加密多个 CSV、解密多个结果，沿用同一密钥；原单文件接口参数继续有效，单文件及批量调用均按源文件名自动命名。校验任务特征名、文件名与任务特征名的独立性，以及非法输出文件名。
5. **推理与资源**：多个 CIFAR-10 样本的密文由现有服务端逐个推理，浏览器解密后每个结果 CSV 含 10 个有限数值；与相同预处理输入的明文模型比较，最大绝对误差不超过 `0.1`。批处理过程中仅保留当前样本内容，文件数量增长只增加元数据和句柄列表。
6. **分类评估**：以已知 logits 和标签验证 argmax、并列最大值取最小下标、逐样本预测报告及 `correct/total`；打乱标注表行序后统计一致。覆盖缺失或多余结果、重复标注、映射重名、越界标签、非法 logits 和空样本集合的校验。真实推理结果按实际解密 logits 计算预测，统计值与对这些 logits 的独立参考计算一致。
7. **交付**：运行既有 JS/Go 测试、真实 Chromium 目录验收及浏览器包构建，核对任务包包含 `task_config.json`、`preprocess.js`、`postprocess.js`，使用独立 `label.csv` 和发布的浏览器包完成上述流程。

## 10. 实施与验收记录

上述功能已在本地 `feat/streaming-key-generation` 分支实现。新增功能测试先在缺失自动命名、预处理、后处理时失败，随后逐模块实现并验证。修改集中于浏览器 JS/TS、两个任务脚本、交付说明和验收代码。

已执行 `npm test`、`npm run build`、`npm run test:browser`、`npm run test:workflow` 和 `npm run package -- /tmp`。真实目录浏览器验收覆盖根层筛选、字符串排序、文件名冲突、中文与空格、串行写入、异常文件关联、空目录、标签映射、并列 argmax 和非法输入。六组训练 transform 参考夹具包含原尺寸、放大、缩小、非正方形、纯色和 JPEG，本机最大数值误差均为 0；预处理同时保留的已解码图像峰值为 1，完成后为 0。

完整验收使用桌面 Chrome、真实 File System Access API 和现有 RTX 5880 Ada GPU 服务端，复用已有 `N16QP1546H192H32` 密钥。CIFAR-10 测试集的前三个样本依次经历图像预处理、批量加密、GPU 推理、页面刷新与恢复、批量解密、标注比较。交付 ZIP 解压后的运行时及任务脚本也通过相同完整流程：

| 指标 | 从交付 ZIP 运行的验收结果 |
| --- | --- |
| 样本数 / 正确数 | 3 / 3 |
| 真实标签 / 预测分类 | `3,8,8` / `3,8,8` |
| 本次小批次准确率 | 100% |
| logits 相对同输入明文推理的最大绝对误差 | `0.0007566196 < 0.1` |
| 逐样本 GPU 推理耗时 | 6.29、6.29、6.06 秒 |
| 同时执行的样本加密数 | 1 |
| 加密会话 WASM 线性内存观测峰值 | 255,852,544 字节，244 MiB |
| 加密会话 Go HeapAlloc 观测峰值 | 236,934,432 字节，约 225.96 MiB |

内存值来自恢复及单文件进度回调的观测，表示本次批量加密会话；采样口径为 WASM 线性内存和 Go HeapAlloc。分类统计同时由独立 Python/NumPy 参考检查，预测表、argmax 和准确率一致。

完整源码验收报告为 `/tmp/latti-workflow.NO5xT6/report.json`；ZIP 验收报告为 `/tmp/latti-workflow.VI6X3Q/report.json`，日志为 `/tmp/latti-browser-batch-package-validation.log`。交付物为 `/tmp/latti-client-wasm-0.1.0.zip` 和 `/tmp/latti-client-tasks.zip`。已检查 ZIP 完整性、包内 `SHA256SUMS`、运行时和任务脚本与源码一致，任务 ZIP 目录为 `tasks/resnet20/client/`，含任务配置、原示例 CSV 和两个脚本。
