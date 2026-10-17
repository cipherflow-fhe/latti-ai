# Retina-DKD degree8_02：10 张图像数据包与浏览器预处理

本示例对应 `/LattiAI-Task-Resources/retina_degree8_02`，将其中
`task/client/gen_csv/gen_client_inputs_seg.py` 翻译为独立的浏览器 ES Module。
`task/client/preprocess.js` 和 `postprocess.js` 的调用方式与 CIFAR-10 示例一致，
使用 File System Access API 目录句柄，逐样本处理。两个模块只依赖浏览器内置 API。

本机交付目录：`/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10/`。
压缩包：`/LattiAI-Task-Resources/retina_degree8_02/retina_degree8_02_demo_10.zip`。

## 数据包

```text
demo_10/
  README.md
  client/
    task_config.json
    ckks_parameter.json
    preprocess.js
    postprocess.js
  dataset/                         ← 传给 preprocess_batch 的 inputDirectory
    images/                        # 10 张原始测试 JPG，保留文件名和原始字节
    disk/<原图主文件名>_disk.jpg     # 10 张新生成的 disk
    lesion/<原图主文件名>_lesion.jpg # 10 张新生成的 lesion
    risk_factor_5.xlsx             # 第一张工作表：表头 + 10 位患者的原始临床指标
    label.csv                      # filename,label，0=DN，1=NDRD
  csv/                             ← JS 已生成，可直接作为待加密输入
    input_0/<原图主文件名>.csv      # 每文件 5 行，每行 262144 个数
    input_1/<原图主文件名>.csv      # 每文件 1 行、2 个数
    input_2/<原图主文件名>.csv      # 每文件 1 行、3 个数
  reference_logits/                # 从所给 CSV 提取的历史 logits，供验证后处理
  reports/
    predictions.csv
    accuracy.json
  selected_samples.csv             # 样本、患者 ID、来源路径、标签、历史输出及归一化指标
  manifest.json                    # 选择规则、ONNX 设置和模型/原图 SHA-256
  validation/
    browser.json
    python.json
  SHA256SUMS
```

样本从 `py_output_all_ub1p75_deg8_ep005.csv` 中选取，10 张对应 10 位不同患者：
4 张正确分类的 DN、1 张错误分类的 DN、5 张正确分类的 NDRD。每组按两个 logits
的绝对差从大到小选取，文件路径作为次排序条件。标签直接使用 `ground_truth`；
工作簿中的“DN合并NDRD”按原测试结果属于类别 1。

**90% 是所给历史模型输出在这 10 张选样上的准确率。**
`reference_logits/` 来自原始 `raw_output_0,raw_output_1`，用于复现后处理；
新推理结果应使用实际解密的 logits 重新统计。本次执行了 ONNX 分割和 JS 前后处理，
分类模型及 FHE 推理需接入现有推理流程。

分割实际调用指定目录中的 `predict_onnx.py`：`lesion.onnx`、`disc.onnx` 和
`cup1_b1.onnx` 至 `cup5_b1.onnx`，CPU 4 线程、fast disk 模式。
lesion 输入尺寸 1024，保留灰度软概率并清除眼底视野外像素；disk 阈值 0.75，
输出背景 255、视盘环 128、视杯 0。两个输出均按原部署代码保存为质量 100 的灰度 JPG。

## 预处理规则与调用

在 Chromium 的 HTTPS 或 localhost 页面导入模块。浏览器需提供 `createImageBitmap`、
`OffscreenCanvas`、`DOMParser`、`DecompressionStream('deflate-raw')` 和目录读写 API。
`inputDirectory` 选择 `dataset/`，`outputDirectory` 选择独立的 CSV 输出目录：

```js
import {preprocess_batch} from '/tasks/retina/client/preprocess.js';

const samples = await preprocess_batch({
  inputDirectory: datasetDirectory,
  outputDirectory: csvDirectory,
});
// samples: [{input_0: FileSystemFileHandle, input_1: ..., input_2: ...}, ...]
```

原图支持根层 `.jpg/.jpeg/.png`，按文件名排序；每张原图对应
`disk/<stem>_disk.jpg` 和 `lesion/<stem>_lesion.jpg`。处理前检查输出名冲突、
患者指标及分割文件对应关系。患者 ID 提取沿用 Python 的 `get_id()`。
临床表读取第一张工作表的前 8 列，保持原顺序：

| 列 | 原始含义 | 模型输入 |
| --- | --- | --- |
| A | 编号 | 不输入模型 |
| B | 患者 ID | 用于匹配原图 |
| C | 病理分组 | 不输入模型 |
| D | 收缩压 `press` | `input_1[0] = press / 202` |
| E | 糖尿病病程（月）`course` | `input_1[1] = course / 564` |
| F | 糖化 `mashing` | `input_2[0] = mashing / 13.8` |
| G | 血红蛋白（原脚本变量名 `protein`） | `input_2[1] = protein / 207` |
| H | `eGFR` | `input_2[2] = eGFR / 132` |

病程为空时补 0；糖化为空时，糖化和血红蛋白的模型输入均置 0，保持原脚本行为。
XLSX 支持数值单元格、inline/shared strings 和稀疏单元格，工作表路径从 workbook
关系中读取。模型只接收处理后的五项指标。

图像按 Pillow 的 RGB8 双线性规则缩放至 `512×512`，下采样扩大滤波核，两次通道计算
分别取整。RGB 除以 255 后按 ImageNet 均值 `[0.485,0.456,0.406]`、标准差
`[0.229,0.224,0.225]` 归一化，中间步骤使用 float32。
掩码先在原尺寸逐通道做 `>80 → 255，其余 → 0`，再双线性缩放、除以 255、
通道求均值、以 `>0.5` 二值化。最终通道顺序是 `[R,G,B,1-disk,lesion]`。
CSV 保留 float32 数值的往返精度。默认 `normalize=true` 对应本任务原命令的
`--normalize`；显式传入 `normalize:false` 对应原 Python 不使用该开关的情况。

三种输入分别加密，特征名必须与任务配置一致，密文也应分别存储：

```js
for (const inputName of ['input_0', 'input_1', 'input_2']) {
  await client.encrypt_batch({
    inputName,
    inputDirectory: await csvDirectory.getDirectoryHandle(inputName),
    outputDirectory: await encryptedDirectory.getDirectoryHandle(inputName, {create: true}),
  });
}
```

同一主文件名的三份密文共同组成一次分类推理输入。

## 后处理

```js
import {postprocess_batch} from '/tasks/retina/client/postprocess.js';

const {predictionsFile, total, correct, accuracy} = await postprocess_batch({
  inputDirectory: decryptedLogitsDirectory,
  labelFile: await datasetDirectory.getFileHandle('label.csv'),
  outputDirectory: reportDirectory,
});
console.log(`${correct}/${total}，准确率 ${(accuracy * 100).toFixed(2)}%`);
```

每个结果 CSV 必须包含两个有限数值，顺序为 `[DN,NDRD]`，主文件名对应原图。
执行 `argmax`，相等时取类别 0；与 `label.csv` 中的真实标签比较，返回
`accuracy=correct/total`，并写入 `predictions.csv`。检查标签范围、重复标注、
缺失或额外结果、非法 logits 和空样本集合。标签行序不影响关联。

将 `reference_logits/` 作为输入时，结果为 `total=10, correct=9, accuracy=0.9`；
错误样本为 `45-Y2747811-5549.jpg`，真实类别 0、历史预测类别 1。

## 在本服务器复现

从仓库根目录执行。分割与 Python 对照使用已安装依赖的 Retina-DKD 环境，
浏览器运行器复用 `inference/interface/wasm` 中的 `playwright-core` 和本机 Chrome。

```bash
/home/wangy/Retina-DKD/myenv/bin/python examples/test_retina_dkd/prepare_dataset.py
/home/wangy/Retina-DKD/myenv/bin/python examples/test_retina_dkd/verify_preprocessing.py --prepare-fixtures
node examples/test_retina_dkd/run_browser.mjs
/home/wangy/Retina-DKD/myenv/bin/python examples/test_retina_dkd/verify_preprocessing.py
```

`prepare_dataset.py` 支持 `--client-dir`、`--xlsx`、`--output-dir`、`--cpu-count`。
`run_browser.mjs` 可接收数据包目录作为位置参数；`verify_preprocessing.py` 支持
`--client-dir`、`--demo-dir`。额外验证用的合成图片及 CSV 放在 `validation/` 内，
正式 ZIP 只包含验证报告。

验收包含 10 张真实图像、4 种额外图像在开启/关闭归一化下的 8 组输出，
覆盖放大、非正方形下采样、同尺寸、RGB 掩码、80 阈值、缺失指标、数字 ID、
ID 前缀去除、XLSX 两种字符串表示及非默认工作表路径。逐值对照原 Python
图像输入；临床输入对照精度为 `1e-15`。浏览器还检查 argmax 并列规则、
无效 logits、非法标签、文件对应和重复输出名，结果记录在 `validation/*.json`。

本机验收已通过：23,592,960 个图像输入数值与原 Python 逐值一致，最大绝对误差为 0；
10 张真实样本的 50 项临床输入误差也为 0。14 项浏览器行为检查全部通过。
