# 浏览器到 FHE Server 端到端联调规范

> 状态：Retina 已完成实际 10 case 联调；本文同时记录 ResNet20/CIFAR10 的现有验证基线和后续最小改动方案。当前已验证的是旧版 WASM 加临时 runner；目标方案只重新发布 latti-ai WASM，`latti-client` 保持不变。

## 1. 验证结论

本次 Retina 使用未重新编译的 `latti-client-wasm-0.1.1`，浏览器完成预处理、三路加密，`latti-client` GPU server 完成 10/10 case 推理，浏览器完成解密和后处理，最终为 **9/10 = 90%**。错误样本为 `45-Y2747811-5549.jpg`（真实标签 0，预测 1）。这证明 Retina 的任务配置、数据、分割结果、FHE 参数和模型服务可以互操作；Retina 的输入加密本身不要求重编译 WASM。旧 WASM 的结果解密仍有 scalar skip 兼容问题，见第 3 节；要取消 runner 补丁，需要发布一次修复后的 WASM。

完整运行记录保存在 `/LattiAI-Task-Resources/retina_degree8_02/task/client/demo_10/validation/server-integration.json`。

ResNet20/CIFAR10 沿用同一 WASM 浏览器批处理接口：一个 `input`、10 个输出 logits，后处理取 `argmax` 并统计准确率。现有 `inference/interface/wasm/tests/workflow.mjs` 和 `workflow-native.py` 已覆盖真实图像预处理、批量加解密、native server、logits 对照和准确率检查；Retina 是在此基础上增加多输入和临床数据的联调。不要把 CIFAR 的既有测试结果当作本次 Retina server 运行结果。

## 2. Retina 交付内容

新增或使用的文件如下：

```text
examples/test_retina_dkd/
  task/client/preprocess.js       # 图像、disk/lesion、XLSX -> 三路 CSV
  task/client/postprocess.js      # 两个 logits -> argmax、predictions.csv、accuracy
  prepare_dataset.py              # 选择样本并生成分割资源
  verify_preprocessing.py         # Python/JS 数值和行为对照
  package_retina.py               # 复用现有 WASM，组装任务包和数据包
  run_browser.mjs                 # Retina 浏览器端自动化验收
  run_server_integration.mjs      # Retina 临时 server 编排器（旧版基线）
examples/e2e/
  run_browser.mjs                 # Retina/ResNet20 公共浏览器入口
  run_server_integration.mjs      # Retina/ResNet20 公共 server 入口
examples/package_e2e_0_1_2.py     # 组装最终五个子 ZIP 的总交付包
```

Retina 的原始验收脚本仍位于 `examples/test_retina_dkd/`，ResNet20 的浏览器工作流仍位于
`inference/interface/wasm/tests/workflow.mjs`；`examples/e2e/` 已提供按
`LATTI_E2E_TASK=retina|resnet20` 选择任务的公共入口。任务目录只保留各自的
`preprocess.js`、`postprocess.js`，两个任务共享同一个联调入口，避免维护两套 ZIP、
文件名和状态轮询逻辑。

数据包的输入目录为 `dataset/`：

```text
dataset/
  images/<original>.jpg
  disk/<stem>_disk.jpg
  lesion/<stem>_lesion.jpg
  risk_factor_5.xlsx
  label.csv                    # filename,label；Retina 标签为 0/1
```

`preprocess.js` 保持原 Python 的 Pillow 双线性、float32、ImageNet 归一化和掩码阈值规则，输出 `input_0`（5 个 512x512 通道）、`input_1`（2 个临床值）和 `input_2`（3 个临床值）。三路文件用相同的样本主文件名，分别以任务配置中的 `input_0`、`input_1`、`input_2` 加密。`postprocess.js` 要求两个有限 logits，平票取 0，按 `label.csv` 的完整原图文件名关联并写出预测和准确率。

已完成的数值验收：23,592,960 个图像输入值与 Python 逐值一致，最大误差为 0；10 张图像的 50 个临床输入值也一致；浏览器行为检查 14 项全部通过。历史参考 logits 和本次实际 FHE logits 都得到 9/10。

### 2.1 同态上下文的生成与复用

ResNet20/CIFAR10 和 Retina-DKD 的客户端参数都使用同一个固定 BTP 预设
`N16QP1546H192H32`：`poly_modulus_degree=65536`、`n_mult_level=9`、
`log_default_scale=40`、`use_btp=true`，任务配置中的参数 ID 都是 `param0`。
因此同一用户只需在同一个密钥目录中执行一次 `setup()`，生成：

```text
N16QP1546H192H32/
  public/eval_context.bin       # 上传给 server 的公开评估上下文
  secret/full_context.bin       # 仅浏览器本地保存的完整密钥上下文
```

切换任务时关闭旧客户端，以新的 `clientDirectory` 打开 WASM 客户端，继续使用同一个
`keyDirectory` 并调用 `restore()`，不需要为每个任务重新生成两个文件。复用的前提是
任务仍引用这个固定参数预设；如果任务使用其他 CKKS 参数、BTP 配置或参数 ID，则必须
另建对应的密钥目录。`full_context.bin` 不上传 server，也不放入交付包。

### 2.2 两个任务的脚本接口

两个任务的客户端脚本文件名和主导出函数一致：

```text
task/client/preprocess.js   -> preprocess_batch({inputDirectory, outputDirectory})
task/client/postprocess.js  -> postprocess_batch({inputDirectory, labelFile, outputDirectory})
```

两者都能按目录批量处理原始图像并生成 CSV，也都用 `postprocess_batch()` 执行
logits 的 `argmax`、标签关联和准确率统计；返回值都包含
`{predictionsFile, total, correct, accuracy}`。差异只在数据目录和输出形状：

| 任务 | `preprocess_batch` 的输入目录 | 生成的 CSV | `postprocess_batch` 的 logits |
| --- | --- | --- | --- |
| ResNet20/CIFAR10 | `dataset/` 或 `dataset/images/`（原图 PNG/JPG） | 一个 `*.csv`，3072 个 CHW 数值 | 10 个 logits，类别 0–9 |
| Retina-DKD | `dataset/`（含 `images/`、`disk/`、`lesion/`、`risk_factor_5.xlsx`） | 同一 case 的 `input_0/`、`input_1/`、`input_2/` 三个 CSV | 2 个 logits，顺序 `[DN,NDRD]` |

因此调用方可以统一按“选择数据目录 → `preprocess_batch()` → 加密 → server 推理 →
`postprocess_batch()`”组织流程；只需依据任务配置处理单输入或多输入结果。

### 2.3 最终交付包

最终交付文件为一个总 ZIP，内部包含五个可独立解压的 ZIP：

```text
latti-e2e-0.1.2.zip
  WASM-0.1.2.zip
  resnet20-task-0.1.2.zip
  resnet20-data-0.1.2.zip
  retina-task-degree8-02-0.1.2.zip
  retina-data-degree8-02-0.1.2.zip
```

任务 ZIP 只包含 `task/client/` 下的配置、预处理和后处理脚本；数据 ZIP 分别包含
原始图像、标签以及 Retina 所需的 disk/lesion/XLSX 资源。WASM ZIP 不包含任何用户密钥。

## 3. 关键 debug

1. **Retina 是三输入模型。** `task_config.json` 的 `task_input_id` 为 `input_0`、`input_1`、`input_2`，输出是标量特征 `output`，含两个 logits。server 的目录模型注册会从 server 侧 `task_config.json` 读取这三个必需输入；不能按 CIFAR 的单 `input` 发送。
2. **`eval_context.bin` 大小不是根因。** Retina 和 CIFAR 都约为 `3,329,929,205` bytes。自动化测试使用目录选择器时，预填充大目录可能让 Chromium picker 停顿；先选择空目录、授权后再复制资源只是测试工具的 workaround，不是生产依赖。
3. **旧 WASM 的 scalar `skip` 解析不兼容 Retina。** native scalar wire 中在偏移 16 和 20 写入同一个 skip。旧 WASM 将两者作为二维数组校验：Retina 为 `16384 * 16384 > 1<<26`，被 `product(skip)` 拒绝；CIFAR 的 `1024 * 1024` 尚未超过上限，所以旧流程看起来正常。当前 `run_server_integration.mjs` 只把结果字节偏移 20 的冗余字段改为 1，密文 payload 未变。这是临时兼容补丁，不能作为正式协议。
4. **server 与浏览器文件契约不同。** WASM 生成 `<stem>.ct.bin`，现有 server 读取 `<case>.ct` 或 `<case>__<input>.ct` 的 `input.zip`，server 返回 `<case>.result.bin` 的 `output.zip`。当前 runner 用 Python 重命名、打 ZIP，并把结果名改回 `.ct.bin` 后才能调用 `decrypt_batch`。这解释了为什么当前需要 `run_server_integration.mjs`；它是验收编排器，不是运行时必需文件。

## 4. 推荐的最小改动

目标是让新版 latti-ai WASM 默认遵循现有 `latti-client` server 的生产契约，同时覆盖 Retina 和 ResNet20/CIFAR10。`latti-client` server 不需要增加目录模式、接口或兼容层；其现有 `input.zip`/`output.zip` 格式就是 WASM 的默认 server-compatible 模式。模型、任务算法和密文 payload 均不变。

### 4.1 latti-ai：修复 wire 解析并发布默认兼容模式

在 `inference/interface/wasm/core/wire.go` 的 `unmarshalFeature` 中，`dim == 0` 仍读取 native scalar wire 的两个字段，但只把第一个字段作为逻辑 scalar skip，内部表示为 `[first, 1]`；`feature.marshal` 继续按 native 格式写两个字段。这样 Retina 的校验变为 `product([16384,1])`，CIFAR 仍保持兼容，且不放宽通用长度上限。

在 WASM archive 的 JS/Worker 适配层（`inference/interface/wasm/src/index.js`、`src/worker.js`、`src/zip.js`）把 server-compatible 设为任务流程的默认模式：

- `encrypt_batch({mode: "server", inputDirectories, outputFile})` 直接生成 server 要求的 `input.zip`，单输入条目为 `<case_id>.ct`，多输入条目为 `<case_id>__<input_name>.ct`；由 WASM 适配层完成扩展名和 ZIP 封装，不由 Retina/CIFAR 联调 runner 重命名或重新打包。
- `decrypt_batch({mode: "server", inputFile, outputName, outputDirectory})` 直接接受 server 返回的 `output.zip`，读取 `<case_id>.result.bin`；适配层将结果交给同一个 wire 解码器并输出 logits CSV。现有按目录处理 `.ct.bin` 的 API 继续保留为兼容入口。
- 加密和解密都按任务配置的 `task_input_id`/`task_output_id` 校验 case 集合；多输入任务缺少或多出任一输入时在 WASM 适配层报错，避免把协议错误留给 server。

重新构建并发布新的 WASM archive（例如 `0.1.2`）。任务脚本和数据集资源包只需随包交付，不需要复制 WASM、修改 skip 字节或处理 server 文件名。新版发布后，`run_server_integration.mjs` 中的 skip 修改、`.ct`/`.ct.bin` 重命名和 ZIP reader/writer 都应删除。旧 `0.1.1` 只作为已验证基线，仍需要临时 runner 才能绕过 Retina 的 scalar 解析问题。

### 4.2 latti-client：保持现有实现

`latti-client` 不做源代码修改。WASM 默认模式直接使用已有契约：

```text
<tasks_dir>/<task_id>/input.zip
  <case_id>.ct                         # 单输入
  <case_id>__<input_name>.ct           # 多输入
<results_dir>/<task_id>/output.zip
  <case_id>.result.bin                 # server 原生结果
```

server 继续负责解压输入、按任务 registry 的 `required_inputs` 调用模型、生成 `output.zip` 并提供现有任务状态和结果下载接口。联调脚本只调用 WASM 的默认 server-compatible API、提交任务、读取结果和调用任务包的后处理，不复制 server 的 ZIP 规则。

## 5. 两个任务的统一联调流程

修改完成后，通用 runner 只需要使用任务包导出的脚本和 WASM API：

```text
选择 dataset/ 和 task/client/
  -> preprocess_batch()
  -> openBrowserClient().restore()
  -> encrypt_batch({mode: "server", inputDirectories, outputFile})
  -> WASM 默认模式生成 server 要求的 input.zip
  -> POST /api/v1/tasks
  -> 从 server 下载 output.zip
  -> decrypt_batch({mode: "server", outputName: "output", inputFile, ...})
  -> postprocess_batch() 输出 predictions.csv 与 accuracy
```

| 任务 | 输入 | 输出 logits | 后处理标签 | 预期检查 |
| --- | --- | --- | --- | --- |
| ResNet20/CIFAR10 | `input`，3x32x32 | 10 个 | 0–9 | `workflow.mjs` 的 native 与 browser logits、argmax、准确率一致 |
| Retina-DKD degree8_02 | `input_0`、`input_1`、`input_2` | 2 个 `[DN,NDRD]` | 0/1 | 10 case 完成，实际 FHE 结果 9/10；三路 case 集合完全相同 |

runner 只负责选择样本、调用 WASM 默认模式、调用 HTTP 状态接口和调用任务脚本；不再知道 scalar skip 的字节布局，也不再实现 `.ct`/`.ct.bin`/`.result.bin` 转换或 ZIP 结构。

## 6. 完成标准与边界

- 新 WASM 对 native 生成的 Retina scalar 结果直接解密；CIFAR 回归测试仍通过。
- WASM 生成的 `input.zip` 满足 server 的单输入/多输入 case、`required_inputs`、重复名、缺失名和额外名检查。
- Retina 三路密文与 CIFAR 单路密文均可由 WASM 原样封装、推理、解密；结果不依赖 runner 的二进制改写、重命名或重新打 ZIP。
- `preprocess.js`、`postprocess.js` 随任务包交付，数据包只提供原图、分割图、XLSX 和 `label.csv`；WASM archive 由 latti-ai 统一发布并可被两个任务复用。
- 端到端验收记录实际解密 logits 的 `total/correct/accuracy`，不能只复用历史参考 logits 的 90%。

