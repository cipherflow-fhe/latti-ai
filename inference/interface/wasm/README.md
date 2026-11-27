# Latti 浏览器 WASM 0.2.0

提供 ONNX 切分、明文前缀推理、两级 FHE 编译、密钥生成/恢复、输入或参数加密，以及结果解密接口。可集成到普通桌面 Chromium 网页，**用户无需安装插件或手动安装 ONNX、Python、Go、Node.js、CUDA**；所需浏览器运行时与依赖已随包提供。

## 网页集成

1. 将完整 `dist/` 部署为网站静态目录，保留内部相对路径。建议与页面同源，正确提供 JS/WASM MIME 类型（WASM 为 `application/wasm`）。
2. 页面导入 `dist/index.js`，为按钮接入接口、文件选择、进度和错误提示。
3. 用户访问页面后，浏览器按接口调用自动下载资源、启动 Worker 并加载 WASM，计算在用户设备上执行。首次加载需要等待；用户只需选择文件并授权目录读写。
4. 原生 FHE 后缀推理由独立部署的 Server/Runner 执行，网页需接入上传和结果回传。包内不包含业务推理服务器。

| 访问方式 | 条件 |
| --- | --- |
| `https://你的域名/…` | 可从其他设备访问，需有效安全连接。 |
| `http://localhost:端口/…` 或 `http://127.0.0.1:端口/…` | 当前用户设备上必须已有静态服务；`localhost` 指用户自己的设备。 |
| 远程普通 HTTP 或 `file://` | 不作为支持的运行入口。 |

浏览器需支持 WebAssembly、模块 Worker、OPFS（含 `SyncAccessHandle`）、Web Locks，以及目录选择所需的 File System Access API。较大任务需要足够内存和磁盘配额。

`openBrowserClient()` 启动加解密 Worker；模型工具首次执行操作时才加载 Pyodide/ONNX/ORT。Worker 由浏览器管理，无需用户启动系统进程，页面关闭后不会继续作为独立服务运行。用完调用 `close()`。

包内 `dist/example.html` 提供密钥、任务绑定和 CSV 加解密示例。ONNX 切分、编译等完整按钮流程需由项目接入。离线缓存需项目另行配置；密钥和任务不会自动跨设备或网站地址同步。

## 核心接口

| 接口 | 用途 |
| --- | --- |
| `openBrowserModelTools()` | 创建模型工具，支持进度回调与超时设置。 |
| `prepareSplit({source, ratio})` | 从 ONNX 和具名 CSV 切分模型；`ratio: '0'` 不生成 prefix，suffix 保留完整原始 ONNX。 |
| `exportPrefix({split, outputDirectory?})` | 从已验证的 split 流式导出独立 prefix 图、必要权重和公共接口。 |
| `runPrefix({prefix, inputDirectory, inputsManifest})` | 只读取独立 prefix 和数据方 CSV，按需执行并生成边界 CSV、样本绑定。 |
| `compileHighLevel({split, deploymentMode, compileConfig?})` | 编译高层任务；也可用 `source` 替代 `split`。 |
| `generateLowLevel({task})` | 生成 MegaAG、签名及密钥需求。 |
| `openBrowserClient({clientDirectory, keyDirectory})` | 打开任务客户端；通过 `setup()` 生成密钥或 `restore()` 恢复。 |
| `exportPublicContext()` / `importPublicContext({inputFile})` | 流式交接仅含参数和公钥的上下文；导入会清除私钥及解密能力。 |
| `encrypt()` / `encryptBatch()` | 用模型方公钥加密输入文件或批次，包括参数加密模式。 |
| `decryptBatch()` | 串行解密完整批次；校验输入批次和结果样本集合一致。 |
| `uploadProcessingScripts()` | 将可信操作方选择的预处理/后处理 JS 上传到所选静态资源目录，生成 SHA-256 manifest。 |
| `provisionRunner({task, parameterDirectory})` | 逐密文加密参数，生成公开 Runner 包。 |
| `decryptOutputs({task, outputFile, run?})` | 校验并解密原生返回的全部输出。 |
| `cancel(requestId?)` / `close()` | 取消当前操作或关闭会话。 |

`ratio: '0'` 时没有 prefix，原始具名 CSV 仍须经公钥加密后送入默认 Runner；后续编译、密钥生成/恢复、参数 provision 和输出解密接口不变。suffix 与原始 ONNX 及外部权重逐字节一致。完整大模型验收使用 `openBrowserModelTools({timeoutMs: 7200000})`；参数生成器的 30 分钟限制按单次请求刷新，持续进行的 provision 可以超过 30 分钟。

High-level 编译在同一个 Worker 内先将权重外置，再仅对图元数据做 shape inference；卷积/矩阵权重按最多 1 MiB 的输出分片转换到 H5，完成后删除编译临时权重。`compileHighLevel({...options, profileMemory: true})` 可启用各子步骤的 Python 分配峰值测量，结果写入 `browser_compile.json` 的 `stages`，并通过 `high-*` progress 事件返回。该选项有计时开销；Python 分配峰值、WASM 容量与浏览器 RSS 是不同口径。

Provision 按当前 slot 向量需要的索引读取 H5，每次最多读取 32768 个值；密文偏移索引和参数 manifest 随密文生成流式写入，Runner 文件 manifest 也逐项写入。参数签名、布局、密文容器和 native Runner 接口保持兼容。

完整参数与返回类型见 `dist/index.d.ts`。模型产物返回 `{directory, manifest}` 文件句柄；可选 `outputDirectory` 指定输出父目录，操作会创建独立子目录。

## 默认主流程：Retina 十样本原图批量验收

默认输入是 `retina_degree8_02/task/client/demo_10/dataset` 的十份眼底、disk、lesion 图像及 `risk_factor_5.xlsx` 原始指标。`label.csv` 仅用于后处理结果指标，不进入模型推理。完整模型为 `model/epoch041.onnx`，切分比例固定 `'0.9'`。

**原图/指标 → preprocess.js 批量 CSV → 独立 prefix 串行推理 → 公钥批量加密输入 → native GPU runtime_lazy 串行推理 → WASM 批量解密 → postprocess.js 指标。** 模型方独立完成 split、suffix high/low-level 编译、一次密钥生成和参数加密。除 native FHE 推理外，主链通过浏览器接口完成；模型工具运行于 Pyodide，ONNX 推理为 ORT WASM CPU，加解密为 Go WASM。隔离的 Python/ONNX oracle 仅用于开发验收数值比较。

| 角色/步骤 | API 或产物 |
| --- | --- |
| 数据方原始批次 | 可信 `preprocess_batch({inputDirectory, outputDirectory})` 将图像、临床指标转换为 `input_0` `[1,5,512,512]`、`input_1` `[1,2]`、`input_2` `[1,3]`；返回十样本 `inputs.json`。 |
| 模型方 split / 编译 | `prepareSplit({source, ratio:'0.9'})` 使用首样本确定接口并验证明文等价；`compileHighLevel()`、`generateLowLevel()` 只编译一次 suffix。 |
| 模型方密钥 | 首次 `setup()` 生成一对密钥，后续 `restore()` 复用。密钥生成与 `provisionRunner()` 参数加密独立计时。 |
| 模型方公开交接 | `exportPrefix({split})` 导出独立 prefix；`exportPublicContext()` 导出 `public_context.bin`，只含 CKKS 参数、公钥，没有评估密钥、私钥。 |
| 数据方 prefix | `runPrefix({prefix, inputDirectory, inputsManifest})` 只接受独立 prefix，逐样本串行、逐节点按需读取权重。 |
| 数据方加密 | 仅持有公开任务 metadata 的客户端调用 `importPublicContext({inputFile})`，再 `encryptBatch({mode:'server',inputDirectories,run})`；输入、参数使用同一个模型方公钥。 |
| native GPU | `evaluate_browser_task --mode server_provisioned_runner --gpu --task-dir ... --batch-manifest ... --output-dir ...`，复用 runner 串行处理批次，结果逐样本写目录。 |
| 模型方解密及后处理 | 返回 `<case_id>.result.bin` ZIP；`decryptBatch({mode:'server',outputName,inputFile,inputBatch})` 验证完整样本集合后串行解密，`postprocess_batch()` 计算 accuracy、specificity、sensitivity、AUC。 |

原始 CSV 已不再是默认流程起点。独立 CSV 入口保留供诊断及已有集成使用。新编译的参数模式默认 `input_mode='client_ciphertext'`；显式声明 `plaintext` 的历史任务仍可读取，但不能作为本次加密输入验收。

### 批次、上下文与角色边界

批量 `inputs.json` 使用：

```json
{"samples":[{"case_id":"case-a","inputs":[{"name":"input_0","csv":"input_0/case-a.csv","shape":[1,5,512,512],"axes":["N","C","H","W"],"dtype":"float32"}]}]}
```

`runPrefix()` 对每个样本校验名称、shape、axes、CSV 内容，依次写 `run/cases/<case_id>/sample_manifest.json` 和 crossing CSV，最后提交 `run_manifest.json`。新 schema `latti-ai.prefix-run.v2` 保存 split/prefix hash、全部 case ID 和逐样本 manifest hash；原单样本格式仍可读取。批量加密同时校验完整 case 集合和每份 crossing hash。批次是逐样本串行调度，ONNX batch 维仍为 `N=1`，不把十份图像合并成一个大张量。

`public_context.bin` 与 `eval_context.bin` 用途不同：前者只用于数据方加密；后者含评估密钥，由 native Runner 使用。公钥导入拒绝 full/eval 包、错误参数、截断和尾随数据，并清除当前客户端原有私钥/解密器。只导入公钥的客户端不能解密、不能执行需要私钥自校验的参数 provision。生产私钥只保留在模型方。

独立 prefix 仅含 `model.onnx`、必要外部权重、`prefix.json` 和产物 manifest，不含 suffix、完整原模型、样本或私钥。导出以 1 MiB 分块落盘；执行按 offset/length 读取当前节点所需权重，复用原有执行器。`ratio='0'` 没有 prefix，`exportPrefix()` 会明确报错。所有 WASM API 和交接 schema 通用，不含 Retina 专用接口名称。

### 处理脚本上传及网页集成

`delivery/batch-workflow.mjs` 的通用 `runParameterBatchWorkflow()` 是可复用接线示例，参数包含模型/原始数据目录、预处理/后处理 URL、密钥目录和 `runNative` 回调。生产项目用回调上传输入密文、调用 native GPU 服务并取回结果 ZIP；交付校验回调读取已绑定的固定 ZIP。

```js
import {uploadProcessingScripts} from '/runtime/index.js';
await uploadProcessingScripts({directory: staticProcessingDirectory,
  preprocessFile, postprocessFile});
// staticProcessingDirectory 是操作方授权的静态资源目录。
// 站点提供该目录后，将对应 URL 传给 runParameterBatchWorkflow。
```

该上传接口保存 `preprocess.js`、`postprocess.js` 和 `processing-manifest.json`，不自动执行脚本。它们是可信操作方提供的可执行代码，应经项目审核后同源部署；不把任意访客上传内容自动作为站点代码执行。远程部署的上传鉴权和发布由项目现有后端负责。Retina 的独立示例脚本位于 `dist/tasks/retina/raw-preprocess.js`、`raw-postprocess.js`，上传后文件名为 `preprocess.js`、`postprocess.js`，没有外部 CDN 依赖。

## 支持范围与验收

已通过 cifar10、retina、BERT、YOLOv5、ViT、ImageNet 六任务在 `ratio='0.9'` 下的输入/参数两模式，共十二单元验收。全部输出最大绝对误差严格小于 `0.1`，连续客户端 RSS 峰值小于 4 GiB。本轮整流程峰值约 2.17 GiB；BERT split 约 1.24 GiB，较上一版下降约 66%。阶段重建会增加初始化耗时，小模型的处理时间可能变长。`ratio=0` 的完整原图参数加密使用磁盘分块低层编译、共享评估密钥和按需公开 mask；各阶段 WASM 内存需单独测量，ratio=0.9 的结果不能替代完整原图验收。

六任务 `ratio='0'` 的完整原图参数加密链路也已通过单样本全部输出验收，max_abs 均小于 `0.1`。high/low 编译与 provision WASM 峰值均不超过 `2.5 GiB`；最大 BERT high-level 线性内存约 `1.77 GiB`、renderer RSS 约 `2.434 GiB`。详细本机证据位于 `/tmp/latti-ratio0/acceptance-report.md` 与 `completion-audit.json`。

CNN 使用 multiplexed 布局，支持已适配的 dense、Conv2D（含 depthwise 和验收模型的大尺寸首层）与多项式参数；BERT/ViT 使用 FeatureMat、`par_block_col_major` PCM 和各自配置。当前不支持 1D 参数布局，其他模型需符合现有编译器算子与结构限制。

资源预检上限：256 节点；initializer 合计 384 MiB、单个 32 MiB；源文件合计 512 MiB；单输入 `2^22` 元素；单 ORT tensor 32 MiB；普通产物单文件 512 MiB、合计 2 GiB（磁盘流式处理的图 JSON 与参数 H5 不计入此限额）；参数索引 16384 个密文。模型权重流式外置到 OPFS，执行节点时只读所需权重；子图提取只处理结构与权重引用。原图、提取及 prefix/suffix 验证之间持久化结果并重建 Worker，阶段指标写入产物 manifest。模型临时文件落 OPFS，操作结束清理临时池；已提交产物由调用方管理。默认模型操作超时 30 分钟。

## 开发与交付校验

默认仅验收 **Retina 十样本 / split=0.9 / 输入与参数均公钥加密**。交付 ZIP 包含：

| 路径 | 内容 |
| --- | --- |
| `WASM-0.2.0.zip` | 独立完整 WASM/JS 运行时包。 |
| `retina/model/epoch041.onnx` | 原始完整 ONNX。 |
| `retina/dataset/` | 十份原图、disk/lesion、原始 XLSX 和 labels。 |
| `retina/prepared/` | 已验收的十样本 CSV 与输入 manifest，供诊断；默认仍重新预处理原图。 |
| `processing/preprocess.js`、`postprocess.js` | 独立预处理和后处理脚本。 |
| `fixture/output.zip` | 真实 native GPU runtime_lazy 批量结果密文。 |
| `fixture/public_context.bin` | 数据方使用的仅公钥上下文。 |
| `fixture/test-keys/` | 固定密文配套的演示密钥，含测试私钥及评估上下文。 |
| `workflow-check.mjs`、`batch-workflow.mjs`、`index.html` | 通用批量接线及默认交付校验页面。 |
| `delivery-manifest.json`、`SHA256SUMS` | 模型、原始数据、配置、逐样本 crossing、密钥身份与结果来源绑定。 |

固定密文回放只替代 native GPU FHE 执行。页面会实际重跑原图预处理、split、独立 prefix、两级编译、参数加密、公钥导入/批量输入加密、批量解密和后处理。每个 crossing 的 case ID/hash 必须与密文来源一致，再检查全部输出误差 `<0.1`。**回放不能证明本次新生成的参数包已经被 native 执行。** 更换模型、样本或密钥后须重新执行 native Runner。

外层 ZIP 汇集双方测试材料，内含公开演示私钥，禁止用于业务数据。生产发给数据方的只有独立 prefix、公开任务 metadata、仅公钥 context；Runner 获得评估上下文和加密参数。整个交付 ZIP 不应作为生产数据方交接包。

解压后运行 `python3 serve.py`，或用现有 Chromium 网页项目同源服务页面、`processing/`，并将嵌套 WASM ZIP 的 `dist/` 部署到 `/runtime/`。Python 只提供本地示例静态服务，不是浏览器运行依赖。打开 `http://localhost:8000/`，选择交付根目录并开始校验。较大的演示评估上下文需要预留磁盘空间。

```sh
npm ci
npm run build
npm test
LATTI_RETINA_DELIVERY_DIRECTORY=/交付根目录 \
CHROME_PATH=/path/to/chromium npm run test:delivery
```

真实 native GPU 完整验收默认同样是上述十样本批次，临时产物放 `/tmp`。开发 oracle 需 NumPy、ONNX/ORT、Pillow、scikit-learn 及 `xlrd==1.2.0`（资源参考脚本读取 XLSX），这些只供独立数值校验，不属于网页运行依赖：

```sh
LATTI_BATCH_ROOT=/tmp/retina-batch-acceptance \
CHROME_PATH=/path/to/chromium npm run test:model-workflow
```

`test:batch` 为同义入口，`test:retina-delivery` 为 `test:delivery` 同义入口。历史六模型单样本诊断保留在 `test:legacy-model-workflow -- --models bert,vit,retina,imagenet,yolov5,cifar10 --modes input,parameter`；它不替代原图批量验收。

打包使用 `npm run package -- /tmp/wasm-package` 生成内层包，然后：

```sh
python3 package_retina_split_delivery.py --wasm /tmp/wasm-package/WASM-0.2.0.zip \
  --acceptance /tmp/retina-batch-acceptance \
  --output /home/linghm/latti-ai-fix/taa-delivery \
  --name retina-batch10-split09-encrypted-input-parameter-wasm-delivery-20261127
```

运行时使用 Pyodide 0.27.7、ORT Web 1.23.2、Lattigo v6，资源/hash 见 `model-resources.lock.json` 和 `dist/manifest.json`。验收报告另存 `latti-ai-reports/`，工作区仅保留实现、规范及测试代码。详细规则见 `docs/spec/wasm_onnx_split_compile_design.md`。
