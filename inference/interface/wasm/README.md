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
| `runPrefix({split, inputsManifest})` | 为新样本生成边界输入和样本绑定信息。 |
| `compileHighLevel({split, deploymentMode, compileConfig?})` | 编译高层任务；也可用 `source` 替代 `split`。 |
| `generateLowLevel({task})` | 生成 MegaAG、签名及密钥需求。 |
| `openBrowserClient({clientDirectory, keyDirectory})` | 打开任务客户端；通过 `setup()` 生成密钥或 `restore()` 恢复。 |
| `encrypt()` / `encryptBatch()` | 加密输入文件或批次。 |
| `provisionRunner({task, parameterDirectory})` | 逐密文加密参数，生成公开 Runner 包。 |
| `decryptOutputs({task, outputFile, run?})` | 校验并解密原生返回的全部输出。 |
| `cancel(requestId?)` / `close()` | 取消当前操作或关闭会话。 |

`ratio: '0'` 时，Runner 直接接收原始具名 CSV；后续编译、密钥生成/恢复、参数 provision 和输出解密接口不变。suffix 与原始 ONNX 及外部权重逐字节一致。完整大模型验收使用 `openBrowserModelTools({timeoutMs: 7200000})`；参数生成器的 30 分钟限制按单次请求刷新，持续进行的 provision 可以超过 30 分钟。

High-level 编译在同一个 Worker 内先将权重外置，再仅对图元数据做 shape inference；卷积/矩阵权重按最多 1 MiB 的输出分片转换到 H5，完成后删除编译临时权重。`compileHighLevel({...options, profileMemory: true})` 可启用各子步骤的 Python 分配峰值测量，结果写入 `browser_compile.json` 的 `stages`，并通过 `high-*` progress 事件返回。该选项有计时开销；Python 分配峰值、WASM 容量与浏览器 RSS 是不同口径。

Provision 按当前 slot 向量需要的索引读取 H5，每次最多读取 32768 个值；密文偏移索引和参数 manifest 随密文生成流式写入，Runner 文件 manifest 也逐项写入。参数签名、布局、密文容器和 native Runner 接口保持兼容。

完整参数与返回类型见 `dist/index.d.ts`。模型产物返回 `{directory, manifest}` 文件句柄；可选 `outputDirectory` 指定输出父目录，操作会创建独立子目录。

## retina ONNX split 参数加密主流程

默认交付校验使用 retina 原始完整 `model/epoch041.onnx`，三个输入为 `input_0=img.csv`、`input_1=non_inv.csv`、`input_2=inv.csv`，shape 分别为 `[1,5,512,512]`、`[1,2]`、`[1,3]`。`inputs.json` 已绑定名称、shape 和语义轴，`compile-config.json` 使用锁定的 storage-aware 配置。

### 模型提供方：切分、编译和参数加密

1. `prepareSplit({source, ratio: '0.9'})` 切分完整 ONNX，使用适配样本验证原图及 prefix/suffix 的明文等价性。
2. `compileHighLevel({split, deploymentMode: 'server_provisioned_runner', compileConfig})`、`generateLowLevel({task})` 编译 suffix，生成任务图、签名及密钥需求。
3. 从任务的 `entry/provisioner` 打开客户端，首次 `setup()` 生成并保存模型方密钥；以后使用 `restore()`。
4. `provisionRunner({task, parameterDirectory: provisionerDirectory})` 加密 suffix 参数，生成公开 Runner 包。将 **prefix ONNX 及其所需权重、输入输出约定、公开 Runner 包**交给数据方。独立 prefix 需转成标准 ONNX，并仅重新打包它引用的权重；不能直接传整个 split 工作目录（含 suffix 明文权重）。

### 数据提供方：本地 prefix 推理和 Runner 推理

1. 用自己的 CSV 和收到的 prefix 做明文推理，无需完整原模型或模型方私钥。交付包提供 `runRetinaPrefix({prefixDirectory, csvDirectory, runtimeURL})`，返回边界 CSV 所在的 `directory` 和输出信息。
2. 将边界 CSV 按 `input_contract.json` 的 `feature_id` 送入收到的 Runner：retina 两项分别为 `_flatten_Flatten_output_0`（512 个值）和 `_mlp_factor_mix_fc2_Gemm_output_0`（32 个值）。这是**参数加密模式，Runner 输入为明文**，不调用输入加密接口。
3. 由数据方设备上的 native Runner 或其接入的原生服务执行 FHE suffix，得到 `output.ct`，回传模型提供方。WASM 网页自身不执行 native Runner；原始 CSV 留在数据方。

已有 native 环境可按下式接线（本交付演示会跳过此步）：

```sh
evaluate_browser_task --mode server_provisioned_runner --task-dir /path/to/runner \
  --input _flatten_Flatten_output_0=/path/to/prefix-output/_flatten_Flatten_output_0.csv \
  --input _mlp_factor_mix_fc2_Gemm_output_0=/path/to/prefix-output/_mlp_factor_mix_fc2_Gemm_output_0.csv \
  --output output.ct
```

### 模型提供方：结果解密

用自己保管的密钥 `restore()`，调用 `decryptOutputs({task, outputFile})` 解密收到的结果密文。开发验收比较原始完整 ONNX 的全部输出，最大绝对误差须严格 `<0.1`。生产私钥/fullContext 始终留在模型方，不随 Runner 发给数据方。

以下为已选择好源目录、密钥目录的生产接口接线；完整可运行的固定样本校验见下节：

```js
import {openBrowserModelTools, openBrowserClient} from '/runtime/index.js';

const models = await openBrowserModelTools({onProgress: console.log});
let client;
try {
  const split = await models.prepareSplit({source: {
    directory: sourceDirectory, modelPath: 'model/epoch041.onnx',
    inputsManifest: await sourceDirectory.getFileHandle('inputs.json'),
  }, ratio: '0.9'});
  let task = await models.compileHighLevel({split,
    deploymentMode: 'server_provisioned_runner',
    compileConfig: await sourceDirectory.getFileHandle('compile-config.json'),
  });
  task = await models.generateLowLevel({task});
  const manifest = JSON.parse(await (await task.manifest.getFile()).text());
  let role = task.directory;
  for (const part of (manifest.entry + '/provisioner').split('/'))
    role = await role.getDirectoryHandle(part);
  client = await openBrowserClient({clientDirectory: role, keyDirectory});
  await client.setup(); // 首次生成；已有密钥时直接 restore()
  await client.close();
  client = await openBrowserClient({clientDirectory: role, keyDirectory});
  await client.restore();
  const bundle = await client.provisionRunner({task, parameterDirectory: role});
  // 模型方发送独立 prefix 和 bundle；数据方用自己的 CSV 跑 prefix，调用 native Runner。
  // 模型方收到数据方回传的 resultFile 后解密：
  // const results = await client.decryptOutputs({task, outputFile: resultFile});
} finally {
  await client?.close();
  await models.close();
}
```

**生产私钥/fullContext 不得上传。** profile-ID 子目录保存密钥，`setup()` 不覆盖已有密钥，`restore()` 校验 hash 与身份。含 uint64 的 CKKS 参数 JSON 保留原始文本，避免 JavaScript 数值转换丢失精度。`runPrefix({split})` 适用于持有完整 split 的本地工作流；双方分离时，数据方使用独立 prefix，项目负责关联样本、任务与回传结果。

## 支持范围与验收

已通过 cifar10、retina、BERT、YOLOv5、ViT、ImageNet 六任务在 `ratio='0.9'` 下的输入/参数两模式，共十二单元验收。全部输出最大绝对误差严格小于 `0.1`，连续客户端 RSS 峰值小于 4 GiB。本轮整流程峰值约 2.17 GiB；BERT split 约 1.24 GiB，较上一版下降约 66%。阶段重建会增加初始化耗时，小模型的处理时间可能变长。`ratio=0` 的完整原图参数加密使用磁盘分块低层编译、共享评估密钥和按需公开 mask；各阶段 WASM 内存需单独测量，ratio=0.9 的结果不能替代完整原图验收。

六任务 `ratio='0'` 的完整原图参数加密链路也已通过单样本全部输出验收，max_abs 均小于 `0.1`。high/low 编译与 provision WASM 峰值均不超过 `2.5 GiB`；最大 BERT high-level 线性内存约 `1.77 GiB`、renderer RSS 约 `2.434 GiB`。详细本机证据位于 `/tmp/latti-ratio0/acceptance-report.md` 与 `completion-audit.json`。

CNN 使用 multiplexed 布局，支持已适配的 dense、Conv2D（含 depthwise 和验收模型的大尺寸首层）与多项式参数；BERT/ViT 使用 FeatureMat、`par_block_col_major` PCM 和各自配置。当前不支持 1D 参数布局，其他模型需符合现有编译器算子与结构限制。

资源预检上限：256 节点；initializer 合计 384 MiB、单个 32 MiB；源文件合计 512 MiB；单输入 `2^22` 元素；单 ORT tensor 32 MiB；普通产物单文件 512 MiB、合计 2 GiB（磁盘流式处理的图 JSON 与参数 H5 不计入此限额）；参数索引 16384 个密文。模型权重流式外置到 OPFS，执行节点时只读所需权重；子图提取只处理结构与权重引用。原图、提取及 prefix/suffix 验证之间持久化结果并重建 Worker，阶段指标写入产物 manifest。模型临时文件落 OPFS，操作结束清理临时池；已提交产物由调用方管理。默认模型操作超时 30 分钟。

## 开发与交付校验

默认交付文件为 **`retina-split-delivery.zip`**，解压后的内容：

| 路径 | 内容 |
| --- | --- |
| `WASM-0.2.0.zip` | 可嵌入网页的完整 WASM/JS/ONNX 依赖包，保持独立 ZIP。 |
| `retina/` | 原始完整 ONNX、三个原始 CSV、输入 manifest 和编译配置。 |
| `handoff/prefix/` | 可单独交给数据方的标准 prefix ONNX、必要权重和输入输出约定。 |
| `fixture/output.ct` | 已通过验收的 native Runner 真实结果密文。 |
| `fixture/test-keys/` | 仅用于固定测试样本的匹配密钥（含测试私钥）。 |
| `index.html`、`retina-check.mjs` | 默认 retina 参数模式的按钮界面与可复用 JS 主流程。 |
| `serve.py`、`delivery-manifest.json`、`SHA256SUMS` | 本地静态服务、结果来源/身份绑定与完整性校验。 |

**固定结果密文必须使用配套测试密钥。** 演示在同一网页模拟双方：模型方重新执行 split、两级编译和参数加密，数据方仅用独立 prefix 与固定 CSV 实际推理，再由模型方解密。仅 Server/Runner 的 FHE 执行使用预生成密文回放。它验证新参数包生成与解密接口，不声称本次新参数包已经被原生执行；新生成的生产密钥不能解密旧结果。更换 CSV 后必须重新执行 native Runner，旧结果不再适用。外层 ZIP 汇集双方测试材料，供集成开发校验；生产向数据方发送的只是 prefix 与公开 Runner，不能发送整个交付 ZIP。测试密钥已随包分享，绝不能用于真实业务数据。约 4.7 GB 的公开 eval context 是 `restore()` 完整性检查所需文件，因此交付 ZIP 使用 ZIP64，解压和运行需预留二十余 GB 磁盘空间。

同事无需源码、Node.js、ONNX 或 native Runner。解压交付 ZIP 后：

```sh
cd retina-split-delivery
sha256sum -c SHA256SUMS  # 可选，Windows 可用对应 SHA-256 工具
python3 serve.py        # 仅本地演示静态服务；已有网页项目可用自己的服务
```

在桌面 Chromium 打开 `http://localhost:8000/`，点击“选择交付根目录”选择该解压目录，再点击“开始校验”。完成后下载报告，结果应为 `passed`、全部输出误差 `<0.1`。`serve.py` 自动解压并校验 WASM 资源，只服务页面与运行时，测试密钥/模型通过本地目录授权读取。若环境没有 Python，可手动将 WASM ZIP 中的 `dist/` 放到网站 `/runtime/`，同源部署页面和 JS；Python 不是浏览器运行依赖。

接入已有网页可直接调用：

```js
import {verifyRetinaDelivery} from './retina-check.mjs';
// 在按钮事件中选择解压交付根目录。
const directory = await showDirectoryPicker();
const {report, bundle, results} = await verifyRetinaDelivery(directory, {
  runtimeURL: new URL('/runtime/index.js', location.href).href,
  onProgress: console.log,
});
```

源码开发的默认验收也仅运行 **retina / ONNX split / 参数加密**：

```sh
npm ci
npm run build
npm test
LATTI_MODEL_ACCEPTANCE_ROOT=/tmp/retina-acceptance \
LATTI_MODEL_NATIVE_RUNNER=/path/to/evaluate_browser_task \
LATTI_MODEL_NATIVE_PYTHON=/path/to/native-venv/bin/python \
CHROME_PATH=/path/to/chromium npm run test:model-workflow
```

该源码验收执行真实 native Runner；交付回放校验则使用 `LATTI_RETINA_DELIVERY_DIRECTORY=/解压目录 CHROME_PATH=/path/to/chromium npm run test:retina-delivery`，不调用 native。需要完整回归时显式指定 `test:model-workflow -- --models bert,vit,retina,imagenet,yolov5,cifar10 --modes input,parameter`。其他回归入口为 `test:browser`、`test:model-failures`、`test:model-numerics`、`test:python-workspace`。

运行时使用 Pyodide 0.27.7、ORT Web 1.23.2、Lattigo v6；资源/hash 见 `model-resources.lock.json`、`dist/manifest.json`。开发者用 `npm run package -- /tmp/wasm-package` 生成独立 WASM 包，再用 `package_retina_split_delivery.py --wasm ... --acceptance ... --test-keys ... --output /tmp/retina-delivery` 生成外层交付 ZIP；测试密钥必须来自同次验收。详细全量规则见源码 `docs/spec/wasm_onnx_split_compile_design.md`。
