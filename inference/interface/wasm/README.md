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
| `prepareSplit({source, ratio})` | 从 ONNX 和具名 CSV 切分模型，验收使用 `ratio: '0.9'`。 |
| `runPrefix({split, inputsManifest})` | 为新样本生成边界输入和样本绑定信息。 |
| `compileHighLevel({split, deploymentMode, compileConfig?})` | 编译高层任务；也可用 `source` 替代 `split`。 |
| `generateLowLevel({task})` | 生成 MegaAG、签名及密钥需求。 |
| `openBrowserClient({clientDirectory, keyDirectory})` | 打开任务客户端；通过 `setup()` 生成密钥或 `restore()` 恢复。 |
| `encrypt()` / `encryptBatch()` | 加密输入文件或批次。 |
| `provisionRunner({task, parameterDirectory})` | 逐密文加密参数，生成公开 Runner 包。 |
| `decryptOutputs({task, outputFile, run?})` | 校验并解密原生返回的全部输出。 |
| `cancel(requestId?)` / `close()` | 取消当前操作或关闭会话。 |

完整参数与返回类型见 `dist/index.d.ts`。模型产物返回 `{directory, manifest}` 文件句柄；可选 `outputDirectory` 指定输出父目录，操作会创建独立子目录。

## ONNX 切分与编译示例

源目录包含模型、external data（如有）、CSV 和 `inputs.json`。路径使用安全相对路径，manifest 必须完整绑定全部输入的名称、shape 和语义轴，例如：

```json
{"inputs":[{"name":"input","csv":"sample.csv","shape":[1,3,32,32],"axes":["N","C","H","W"],"dtype":"float32"}]}
```

在按钮点击事件中选择目录后调用：

```js
import {openBrowserModelTools, openBrowserClient} from '/dist/index.js';

const sourceDirectory = await showDirectoryPicker();
const models = await openBrowserModelTools({onProgress: console.log});
try {
  const source = {
    directory: sourceDirectory,
    modelPath: 'model.onnx',
    inputsManifest: await sourceDirectory.getFileHandle('inputs.json'),
  };
  const split = await models.prepareSplit({source, ratio: '0.9'});
  let task = await models.compileHighLevel({
    split, deploymentMode: 'client_encrypted_input',
  });
  task = await models.generateLowLevel({task});
  // 保存 task，供后续按钮执行密钥与加解密流程。
} finally {
  await models.close();
}
```

后续从任务 manifest 的 `entry/client` 目录打开客户端，调用 `setup()` 或 `restore()`，再按 `input_contract.json` 中的 `feature_id` 和 CSV 加密**全部**边界输入。原生 Server 执行后，将返回文件交给 `decryptOutputs()`。

参数模式改用 `deploymentMode: 'server_provisioned_runner'`，传入适用的 `compileConfig`，使用 `entry/provisioner` 角色目录打开客户端。密钥准备后调用 `provisionRunner()`，将公开 Runner 包与具名明文边界输入交给原生 Runner。

**仅公开 eval context 和密文可上传；`fullContext` 含私钥，不得上传。** 密钥保存在 `keyDirectory` 下的 profile 子目录，`setup()` 拒绝覆盖已有密钥，恢复与任务绑定会校验 hash、参数和密钥能力。含 uint64 的 CKKS 参数 JSON 应保留原始文本，避免经 JavaScript 数值转换丢失精度。

新样本先调用 `runPrefix()`，CSV 放在 `split.directory` 下并由新 manifest 引用；加密和 `decryptOutputs()` 都传同一 `run`。直接覆盖已编译 CSV 会因内容绑定不匹配而失败。

## 支持范围与验收

已通过 cifar10、retina、BERT、YOLOv5、ViT、ImageNet 六任务在 `ratio='0.9'` 下的输入/参数两模式，共十二单元验收。全部输出最大绝对误差严格小于 `0.1`，连续客户端 RSS 峰值小于 4 GiB。未切分完整 ONNX 的参数加密内存问题单独记录，不属于本次强制验收。

CNN 使用 multiplexed 布局，支持已适配的 dense、非 big Conv2D（含 depthwise）与多项式参数；BERT/ViT 使用 FeatureMat、`par_block_col_major` PCM 和各自配置。当前不支持 big convolution 和 1D 参数布局，其他模型需符合现有编译器算子与结构限制。

资源预检上限：256 节点；initializer 合计 384 MiB、单个 32 MiB；源文件合计 512 MiB；单输入 `2^22` 元素；单 ORT tensor 32 MiB；产物单文件 512 MiB、合计 2 GiB；参数索引 16384 个密文。模型临时文件落 OPFS，操作结束清理临时池；已提交产物由调用方管理。默认模型操作超时 30 分钟。

## 开发与交付校验

以下命令在源码仓库中供开发者使用，终端用户无需执行：

```sh
npm ci
npm run build
npm test
CHROME_PATH=/path/to/chromium npm run test:browser
npm run package -- /tmp/latti-packages
```

六任务验收需先构建原生 `evaluate_browser_task`，另备 Python oracle 环境：

```sh
LATTI_MODEL_ACCEPTANCE_ROOT=/tmp/latti-model-acceptance \
LATTI_MODEL_NATIVE_RUNNER=/path/to/evaluate_browser_task \
LATTI_MODEL_NATIVE_PYTHON=/path/to/native-venv/bin/python \
CHROME_PATH=/path/to/chromium npm run test:model-workflow
```

默认串行运行六任务、两种模式；可用 `--models`、`--modes` 选择子集。失败、数值和临时工作区回归分别使用 `test:model-failures`、`test:model-numerics`、`test:python-workspace`。详细验收规则见源码中的 `docs/spec/wasm_onnx_split_compile_design.md`。

运行时固定使用 Pyodide 0.27.7、ORT Web 1.23.2、Lattigo v6。依赖版本/hash 见 `model-resources.lock.json`，源码和资源清单见 `dist/manifest.json`。构建时可设置 `LATTI_MODEL_RESOURCE_CACHE` 复用缓存；wheel 重建入口为 `build-model-wheels.py`（Python 3.12、pyodide-build 0.30.7、Emscripten 3.1.58、protoc 3.21.12）。

交付包为 `WASM-0.2.0.zip` 和同内容的 `latti-client-wasm-0.2.0.zip`。解压后执行 `sha256sum -c SHA256SUMS` 校验完整性。模型、样本与任务配置由业务项目另行提供。
