# Chromium WASM ONNX 切分、编译与两种加密模式设计规范

日期为 2026-11-19，状态为四项降峰已实施，并通过新一轮六任务、两模式十二单元验收。新增接口、文件和测试入口均为实施要求。依据见 [浏览器客户端](taa_v6_browser_client_design.md)、[ONNX split](onnx_split_plaintext_prefix_fhe_suffix_design.md)、[流式密钥](taa_v6_streaming_context_design.md) 和 [参数 Runner 流程](../zh/server_provisioned_runner.md)。

## 1. 目标与边界

在 Chromium 网页客户端提供通用 JS API，完成原始 ONNX 切分、prefix 推理、两级 FHE 编译、密钥生成/恢复、输入加密、参数 Provisioner 和输出解密。输入为用户的 FHE 适配 ONNX、external data、具名输入 manifest 与编译配置，算法按原图和 contract 工作，禁止依赖模型名称、固定节点数或预存 crossing 值。

| 执行位置 | 职责与产物 |
| --- | --- |
| Chromium Pyodide / Python ONNX / NetworkX | 加载/checker/shape inference/extract、稳定切分、ONNX→`pt.json`、High-Level、H5、模式目录、contract、Low-Level MegaAG/signature/key requirements |
| Chromium ONNX Runtime Web | 原图观测、prefix、suffix 明文推理与等价验证，使用 `wasm` provider，初始 `numThreads=1` |
| Chromium Python 参数物化 + Go/WASM | setup/restore、Feature 输入加密、逐密文参数 bundle、输出解密与 `output_routes` 汇合 |
| native `InferenceServer` / `InferenceRunner` | 执行浏览器生成的 suffix 任务，返回具名输出密文 |

复用现有 split/compiler 的算子、结构安全、DP/BTP、level/scale、packing、NetworkX、目录与文件语义。发布包列明算子、layout、dtype、profile 和资源支持范围，范围外任务明确失败。保留 `client_encrypted_input`、`server_provisioned_runner` 两个 mode 值及角色目录。

native 编译器、ONNX Runtime 和 Provisioner 可作为隔离 oracle。主链的 split、任务图、H5、参数 bundle、crossing 密文和解密值来自本次浏览器公开 API。验收 harness 负责静态文件/产物传输、native Runner 调用及证据校验，生产客户端通过文件接口集成。

## 2. 代码修改与依赖交付

以下为修改范围与建议新增位置，可合并同职责文件，CLI 与浏览器共享算法实现。

| 位置 | 修改要求 |
| --- | --- |
| `tools/prepare_split_onnx.py`、`evaluate_onnx_csv.py` | 保留 `split_graph`、`tensor_semantics`、extraction、安全规则和 `expand_polynomials`，抽出可等待 ORT Web 结果的准备阶段，native ORT import 留在 native 入口 |
| `tools/onnx_split_contract.py` | 保留 schema、packing/signature 检查，适配路径映射、流式 CSV 校验和模型/样本内容绑定 |
| `training/run_compile.py`、`training/model_compiler/` | 抽出可调用编译/布局入口，保留 ONNX 转换、NetworkX pipeline、DP/BTP、storage-aware 评分、H5、contract 与错误传播，处理资源路径和全局状态 |
| `training/nn_tools/{__init__,export,feature_mat_h5,eval_fn_hat_for_aespa}.py` | 按第 3 节解耦训练与 ONNX 参数导出的 torch 依赖 |
| `inference/interface/gen_mega_ag.py`、`inference/model_generator/`、`inference/lattisense/frontend/` | 复用构图/linker，抽出函数入口和状态重置，共享两模式 key requirements 导出，保留 use-site/seed 元数据 |
| 新增 `inference/interface/wasm/python/browser_pipeline.py` | Python 薄入口，调度共享 split/compile/export、异步结果交接和工作区映射 |
| 新增 `inference/interface/wasm/src/{model,model-worker}.js`，修改 `src/index.js`、`index.d.ts` | 模型 API、Pyodide/ORT、FFI、串行调度、取消、OPFS 提交和类型声明 |
| `inference/interface/wasm/src/worker.js`、`cmd/latti-client-wasm/main.go`、`core/` | 保留 v2 协议、CKKS、packing、key context、`OutputStream`，增加参数编码、逐密文加密、output bundle 和模式校验 |
| `inference/interface/wasm/{build.sh,build-info.mjs,build-manifest.mjs,package.json}` 及 lock | 构建/复制 Worker、Python、wheel、ORT/Pyodide 与资源锁，全部纳入 build ID、清单与 hash |
| `inference/interface/wasm/tests/browser-harness.mjs`，新增 `tests/model-workflow.{mjs,sh}`、`model-workflow-native.py` | 真实 Chromium 阶段测量、端到端编排、native adapter、隔离 oracle 和 npm `test:model-workflow` |
| `inference/interface/wasm/README.md` | 依赖构建、API、支持范围、模式目录、运行及验收命令 |

锁定 Pyodide、Python ABI、Emscripten、ORT Web、ONNX wheel、NetworkX、NumPy、SciPy、h5py、protobuf、源码及 JSON cost/profile 表的版本和 SHA-256，随包发布可重现静态资源。只加载实际所需依赖，解耦 torch、CUDA、绘图和数据下载。保留 `pt.json`、H5、MegaAG 格式。

ONNX C++ 扩展可在构建机交叉编译为匹配 Pyodide ABI 的 wheel。浏览器先验证 import、load/save、checker、shape inference、extract 与 external data，失败即前置项未完成。ORT 按锁定版本检测 SIMD，保持输入 dtype、opset、多项式展开和 requested output names 语义，WebGPU 另作扩展测试。

## 3. Python 编译与 torch 解耦

High-Level 在 Pyodide 调用既有 `onnx_to_json`、`run_pipeline`，接受原始或 suffix ONNX。`pt.json` 保留作中间产物/诊断。当前 BTP 直接调用 `run_single_compile`，沿用串行路径并记录 temperature、random seed 和实际配置，仅对实际不可用的多进程 import/调用作必要适配。GPU cost table 是目标成本模型，构图可在浏览器 CPU 执行。

| torch 依赖点 | 解耦要求 |
| --- | --- |
| `run_compile.py`→`nn_tools.__init__` | ONNX 入口避免立即加载 activations、modules、replace 等训练模块 |
| `export.py` 顶层 import、`nn.Module` 注解、训练模型导出 | 处理间接导入和注解求值，拆分或延迟加载训练功能，ONNX 导入/执行路径无需 torch |
| 普通 H5、`feature_mat_h5.py` 的 `_resolve_activation_cls` 和系数调用 | 按激活名称选共享数值函数，覆盖已支持的 ReLU、ReLU6、SiLU、GELU 等语义 |
| `eval_fn_hat_for_aespa.py` 的 torch forward、绘图 | 提取接收数值函数的 NumPy/SciPy 积分/Hermite 核心，训练与绘图适配按功能调用加载 |

保留训练侧公开导入和 `get_hermite_coeffs_for_module` 等行为，由薄适配调用共享核心。ONNX 路径实际使用无 torch 数值实现，核对激活参数、GELU 近似方式、float64、积分区间/容差、Hermite 归一化、degree、RangeNorm scale、eps 和名称映射，禁止预存样例系数替代计算。

H5 按 layer/dataset 导出，控制全量 initializer float64 字典、output 字典和融合副本。导出失败使整个编译失败。Low-Level 复用 frontend/linker 生成 MegaAG/signature，保留 `runtime_lazy` 参数列表、generator、seed layout、flat index 和 use-site level。两模式角色配置均从最终 MegaAG key nodes 导出 `evaluation_keys`，包含角色、levelQ/P、GaloisElement、压缩和分解参数。任务之间重置 compiler config/frontend DAG 或重建 Worker，保护已提交任务。

## 4. API、工作区和模型流程

模型入口 `openBrowserModelTools` 独立于 `openBrowserClient`，按需启动一个模型 Worker，页面协调它与既有 FHE Worker。每 handle 串行、最多一个活跃操作，排队/拒绝行为明确，沿用 requestId/jobId、progress、result/error/cancelled、cancel/close 和唯一终态。

公共类型 `ArtifactDirectory` 包含 `directory: FileSystemDirectoryHandle`、`manifest: FileSystemFileHandle`，`ModelSources` 包含 `directory: FileSystemDirectoryHandle`、相对 `modelPath`、`inputsManifest: FileSource`。下表为接口契约，名称确定后同步声明、示例与测试。

| 接口 | 参数与结果 |
| --- | --- |
| `prepareSplit` | `ModelSources`、十进制字符串 `ratio`，返回 split `ArtifactDirectory` |
| `exportPrefix` | 已验证的 `split`、可选输出父目录；流式导出独立 prefix 图、必要权重和公共接口，不包含 suffix、原模型或样本 |
| `runPrefix` | 独立 `prefix`、数据方 `inputDirectory` 和 `inputsManifest`；复用逐节点按需权重读取，返回 crossing/公开输出和样本绑定目录，不接受完整 split |
| `compileHighLevel` | 二选一 `source: ModelSources` 或 `split: ArtifactDirectory`，加 `deploymentMode`、可选 `compileConfig: FileSource`，返回 task 目录 |
| `generateLowLevel` | `task: ArtifactDirectory`，返回具备 MegaAG/signature/requirements 的任务目录 |
| `cancel` / `close` | 可选 requestId 的 cancel 返回 `{accepted: boolean}`，close 结束会话，均为 Promise |
| FHE `provisionRunner` | `task`、`parameterDirectory`，使用已 setup/restore 的密钥，返回公开 Runner 目录及 manifest |

以上产物接口均为 Promise，可指定 `outputDirectory`。既有 setup/restore/bindTask/encrypt/encryptBatch/decrypt/decryptBatch 继续按真实 feature ID 使用。参数物化与加密逐元素交接。

1. 浏览器 Python ONNX 绑定全部具名输入、dtype、完整 shape/axes，执行 checker 与现有稳定 split、安全检查。
2. 复用多项式展开和具名观测 outputs，通过异步 JS `session.run(feeds)` 获得原图全部输出与 crossing，回填 ValueInfo，再 checker/extract prefix/suffix 和 external data。源模型 hash 保持可追溯。
3. ORT Web 独立运行 prefix，crossing 按原名称、shape/dtype、C-order 写 CSV 和二进制工作文件，随后运行 suffix 明文图并按 `output_routes` 汇合。名称、元素数、finite 和明文等价全部通过后提交 split manifest。
4. 编译生成模式独立的 task/contract，完成 Low-Level、密钥绑定和加密。native 返回输出密文后在浏览器解密、恢复原始顺序/shape/语义 dtype 并汇合全部输出。

Python 同步 executor 由浏览器入口桥接异步 ORT，CLI 共享算法并保留 native executor。模型 Worker 可串行使用 Pyodide/ORT，重叠峰值过大时在持久化边界重建。释放 PyProxy、FFI view 和 session，跨运行时副本计入预算。

原始输入只读，以安全相对路径定位 ONNX/external data/CSV。OPFS 分别保存 source、split、两模式 task、profile-ID keys 和 results。新样本单独写 run manifest，绑定已编译接口与内容 hash，不能覆盖 CSV 绕过 contract。文件写完并通过 hash/引用校验后最后提交 manifest，异常只清理当前临时目录，已提交 split/task 可用于重试，中断参数 bundle 可清理后重做。

既有绝对路径用稳定 Python 虚拟根及可移植定位信息映射到 native 目录，已 hash manifest 不得静默改写。如扩展相对路径，须兼容 native absolute-path 读取并共享引用/hash 更新逻辑与回归。CKKS uint64、GaloisElement、byte offsets 使用原 JSON 文本交 Go/Python 或十进制字符串，避免普通 JS JSON 往返丢精度。ORT externalData 显式映射 location 与文件。

## 5. 加密、解密与 native 接线

Provisioner 按实际 signature、layer、kind、layout/generator 编码 conv/dense weight、bias、多项式系数、mult_scalar、structural mask 和 seed/tile，记录 arg ID、count、level、scale 等任务清单。未支持类别在加密前失败，禁止按模型名或硬编码 arg ID 分派。

优先复用 Python/H5 按元素物化与 Go packing，以 C++ `InferenceProvisioner`、`parameter_layout_metadata` 为互操作参考。若抽取纯 C++ 布局到 Emscripten 更省修改，可局部复用并说明依据，密码学仍由既有 Go/WASM 提供。遵循原 ringt→use-site plaintext materialization，保留 bias 在 rescale 后的 level、scale 和同参数多 use-site 实例，普通输入 encrypt 不能直接替代参数编码。

`OutputStream` 按 argument/flat index 获取有界参数值，一次编码、加密、序列化并写一个 ciphertext，记录 offset/length 后释放对象，单个在途块上限 4 MiB。禁止缓冲整个 ciphertext vector。输出兼容 `latti-ai.encrypted-parameter-store.v2`，通过 signature/layout/`validate_parameter_manifest` 校验并最后提交 manifest。Runner 包只含 sanitized config/graph、MegaAG、signature、eval context、encrypted store，私有 H5、secret/full context 和明文参数保留在浏览器工作区。

两模式均用 setup/restore 管理持久化密钥，按最终 task requirements 验证 profile、实际密钥身份和覆盖能力，兼容任务可复用。`bindTask` 校验模式配置、output signature 与 evaluation keys，禁止跳过检查或修补 ciphertext header/scale/level。浏览器有界解析 `latti-ai.runner-output.v1` 外层 framing，校验 ID 集合、重复、长度、截断、尾随数据，逐输出将原始 Feature wire 交 Go 解密。

| 模式 | native 只执行的接线 |
| --- | --- |
| 输入加密 | 浏览器 `task/server`、公开 eval context、`{feature_id: ciphertext_file}`→`InferenceServer::import_eval_context_file/load_model/evaluate`→原始输出密文文件或 named bundle |
| 参数加密 | 浏览器 Runner bundle、公钥加密的 crossing→`InferenceRunner::load/evaluate_inputs`（GPU `runtime_lazy`）→逐样本结果密文目录/ZIP |

必要时补窄范围 C++/Python binding adapter，记录修改。现有 `examples/inference.cpp` 输入模式内含 native setup/encrypt/decrypt，主链须改用只执行 Server 的入口。参数 example 可复用，但主链不得用 native provision/decrypt 替代浏览器。CPU/GPU 均可，优先复用既有 GPU lazy 环境并记录 backend、设备、构建、lazy mode、日志与 native 独立峰值。测试 HTTP 仅供静态资源和产物传输，隔离 oracle 需标记来源。

## 6. 内存与失败处理

客户端相关 renderer/Worker 同时驻留峰值严格 `< 4294967296` bytes。主要口径为 renderer RSS 合计，同时记录干净浏览器基线、进程集合与浏览器总 RSS，核查其他进程资源。覆盖 Pyodide、ORT、Go、JS heap、external ArrayBuffer、FFI 与临时工作集，native RSS 单独报告。

复用 browser harness 的 CDP/RSS 采样，记录 OS/Chromium/工具链、周期、阶段前后采样、缺口与余量。分项记录各 WASM memory 容量、Go HeapAlloc、JS heap、最大 tensor/key/ciphertext、in-flight chunk 和 OPFS 磁盘占用，堆包含于 linear memory 的部分避免重复相加，分项容量不能替代 RSS。

按以下顺序降峰并在生产端用已验证 profile/shape/图规模/单对象上限预检。

- 分块 I/O、有限 CSV buffer、逐 ciphertext、逐 H5 layer/dataset，减少图和权重副本，已持久化阶段及时释放。
- MEMFS 有界暂存仅用于实测达标的路径，超限时增加局部 OPFS `SyncAccessHandle` 或 external-data 范围读取。完整文件落 MEMFS 不代表流式，ORT external data 也不保证按需磁盘推理。
- linear memory/ORT arena 不一定随 GC、引用释放缩小，必要时关闭模型 Worker。FHE 密钥基线计入重叠峰值，关闭前持久化并验证 restore。
- 不可分 tensor/key/ciphertext 超预算时分配前拒绝，必测范围的完成条件见第 7 节。key/bundle 总文件允许超过 4 GiB，磁盘另作配额预算。

本轮四项降峰要求：

1. 保持原始输入只读，以有界 protobuf 字段扫描将 initializer 流式外置到 OPFS。工作图使用 `load_external_data=False`，执行节点时只按 offset/length 读取其权重；原模型、CSV 和 external data 的 hash 仍绑定原始来源。
2. 子图提取复用 ONNX `Extractor` 的依赖闭包，仅处理图结构与共享权重引用，不再反复加载、复制、序列化全量权重。权重文件置于 split 目录，prefix/suffix 的 checker 与 portable contract 校验仍执行。
3. source 外置、原图观测、Transformer 原始 float64 CSV reference、子图提取、prefix 验证、suffix 验证分别将结果持久化到私有 OPFS 检查点后重建模型 Worker。外部请求只有最终唯一终态；各阶段容量、Worker 次数必须记录，取消、quota/I/O、重试及提交保护必须覆盖阶段边界。
4. 标准算子与 Transformer 均按节点交接实时权重及激活，及时回收无后续使用的值；大权重以 feed 交给 ORT，避免 TensorProto、序列化与 externalData 的重复权重副本。float64 使用标准内核，不能因融合算子缺 double 内核而降为 float32。FFI 结果、feeds、session 在失败和成功路径均释放。

本轮统一按 BERT→ViT→retina→ImageNet→YOLOv5→CIFAR10 执行十二单元，并在末尾重复 CIFAR10 检查状态隔离。报告与上一版同口径 RSS、单独 WASM 容量及阶段峰值对比；切点、样本、完整输出和原有阈值保持不变。

模型同步运算可终止 Worker 取消，FHE 复用协作取消，所有请求产生唯一终态。OOM、runtime exit、有限超时、quota、I/O 错误均失败，保留阶段与最后指标，清理临时引用/目录并保护已提交文件。生产端无法精确读取进程峰值，依靠预检、运行时检查和开发验收共同约束。

## 7. 验收

### 7.1 固定输入与配置

CIFAR10、retina、BERT、YOLOv5、ViT、ImageNet 是必须通过的六个任务，各跑输入加密和参数加密，共十二单元。任务选择器可使用小写 `bert`、`vit`，实际仓库目录分别为 `runs/BERT/`、`runs/ViT/`。仅完成原 CIFAR10/retina 四单元不能算完成目标。每单元从原 ONNX/具名 CSV 经真实公开 API、Worker/WASM 生成主链产物，native 只执行 suffix。每单元至少使用下表固定样本，报告实际覆盖范围，模型更新须更新 hash 与结构/数值基线。

| 模型 | 原始模型及 CSV | FLOAT 输入、轴与输出 | 非 Constant 节点 / 切分 / crossing |
| --- | --- | --- | --- |
| cifar10 | `runs/cifar10/model/trained_poly.onnx`，`input=runs/cifar10/img.csv` | 输入 `[1,3,32,32]` NCHW，`output [1,10]` | 52 / 46+6 / 两个 Feature2D `[1,64,8,8]` |
| retina | `runs/retina/model/epoch041.onnx`，同目录 `input_0=img.csv`、`input_1=non_inv.csv`、`input_2=inv.csv` | 输入 `[1,5,512,512]` NCHW、`[1,2]` NF、`[1,3]` NF，`output [1,2]` | 63 / 56+7 / Feature0D `[1,512]` 与 `[1,32]` |
| BERT | `runs/BERT/model/trained_poly.onnx`，`inputs_embeds=runs/BERT/input.csv` | 输入逻辑 `[1,44,768]` NTD，`logits [1,2]`；按模型 attention mask 的有效 44 tokens 绑定原始 CSV | 99 / 89+10 / 两个 FeatureMat `[1,44,768]` |
| YOLOv5 | `runs/yolov5/model/best_conv_outputs.onnx`，`images=runs/yolov5/img.csv` | 输入 `[1,3,512,512]` NCHW；三个具名 Conv 输出 `[1,24,64,64]`、`[1,24,32,32]`、`[1,24,16,16]`，共 129024 元素 | 142 / 127+15 / 四个 Feature2D `[1,128,16,16]`、`[1,64,64,64]`、`[1,128,32,32]`、`[1,128,16,16]` |
| ViT | `runs/ViT/model/trained_poly.onnx`，`input=runs/ViT/patch_input.csv` | 输入 `[1,197,768]` NTD，原图声明 `output [1,197,10]`；实际输出及轴语义由原始完整模型执行确认 | 99 / 89+10 / FeatureMat `[1,197,192]` |
| ImageNet | `runs/imagenet/model/trained_poly.onnx`，`input=runs/imagenet/img.csv` | 输入 `[1,3,256,256]` NCHW，`output [1,1000]` | 100 / 90+10 / Feature2D `[1,160,8,8]` |

retina、ViT 动态 batch 显式绑定 `N=1`，retina 三个原始输入均进入 prefix。BERT 的 ONNX 输入声明为 `[1,128,768]`，现有任务语义以 attention mask 有效前缀绑定 `[1,44,768]`，须复用既有 contiguous attention-mask 校验及 token 适配，不得补零或换样本绕过。保留原预处理语义，crossing 名称、顺序、producer port、consumer 和数值从原图执行取得，禁止预制。锁定 CSV、config、源码/资源 hash 及以下模型 SHA-256。

```text
cifar10  4e4857a7af897707aa977ee824e1d0d425098cfd912a081f3c922ee546d59218
retina   cf91f21f43d903ad5a4102eb899201290b35082f583aae22af48ec6cab94a898
bert     850684974c16b0a6afe513ed347d7cbd4ab790dbc7672f2d704ec730a211073b
yolov5   7f924c75ed124391b6847f8d037ec6eb7c60da0032f39292fa7dd558c43b9a3d
vit      96cdfaa5c03c55171ff280f37be438cbe2621f5ba0c3f571f1f9d4b78848529c
imagenet fa59a8d129c11ecd58d43f651228a02f90e49139944131de34ac36d31f051cc7
```

配置固定为 Decimal 字符串 `"0.9"`、`top_level_non_constant_v1`、按原节点序号的 `lexicographical_topological_sort`，拒绝 forwarded original input 和无损旁路，不得为内存移动切点。CNN/MLP 两模式采用 multiplexed、standard、PN15Default CI 的真实 Q/P/ring/scale/BTP；参数模式显式加载锁定的 `training/config/storage_aware_config.json` 并使用 `runtime_lazy`，输入模式沿用 plaintext lazy。BERT 两模式使用锁定的 `training/config/BERT_pcm_config.json`（12 heads、head_dim=64、block=64），ViT 两模式使用 `training/config/ViT_pcm_config.json`（3 heads、head_dim=64、block=64）；均保留 `par_block_col_major` PCM、PN15Default CI、既有 BTP 与 runtime_lazy 参数语义。本次必测矩阵不以 PDM 替代 PCM。同模型可共享 split，两模式独立编译 contract，level/scale 不混用。

基线见 [CIFAR10 流程](../../runs/cifar10/run_pipeline.md)、[retina 流程](../../runs/retina/run_pipeline.md)、[native split 报告](../report/onnx_split_gpu_lazy_acceptance.md)，历史性能仅作诊断参考。

### 7.2 必测链路与数值判据

十二单元均完成浏览器 split、原图/prefix/suffix ORT 验证、High/Low-Level、H5、setup/restore、native 执行和浏览器全部输出解密/汇合。输入模式必须加密该任务全部 crossing（CIFAR10/retina/BERT 各两个、ImageNet/ViT 各一个、YOLOv5 四个），参数模式必须在浏览器生成涵盖实际签名全部参数的 bundle。枚举 arg ID、kind/count/level/scale/layout/generator，缺类别、skip 或门禁拒绝任一必测模型均计未完成。参数编码必须覆盖新增 CNN suffix 的全部实际参数，以及 BERT/ViT PCM 签名中的 matrix/affine/poly/structural 与 seed/recovery 类别；不得沿用仅支持原两个 suffix 的范围声明拒绝新增必测项。

每种模式 setup 持久化后关闭并 restore，再完成真实加解密，验证密钥身份与数值。六个任务仅要求默认 `ratio="0.9"` 的 ONNX split 链路通过十二单元验收，不强制未切分原始完整 ONNX 的 WASM 参数加密编译/provision/执行。`compileHighLevel({source, ...})` 保留为 API 能力；完整模型参数模式已发现的内存问题单独记录阶段、错误、实测峰值和复现入口，不阻塞本次 split 验收。所有必测执行均关联本次浏览器产物，缺 native 实际执行或以 oracle 替代主链均失败。

| 验证对象 | 通过条件 |
| --- | --- |
| split | 原图/prefix/suffix 在浏览器通过 checker，重复节点分区、crossing/source/port/routes 与 native oracle 一致 |
| 明文 | 浏览器原始完整 ONNX 全部输出作主 reference，与隔离 native ONNX 交叉检查，prefix/suffix 汇合按既有 `compare_arrays` 满足 shape/dtype、finite、`atol=1e-7, rtol=1e-6` |
| High/Low-Level | 对照 CT 图、packing/profile/level/scale/layout、H5 datasets、MegaAG 接口/拓扑、signature、key requirements，固定搜索随机源，规范化路径、耗时和无语义 ID，解释合法配置差异 |
| 参数 bundle | argument 集合、flat count、offset/length、level/scale/layout 与解密语义一致，随机密文不要求字节相同，传输保持原始 wire |
| 最终结果 | 按 `output_routes` 汇合原始全部输出，校验名称、顺序、shape、语义 dtype、元素数、finite，保留解密 float64 直接计算全部元素 `max_abs_error`，严格 `< 0.1`，等于阈值失败 |
| 资源 | 第 6 节连续流程及阶段峰值达标，报告全部运行时、单对象、OPFS、采样缺口和余量 |

ORT 版本差异先定位展开、算子和精度路径，禁止静默放宽容差或先将解密值转 float32 掩盖误差。

BERT/ViT 的 `CustomMultiHeadAttention`、`CustomLayerNorm`、`Linear`、`CustomGELU`、`CustomTanh`、`PolyActRN` 必须按图属性与既有 float64 参考语义提供通用浏览器算子适配。原图、prefix、suffix 的浏览器主链仍须实际执行，ORT Web 的算子展开、dtype 与 requested outputs 均记录；隔离 native NumPy float64 参考只作 oracle，不能替代浏览器主链。若浏览器 runtime 无法保留必需 dtype/算子语义，该项记录未完成，禁止改用历史结果、预存 crossing 或放宽阈值。YOLOv5 的三个原始输出全部解密/汇合并按顺序比较，禁止只验收最后一个输出。

原有资源预检范围必须扩展到六任务实际模型、initializer、输入、tensor、graph 和 PCM profile，仍须分配前校验。BERT 原 ONNX 约 343 MB、YOLOv5 有 142 个非 Constant 节点；不能用旧的 128 MiB source / 128-node 门禁跳过。内存严格 `< 4294967296` bytes、固定切点、真实 browser/native 执行和所有输出严格 `< 0.1` 的条件不变。超预算须按第 6 节完成流式处理、释放或阶段重建并实测，不算豁免。


### 7.3 回归与失败验收

- 在真实 Chromium 验证 ONNX wheel 的 import、checker/shape inference/extract、external data 和 H5 读写。
- 在无 torch Python 环境及实际 Pyodide 跑完整 ONNX→CT→H5，检查模块列表无间接 torch。小 fixture 覆盖普通与 FeatureMat 激活/系数分支，对照原 torch 路径的激活、Hermite、dataset 名称/shape/dtype/值，记录固定配置、容差和最大误差，回归原训练 API。
- BERT→ViT→retina→ImageNet→YOLOv5→CIFAR10→CIFAR10 验证状态隔离与 CNN/FeatureMat 配置切换，重复准备使用新目录并对照规范化内容。
- cycle、原始输入直通、Identity/Reshape/Transpose 旁路、歧义 shape/axes、不支持控制流、缺输入均失败。
- 模型/CSV/contract/hash 篡改、模式 contract 互换、缺 key、ring/scale/profile 不匹配在加密前失败。
- 参数缺元素、重复 flat index、非法布局/offset 越界，output bundle 截断、重复/未知 ID、非法长度/尾随数据均拒绝。
- 参数生成、CSV 加密、解密覆盖跨 chunk。setup/restore、编译、provision 注入取消、I/O/quota 错误，验证唯一终态、临时清理、已提交保护和重试。
- 连续测量内存并保持既有 Go/JS 单测与 browser client 工作流通过，新增测试聚焦 ORT 桥接、torch 解耦、状态、参数互操作、模式绑定和 framing。

### 7.4 入口与证据

新增脚本和 npm `test:model-workflow`，开发及交付默认跑 retina 的 split 参数加密模式；完整验收按下列命令显式选择六任务、两模式，串行跑十二单元。支持单元诊断及 Chromium、native Python、backend/设备、build/证据目录配置。依赖 wheel 构建步骤、有限可配置超时和当前阶段输出写入 README。

```bash
cd inference/interface/wasm
npm ci
npm run build
npm test
npm run test:browser
LATTI_MODEL_ACCEPTANCE_ROOT=/tmp/latti-browser-model-acceptance \
  npm run test:model-workflow -- --models cifar10,retina,bert,yolov5,vit,imagenet --modes input,parameter
```

`tools/run_split_inference.py` 含 native Client/Provisioner/decrypt，主链只复用其比较/路由逻辑。证据按根目录环境/汇总、每模型 source/split、每模式 task/bundle/browser日志/native输出/results/memory、独立 oracle 保存。

| 证据 | 必需内容 |
| --- | --- |
| `environment.json`、`source-manifest.json`、产物清单 | commits、工具链/OS/Chromium/设备、资源/build ID，source/suffix/contract/signature/config/CSV hash，mode、profile、公开密钥身份、样本绑定、执行位置及文件大小 |
| split/task/bundle、`browser.jsonl`、native 日志/输出密文 | 原始/子图和 external data、全部 crossing/reference、H5、图/签名/contract、参数索引与校验、API 进度/终态、backend/lazy mode、duration |
| `comparison.json`、浏览器解密与汇合输出 | `reference=original_full_onnx_csv_plaintext`、每 output 名称/shape/逻辑 dtype/count、max_abs/max_error_index、总体 `max_abs_error` 和严格阈值 |
| `memory.json`、`summary.json` | 阶段/连续指标、RSS/容量口径、周期/缺口、磁盘、十二单元各 browser 阶段与 native evaluate 的完成/失败及未完成项 |

数值、时间、错误与产物来源记录真实值，oracle 单独标记。secret key 内容禁止进入日志/公开报告，密钥存于本地私有目录，证据只引用身份和必要 hash。

## 8. 实施与完成

顺序为依赖/ONNX wheel 与包清单→训练解耦和模型准备/两级编译→既有 Go/WASM 输入模式闭环→签名驱动的逐密文 Provisioner/参数模式→内存与失败回归、包交付及第 7 节证据。

完成要求六个任务在默认 split ratio="0.9" 下通过同一公开 API 实现本规范的全部 split 客户端能力，模式产物可供 native 执行并正确回传解密，模型名和验收值不进入算法分支。依赖构建、类型/API、脚本、README、包内容和报告齐全，资源达标且第 7 节六任务 × 两模式十二单元及全部回归验收通过。只修改规范、只运行 native oracle、只通过原四单元或任一新增任务被门禁拒绝，均不算完成。


## 9. 四项降峰本轮验收结果（2026-11-19）

主证据位于 `/tmp/latti-browser-model-acceptance/on-demand-final-v2/`，按 BERT→ViT→retina→ImageNet→YOLOv5→CIFAR10 顺序完成十二单元，末尾 CIFAR10 重复检查通过。214 个主流程请求没有重复终态；23 组真实 Worker 失败回归通过，48 请求没有重复终态，临时池为空。新构建实际执行原图、prefix、suffix，不使用历史模型结果替代。

| 模型 | 输入模式 max_abs_error | 参数模式 max_abs_error |
| --- | ---: | ---: |
| bert | 7.91941356266e-05 | 3.98940919082e-05 |
| vit | 0.013026928108 | 0.00307504975793 |
| retina | 1.69650047788e-07 | 2.62993390088e-06 |
| imagenet | 3.8466350194e-06 | 9.69193244149e-05 |
| yolov5 | 1.30679415866e-05 | 3.22220920586e-05 |
| cifar10 | 3.18750796424e-06 | 4.85827441921e-06 |

全部原始输出严格 `<0.1`，YOLOv5 三输出共 129024 元素、ViT 全部 1970 元素、ImageNet 1000 元素均完整比较。CT/H5/MegaAG/signature/key requirements 与独立 oracle 对照通过；参数 bundle 的实际参数集合、flat count、布局、level/scale 和全部 slots 校验通过。

整轮 renderer/Worker 合计 RSS 峰值从 3968622592 bytes 降为 **2328748032 bytes**（下降约 **41.3%**）；余量 1966219264 bytes（约 1.83 GiB），最高阶段转为 `bert-parameter/provision`。采样周期 250 ms，15961 次采样，最大间隔 252 ms，无进程读取错误。分项容量不与 RSS 重复相加，native RSS 独立报告。

| 模型 | 原 split RSS（GiB） | 本轮 split RSS（GiB） | 本轮 Pyodide 最大容量（MiB） | 原/本轮 split 耗时（秒） |
| --- | ---: | ---: | ---: | ---: |
| bert | 3.696 | 1.244 | 103.5 | 60.89 / 69.68 |
| vit | 1.092 | 0.920 | 71.9 | 13.72 / 50.00 |
| retina | 1.640 | 1.221 | 214.8 | 18.37 / 81.50 |
| imagenet | 1.130 | 1.041 | 149.1 | 11.88 / 55.80 |
| yolov5 | 1.022 | 1.030 | 179.0 | 12.61 / 64.31 |
| cifar10 | 0.873 | 0.945 | 103.5 | 8.56 / 46.82 |

BERT split RSS 下降约 66.3%，Pyodide 最大容量由 3076128768 bytes 降至 108527616 bytes（约 103.5 MiB）。Transformer 使用 7 个阶段 Worker，标准图使用 6 个。逐阶段指标随 artifact manifest 保存。YOLOv5/CIFAR10 的 RSS 未下降；统一生命周期带来初始化开销，较小模型的 split 耗时明显增加，不能将本次结果描述为所有模型都更快或所有阶段都降峰。

四项实现对应 `tools/onnx_external_weights.py`、共享 `prepare_steps` 的权重引用路径、模型 API 的私有检查点/Worker 续接、按节点的权重 feed 与生命周期释放。小型 shape/control 常量保留或物化为有界常量，使 Reshape 等节点可在分配前推断 shape；float64 采用 ORT 标准内核，未降低精度。

详细资源与前后对比见 `resource-audit.json`、`memory-comparison.json`；失败回归在相邻的 `on-demand-final-v2-failures/`。交付包和完成审计生成于相邻 `/tmp` 目录。完整原始 ONNX 参数加密的既有内存问题仍为非阻塞项。


## 10. 原图批量默认链路与仅公钥交接（2026-11-27）

本节更新默认开发/交付场景。历史六模型单样本记录保持其原有范围，不能替代本节十样本验收。

默认原始输入为 Retina `demo_10/dataset` 的十份眼底、disk、lesion 图像和原始 XLSX 指标。可信 `preprocess.js` 生成三个输入目录及 `inputs.json.samples[]`；label 仅供后处理，不进入 prefix/suffix。ONNX batch 维为 N=1，批次逐样本串行调度，不合并成 N=10 大张量。首样本用于 split=0.9 的接口/明文等价检查；suffix high/low-level 只编译一次。

1. 模型方一次 `setup()` 生成同一 SK/PK 对及评估上下文。密钥生成和参数加密分别计时，后续复用。
2. `exportPublicContext()` 输出裸 CKKSCTX1 公共参数/PK envelope，两项 evaluation-key present 标志均为 false，不含 full/SK 外层容器。`importPublicContext()` 按可信 profile 校验长度、参数、键维度、EOF；拒绝 full、eval、截断及尾随内容。导入开始即清除原 SK、decryptor 和旧 PK，失败不恢复私钥能力。公钥导入后只能加密，不能解密或执行要求私钥自校验的参数 provision。
3. 模型方 `exportPrefix()` 生成独立 prefix；数据方只取得该产物、公共任务配置/输入 contract、公钥 context。`runPrefix()` 不接受完整 split。权重按当前节点 offset/length 读取，使用原有执行器，逐样本落盘。
4. `latti-ai.prefix-run.v2` 保存 split/prefix SHA-256、完整 case 集合、逐样本 manifest SHA-256 和耗时。每个 `run/cases/<case_id>/` 保存 crossing CSV 和公开路由。旧单样本 inputs manifest 与根 sample_manifest.json 保留兼容。
5. 新参数模式默认 `input_mode=client_ciphertext`、`parameter_mode=encrypted_offline`。输入/参数均用同一个模型方 PK 加密；`encryptBatch({mode:'server',inputDirectories,run})` 验证精确样本集合、shape/dtype/representation 和 crossing 内容 hash，串行生成输入密文 ZIP。显式 plaintext 历史任务仍可执行，但不计为本节验收。
6. native `evaluate_browser_task --batch-manifest ... --output-dir ... --gpu` 复用 Runner，按 manifest 顺序串行执行 ciphertext 输入与 encrypted parameters 的 GPU runtime_lazy 图；完整 case/输入路径预检后逐样本保存 `<case_id>.result.bin`，最后提交 batch manifest。结果 ZIP 只含约定的样本结果密文。
7. 模型方 `decryptBatch()` 校验输入批次的 profile/keyIdentity 和完整输出 case 集合，串行解密。可信 `postprocess.js` 依据 filename 与 label 对齐，计算 accuracy、specificity、sensitivity 和 AUC；ties 按平均秩计 AUC，单类时对应指标为 null。

`uploadProcessingScripts({directory,preprocessFile,postprocessFile})` 向操作方授权的静态资源目录写 JS 和 SHA-256 manifest，不自动执行。脚本视为可信操作方代码，远程鉴权/发布复用集成项目的后端；不让任意访客上传脚本后获得站点执行权限。

开发默认 `npm run test:model-workflow` / `test:batch`，从原图跑完整 WASM 主链及 native GPU；`test:delivery` 使用同一批量网页流程，仅以已绑定的真实 native 密文 ZIP 替代 FHE 执行。对照原始完整 ONNX 比较十样本全部输出，最大绝对误差严格小于 0.1；独立 Python 预处理 oracle 与浏览器 CSV 逐元素比对。报告记录阶段耗时、renderer RSS 与 native RSS/GPU 显存峰值，以及 case/模型/配置/参数/密钥身份/结果绑定。

交付必须包含独立 WASM ZIP、完整原 ONNX、原图/指标/标签、生成的批量 CSV、独立 pre/post JS、真实 GPU 批量结果密文、配套演示密钥及逐样本 crossing hash。页面重新执行预处理、split、编译、prefix、公钥导入/批量加密、参数 provision、批量解密和后处理；回放不证明新生成的参数 bundle 被 native 执行。生产数据方只能取得 prefix/公共 metadata/仅公钥 context，不能取得汇集双方测试材料的整个交付包。

验收产物在 /tmp，空间不足时只删除已有通过记录的旧产物；报告另存 latti-ai-reports，交付 ZIP 在 taa-delivery，工作区仅保留代码和规范。
