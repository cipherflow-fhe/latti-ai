# 固定参数的浏览器密钥预生成 implementation plan

## 1. 实施依据

严格依据[固定参数的浏览器密钥预生成设计规范](../specs/latti_parameter_key_pregeneration_design_spec.md)。模块按以下顺序在 Latti-AI 的本地 `feat/streaming-key-generation` 分支实现和提交；Lattigo/LattiSense 的现有接口作为依赖复用。

## 2. 文档基线

提交：`docs: specify parameter key pregeneration`

- 提交设计规范及本计划，明确参数目录、双文件内容、API 选项、兼容性和验收标准。
- 核对规范中的接口与现有流式/持久化代码对应关系。

## 3. Go 核心与 WASM 桥接

提交：`feat: initialize browser keys independently of tasks`

对应规范：§1、§2、§4。

1. 先增加行为测试：参数配置单独初始化并生成 SK/PK 和公开 eval header；生成的 full context 能在不同兼容布局中恢复并加解密；错误配置、模式和参数 ID 拒绝。
2. 运行新增测试并确认在功能缺失处失败。
3. 新增固定参数初始化函数；复用现有 Keys、流式生成和 full context 编码。
4. 为任务初始化校验显式指定的固定预设，并检查输入输出参数引用。
5. WASM 指令桥接新增参数初始化分支；只有密钥的会话在加解密时明确要求任务配置。
6. 运行包内完整 `npm test`，执行 WASM 发布构建后提交。

## 4. Worker 与 JS/TS 目录接口

提交：`feat: reuse pregenerated browser contexts across tasks`

对应规范：§2、§3、§4。

1. 先编写真实 Chromium 测试：缺少 task 配置仍能打开参数会话、校验目录/配置、写入新目录、已有任一密钥保护，以及刷新后绑定新任务；确认新增接口在实现前失败。
2. 扩展现有 `openBrowserClient()` 可选 `keyDirectory`，保留原任务入口。
3. Worker 根据参数目录或任务目录选择初始化入口，将新模式的双文件分别路由到 secret/public；恢复只读取 secret 文件。
4. 沿用原单 Worker 队列、chunk 写入和文件关闭语义。
5. 构建并运行旧浏览器回归及新目录流程验证后提交。

## 5. 端到端验收与交付

提交：`test: validate parameter key pregeneration delivery`

对应规范：§5、§6。

1. 完成真实目录的完整 N16 BTP setup，记录文件大小、流式指标、耗时和浏览器内存。
2. 现有 CPU/GPU 服务端直接导入 eval 文件并运行 ResNet-20，浏览器刷新恢复并解密；检查任务误差容限和同一密钥跨布局复用。
3. 新增参数目录 JSON 模板，更新中文 README 的调用流程及发布 ZIP 内容。
4. 运行完整包测试、旧/新浏览器验收、构建、ZIP 内容/校验验证及 `git diff --check`。
5. 将实际命令、结果和指标写入验收记录，更新本计划执行记录并提交；构建产物与测试密钥保存到 `/tmp`。

## 6. 执行记录

实现时按上述模块填写实际提交和验证结果。
