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

全部模块已完成，实现和提交位于 Latti-AI 本地 `feat/streaming-key-generation` 分支。

| 模块 | 提交 | 验证 |
| --- | --- | --- |
| 文档基线 | `3978465` | 规范与现有接口核对 |
| Go 核心与 WASM 桥接 | `b1f8fda` | 新用例先失败，随后 Go/CSV 测试与 WASM 构建通过 |
| Worker 与 JS/TS 目录接口 | `15df1e0` | 真实 Chromium 新接口先失败，随后新旧目录流程通过 |
| 普通 CKKS 参数兼容性 | `fab2a59` | `PN14QP438` 回归用例先失败，修复后加解密通过 |
| 任务文件边界校验 | `b4fe232` | 空任务的浏览器用例先失败，修复后拒绝无效任务 |
| 端到端与交付 | 本记录所在的 `test: validate parameter key pregeneration delivery` 提交 | 真实目录 N16 BTP、跨刷新/布局、GPU 推理、发布包校验通过 |

已执行：

- `npm test`：包内 Go 测试与 3 个 CSV 测试通过；
- `npm run build`、`npm run test:browser`：发布构建、原普通 CKKS 回归与参数目录边界验证通过；
- `npm run test:pregeneration`：真实 File System Access 目录生成 3,329,929,205 字节 eval 和 25,494,056 字节 full；跨刷新解密旧密文、加密新 CSV、同一密钥绑定标量/ResNet-20 布局通过；现有 GPU 服务端直接导入并推理，最大绝对误差 `0.0004500023 < 0.1`；
- `npm run package -- /tmp`：ZIP 文件完整性、运行时与源码一致性、中文 README/模板和全部 `SHA256SUMS` 条目通过；
- Python 验收脚本的 Ruff 检查/格式检查、shell/JS 语法检查与 `git diff --check` 通过。

`setup()` 为 173.10 秒；WASM 线性内存峰值 940 MiB，最大 payload 约 60.78 MiB，chunk 4 MiB；250 ms 采样的渲染进程 RSS 合计峰值约 1.28 GiB。指标口径及交付物校验值详见设计规范第 7 节，原始结果为 `/tmp/latti-pregeneration.d0iDGE/report.json`，交付包为 `/tmp/latti-client-wasm-0.1.0.zip`。
