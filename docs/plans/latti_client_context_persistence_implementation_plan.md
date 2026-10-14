# 客户端双文件持久化 implementation plan

## 1. 实施依据

严格依据 [客户端双文件持久化设计规范](../specs/latti_client_context_persistence_design_spec.md)。三个仓库均在本地 `feat/streaming-key-generation` 分支实施，按 Lattigo → LattiSense → Latti-AI 顺序提交依赖。

交付内容为：默认 setup 保存 `eval_context.bin` 和仅含参数/SK/PK 的 `full_context.bin`；原生客户端与浏览器客户端恢复同一密钥；浏览器本地加密、解密与文件落盘。

## 2. Lattigo：共享纯 Go context 与 SDK 适配

提交：`feat: add persistent client contexts and shared streaming primitives`

1. 在纯 Go 包中实现仅含参数/SK/PK 的 full context 编码和恢复检查，沿用普通 CKKS/BTP 字段顺序、长度宽度及 SK/PK serializer。
2. 导出适配使用独立局部视图，保证从旧全套 context 导出时评估键标志均为 `0`，原 context 保持可用。
3. 复用现有 Galois 集合推导与单键序列化，提供浏览器可逐 payload 消费的串行生成入口；native SDK 与 WASM 使用相同纯 Go辅助逻辑。
4. Go SDK 增加 client-context 导出与按有效参数恢复的 CGO 符号，沿用错误返回约定。
5. 测试先验证双格式、SK/PK 恢复、参数不匹配、评估键过滤及源 context 功能，再验证现有流式生成回归和无 cgo WASM 编译。

对应规范：§3、§4.2、§5、§6。

## 3. LattiSense：C++ 薄包装

提交：`feat: wrap persistent client context serialization`

1. 更新 Lattigo submodule 指针。
2. 为 CKKS/BTP context 增加 client-context 导出和按期望参数导入薄包装，检查 Go SDK 状态返回。
3. 复用现有加解密对象及 generic serializer。
4. C++ 测试覆盖新客户端恢复原密钥、加解密往返和恢复错误传播；重新编译 SDK。

对应规范：§4.2、§5、§8。

## 4. Latti-AI：原生双文件保存与 Node 恢复

提交：`feat: persist client keys after streaming setup`

1. 更新 LattiSense submodule 指针。
2. `InferenceClient::setup()` 在任何上下文写入前检查已有 full 文件，沿用流式 create，并在完成后普通写入 full 文件。
3. full 导出调用新的局部视图适配；load 用任务的有效 scheme 参数检查并恢复 SK/PK。
4. 保留 Node 方法名、AsyncWorker 和 Promise 行为，更新使用说明。
5. 测试覆盖两个文件、已有 full 文件保护、写入失败、fresh-client 恢复、加解密、参数不匹配；运行 Node 测试和 CPU/GPU 回归。

对应规范：§4、§5、§7、§8。

## 5. Latti-AI：浏览器 WASM 核心与 Worker 包

提交：`feat: add browser wasm client with persistent keys`

1. 新增 `inference/interface/wasm` 浏览器包，固定 Go toolchain 并随构建复制匹配的 `wasm_exec.js`。
2. Go WASM 入口以 `GOOS=js GOARCH=wasm CGO_ENABLED=0` 构建，调用共享 context/逐键原语。
3. 按 native task config 读取有效参数，移植已验证的 CSV 数值 packing、ciphertext 结构、输出 unpack；用 native/browser 对照测试约束布局。
4. JS/TS API 实现 `openBrowserClient()`、`setup()`、`restore()`、`encrypt()`、`decrypt()` 和 `close()`。
5. Worker 使用用户授权文件句柄，串行处理操作；eval 文件按有界 chunk/背压写入，full 文件在 Worker 内保存与恢复。
6. setup 返回两个文件句柄，encrypt 返回密文文件，decrypt 返回 CSV 文件；任务名称负责映射。
7. Chromium 测试使用真实 WASM 与本地磁盘，覆盖 setup、关闭/刷新、restore、旧密文解密及新输入加密；测试文件失败和状态约束。

对应规范：§3、§4、§5、§6、§7。

## 6. Latti-AI：端到端与内存验收

提交：`test: verify persistent browser client interoperability`

1. 浏览器加密的密文由 native CPU/GPU 服务端读取，输出在浏览器新会话恢复后解密，按任务误差容限检查。
2. 记录独立客户端流式生成、full 导出及恢复的峰值内存；浏览器同时记录 WASM linear-memory 峰值、在途 chunk 与耗时。
3. 报告记录参数、环境、实际命令及结果；full 文件大小仅由参数/SK/PK 决定。
4. 检查各仓库 diff、测试结果、提交和 submodule 指针。构建输出及私钥样本保存在构建目录或 `/tmp`，提交源码、计划和验证记录。

对应规范：§5、§8。

## 7. 执行记录

| 模块 | 本地提交 | 验证 |
|---|---|---|
| Lattigo | `6950b19e` | clientcontext/Go SDK 定向测试及 race 测试通过；共享 BTP 流由 native SDK 导入并完成 bootstrap；无 cgo WASM 编译通过 |
| LattiSense | `318bfa1` | Release 编译通过；83 个 C++ 用例、20481 个断言通过，含 persistence/streaming |
| Latti-AI 原生 | `b32ef68` | GPU Release 与 Node Release 编译通过；Node 与 Python 测试通过，验证 full 文件权限、原密文恢复、参数拒绝和已有私钥文件保护 |
| Latti-AI 浏览器 | `0d81c81` | Go/WASM 核心、CSV 流解析和 Chromium Worker 测试通过；刷新后 restore、旧密文解密、新输入加密、进度/内存事件和非法名称拒绝通过 |

Lattigo 全量 `go test ./...` 仍报告基线故障：BFV/CKKS `Marshaller/Parameters/Binary` 长度差异，Ring `ModUp`/`ModDown`，RLWE `SwitchingKey.MarshalBinary` 无限递归。新增持久化和流式定向测试均通过。最终验收报告记录完整命令及限制。

浏览器发布构建由 `inference/interface/wasm/build.sh` 生成 `dist/latti_client.wasm`、匹配 Go 版本的 `wasm_exec.js` 和 Worker/TypeScript API；生成物保留在构建目录，不提交密钥或构建产物。

补充验收证据：`build/gpu/examples/inference --task-dir /tmp/latti-persistence-gpu-mnist-task --gpu --verify` 首次生成并完成 GPU 推理，最大误差 `0.00000601`；同一目录第二次运行从 `full_context.bin` 恢复，最大误差 `0.00000786`。Chromium Worker 刷新恢复测试记录 WASM linear memory `8388608` bytes，恢复后旧密文解密和新输入加密均通过。
