# 固定参数的浏览器密钥预生成设计规范

> 状态：已实现并完成验收（2026-10-15）。模块提交及验证记录见[实施计划](../plans/latti_parameter_key_pregeneration_implementation_plan.md)。

## 1. 目标与参数

桌面 Chrome/Edge 普通网页在 Dedicated Worker 中，自动使用内置的 `N16QP1546H192H32` 完整预设提前生成一套用户密钥。用户选择密钥保存目录即可开始生成；收到具体任务后，加载同一套密钥完成 CSV 加密和结果解密。

内置参数 ID 为 `param0`，对应现有 Lattigo `bootstrapping.N16QP1546H192H32` 的 Scheme/BTP 参数、`N=65536`、乘法层级 9、默认 scale `2^40`、`LogSlots=15` 和现有评估键集合。Q/P 使用编译预设定义。任务中的 `block_shape`、`pack_num`、输入输出形状和 packing 留在 `task_config.json` 中。

## 2. 文件与复用

用户选择密钥保存目录，客户端自动创建或打开其中的 `N16QP1546H192H32` 子目录；用户直接选择已有的同名参数目录时，客户端使用该目录。生成后的文件为：

```text
<用户目录>/N16QP1546H192H32/
  secret/
    full_context.bin
  public/
    eval_context.bin
```

`full_context.bin` 保存参数、SK 和 PK，评估键标志均为 0。`eval_context.bin` 保存参数、PK、RLK、旋转/共轭键、BTP 参数和 DtS/StD，`has_sk` 固定为 0。两者复用现有二进制格式，来自同一次生成的 SK/PK。

目录由用户隔离，同一用户在参数和评估键需求兼容的任务间复用这两个文件。目录中已有任一密钥文件时，`setup()` 拒绝覆盖；已有文件通过 `restore()` 恢复。网页上传接口只选取 `public/eval_context.bin` 和密文，`secret/full_context.bin` 留在本地。

任务目录单独保存 `task_config.json` 和 CSV。任务输入输出的参数 ID 必须为内置的 `param0`，解析到上述固定预设，`use_btp` 为 true；恢复时校验完整参数和 BTP 配置。输入输出名称、形状、level、packing 和布局在任务加载阶段检查。

## 3. 浏览器接口

扩展现有 `openBrowserClient()` 的目录选项，继续使用 `setup()`、`restore()`、`encrypt()`、`decrypt()` 和 `close()`：

```ts
openBrowserClient({
  clientDirectory?: FileSystemDirectoryHandle,
  keyDirectory?: FileSystemDirectoryHandle,
  onProgress?: (progress: Progress) => void,
}): Promise<BrowserClient>
```

至少传入一个目录。`keyDirectory` 表示用户选择的密钥保存目录或同名参数目录；提供它时，Worker 使用内置参数，将双文件路由到上述参数目录，任务配置从 `clientDirectory` 读取。

提前生成：

```js
const keys = await openBrowserClient({ keyDirectory, onProgress });
try {
  const { evalContext, fullContext } = await keys.setup();
} finally {
  await keys.close();
}
```

收到任务后：

```js
const client = await openBrowserClient({ clientDirectory, keyDirectory });
try {
  await client.restore();
  await client.encrypt({ inputName: 'input', inputFile, outputDirectory });
  // 得到服务端输出后，使用同一客户端的 decrypt() 写出 CSV。
} finally {
  await client.close();
}
```

仅传 `keyDirectory` 的会话支持生成、保存和恢复密钥；加解密要求已加载任务配置。仅传 `clientDirectory` 时，沿用现有任务目录配置和双文件位置，普通 CKKS/BTP 的调用保持兼容。

## 4. 核心生成与恢复

Worker 内置固定的 `param0` 参数描述，并通过已有的 `init-keys` 或 `init` 指令初始化 Go 核心。Go 核心根据编译预设创建 `clientcontext.Keys`，复用现有 `BeginSetup()`、`clientcontext.NewEvaluationStream()` 和 `clientcontext.MarshalFull()`。任务绑定时继续校验 BTP 模式、参数 ID 和数据布局。

评估键沿用现有正负幂次旋转、共轭和 `RotationsForBootstrapping()` 的去重集合；BTP 参数在该推导完成后序列化。RLK 和每把 Galois 键串行生成，复用单 payload 序列化 buffer。DtS/StD 继续通过现有成对接口生成。

Worker 继续以 4 MiB chunk 和背压顺序写入 `public/eval_context.bin`，关闭后保存 `secret/full_context.bin`。`setup()` 等两个文件均关闭后返回文件句柄。新任务会话从 secret 目录读取完整客户端 context，复用现有恢复检查及加解密对象。

## 5. 服务端与代码边界

服务端收到完整 `eval_context.bin` 后，直接使用现有 `import_eval_context_file()` 和任务加载、推理接口。任务文件继续描述模型与输入输出布局。

实现修改集中在 Latti-AI 的 `inference/interface/wasm`：Go 核心参数初始化、WASM 指令桥接、Worker 目录路由、JS/TS 接口和交付说明。Lattigo 的 `clientcontext` 及 LattiSense 的序列化/导入接口直接复用。修改和提交在本地 `feat/streaming-key-generation` 分支完成。

## 6. 验收与交付

1. 用户选择保存目录后，真实 WASM Worker 自动使用内置预设，在 `N16QP1546H192H32/` 中生成两个文件；验证目录自动路由、已有参数目录复用和已有文件保护。
2. 浏览器刷新后，同一密钥分别绑定两个布局不同的兼容任务；旧密文解密、新 CSV 加密成功；不匹配的参数、模式或参数 ID 被拒绝。
3. ResNet-20/CIFAR-10 的 BTP 文件由现有服务端直接导入、推理，浏览器新会话恢复后解密，并检查与明文结果的误差。
4. 真实用户目录的 File System Access API 完成大于 3 GiB 的 eval 文件写入；记录文件大小、chunk/payload 峰值、WASM 线性内存、Go HeapAlloc、耗时及可取得的浏览器 RSS 指标。
5. Go/CSV 测试、既有浏览器回归、新参数目录浏览器验收、发布构建和 ZIP 校验通过。发布包内置固定参数，并附中文 README 说明目录选择、提前生成、任务绑定及恢复流程。

## 7. 实施与验收记录

实现位于 `inference/interface/wasm/`，复用 Lattigo `clientcontext` 和 LattiSense 服务端接口。Go 核心及 WASM 桥接支持参数独立初始化；Worker 和 JS/TS 接口支持参数目录、任务绑定、双文件保存及恢复。固定参数内置到 Worker，中文 README 随浏览器 ZIP 交付。

验收命令在该源码目录执行：`npm test`、`npm run build`、`npm run test:browser`、`npm run test:pregeneration`、`npm run package -- /tmp`，均返回成功。Go/Chromium 测试覆盖参数独立初始化、保存目录自动路由、内置参数绑定、已有任一文件保护、空任务拒绝、错误参数 ID/模式拒绝和原普通 CKKS 路径。

完整验收使用 Linux 桌面 Chrome `147.0.7727.101`、Go `1.24.11` 和 RTX 5880 Ada 的现有 GPU 服务端构建。隔离 D-Bus/Xvfb 自动完成目录选择与浏览器授权，生产 Worker 通过真实 File System Access API 写入 `/tmp/latti-pregeneration.wtniIX/N16QP1546H192H32/`。用户只选择保存目录，客户端自动建立参数子目录并完成 `setup()`；随后同一密钥分别绑定标量和 CIFAR-10/ResNet-20 任务。页面刷新后旧标量密文解密及新 CSV 加密通过；GPU 服务端直接导入 eval 文件，推理后再次刷新恢复并解密，10 个输出与本次明文计算的最大绝对误差为 `0.0004983101`，低于验收阈值 `0.1`。

| 指标 | 本次结果 |
| --- | --- |
| `setup()` 耗时 | 174.73 秒 |
| `eval_context.bin` | 3,329,929,205 字节，约 3.10 GiB |
| `full_context.bin` | 25,494,056 字节，约 24.31 MiB |
| 最大序列化 payload | 63,734,300 字节，约 60.78 MiB |
| 最大 chunk / 同时写入 chunk | 4 MiB / 1 |
| 实际写入 chunk 数 | 838 |
| WASM 线性内存峰值 | 985,661,440 字节，940 MiB |
| Go HeapAlloc 观测峰值 | 974,063,856 字节，约 928.94 MiB |
| 渲染进程 RSS 合计观测峰值 | 1,344,736 KiB，约 1.28 GiB |
| CDP 所列 Chrome 进程 RSS 合计观测峰值 | 1,871,184 KiB，约 1.78 GiB |
| GPU 推理耗时 | 6.26 秒 |

RSS 在参数会话初始化及 setup 阶段，以 CDP `SystemInfo.getProcessInfo` 的 PID 读取 `/proc/<pid>/status`，每 250 ms 采样，共 701 次。合计包括相应进程中的浏览器开销及共享页；该口径是进程 RSS 采样值。Worker/WASM 的内存另由表中的线性内存和 Go HeapAlloc 表示。磁盘 eval header 已检查为 `has_sk=0`、`has_pk=1`；full 文件经过现有恢复校验，评估键标志为 0。

完整指标、输出及测试文件保存在 `/tmp/latti-pregeneration.wtniIX/report.json`，执行日志为 `/tmp/latti-builtin-preset-validation.log`。交付物 `/tmp/latti-client-wasm-0.1.0.zip` 包含内置固定参数的运行时及中文 README，已核对 ZIP 内容与源码构建产物一致，并通过内部 `SHA256SUMS` 校验；ZIP SHA-256 为 `874c554dd399f8be420c607c02710b68b297edd2697db1c2c7648dc656b1d42a`。
