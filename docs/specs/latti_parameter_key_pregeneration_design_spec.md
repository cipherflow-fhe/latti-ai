# 固定参数的浏览器密钥预生成设计规范

## 1. 目标与参数

桌面 Chrome/Edge 普通网页在 Dedicated Worker 中，使用固定的 `N16QP1546H192H32` 完整预设提前生成一套用户密钥。生成过程仅读取参数配置；收到具体任务后，加载同一套密钥完成 CSV 加密和结果解密。

参数名称对应现有 Lattigo `bootstrapping.N16QP1546H192H32` 的 Scheme/BTP 参数、`LogSlots=15` 和现有评估键集合。配置内容如下：

```json
{
  "param0": {
    "poly_modulus_degree": 65536,
    "n_mult_level": 9,
    "log_default_scale": 40,
    "param_name": "N16QP1546H192H32"
  }
}
```

参数配置包含一个条目。名称、环维度、乘法层级和默认 scale 与该预设一致；显式提供的 `log_slots` 必须为 15。Q/P 使用预设定义。任务中的 `block_shape`、`pack_num`、输入输出形状和 packing 留在 `task_config.json` 中。

## 2. 文件与复用

用户选择名称为 `N16QP1546H192H32` 的参数目录，预先放入配置文件：

```text
<用户目录>/N16QP1546H192H32/
  ckks_parameter.json
  secret/
    full_context.bin
  public/
    eval_context.bin
```

`full_context.bin` 保存参数、SK 和 PK，评估键标志均为 0。`eval_context.bin` 保存参数、PK、RLK、旋转/共轭键、BTP 参数和 DtS/StD，`has_sk` 固定为 0。两者复用现有二进制格式，来自同一次生成的 SK/PK。

目录由用户隔离，同一用户在参数和评估键需求兼容的任务间复用这两个文件。目录中已有任一密钥文件时，`setup()` 拒绝覆盖；已有文件通过 `restore()` 恢复。网页上传接口只选取 `public/eval_context.bin` 和密文，`secret/full_context.bin` 留在本地。

任务目录单独保存 `task_config.json` 和 CSV。任务输入输出的参数 ID 必须能解析到上述固定预设，`use_btp` 为 true；恢复时校验完整参数和 BTP 配置。输入输出名称、形状、level、packing 和布局在任务加载阶段检查。

## 3. 浏览器接口

扩展现有 `openBrowserClient()` 的目录选项，继续使用 `setup()`、`restore()`、`encrypt()`、`decrypt()` 和 `close()`：

```ts
openBrowserClient({
  clientDirectory?: FileSystemDirectoryHandle,
  keyDirectory?: FileSystemDirectoryHandle,
  onProgress?: (progress: Progress) => void,
}): Promise<BrowserClient>
```

至少传入一个目录。`keyDirectory` 表示上述参数目录；提供它时，参数和双文件使用该目录，任务配置从 `clientDirectory` 读取。

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

只有参数目录的会话可以生成、保存和恢复密钥；加解密要求已加载任务配置。仅传 `clientDirectory` 时，沿用现有任务目录配置和双文件位置，普通 CKKS/BTP 的调用保持兼容。

## 4. 核心生成与恢复

在浏览器 Go 核心中增加固定参数的初始化入口，直接创建 `clientcontext.Keys`，再复用现有 `BeginSetup()`、`clientcontext.NewEvaluationStream()` 和 `clientcontext.MarshalFull()`。

评估键沿用现有正负幂次旋转、共轭和 `RotationsForBootstrapping()` 的去重集合；BTP 参数在该推导完成后序列化。RLK 和每把 Galois 键串行生成，复用单 payload 序列化 buffer。DtS/StD 继续通过现有成对接口生成。

Worker 继续以 4 MiB chunk 和背压顺序写入 `public/eval_context.bin`，关闭后保存 `secret/full_context.bin`。`setup()` 等两个文件均关闭后返回文件句柄。新任务会话从 secret 目录读取完整客户端 context，复用现有恢复检查及加解密对象。

## 5. 服务端与代码边界

服务端收到完整 `eval_context.bin` 后，直接使用现有 `import_eval_context_file()` 和任务加载、推理接口。任务文件继续描述模型与输入输出布局。

实现修改集中在 Latti-AI 的 `inference/interface/wasm`：Go 核心参数初始化、WASM 指令桥接、Worker 目录路由、JS/TS 接口和交付说明。Lattigo 的 `clientcontext` 及 LattiSense 的序列化/导入接口直接复用。修改和提交在本地 `feat/streaming-key-generation` 分支完成。

## 6. 验收与交付

1. 参数目录只有上述 JSON 时，真实 WASM Worker 成功生成两个文件；验证目录名称、参数配置和已有文件保护。
2. 浏览器刷新后，同一密钥分别绑定两个布局不同的兼容任务；旧密文解密、新 CSV 加密成功；不匹配的参数、模式或参数 ID 被拒绝。
3. ResNet-20/CIFAR-10 的 BTP 文件由现有服务端直接导入、推理，浏览器新会话恢复后解密，并检查与明文结果的误差。
4. 真实用户目录的 File System Access API 完成大于 3 GiB 的 eval 文件写入；记录文件大小、chunk/payload 峰值、WASM 线性内存、Go HeapAlloc、耗时及可取得的浏览器 RSS 指标。
5. Go/CSV 测试、既有浏览器回归、新参数目录浏览器验收、发布构建和 ZIP 校验通过。发布包附中文 README 和参数目录配置模板，说明提前生成、任务绑定及恢复流程。
