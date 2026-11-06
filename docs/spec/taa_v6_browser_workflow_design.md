# TAA v6 浏览器任务、批处理与 WASM 0.2.0 交付设计

日期：2026-11-06。状态：已实现并通过验收，CIFAR/Retina 当前端到端流程、六任务 14 配置回归与最终 clean revision 发布包核对均完成，证据见[验收记录](../research/taa_v6_acceptance_report.md)。依赖[浏览器客户端](taa_v6_browser_client_design.md)、[OPFS ZIP64](taa_v6_opfs_zip64_design.md)和[流式密钥](taa_v6_streaming_context_design.md)；旧任务日志、参数和准确率不是本规范的验收结果。

## 1. 完整流程与职责

```text
发布的 v6 profile → 无任务 setup / full restore
→ 绑定当前生成器生成的任务配置
→ 任务预处理脚本 → CSV
→ encryptBatch → OPFS input.zip
→ 上传公开 eval + input.zip → 真实 native/HTTP server
→ 下载 output.zip → OPFS reader → decryptBatch → logits CSV
→ 任务后处理 + 独立标签表 → 逐例预测/accuracy
```

WASM 负责 v6 packing/wire、密钥复用、单文件生命周期和 server 文件契约；任务 ES Module 负责模型预/后处理；server 负责 registry、required_inputs、模型执行与结果下载；联调 runner 只负责调用和记录证据。不在 runner 实现 ZIP、重命名结果、改 scalar skip 字节或修补密文 payload。

## 2. 单文件与目录批次

单文件操作显式给出任务 inputName/outputName、源 File/handle 和可选用户导出目录；返回 OPFS committed ArtifactRef。目录加密从 CSV stem 生成 `<stem>.ct.bin`，目录解密从密文 stem 生成 `<stem>.csv`，预处理图像只替换最后一个受支持扩展名。核心逻辑与 server 模式共用，不为两种模式各写一套编码器。

拟定批次参数如下；结果统一为浏览器规范中的 BatchResult，不返回全部文件内容或无限 handle 数组：

| 操作 | 必须参数 | 产物 |
| --- | --- | --- |
| encryptBatch directory | inputName、inputDirectory | OPFS 逐例密文 + manifest，可逐块导出到用户目录 |
| decryptBatch directory | outputName、inputDirectory | OPFS 逐例 CSV + manifest |
| encryptBatch server | inputDirectories: Record<inputName,directory> | OPFS input.zip + case/entry manifest；outputFile/outputDirectory 仅作流式导出目的地 |
| decryptBatch server | inputFile: File/handle、outputName | 直接消费 output.zip，OPFS 逐例 logits CSV + manifest，可流式导出 |

目录只枚举根层受支持文件，使用确定的非 locale 字符串顺序；预检空集合、重名、大小写/规范化冲突、输出已存在和缺少权限。所有阶段一次处理一个样本/输入，关闭当前文件、释放 payload/图像后再开始下一个；有限文件名索引计入 ZIP64 规范的 metadata 预算。

批次默认遇到错误即停止，错误包含 phase/case/input/源文件名。已完成文件不作为整个批次成功结果发布；只有批次清单提交后调用方才得到成功 BatchResult。重试使用新 job，或先明确清理当前未提交工作区；不加入复杂的断点续算协议。

## 3. server ZIP 的精确文件契约

| 归档 | 条目命名 |
| --- | --- |
| 单输入 input.zip | `<case_id>.ct` |
| 多输入 input.zip | `<case_id>__<input_name>.ct` |
| output.zip | `<case_id>.result.bin` |

case_id 从 CSV basename 去掉最后 `.csv` 得到；非空且满足扁平路径校验，多输入分隔符 `__` 不允许出现在 case_id，避免歧义。input_name 必须是 task_config/服务端 required_inputs 的真实 ID，不能从目录名猜测。任务配置列出的输入集与传入 inputDirectories 完全相等，所有输入目录的 case 集合完全相同；缺少、多出、重复或命名冲突均在 keygen/加密之前报告。

`task_input_id`/`task_output_id` 与其参数表一起决定特征名称；不得仅用对象首项替代。当前 ResNet/Retina server 契约为每 case 一个已选定任务输出，decryptBatch 必须验证 outputName 是该任务输出且与预期结果匹配。多个输出的任务如无明确 server 命名协议则拒绝，不臆造新后缀映射。

输出 ZIP 与输入 case manifest 对照，拒绝缺失、多余、重复和非法结果条目；manifest 是磁盘小型元数据，不泄漏私有 full。WASM 内部直接生成/消费上述条目名，调用方不需要 `.ct`/`.ct.bin`/`.result.bin` 转换。scalar 结果按 v6 wire 原样解析，禁止 offset 20 改写等历史 workaround。

ZIP32 和 ZIP64 均按同一个 server 文件契约处理，格式选择不改变 case 对应关系。小归档可在快速测试中强制 ZIP64；真实大归档仍需单独验收。外部 HTTP server 的端点/认证沿用实际部署配置，本规范不臆造仓库外 API，也不承诺未验证的 ZIP64 能力。

## 4. CIFAR/ResNet 预处理与后处理

任务包 client 目录提供独立 `preprocess.js`、`postprocess.js` 与当前 task_config/ckks_parameter；脚本作为明确导入的可信 ES Module 发布，不自动执行用户任意文件。任务脚本不携带密钥或 WASM 副本。

CIFAR-10/ResNet20 的转换以当前 `examples/test_cifar10/train.py` 的 eval/test_transform 为唯一参考，源分支行为仅作对照：

| 阶段 | 规定 |
| --- | --- |
| 输入 | 根层单帧、不透明 RGB PNG/JPG/JPEG；后缀大小写不敏感；预检输出 basename 冲突 |
| 解码/Resize | RGB8；32×32，同尺寸不缩放；与 torchvision PIL 默认双线性及下采样抗锯齿语义对照，不以浏览器默认 Canvas 缩放替代保证 |
| ToTensor | 除 255、CHW 顺序、float32 |
| Normalize | R/G/B mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]；逐步 float32，值与当前训练脚本再次核对 |
| CSV | c→y→x 的 3072 个数，一行、末尾换行，保存可往返数值精度；输出形状与 task_config 一致 |

每次只保留一张解码图像/当前工作 buffer，CSV 按块写，写完关闭并释放 bitmap/像素再处理下一张。若目标训练脚本语义发生变化，应同步脚本与参考夹具，不能继续引用旧常量却声称对齐。

后处理每例读取恰好 10 个有限 logits，argmax 并列取最小类别索引；标签按 case_id 关联，与文件/标签行顺序无关。标签与结果必须一一对应、无重复、类别范围有效，样本数至少为 1。逐例预测流式写报告，返回 total/correct/accuracy=correct/total；不得用历史参考 logits 代替本次实际解密值。

## 5. Retina 三输入

Retina-DKD 保留任务既有三路输入语义：input_0、input_1、input_2；原图、disk/lesion 分割结果、risk_factor 表格与标签由任务包脚本按当前训练/评估规则关联。只有相同 case_id 的三路 CSV 才能组成一个样本，shape/数值范围/输入 ID 在打包时与当前 task_config 校验。

复用已有任务脚本时只改必要的配置/输出入口与 v6 参数生成，不改变模型、分割算法或特征计算。若预处理依赖仓库已有离线工具，任务包清楚声明并交付其输出，不因浏览器链路而增加新的浏览器推理框架或偷偷替换算法。

输出为两个 logits，顺序 `[DN,NDRD]`、标签 0/1；argmax 和一一对应规则与 CIFAR 一致。10-case 真实联调用例记录本次 logits、预测、正确数和准确率，来源 9/10 不是当前通过条件或可直接复制的结果。

## 6. WASM 0.2.0 版本、构建与打包

版本唯一来源为拟新增 `inference/interface/wasm/package.json` 的 `version: "0.2.0"`。构建器读取它生成 JS/Worker/WASM build info、类型声明对应说明与 dist manifest；package-lock 顶层及 packages 根节点保持一致，不在 runner/打包脚本复制独立版本常量。

| 交付项 | 要求 |
| --- | --- |
| 独立浏览器包 | `latti-client-wasm-0.2.0.zip` |
| 总交付内 WASM 包 | `WASM-0.2.0.zip`；总 manifest 单独写 runtimeVersion=0.2.0、protocolVersion=2 |
| 代码与声明 | index、worker、zip、OPFS/协议辅助模块、类型声明、WASM、匹配的 wasm_exec.js、README、示例入口都进入复制清单 |
| 可复现信息 | 源码/四仓库 gitlink、Go/Node toolchain、profile 来源、协议/版本、文件 SHA256；附 SHA256SUMS |
| 任务与数据 | 保留各自独立版本，交付任务脚本与真实当前配置；数据包只含授权样本/标签/所需分割或表格，不内嵌私钥 |

Lattigo 当前 go.mod 为 Go 1.25.0 基线，WASM module/toolchain 跟随它；用实际构建 Go toolchain 附带的 `wasm_exec.js`，不能沿用来源 Go1.24.x runtime。验证 `GOOS=js GOARCH=wasm CGO_ENABLED=0`；native CGO 库/headers 和 Node addon 同样按实际新源码重建。

打包脚本使用通用命名并读取版本，不引入 `package_e2e_0_1_2.py` 这类旧版本专用入口；联调 runner 默认路径从 manifest/package 推导。JS、Worker、WASM 通过真实握手比较版本、协议及本次 build 资源集标识，错误组合启动即失败；只改目录名或 README 不构成版本升级完成。

dist 可以是普通运行时文件目录，但**浏览器中的 ZIP 操作**都使用 OPFS 规范；构建机打包不要求浏览器 OPFS。最终验收从实际解压的 0.2.0 发布包启动，防止开发目录通过、归档漏模块。

## 7. 修改范围

新增 WASM build/test/package 入口与任务模块；修改 native/examples/Node/Python 默认 setup 调用以使用文件产物；调整 CIFAR/Retina 的任务生成、联调和总交付入口。任务必须由当前 v6 generator 在临时目录重新生成，不复制来源 task_config、旧 preset/缺省 Q/P 或旧 dist 作为 fixture。

不重构无关六任务训练代码，不改变 compiler/runtime_lazy/encparams 路线，不新增线上服务框架，不将来源旧接口包装为默认 API。

## 8. 验收

1. 预处理与 Python/torchvision 参考：同尺寸无损 RGB 3072 个值最大绝对误差 ≤1e−6；放大/缩小/非正方形在同解码像素条件下每通道 RGB8 差≤1，归一化差≤1/(255×std[c])+1e−6；JPEG 解码差单独记录。覆盖通道/像素位置/纯色。
2. 目录模式逐例加解密、单文件自动命名、空目录/重复名/损坏文件/取消；server 模式单输入与三输入匹配，缺/多任一输入、结果或 label 均失败，输出不会串 case。
3. ResNet20/CIFAR10 与 Retina 各完成一组真实当前 v6 端到端：profile/setup/restore→OPFS ZIP→真实 server→结果 ZIP→解密→标签统计；逐例 logits 与 native 路径在既有误差标准内，无 payload 修改。
4. 从 OPFS 上传的真实 ZIP64 被 server 接受，结果 ZIP 原样被 0.2.0 reader 消费；独立大文件/offset/count 测试按 ZIP64 规范执行。外部能力未完成必须明确保留未完成项。
5. 分类后处理用独立实现核对实际 logits 的 argmax/accuracy；覆盖并列最大、标签打乱、缺/多/重复标签、NaN/Inf、越界和空集合。
6. 正式包解压后跑浏览器/OPFS/两任务，核对全部版本/manifest/hash；故意混入旧 Worker/WASM/wasm_exec.js，启动失败且无永久 pending。
7. 所有测量带本次 SHA/参数/输入，目录与 ZIP 分别报告内存和磁盘；六任务及 encparams 完整回归按总则执行，不以这两模型替代全基线验收。
