# 浏览器预处理、批量加解密与分类评估 implementation plan

## 1. 实施依据

严格依据[设计规范](../specs/latti_browser_preprocessing_batch_design_spec.md)，在 Latti-AI 本地 `feat/streaming-key-generation` 分支实施。密钥、packing、Go/WASM 核心和服务端接口直接复用现有实现。

设计规范基线提交：`b3f5e4b`（`docs: specify browser batch inference workflow`）。本计划单独提交，随后按模块分别实现、验证和 commit。

## 2. 浏览器批量加解密

提交：`feat: add sequential browser batch encryption`

对应规范：§4、§5、§9.2–§9.4。

1. 在真实 Chromium 回归中先增加源文件自动命名、多个 CSV 按顺序加解密、根层筛选、重名拒绝、中文/空格名称及空目录用例；确认缺失功能导致失败。
2. `src/worker.js` 按输入文件名生成输出名，任务特征名保持原用途，复用现有核心调用和写文件过程。
3. `src/index.js` 包装现有 `encrypt()`、`decrypt()`，根层枚举后排序，以 `for…of` 和 `await` 顺序处理，预先检查名称冲突并关联错误源文件。
4. `src/index.d.ts` 增加批量方法声明，保留单文件参数结构。
5. 更新既有验收脚本中依赖固定 `input.ct.bin` 的引用，使其对应实际输入文件名。
6. 运行包内 `npm test`、构建和真实浏览器回归，检查 diff 后提交。

## 3. CIFAR-10 任务预处理

提交：`feat: preprocess CIFAR images sequentially in browsers`

对应规范：§2、§3、§9.1–§9.3。

1. 从 `train.py` 的真实 eval transform 生成无损 RGB、放大、缩小、非正方形及 JPEG 参考夹具。新增浏览器测试，确认预处理缺失时失败。
2. 新增任务独立模块 `examples/test_cifar10/task/client/preprocess.js`，实现 RGB 解码、Pillow 语义双线性 Resize、float32 归一化、CHW 展平和单行 CSV 写出。
3. 根层按文件名排序，预先检查输出重名，一张图像处理并关闭 CSV 后再处理下一张；沿用既定自动命名和两目录接口。
4. 用真实浏览器文件句柄核对数值误差、CSV 数量/顺序、图像资源释放及异常行为。
5. 运行包内测试及预处理浏览器验收后提交。

## 4. 独立标注与分类后处理

提交：`feat: evaluate browser logits against image labels`

对应规范：§2、§6、§9.6。

1. 先增加已知 logits/标签、打乱标注顺序、并列最大值、中文/引号文件名以及缺失/重复/非法输入的浏览器用例，确认后处理缺失时失败。
2. 新增任务独立模块 `examples/test_cifar10/task/client/postprocess.js`，读取用户准备的 `filename,label` CSV，按源图名称关联 logits，检查样本集合及类别范围。
3. 逐文件读取 10 个有限 logits，执行 argmax；顺序写出 `predictions.csv`，关闭后返回文件句柄、total、correct 和 accuracy。
4. 保持预处理接口；标注表由用户提供并由后处理读取。
5. 运行包内测试及浏览器分类评估验收后提交。

## 5. 端到端验收与交付

提交：`test: validate browser batch inference delivery`

对应规范：§7–§9。

1. 使用多个 CIFAR-10 测试集原图和真实标签，在 Chromium 中串行执行预处理、同一恢复会话的批量加密及磁盘落盘。
2. 复用既有固定参数密钥文件和 GPU 服务端，逐样本导入密文并推理。浏览器恢复后批量解密，检查每个样本的 10 个 logits 与同预处理输入明文结果最大绝对误差小于 `0.1`。
3. 调用后处理并独立核对 argmax、样本关联、预测表及准确率；保留验收报告和当前样本资源指标。
4. 更新中文 README，说明脚本静态部署、目录选择、自动命名、批量调用、独立标注表和后处理。
5. 沿用浏览器包构建并补充任务 ZIP，任务目录为 `tasks/resnet20/client/`，包含任务配置、预处理和后处理模块；产物保存在 `/tmp`。
6. 运行 `npm test`、`npm run build`、`npm run test:browser`、新增工作流验收和发布包检查；核对 ZIP 内容、校验值及 `git diff --check`。
7. 将实际命令、结果及各模块提交写入执行记录，更新设计规范状态后提交。

## 6. 执行记录

| 模块 | 状态/提交 | 验证 |
| --- | --- | --- |
| 设计规范 | `b3f5e4b` | 格式、引用及现有接口核对 |
| 实施计划 | 待提交 | 与规范逐模块对应 |
| 浏览器批量加解密 | 待实施 | 真实 Chromium 与包内测试 |
| 图像预处理 | 待实施 | 训练 transform 参考与串行落盘 |
| 分类后处理 | 待实施 | 标签关联、argmax 和准确率 |
| 端到端与交付 | 待实施 | 多样本 GPU 推理、浏览器后处理及 ZIP |
