# latti-ai / lattisense dev 合并 main 后回归测试方案

## 1. 目标

用于 `dev` 合并到 `main` 后，从外部测试视角确认 latti-ai / lattisense 是否具备发布条件。重点不是穷举命令说明，而是明确回归需要覆盖哪些内容；具体 pytest 参数和执行命令以 `regression-assets/tests/README.md` 为准。

## 2. 执行约束

- 正式回归只拷贝 `regression-assets/tests/` 到待测 `latti-ai` 仓库根目录执行。
- `regression-assets/self-tests/` 仅用于测试资产自测，不进入正式回归目录。
- CPU / GPU / Docker / performance 回归在独立 Linux 测试机执行，不在本地 Windows 环境执行。
- 测试脚本不自动发送飞书、webhook 或其他外部通知。
- 结果以 `final-report.md` 为主，结合 `summary.md`、模块日志和人工检查结论判断。

## 3. 必测内容

### 3.1 环境与版本信息

需要记录并确认：

- latti-ai 当前 commit、branch、working tree 状态。
- `inference/lattisense` submodule commit 与状态。
- 测试机 OS、CPU、内存、磁盘、Python、CMake、编译器、CUDA / GPU 信息。
- pytest 日志目录和报告产物是否正常生成。

### 3.2 代码质量检查

需要覆盖：

- C/C++ 格式检查。
- Python ruff lint / format 检查。
- C/C++ 静态分析检查。

### 3.3 CPU 回归

需要覆盖：

- submodule 初始化和 lattisense 关键依赖同步。
- latti-ai CPU Release 构建。
- CPU CTest 核心用例，例如 data structs。
- hetero layer 生成、产物同步和 FHE layer 核心路径验证。
- Python compiler tests CPU 路径。
- CPU E2E batch 测试。

### 3.4 GPU 回归

需要覆盖：

- `nvidia-smi` 能识别 GPU 和 compute capability。
- HEonGPU configure / build / install。
- latti-ai GPU Release 构建，CUDA arch 需与测试机 GPU 匹配。
- Python compiler tests GPU 路径。
- GPU E2E batch 测试。
- MNIST / CIFAR10 mega ag 生成。

### 3.5 Docker 回归

需要覆盖：

- `Dockerfile` 能从干净环境成功构建镜像。
- 镜像构建阶段的 apt source 清理、`root` 用户和系统依赖安装逻辑无异常。
- 容器内能完成基础 CPU 验证：CMake 配置构建、hetero layer 生成、核心 FHE layer 测试和 Python compiler tests。

### 3.6 lattisense standalone 回归

需要覆盖：

- `inference/lattisense` 能 standalone CMake configure。
- standalone build 成功。
- standalone `ctest --output-on-failure` 通过。
- 报告中能体现 lattisense 当前 commit，便于确认 submodule 变更是否已被验证。

### 3.7 性能回归

需要覆盖：

- lattisense CPU / GPU FHE 算子 benchmark。
- convolution CPU / GPU benchmark。
- MNIST、CIFAR10、ImageNet 示例 CTest benchmark。
- 有 baseline 时对比耗时或吞吐量变化，超过阈值需标记疑似回归。
- 无 baseline 时只采集当前 main 指标，报告中明确标记“数据不足”。

### 3.8 文档与入口检查

需要覆盖：

- README、关键使用说明、Dockerfile 和示例入口文件仍存在。
- 生成 `docs-consistency-checklist.md`，由测试人员人工确认文档内容是否仍匹配当前构建、运行和测试方式。

## 4. 建议执行范围

| 场景 | 建议 scope | 覆盖重点 |
| --- | --- | --- |
| 环境确认 | `environment` | 版本、submodule、测试机和报告产物。 |
| 质量检查 | `quality` | 格式、lint、静态分析。 |
| CPU 回归 | `cpu` | CPU 构建、CTest、hetero / FHE layer、compiler tests、CPU E2E。 |
| GPU 回归 | `gpu` | HEonGPU、GPU 构建、GPU compiler tests、GPU E2E、mega ag。 |
| Docker 验证 | `docker` | Docker 镜像构建和容器内 CPU 基础验证。 |
| lattisense 验证 | `lattisense` | standalone configure / build / CTest。 |
| 性能回归 | `performance` | CPU/GPU/conv/examples benchmark 与 baseline 对比。 |
| 文档检查 | `docs` | 关键文档入口存在性和人工检查清单。 |
| 主流程回归 | `main-flow` | environment + quality + CPU + GPU + performance。 |
| 完整回归 | `full` | 全部阶段；可按需要跳过 Docker、lattisense 或 performance。 |

## 5. 通过标准

- 必测 scope 中没有阻塞失败。
- `final-report.md`、`summary.md` 和相关日志完整生成。
- Docker、lattisense standalone、CPU/GPU 核心路径均通过。
- 性能报告没有未解释的明显回退；若缺少 baseline，需在报告中明确说明只完成当前指标采集。
- 文档一致性检查清单已人工确认，无影响发布的文档入口缺失或内容错误。
