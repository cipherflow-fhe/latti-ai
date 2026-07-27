# Post-merge Regression Pytest Suite

这是基于资产包内 `../docs/dev合并main后回归测试方案.md` 编写的独立 pytest 命令编排脚本目录，用于整体拷贝到独立 Linux 测试机后，手动执行 `dev` 合并 `main` 后的回归验证。

本目录按外部测试资产维护，不作为 `latti-ai` / `lattisense` 仓库代码提交项；测试代码不会自动跟随被测仓库更新。每次回归前，测试人员需要人工确认脚本仍匹配当前仓库结构、构建命令、CTest label、Dockerfile 和 benchmark 输出格式。

## 1. 设计约束

- 作为外部测试资产维护，不提交到 `latti-ai` / `lattisense` 仓库。
- 使用时由测试人员手动拷贝到待测仓库根目录；仓库更新后需要人工评估是否同步更新本测试资产。
- 不依赖仓库已有 `scripts/` 或 `tests/` 下的任何脚本。
- 只负责编排项目原生命令，例如 `git`、`cmake`、`ctest`、`ruff`、`cppcheck`、`docker`、`nvidia-smi` 和项目内测试二进制。
- 同一测试模块内的多个命令会合并写入一个模块级日志文件，减少日志文件数量；日志中仍按 step 记录实际命令、工作目录、退出码和耗时。
- 默认 scope 是 `environment`，避免误触发 CPU/GPU/Docker 等重型回归。
- 本套脚本应在独立 Linux 测试机上手动运行，不用于本地 Windows 环境执行 latti-ai 回归命令。
- 本套脚本不配置 GitHub Actions 触发器，不自动生成飞书摘要，也不自动发送外部通知；最终测试结果以日志目录中的 Markdown 报告为准。

## 2. 目录结构

```text
tests/
  __init__.py
  conftest.py
  regression_config.py
  regression_runner.py
  report_writer.py
  performance_benchmark.py
  resource_monitor.py
  test_00_environment.py
  test_10_quality.py
  test_20_cpu.py
  test_30_gpu.py
  test_40_docker.py
  test_50_lattisense.py
  test_60_docs_consistency.py
  test_70_performance.py
  README.md
```

`self-tests/` 已删除，本资产包当前只保留正式回归用的 `tests/` 目录。正式回归时将本目录整体作为待测仓库根目录下的 `tests/` 使用即可。

## 3. 执行方式与推荐顺序

默认 `--scope` 是 `environment`，不会误触发 CPU/GPU/Docker/performance 等重型回归。pytest session 结束时会自动生成 `summary.md` 和 `final-report.md`；最终结果以 `final-report.md` 为主，必要时再查看模块级 `*.log`。

### 3.0 测试环境准备要点

- `quality` 和包含 `quality` 的 scope 需要测试机 PATH 中存在 `pre-commit`、`ruff` 和 `cppcheck`。仓库 `.pre-commit-config.yaml` 使用新版 stage 名称 `pre-commit`，建议使用 `pre-commit>=3.5`；旧版 `pre-commit 2.x` 会报 `Expected one of commit ... but got: 'pre-commit'`。
- `ruff` 建议使用 `0.9.10`，与仓库 pre-commit 配置中的版本保持一致。
- `cppcheck` 需要支持 `--std=c++20`；系统源版本过旧时会报 `unrecognized command line option: "--std=c++20"`，需要升级到支持 C++20 的版本。
- 麒麟 V10 ARM/aarch64 测试机如果系统源缺少 GCC/G++ 12，可使用 Miniforge/conda 环境提供 GCC/G++ 12，并通过 `CC`、`CXX` 指定编译器；不要把系统自带 GCC 7.x 直接软链为 `gcc-12`。
- ARM 测试环境建议通过环境脚本统一激活 conda、Go、CMake prefix 和编译器变量，例如：

```bash
source /opt/latti-arm-env.sh
cd /home/latti-ai
python -m pytest tests/test_20_cpu.py \
  --scope cpu \
  --python-bin "$(which python)" \
  --continue-on-error \
  -v
```

- CPU scope 的最终 `./test_e2e '[batch][cpu]'` 是单进程 batch E2E，会连续运行多个较大的 FHE case。低内存测试机可能被 OOM killer 杀掉；例如 6.5GiB 内存即使追加 32GiB swap 仍可能无法支撑该步骤。这类结果应记录为测试机资源限制，不应通过拆分 case 改变正式回归脚本语义。

### 3.1 推荐主流程

main 合并后的默认主流程建议先跑 `main-flow`，覆盖 environment、quality、CPU、GPU 和 performance；Docker、lattisense standalone 和 docs 可按需单独补跑。GPU 阶段会先检查 `/home/qianc/latti-ai-deps/cccl-src`，如果该目录存在，会自动作为 HEonGPU 的本地 CCCL FetchContent 源码目录传入；只有使用其他路径时才需要显式传 `--cccl-source-dir`。

```bash
python -m pytest \
  tests/test_00_environment.py \
  tests/test_10_quality.py \
  tests/test_20_cpu.py \
  tests/test_30_gpu.py \
  tests/test_70_performance.py \
  --scope main-flow \
  --cuda-arch 89 \
  --log-dir logs/main-flow-manual-check \
  --python-bin python3 \
  --benchmark-modules cpu,gpu,conv-cpu,conv-gpu,examples-cpu,examples-gpu \
  --benchmark-repeat 3 \
  --continue-on-error \
  -v
```

### 3.2 分阶段执行命令

- **环境信息**：`python -m pytest tests --scope environment -v`。只采集 commit、submodule、系统、工具链和 GPU 信息。
- **质量检查**：`python -m pytest tests --scope quality --continue-on-error -v`。执行格式、lint 和静态分析检查。
- **CPU 回归**：`python -m pytest tests --scope cpu --continue-on-error -v`。执行 CPU 构建、CTest、hetero/FHE layer、compiler tests、CPU E2E；低内存测试机可能无法完成最终 batch E2E。
- **GPU 回归**：`python -m pytest tests --scope gpu --cuda-arch 89 --continue-on-error -v`。执行 HEonGPU、GPU 构建、GPU compiler tests、GPU E2E、mega ag。
- **Docker 验证**：`python -m pytest tests --scope docker --continue-on-error -v`。执行 Docker 镜像构建和容器内 CPU 基础验证。
- **lattisense standalone**：`python -m pytest tests --scope lattisense --continue-on-error -v`。执行 lattisense standalone configure/build、生成 CPU 测试数据并执行 CTest。
- **文档检查**：`python -m pytest tests --scope docs -v`。检查关键输入文件存在性，并生成 `docs-consistency-checklist.md` 人工清单。

### 3.3 性能 benchmark

`performance` scope 会直接执行 benchmark，并把 benchmark step、性能指标、examples 单 case OpenMP线程数、扣减运行前系统基线后的内存/显存增量峰值和 baseline 对比统一写入 `final-report.md`；不再生成 `performance.md`、`performance-benchmark.md`、`performance-benchmark-results.jsonl` 或 `performance-comparison.json`。只有运行 `examples-cpu` 或 `examples-gpu` 模块时，资源增量原始记录才会写入 `performance-resource-metrics.jsonl`。

无 baseline（首次运行或仅采集当前 main 性能）时：

```bash
python -m pytest tests/test_70_performance.py \
  --scope performance \
  --benchmark-modules all \
  --benchmark-repeat 3 \
  --cuda-arch 89 \
  --continue-on-error \
  -v
```

有 baseline 时：

```bash
python -m pytest tests/test_70_performance.py \
  --scope performance \
  --benchmark-modules all \
  --benchmark-repeat 3 \
  --cuda-arch 89 \
  --baseline <previous-main-commit-or-release-tag> \
  --baseline-log-dir logs/<baseline-run> \
  --performance-regression-threshold 10 \
  --continue-on-error \
  -v
```

只跑单个模块，例如 CPU 算子 benchmark：

```bash
python -m pytest tests/test_70_performance.py \
  --scope performance \
  --benchmark-modules cpu \
  --benchmark-repeat 3 \
  -v
```

### 3.4 完整回归

完整回归覆盖全部阶段；如需快速排查或分阶段执行，可结合 `--skip-performance`、`--skip-docker` 或 `--skip-lattisense` 跳过对应重型阶段。完整回归中的 GPU 阶段同样会自动检查默认 CCCL 源码目录 `/home/qianc/latti-ai-deps/cccl-src`。

```bash
python -m pytest tests \
  --scope full \
  --cuda-arch 89 \
  --log-dir logs/$(date +%Y%m%d)/$(date +%H%M%S) \
  --continue-on-error \
  -v
```

有 baseline 时追加：

```bash
--baseline <previous-main-commit-or-release-tag> \
--baseline-log-dir logs/<baseline-run> \
--performance-regression-threshold 10
```

## 4. pytest 参数说明

参数默认值：

- `--scope`：`environment`
- `--cuda-arch`：None
- `--log-dir`：`logs/YYYYMMDD/HHMMSS`
- `--repo-root`：repo root
- `--continue-on-error`：`False`
- `--python-bin`：`python3`
- `--install-python-deps`：`False`
- `--baseline`：None
- `--target`：`git rev-parse HEAD`
- `--baseline-log-dir`：None
- `--performance-regression-threshold`：`10.0`
- `--skip-performance`：`False`
- `--skip-docker`：`False`
- `--skip-lattisense`：`False`
- `--benchmark-modules`：`all`
- `--benchmark-repeat`：`3`
- `--benchmark-timeout-seconds`：`3600`
- `--cccl-source-dir`：如果 `/home/qianc/latti-ai-deps/cccl-src` 存在则默认使用该本地 CCCL 源码目录；否则不传入 `FETCHCONTENT_SOURCE_DIR_CCCL`

参数说明：

- `--scope`：指定执行范围，避免默认触发重型回归。
- `--cuda-arch`：GPU scope 必填，例如 Ada GPU compute capability 8.9 对应 `89`。
- `--log-dir`：命令日志和报告输出目录；未指定时自动创建 `logs/YYYYMMDD/HHMMSS` 目录，同一天运行的日志会先按日期归档，再按运行时间分目录，避免覆盖历史日志。
- `--repo-root`：指定被测仓库根路径。
- `--continue-on-error`：同一 pytest 用例内关键命令失败后是否继续执行后续命令。最终仍会 fail。
- `--python-bin`：Python 编译器测试和生成脚本使用的 Python 可执行文件。
- `--install-python-deps`：默认不安装 `training/requirements.txt`；仅在显式传入该参数时才安装，避免测试机网络慢时长时间下载大包。
- `--baseline`：性能报告中的基线版本；没有 baseline 时不要传。
- `--target`：性能报告中的合并后 main 版本；未传入时自动识别当前 `--repo-root` 的 HEAD commit，识别失败时留空。
- `--baseline-log-dir`：可选的基线性能日志目录，例如 `logs/20260623/182821`；没有 baseline 时不要传。传入后脚本会读取该目录与当前 `--log-dir` 自动填充 `final-report.md` 的基线列、main 列、变化比例和结论。
- `--performance-regression-threshold`：性能回归阈值百分比；耗时类指标 target 比 baseline 慢超过该值时标记为疑似回归，吞吐量类指标 target 比 baseline 低超过该值时标记为疑似回归。
- `--skip-performance`：full scope 中跳过 performance 阶段。
- `--skip-docker`：full scope 中跳过 docker 阶段。
- `--skip-lattisense`：full scope 中跳过 lattisense 阶段。
- `--benchmark-modules`：逗号分隔的 benchmark 模块：`all`、`cpu`、`gpu`、`conv-cpu`、`conv-gpu`、`examples-cpu`、`examples-gpu`。
- `--benchmark-repeat`：每个 benchmark 运行命令的重复次数，必须大于等于 1。
- `--benchmark-timeout-seconds`：每个 benchmark step 的超时时间，单位秒。
- `--cccl-source-dir`：可选，指定本地 CCCL 源码目录。传入时目录必须存在；GPU scope 会把它传给 HEonGPU CMake 的 `-DFETCHCONTENT_SOURCE_DIR_CCCL=<path>`。不传时会检查默认路径 `/home/qianc/latti-ai-deps/cccl-src`；该目录存在则自动使用，不存在则让 HEonGPU 按自身 FetchContent 逻辑处理。

## 5. Scope 说明

- `environment`：启用 environment + report，只记录测试机环境和版本信息。
- `quality`：启用 environment + quality + report，执行格式、lint、静态分析检查。
- `cpu`：启用 environment + quality + cpu + report，执行 CPU-only 构建和核心 CPU 测试回归。
- `gpu`：启用 environment + gpu + report，执行 GPU 构建、HEonGPU、GPU E2E 和 mega ag 生成；GPU examples 由 performance scope 的 `examples-gpu` benchmark 覆盖。
- `docker`：启用 environment + docker + report，执行 Docker 镜像构建和容器内 CPU 验证。
- `lattisense`：启用 environment + lattisense + report，执行 lattisense standalone configure/build/CTest。
- `performance`：启用 environment + performance + report，执行性能 benchmark 并自动生成性能对比报告。
- `docs`：启用 environment + docs + report，检查文档输入文件存在性并生成文档一致性清单。
- `main-flow`：启用 environment + quality + cpu + gpu + performance + report，用于 main 合并后手动主流程回归。
- `full`：启用全部阶段，用于合并后完整回归。

## 6. 公共模块说明

本节只说明公共模块在回归脚本中的职责。它们通常不需要单独执行，由 pytest 收集测试时自动加载或被各测试用例调用。

### 6.1 `__init__.py`

将 `tests` 标记为 Python 包，保证 `conftest.py`、公共模块和各测试脚本之间可以稳定导入。

### 6.2 `conftest.py`

pytest 插件入口，负责：

- 注册回归参数，例如 `--scope`、`--cuda-arch`、`--log-dir`、`--baseline-log-dir`、`--benchmark-modules` 等。
- 注册 `regression_scope(name)` 和 `heavy` marker。
- 根据 `--scope` 自动 skip 不属于当前执行范围的测试。
- 提供 session 级 `regression_config` 和 `regression_runner` fixture。
- 在 session 结束时生成 `summary.md` 和 `final-report.md`。

### 6.3 `regression_config.py`

集中维护回归配置和 scope 映射：

- `SCOPE_STAGES` 定义 `environment`、`quality`、`cpu`、`gpu`、`docker`、`lattisense`、`performance`、`docs`、`main-flow`、`full` 对应的执行阶段。
- `RegressionConfig` 保存仓库根目录、日志目录、CUDA arch、baseline、benchmark 参数和跳过选项。
- `build_config(...)` 负责解析 pytest 参数，并在执行前校验非法 scope、缺失 baseline 目录、非法 CCCL 源码目录等配置问题。

### 6.4 `regression_runner.py`

统一封装外部命令执行，避免各测试脚本重复处理日志和失败逻辑：

- `CommandStep` 描述一个命令 step，包括名称、命令、日志文件、工作目录、环境变量、是否关键和超时时间。
- `RegressionRunner.run_many(...)` 按顺序执行 step，并按 `--continue-on-error` 决定关键命令失败后是否继续。
- 每个 step 会记录命令、工作目录、开始/结束时间、退出码和耗时。
- 构建类命令会使用 `<log-dir>/tmp` 作为临时目录，降低测试机系统 `/tmp` 空间不足导致误失败的概率。
- 报告所需的测试人员和 submodule commit 也在这里采集。

### 6.5 `report_writer.py`

负责生成最终可读报告：

- `summary.md` 汇总所有 step 的执行状态、退出码、耗时和日志文件。
- `final-report.md` 是主要回归报告，包含基本信息、命令执行结果，以及 performance scope 的 benchmark step 和性能指标对比。
- 单独运行 quality、GPU、lattisense 用例时，会生成更聚焦的专项报告。
- 不再生成 `feishu-summary.md`、`feishu-message.json`、`step-results.jsonl`、`performance.md`、`performance-benchmark.md`、`performance-benchmark-results.jsonl` 或 `performance-comparison.json`。

### 6.6 `performance_benchmark.py`

负责 performance scope 的 benchmark 编排和指标解析：

- 支持的模块包括 `cpu`、`gpu`、`conv-cpu`、`conv-gpu`、`examples-cpu`、`examples-gpu`。
- `all` 会展开为全部 benchmark 模块；GPU 相关模块必须传 `--cuda-arch`。
- 每个运行类 step 会按 `--benchmark-repeat` 重复执行；CPU/GPU 算子和 convolution benchmark 只记录 wall time、FHE 算子耗时和吞吐量。
- `examples-cpu` 和 `examples-gpu` 运行类 step 会直接运行 `./examples/inference`，并通过 `resource_monitor.py` 解析 inference 输出中的 OpenMP 实际线程数标记，额外记录该 case 的 OpenMP线程数，以及扣减 case 启动前系统基线后的 CPU 内存增量峰值；`examples-gpu` 还会采样 `nvidia-smi` 记录扣减 case 启动前系统基线后的显存增量峰值。
- `final-report.md` 会解析 `performance.log` 中的 `wall_seconds`、FHE 算子耗时和吞吐量，解析 `performance-resource-metrics.jsonl` 中的 examples 资源增量，并在提供 `--baseline-log-dir` 时自动生成 baseline 对比。

## 7. 测试脚本说明

### 7.1 `test_00_environment.py`

**整体介绍**

环境采集脚本，用于记录待测代码版本、submodule 状态、Linux 系统信息、工具链、Docker 和 GPU 信息。该脚本是所有 scope 的基础阶段。

**测试用例**

#### `test_collect_git_and_submodule_versions`

- **功能**：记录合并后 main 的 commit 和 submodule commit。
- **测试内容**：确认待测仓库 HEAD 和各 submodule commit 可被完整采集，作为回归报告中的版本追溯依据；不再记录最近 20 条 git history。
- **执行命令**：
  - `git rev-parse HEAD`：读取待测仓库当前 HEAD commit，作为本次回归的主版本标识。
  - `git submodule status`：记录所有 submodule 的 commit 和状态，便于追溯 lattisense、HEonGPU、Lattigo 等依赖版本。
- **输出日志**：`environment.log`，该用例的 2 个 git 命令按 step 合并写入同一个日志文件。
- **失败行为**：任一 git 命令失败会使该 pytest 用例失败。

#### `test_collect_linux_environment`

- **功能**：记录 Linux 测试机环境、编译工具链、Python、Go、Docker 和 GPU 信息。
- **测试内容**：确认测试机 OS、CPU、内存、编译工具链、Python、Go、Docker、GPU 信息以及 `final-report.md` 基本信息所需的测试人员和 submodule commit 可被采集，便于后续定位环境差异；GPU 信息缺失只作为非阻塞项记录。
- **执行命令**：
  - `uname -a`：记录内核版本和系统架构，辅助定位 Linux 内核或平台差异。
  - `cat /etc/os-release`：记录 Linux 发行版和版本号，辅助判断系统依赖兼容性。
  - `lscpu`：记录 CPU 型号、核心数和指令集信息，辅助分析 CPU 构建和性能结果。
  - `free -h`：记录内存容量和可用内存，辅助判断构建、Docker 或大模型示例是否受内存影响。
  - `collect_environment_report_info()`：在 pytest session 内存中采集测试人员、lattisense/HEonGPU/Lattigo commit，以及 GCC、G++、CMake、Go、Python3、Docker 版本；不写出额外 JSON 文件，`final-report.md` 只渲染测试人员和 commit 等基本信息，不再输出模块版本表。
  - `nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader`：记录 GPU 型号、驱动、显存和 compute capability；失败时作为非阻塞项记录，表示该测试机可能不适合 GPU scope。
- **输出日志**：`environment.log`，Linux 环境、环境报告信息采集结果和 GPU 信息采集命令按 step 合并写入同一个日志文件。
- **失败行为**：`nvidia-smi` 为非阻塞采集项，失败会记录为 `NON_BLOCKING_FAIL`；其他命令失败会使该 pytest 用例失败。

**执行方式**

```bash
python -m pytest tests/test_00_environment.py --scope environment -v
```

---

### 7.2 `test_10_quality.py`

**整体介绍**

质量检查脚本，参考合并后回归方案中的质量检查命令执行 clang-format、ruff 和 cppcheck。该脚本标记为 `heavy`，因为会扫描大量文件。

**测试用例**

#### `test_quality_checks_match_release_plan`

- **功能**：执行 C/C++ 格式检查、Python lint、Python format check 和 C++ 静态分析。
- **测试内容**：确认合并后代码满足 clang-format、ruff lint、ruff format 和 cppcheck 的质量门禁；同时会覆盖拷贝到仓库根目录的外部测试脚本，避免测试资产自身格式不合规。
- **执行命令**：
  - `pre-commit run clang-format --all-files`：运行项目已有 clang-format pre-commit hook，确认 C/C++ 代码格式符合仓库规则。
  - `ruff check . --exclude='inference/lattisense,build,venv,.venv,.git,logs'`：运行 Python lint，排除第三方子模块、构建目录、虚拟环境和历史日志，检查主仓库及外部测试脚本的 Python 质量问题。
  - `ruff format --check . --exclude='inference/lattisense,build,venv,.venv,.git,logs'`：检查 Python 文件是否已按 ruff 格式化；不会自动改文件，用于阻止未格式化测试脚本进入正式回归。
  - `cppcheck`：按回归方案执行 C++ 静态分析，扫描 `inference/data_structs`、`inference/fhe_layers`、`inference/inference_task`、`inference/util`、`inference/util.cpp`、`inference/common.h`、`examples`；额外排除 `.venv`、`logs`，并 suppress 第三方 `inference/lattisense/lib/nlohmann/json.hpp` 的 `preprocessorErrorDirective`。
- **输出日志**：`quality.log`，clang-format、ruff 和 cppcheck 命令按 step 合并写入同一个日志文件。
- **报告输出**：单独运行 `test_10_quality.py` 后，`final-report.md` 不输出“基本信息”和“模块版本”，只输出“格式检查和静态分析”表格，展示 C/C++ 格式检查、Python lint、Python 格式检查和 C++ 静态分析的结果、退出码、耗时和日志。
- **失败行为**：任一质量检查命令非 0 退出会使该 pytest 用例失败；使用 `--continue-on-error` 时会继续执行后续命令并在最后汇总失败。

**执行方式**

```bash
python -m pytest tests/test_10_quality.py --scope quality --continue-on-error -v
```

也可以通过 CPU 或 full scope 间接执行；如果 CPU compiler tests 需要使用当前环境中的 Python，需同步传入 `--python-bin`：

```bash
python -m pytest tests \
  --scope cpu \
  --python-bin "$(which python)" \
  --continue-on-error \
  -v
```

---

### 7.3 `test_20_cpu.py`

**整体介绍**

CPU 回归脚本，覆盖 submodule 初始化、CPU-only CMake configure/build、CTest、hetero layer 数据生成、FHE layer 检查、Python compiler tests 和 CPU E2E。MNIST/CIFAR-10 CPU examples 由 `test_70_performance.py` 的 `examples-cpu` benchmark 覆盖。

**测试用例**

#### `test_cpu_build`

- **功能**：准备 submodule 并执行 CPU-only 构建。
- **测试内容**：确认主仓库 submodule 和 lattisense/Lattigo 依赖可初始化，旧 `build-cpu` 会在配置前被清理，CPU-only CMake configure 与构建流程可在测试机上完整通过。
- **执行命令**：
  - `git submodule update --init --recursive`：初始化并更新主仓库所有 submodule，确保 CPU 构建所需源码依赖完整。
  - `git -C inference/lattisense submodule update --init fhe_ops_lib/lattigo`：单独初始化 lattisense 下的 Lattigo 依赖，避免 Go/FHE 后端源码缺失。
  - `rm -rf build-cpu`：删除已有 CPU 构建目录，避免旧 CMake cache 或旧构建产物影响本次回归。
  - `cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release`：生成 CPU-only Release 构建目录，验证 CMake 配置阶段可通过。
  - `cmake --build build-cpu -j$(nproc)`：并行构建 CPU 目标，验证核心 C++ 代码和依赖能完整编译链接。
- **输出日志**：`cpu.log`，submodule、清理、CPU configure 和 CPU build 命令按 step 合并写入同一个日志文件。
- **失败行为**：任一命令失败会使该 pytest 用例失败。

#### `test_cpu_ctest_and_core_paths`

- **功能**：执行 CPU 构建产物上的核心测试路径。
- **测试内容**：确认 CPU 构建产物中的基础 CTest、hetero layer 生成与读取链路、FHE layer sq 路径、Python compiler tests 和 CPU E2E 均可运行；MNIST/CIFAR-10 CPU examples 不在该用例中重复执行，由 performance scope 的 `examples-cpu` benchmark 覆盖。CPU E2E 使用 `./test_e2e '[batch][cpu]'` 单进程 batch 模式，低内存测试机可能被 OOM killer 杀掉；该情况应按测试机资源限制记录，不通过拆分 case 改变正式回归语义。
- **执行命令**：
  - 在 `build-cpu` 下执行 `ctest -R "^test_data_structs$" --output-on-failure`：只运行 `test_data_structs` CTest 目标，验证基础数据结构测试通过。
  - 设置 `PYTHONPATH` 和 `LATTI_HETERO_BASE_PATH` 后执行 `test_gen_layers.py TestLayerExport.test_sq`：生成 sq layer 相关 hetero 输出，验证 Python 生成脚本与输出路径可用。
  - 执行 `./test_fhe_layers_hetero 'sq*'`：运行 sq 相关 FHE layer C++ 测试，验证刚生成的 hetero 数据能被测试二进制读取。
  - 默认执行 `python-deps-check`：只 import `training/requirements.txt` 对应关键包，不下载依赖，用于快速确认 Python 编译器测试所需依赖已存在。
  - 如果显式传入 `--install-python-deps`，才会先执行 `${python_bin} -m pip install -r training/requirements.txt`：按需安装 Python 依赖，避免默认回归因网络或大包下载耗时过长。
  - 设置 `LATTI_E2E_BASE_PATH` 后执行 `${python_bin} -m pytest training/model_compiler/unittests/test_compiler.py -v`：运行模型编译器单测，验证 CPU E2E 输出路径和 compiler tests 可用。
  - 执行 `./test_e2e '[batch][cpu]'`：运行 CPU batch E2E 测试，验证 CPU 推理端到端路径。
- **输出日志**：`cpu.log`，CTest、hetero layer、Python 依赖检查、Python compiler tests 和 CPU E2E 命令按 step 合并写入同一个日志文件。
- **失败行为**：任一命令失败会使该 pytest 用例失败；使用 `--continue-on-error` 时会尽量收集更多失败日志。

**执行方式**

```bash
python -m pytest tests/test_20_cpu.py \
  --scope cpu \
  --python-bin "$(which python)" \
  --continue-on-error \
  -v
```

如果希望脚本自动安装 `training/requirements.txt`，显式增加：

```bash
python -m pytest tests/test_20_cpu.py \
  --scope cpu \
  --python-bin "$(which python)" \
  --install-python-deps \
  --continue-on-error \
  -v
```

完整 CPU scope 推荐执行整个目录，让 environment 和 quality 一起运行：

```bash
python -m pytest tests \
  --scope cpu \
  --python-bin "$(which python)" \
  --continue-on-error \
  -v
```

---

### 7.4 `test_30_gpu.py`

**整体介绍**

GPU 回归脚本，覆盖 CUDA arch 记录、HEonGPU configure/build/install、latti-ai GPU configure/build、GPU compiler tests、GPU E2E 和 mega ag 生成。GPU examples 由 `test_70_performance.py` 的 `examples-gpu` benchmark 覆盖。

**测试用例**

#### `test_gpu_build`

- **功能**：构建 HEonGPU 并构建启用 GPU 的 latti-ai。
- **测试内容**：确认测试机 CUDA arch 可采集，旧 HEonGPU `build` 和 latti-ai `build-gpu` 会在配置前被清理，HEonGPU configure/build/install 可通过，latti-ai GPU 构建可通过；如果显式传入 `--cccl-source-dir`，或默认路径 `/home/qianc/latti-ai-deps/cccl-src` 存在，会验证 HEonGPU CMake 使用仓库外 CCCL 源码目录，不再依赖网络下载。单独运行 GPU 用例时，`final-report.md` 不输出“基本信息”和“模块版本”，只输出“GPU 构建与核心验证”表格。
- **执行命令**：
  - `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`：采集 GPU 型号和 compute capability，确认 `--cuda-arch` 应与测试机 GPU 匹配。
  - 在 `inference/lattisense/HEonGPU` 下执行 `rm -rf build`：删除旧 HEonGPU 构建目录，避免旧 CMake cache 或旧构建产物影响本次回归。
  - 在 `inference/lattisense/HEonGPU` 下执行 HEonGPU CMake configure：配置 HEonGPU 构建目录，验证 CUDA compiler、CUDA arch、安装目录和 Thrust/CCCL 相关配置可用；当显式 `--cccl-source-dir` 或默认 `/home/qianc/latti-ai-deps/cccl-src` 解析到可用目录时追加 `-DFETCHCONTENT_SOURCE_DIR_CCCL=<path>`。
  - 在 `inference/lattisense/HEonGPU` 下执行 `cmake --build build -j$(nproc)`：构建 HEonGPU，验证 GPU FHE 后端依赖可编译。
  - 在 `inference/lattisense/HEonGPU` 下执行 `cmake --install build`：安装 HEonGPU 到本地 install 目录，验证后续 latti-ai GPU 构建可链接该依赖。
  - 执行 `rm -rf build-gpu`：删除旧 latti-ai GPU 构建目录，避免旧 CMake cache 或旧构建产物影响本次回归。
  - 执行 `cmake -B build-gpu -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=<cuda_arch> -DCMAKE_BUILD_TYPE=Release`：配置启用 GPU 的 latti-ai 构建目录，验证 GPU 编译选项和 CUDA arch 生效。
  - 执行 `cmake --build build-gpu -j$(nproc)`：构建 latti-ai GPU 目标，验证 GPU 路径完整编译链接。
- **输出日志**：`gpu.log`，CUDA arch、HEonGPU 清理/configure/build/install 和 latti-ai GPU 清理/configure/build 命令按 step 合并写入同一个日志文件。
- **失败行为**：未传 `--cuda-arch` 会直接 pytest fail；任一构建命令失败会使该 pytest 用例失败。

#### `test_gpu_tests_and_examples`

- **功能**：执行 GPU compiler tests、GPU E2E 和 MNIST/CIFAR-10 mega ag 生成。
- **测试内容**：确认 GPU 构建产物可支持 compiler tests、batch GPU E2E 和 MNIST/CIFAR-10 mega ag 任务生成；MNIST/CIFAR-10/ImageNet GPU examples 不在该用例中重复执行，由 performance scope 的 `examples-gpu` benchmark 覆盖。
- **执行命令**：
  - 设置 `LATTI_E2E_BASE_PATH` 后，在 `build-gpu/inference/unittests` 下执行 `${python_bin} -m pytest ../../../training/model_compiler/unittests/test_compiler.py -v`：运行模型编译器单测，验证 GPU E2E 输出路径和 compiler tests 可用。
  - 执行 `./test_e2e '[batch][gpu]'`：运行 GPU batch E2E 测试，验证 GPU 推理端到端路径。
  - 执行 `${python_bin} inference/interface/gen_mega_ag.py --task-dir examples/test_mnist/task`：为 MNIST 示例生成 mega ag 输入，验证示例前置任务生成脚本可用。
  - 执行 `${python_bin} inference/interface/gen_mega_ag.py --task-dir examples/test_cifar10/task`：为 CIFAR-10 示例生成 mega ag 输入，验证较大示例任务生成路径可用。
- **输出日志**：`gpu.log`，GPU compiler tests、GPU E2E 和 mega ag 生成命令按 step 合并写入同一个日志文件。
- **失败行为**：任一命令失败会使该 pytest 用例失败。

**执行方式**

如果测试机已经准备好默认 CCCL 源码目录 `/home/qianc/latti-ai-deps/cccl-src`，或接受 HEonGPU 按自身 FetchContent 逻辑处理 CCCL，可直接运行：

```bash
python -m pytest tests/test_30_gpu.py \
  --scope gpu \
  --cuda-arch 89 \
  --continue-on-error \
  -v
```

脚本会先检查默认目录是否存在；存在时会自动追加 `-DFETCHCONTENT_SOURCE_DIR_CCCL=/home/qianc/latti-ai-deps/cccl-src`。如需避免回归执行过程中依赖网络下载，但默认目录还未准备，可先在测试机准备 CCCL 源码缓存：

```bash
mkdir -p /home/qianc/latti-ai-deps
git clone https://github.com/NVIDIA/cccl.git /home/qianc/latti-ai-deps/cccl-src
cd /home/qianc/latti-ai-deps/cccl-src
git checkout e21d607157218540cd7c45461213fb96adf720b7
```

然后运行：

```bash
python -m pytest tests/test_30_gpu.py \
  --scope gpu \
  --cuda-arch 89 \
  --cccl-source-dir /home/qianc/latti-ai-deps/cccl-src \
  --continue-on-error \
  -v
```

---

### 7.5 `test_40_docker.py`

**整体介绍**

Docker 验证脚本，覆盖镜像构建和容器内 CPU 验证路径，重点补充 `main` 合并后 Docker 发布路径的本地或测试机验证。

**测试用例**

#### `test_docker_build_and_cpu_verify`

- **功能**：构建 `latti-ai:post-merge-test` 镜像，并在容器内执行 CPU 构建与基础测试。
- **测试内容**：确认 Dockerfile 可构建出测试镜像，容器内能完成 CPU configure/build、hetero layer 生成同步、基础 C++ 测试、FHE layer sq 路径和 Python compiler tests。
- **执行命令**：
  - `docker build -t latti-ai:post-merge-test .`：使用仓库根目录 Dockerfile 构建回归测试镜像，验证 Docker 构建链路和镜像依赖安装可通过。
  - `docker run --rm latti-ai:post-merge-test bash -c '...'`：启动一次性容器，在容器内执行 CPU 构建和核心验证，验证镜像内环境不仅能构建，还能运行基础测试。
- **容器内验证内容**：
  - 使用 `set -e`：确保容器内任一验证命令失败都会使 `docker-cpu-verify` 返回非 0，避免失败被后续命令掩盖。
  - 设置 `PYTHONPATH=/workspace:${PYTHONPATH:-}`：让容器内 Python 能导入仓库源码模块。
  - 设置 `LATTI_HETERO_BASE_PATH=/workspace/build/inference/hetero`：指定 FHE layer hetero 测试读取生成结果的位置。
  - 设置 `LATTI_E2E_BASE_PATH=/workspace/build/inference/hetero_e2e`：指定 compiler tests 和 E2E 测试使用的输出位置。
  - 在 `/workspace/build` 下执行 `cmake ..` 和 `make -j$(nproc)`：验证容器内 CPU 构建流程可完成。
  - 执行 `python test_gen_layers.py TestLayerExport.test_sq`：生成 sq layer 测试所需 hetero 输出。
  - 校验 `/workspace/build/build/inference/hetero` 已生成，并同步到 `/workspace/build/inference/hetero`：确保 `test_fhe_layers_hetero` 读取到生成结果。
  - 执行 `./test_data_structs`：验证基础 C++ 数据结构测试通过。
  - 执行 `./test_fhe_layers_hetero "sq*"`：验证容器内 FHE layer sq 路径可运行。
  - 执行 `python -m pytest training/model_compiler/unittests/test_compiler.py -v`：验证容器内 Python compiler tests 可运行。
- **输出日志**：`docker.log`，Docker build 和容器内 CPU 验证命令按 step 合并写入同一个日志文件。
- **失败行为**：Docker build 或容器内任一命令失败会使该 pytest 用例失败；不会因为后续命令通过而掩盖前面命令失败。

**执行方式**

```bash
python -m pytest tests/test_40_docker.py \
  --scope docker \
  --continue-on-error \
  -v
```

---

### 7.6 `test_50_lattisense.py`

**整体介绍**

lattisense standalone 验证脚本，用于避免 lattisense 子项目只被主工程间接覆盖，单独验证其 configure/build/CTest 路径。

**测试用例**

#### `test_lattisense_standalone_build_and_ctest`

- **功能**：独立构建并测试 `inference/lattisense`。
- **测试内容**：确认 lattisense 子项目脱离主工程间接路径后，会先清理旧 `build-lattisense`，再独立完成 CMake configure、build、CPU 测试数据生成和 CTest，覆盖 standalone 集成风险。单独运行 lattisense 用例时，`final-report.md` 不输出全局“基本信息”和“模块版本”表，但会显示 lattisense commit，并输出“lattisense standalone 构建与 CTest”表格。
- **执行命令**：
  - `rm -rf build-lattisense`：删除旧 lattisense standalone 构建目录，避免旧 CMake cache 或旧构建产物影响本次回归。
  - `cmake -S inference/lattisense -B build-lattisense -DLATTISENSE_BUILD_TESTS=ON`：为 lattisense 子项目单独生成构建目录，显式启用 lattisense 自带测试，验证 standalone configure 阶段可通过。
  - `cmake --build build-lattisense -j$(nproc)`：单独构建 lattisense，验证子项目离开主工程构建路径后仍可编译链接。
  - 在 `inference/lattisense/unittests` 下执行 `${python_bin} test_cpu_bfv.py`：生成 BFV CPU 测试数据，供 `test_cpu_bfv` 读取 `task_signature.json` 等文件。
  - 在 `inference/lattisense/unittests` 下执行 `${python_bin} test_cpu_ckks.py`：生成 CKKS CPU 测试数据，供 `test_cpu_ckks` 读取 `task_signature.json` 等文件。
  - 在 `build-lattisense` 下执行 `ctest --output-on-failure`：运行 lattisense 自带 CTest，验证 standalone 测试集通过。
- **输出日志**：`lattisense.log`，清理、configure、build、测试数据生成和 CTest 命令按 step 合并写入同一个日志文件。
- **失败行为**：清理、configure、build 或 CTest 任一失败会使该 pytest 用例失败。

**执行方式**

```bash
python -m pytest tests/test_50_lattisense.py \
  --scope lattisense \
  --python-bin "$(which python)" \
  --continue-on-error \
  -v
```

---

### 7.7 `test_60_docs_consistency.py`

**整体介绍**

文档一致性辅助脚本，不自动判断文档内容是否完全正确，而是检查回归方案要求的关键文档和配置输入是否存在，并生成可人工填写的文档一致性检查清单。

**测试用例**

#### `test_release_documentation_inputs_exist`

- **功能**：检查关键文档和 Dockerfile 是否存在。
- **测试内容**：确认 release 回归需要人工核对的 README、build guide、API、examples、lattisense 文档和 Dockerfile 仍在预期路径，避免文档一致性检查清单引用失效路径。
- **检查路径**：
  - `README.md`：确认项目主 README 存在，供人工核对 release 说明、安装和使用入口是否同步。
  - `docs/en/build-guide.md`：确认英文构建指南存在，供人工核对构建命令、依赖和平台说明。
  - `docs/en/APIs_Reference.md`：确认英文 API 参考存在，供人工核对接口变更是否同步。
  - `examples/README.md`：确认 examples 文档存在，供人工核对示例运行方式和支持范围。
  - `inference/lattisense/README.md`：确认 lattisense 英文 README 存在，供人工核对子项目说明。
  - `inference/lattisense/README_zh.md`：确认 lattisense 中文 README 存在，供人工核对子项目中文说明。
  - `Dockerfile`：确认 Dockerfile 存在，供人工核对容器构建说明和实际 Docker 验证路径。
- **输出**：pytest assertion 结果；缺失文件会在失败消息中列出。
- **失败行为**：任一必需路径缺失会使该 pytest 用例失败。

#### `test_write_documentation_consistency_checklist`

- **功能**：生成文档一致性人工检查清单。
- **测试内容**：确认脚本能在日志目录写出面向人工评审的清单，覆盖 CHANGELOG、README、build guide、API、examples、lattisense 文档和 Dockerfile 等同步检查项。
- **输出文件**：`docs-consistency-checklist.md`
- **输出内容**：CHANGELOG、README、build guide、API、examples、lattisense 文档、Dockerfile 等检查项。
- **失败行为**：如果清单文件无法写入或不存在，pytest 用例失败。

**执行方式**

```bash
python -m pytest tests/test_60_docs_consistency.py --scope docs -v
```

---

### 7.8 `test_70_performance.py`

**整体介绍**

性能回归 benchmark 执行脚本。performance scope 会执行指定 benchmark 模块，并在 pytest session 结束时把 benchmark step、性能指标和 baseline 对比统一写入 `final-report.md`。

**测试用例**

#### `test_run_performance_benchmarks`

- **功能**：运行性能 benchmark，并让 session 级报告生成逻辑把性能结果合并进 `final-report.md`。
- **测试内容**：确认 CPU/GPU 算子、convolution CPU/GPU 和 examples CPU/GPU 等性能路径可按模块执行，重复运行后能采集 wall time、解析关键指标，并与 baseline 日志目录自动生成性能对比结论；只有 examples 性能测试会额外采集单 case OpenMP线程数、扣减运行前系统基线后的 CPU 内存增量峰值和 GPU 显存增量峰值（仅 `examples-gpu`）。
- **模块**：
  - `cpu`：分 step 运行 lattisense CPU 算子 benchmark：BFV mult、CKKS mult、BFV rotate。
  - `gpu`：分 step 运行 lattisense GPU 算子 benchmark：BFV mult、CKKS mult、BFV rotate，需要 `--cuda-arch`。
  - `conv-cpu`：运行 convolution CPU benchmark。
  - `conv-gpu`：运行 convolution GPU benchmark，需要 `--cuda-arch`。
  - `examples-cpu`：分别直接运行 MNIST 和 CIFAR-10 的 `./examples/inference` CPU 示例耗时，并记录单个 `inference` 计算进程的 OpenMP线程数和 CPU 内存增量峰值。
  - `examples-gpu`：分别直接运行 MNIST、CIFAR-10 和 ImageNet 的 `./examples/inference --gpu` 示例耗时，并记录单个 `inference` 计算进程的 OpenMP线程数、CPU 内存增量峰值和 GPU 显存增量峰值，需要 `--cuda-arch`。
- **输出文件**：
  - `performance.log`：原始命令输出，包含每次 repeat 的 `wall_seconds`；examples benchmark 额外包含 `resource_openmp_actual_threads`、`resource_memory_delta_peak_mib` 和 `examples-gpu` 的 `resource_gpu_memory_delta_peak_mib`。
  - `performance-resource-metrics.jsonl`：仅运行 `examples-cpu` 或 `examples-gpu` 时生成；每个 examples benchmark case 的每次 repeat 一行 JSON，包含 step、repeat、命令、退出码、wall time、OpenMP线程数、CPU 内存增量峰值和 GPU 显存增量峰值（仅 `examples-gpu`）。
  - `final-report.md`：包含 benchmark step 执行结果、性能指标、examples 资源增量和 baseline 对比。
- **失败行为**：未知模块、GPU 模块缺少 `--cuda-arch`、`--benchmark-repeat < 1` 或关键命令失败时 pytest 用例失败；失败时仍会在 pytest session 结束时基于已记录 step 生成 `final-report.md`。

**执行方式**

运行全部 benchmark：

```bash
python -m pytest tests/test_70_performance.py \
  --scope performance \
  --benchmark-modules all \
  --benchmark-repeat 3 \
  --cuda-arch 89 \
  --continue-on-error \
  -v
```

只运行部分模块：

```bash
python -m pytest tests/test_70_performance.py \
  --scope performance \
  --benchmark-modules cpu,conv-cpu,examples-cpu \
  --benchmark-repeat 3 \
  -v
```

## 8. 输出产物

默认输出到：

```text
logs/YYYYMMDD/HHMMSS/
```

主要产物：

- `*.log`：模块级合并日志，记录 step 原始输出、命令、工作目录、退出码和耗时。
- `summary.md`：pytest session 结束时自动生成的命令结果汇总。
- `final-report.md`：pytest session 结束时自动生成的回归测试报告；environment/main-flow/full/performance 等回归会自动填充测试人员、测试开始/结束时间、lattisense/HEonGPU/Lattigo commit；quality/GPU/lattisense 单模块运行会生成对应专项报告；其中“建议测试结论”主要基于命令执行状态，CI 结果、文档一致性人工检查和性能“数据不足”影响需由测试人员补充评审。
- `performance.log`：performance scope 执行 benchmark 时生成，记录 benchmark 原始命令输出；benchmark step、性能指标、examples 资源增量和 baseline 对比统一写入 `final-report.md`。
- `performance-resource-metrics.jsonl`：仅运行 `examples-cpu` 或 `examples-gpu` 时生成，每个 examples benchmark case 的每次 repeat 记录一行 JSON，包含 OpenMP线程数、扣减运行前系统基线后的 CPU 内存增量峰值；`examples-gpu` 额外记录扣减运行前系统基线后的显存增量峰值。
- `docs-consistency-checklist.md`：文档一致性检查清单，仅 docs/full scope 生成。

## 9. 失败处理规则

- 默认情况下，同一个 pytest 用例中的关键命令失败后，后续命令不再继续执行，该用例失败。
- 加上 `--continue-on-error` 后，同一个 pytest 用例会继续执行后续命令，最后统一汇总关键失败并使该用例失败。
- `CommandStep(critical=False)` 的命令失败会记录为 `NON_BLOCKING_FAIL`，不直接阻塞当前用例。
- `nvidia-smi-gpu-info` 在环境采集阶段是非阻塞项；无 GPU 的机器会记录日志，但不阻塞 environment scope。
- GPU scope 必须传入 `--cuda-arch`；否则 `test_gpu_build` 会直接失败并提示示例参数。
- performance scope 会执行 benchmark 并自动生成性能对比报告。
- benchmark 模块 `gpu`、`conv-gpu`、`examples-gpu` 必须传入 `--cuda-arch`；否则 `test_run_performance_benchmarks` 会失败并提示补参数。
- `--benchmark-repeat` 必须大于等于 1；未知 benchmark 模块会直接失败并列出合法模块。

