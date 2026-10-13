# 密钥流式生成落盘验收记录

2026-10-13，按 [implementation plan](streaming_eval_context_file_implementation_plan.md) 完成 M1–M7。C++ `InferenceClient::setup()` 和 Node `await client.setup()` 默认逐键生成评估键，成功返回前发布 `client_dir/eval_context.bin`。服务端使用 `import_eval_context_file(path)`，随后仍走原 deserializer。独立客户端实测满足密钥生成阶段峰值 RSS 显著降低的首要条件。

## 实现与提交

三个仓库均直接在本地 `feat/streaming-key-generation` 分支开发，未使用 worktree 或 push。顶层 gitlink 指向 LattiSense `d1203df`，LattiSense 的嵌套 gitlink 指向 Lattigo `cc39b5ae`。

| 步骤 | 仓库 | 实现提交 |
| :--- | :--- | :--- |
| M1 | Lattigo | `a07aa5f5`：普通 CKKS、逐键 writer、临时文件发布、错误传播及协议/失败测试 |
| M2 | Lattigo | `0891370c`：BTP 参数、PK、完整 Galois 并集、DtS/StD 与 bootstrap 测试 |
| M3 | LattiSense | `9976894`：C++ 流式创建/导出薄包装、子模块指针与回归测试 |
| M4 | Latti-AI | `62d6edb`：服务端文件导入、读取失败与大小边界测试 |
| M5 | Latti-AI | `a243d8f`：默认 setup、客户端恢复后文件导出、示例迁移及最终子模块指针 |
| M6 | Latti-AI | `2e0e24b`：Node 默认 Promise、文件发布和失败 reject 用例 |
| M7 | Latti-AI | 本验收记录所属提交：独立客户端测量工具、原始日志及完成状态 |

必要的补充修复也已分别提交：Lattigo `57ec882f` 使没有评估键 bundle 的 BTP 客户端可保存 SK/PK；`c73b0fab` 由现有构建重新生成已被跟踪的 CGO 头；`cc39b5ae` 修复恢复后客户端的浅拷贝，以实际 `bootstrapper != nil` 判断是否复制 bootstrapper。LattiSense `d1203df` 更新指针并验证恢复、浅拷贝加密和重新导出。顶层 `9db7840` 将剩余 C++ e2e 调用及 README 示例迁移到文件导入。

实现复用现有单键生成器、serializer、句柄和错误机制，核心 `rlwe/keygenerator.go`、`rlwe/bit_stream.go`、bootstrapper 算法均未修改。流式过程复用一个局部 `bytes.Buffer`，没有全套键预分配或整文件 Go/C++ 序列化副本；BTP DtS/StD 沿用成对生成。客户端保留 SK/PK 和加解密状态，未挂载 RLK、RTKs、完整 BTP evaluation bundle 或 bootstrapper。生产路径没有新增强制 GC、公开遥测接口、独立 keypair-only API 或 Node 模式开关。

## 客户端内存实测

原始数据保存在 [memory 目录](streaming_key_generation_results/memory/environment.json)，包含环境、每阶段 JSONL 和六次 `/usr/bin/time -v` 输出。环境为 Linux 5.15.0-124、glibc 2.39、Intel Xeon Gold 5415+、16 个逻辑 CPU、约 251.37 GiB 内存。实际模块 toolchain 为 Go `1.24.11`，固定 `GOMAXPROCS=16`、`GOGC=100`、`GOMEMLIMIT=off`；采样周期为 250 ms，另在每把 RLK/RTK 生成及写出、封装键生成和文件发布时采样。

测量工具通过临时 Go source overlay 插入采样并调用真实 SDK 入口，不修改仓库内生产源码。每个模式独立运行一个进程，未包含服务端导入或推理：

- `batch-generate`：原随机构造及批量旋转键生成，作为旧密钥生成基线；BTP 包含原构造器的 bootstrapper 初始化。
- `batch`：同一旧路径再调用原完整内存 serializer，并以 native `malloc`/`memcpy` 复现 `export_raw_data` 的完整 `Bytes` 分配和复制。
- `stream`：真实 `StreamSetupCkksContextToFile` / `StreamSetupCkksBtpContextToFile`，从客户端初始化到文件关闭、Rename 发布。

旧路径的 native copy 是 SDK 层的 C++ 导出开销重放，不是完整 Node 进程测量。RSS 包括参数准备、测量器、未回收对象和最后诊断 GC 的整个进程；`result.elapsed_seconds` 则只统计初始化/生成/导出阶段。诊断 GC 在该阶段结束后执行，生产路径及逐键边界均不强制 GC。表中时间取 `/usr/bin/time` 的完整进程 wall time。

| 参数 / 模式 | 峰值 RSS（KiB） | 峰值 RSS（GiB） | wall time（秒） |
| :--- | ---: | ---: | ---: |
| CKKS `PN16QP1761` / batch-generate | 11,086,724 | 10.573 | 48.37 |
| CKKS / batch + 内存导出及 native copy | 27,661,492 | 26.380 | 65.68 |
| CKKS / stream | 2,047,028 | 1.952 | 58.82 |
| BTP `N16QP1546H192H32` / batch-generate | 14,119,632 | 13.466 | 6.51 |
| BTP / batch + 内存导出及 native copy | 21,462,912 | 20.469 | 16.84 |
| BTP / stream | 1,194,508 | 1.139 | 39.38 |

相比旧**密钥生成**基线，CKKS 峰值下降 **81.5%**，BTP 下降 **91.5%**；相比旧生成加完整导出，分别下降 **92.6%**、**94.4%**。流式时间包含落盘，旧内存导出不包含落盘。fork 的旧 BTP 路径并行批量生成，新的顺序逐键生成耗时明显增加；本次按降低峰值内存和最小修改优先接受这一取舍。这是固定环境下各模式一次运行的结果，不作为其他参数或部署环境的内存保证。

三个模式的 metadata 已逐项核对一致：`N=65536`、`LogSlots=15`、全部 Q/P、Galois 集合、Go/GC/线程配置和参数 SHA256 均相同；旧 context 的实际 Galois key map 也逐项检查。CKKS 有 30 把 RTK，BTP 有 50 把（含共轭）。metadata 中为比较而排序，不改变生产路径键顺序。文件尺寸与旧完整序列化长度一致：

| 参数 | Q/P 数量 | Scheme SHA256 | 文件字节数 |
| :--- | :--- | :--- | ---: |
| CKKS `PN16QP1761` | 34 / 4 | `b1bbc7102a5302fc713eb0448a8f81d322988e2b433e009ba4670e6a5a4cb6db` | 4,072,914,629 |
| BTP `N16QP1546H192H32` | 25 / 5 | `001ee01901b7ad925029d6f6121660e7047e980d7acf92a3e18af498a88115b6` | 3,329,929,205 |

BTP 参数 SHA256 为 `95519820d481212a148b9a5fe2fe6c04384a7806507631af9c34b14dfd48a2d8`。尺寸只用于辅助复核；无 SK 的结论来自协议结构测试，而非随机文件字节搜索或尺寸相等。

| 流式方案 | buffer 最大实际容量（MiB） | 发布时 HeapAlloc / HeapInuse（MiB） | 诊断 GC 后 HeapAlloc / HeapInuse（MiB） | 采样 HeapAlloc 峰值（MiB） |
| :--- | ---: | :--- | :--- | ---: |
| CKKS | 192 | 2377.72 / 2378.34 | 810.65 / 811.43 | 2377.72 |
| BTP | 80 | 1225.81 / 1226.42 | 575.53 / 576.23 | 1568.72 |

buffer 容量包含现有 `bytes.Buffer` 的扩容余量，不能用单键压缩长度替代。堆采样计入 key generator 工作区、加解密状态、DtS/StD 成对生成和待回收对象；引用释放不等于物理内存立即回收。Go 堆与 RSS 是不同指标，部分 HeapAlloc 样本高于 RSS，不能将二者直接相加或互相替代。实测在默认 GC 下已显著降峰，无需增加局部 GC 策略。

测量时 Lattigo 为 `cc39b5ae`，顶层为 `2e0e24b`；后续 `9db7840` 仅迁移辅助调用和 README，未改变被测 SDK。overlay 测量入口 SHA256 为 `13ca590e698c53a6b3fc54ac8c4132bdec30b1961047de2a11344e7268ef6721`，与提交的 Go 探针一致。可在 Linux 下从顶层复现，需现有 Go/CGO、C 编译器、`ar`、Python 3 和 GNU time：

```bash
python3 examples/measure_streaming_key_memory.py \
  --output-dir /tmp/latti-key-memory-results \
  --ckks-n 65536 --gomaxprocs 16
```

脚本在临时目录编译探针，依次完成六个独立进程，并清理评估键文件。`--schemes ckks --ckks-n 4096` 可用于工具的轻量试跑；BTP 保持现有默认 `N=65536`。提交的原始结果目录仅含日志和参数，不包含密钥、临时文件或二进制。

## 功能、失败边界与精度

| 验收项 | 结果与证据 |
| :--- | :--- |
| Go CKKS 协议与功能 | [Go 日志](streaming_key_generation_results/tests/go-streaming.txt)：BigEndian 参数长度、完整读完、无 SK/PK、集合一致，使用同一 SK/密文比较正负旋转、共轭和重线性化 |
| Go BTP 协议与 bootstrap | 同一日志：输入 context 含 SK，仍固定 `has_sk=0`，PK/字段顺序/完整集合、sparse/full DtS/StD、ephemeral weight 32 和 0、原 decoder 与 bootstrap 均通过 |
| 文件失败处理 | Go 注入短写、写入、Close、Rename、清理和 panic 失败，验证旧目标保留、临时文件清理或附带临时路径；另使用 `/dev/full` 验证 ENOSPC 写错误 |
| 恢复后 BTP 浅拷贝 | [Go 恢复日志](streaming_key_generation_results/tests/go-restored-btp.txt)：没有 bootstrapper 的恢复客户端可浅拷贝并加解密 |
| C++ CKKS/BTP 包装 | [C++ 日志](streaming_key_generation_results/tests/cpp-streaming.txt)：13 个断言、2 个用例，覆盖流式创建、导出、导入、bootstrap、SK/PK 持久化及恢复后重新导出；普通 CKKS 原完整 context 内存接口也通过 |
| 顶层文件导入与默认 setup | [CTest 日志](streaming_key_generation_results/tests/context-file.txt)：缺失/空文件、负/过大长度、短读、正常读取、默认文件发布、释放后拒绝导出和恢复后重新导出均通过 |
| Node 默认流程 | [普通 CKKS](streaming_key_generation_results/tests/node-ckks.txt) 与 [BTP](streaming_key_generation_results/tests/node-btp.txt)：实际重编译 addon，setup 返回 Promise，发布后 resolve，加解密、SK/PK 保存恢复、release/reload 及发布失败 reject 均通过，失败保留旧目标并清理临时文件 |
| CIFAR-10 CPU BTP e2e | [CTest 日志](streaming_key_generation_results/tests/e2e-initial.txt)：真实文件 setup/import，`output` 的全部 10 个值有限，最大绝对误差 `0.00044756 < 0.1`，通过，409.25 秒 |
| MNIST CPU 普通 CKKS e2e | [最终 CTest 日志](streaming_key_generation_results/tests/e2e-mnist.txt)：真实文件 setup/import，全部 10 个值有限，最大绝对误差 `0.00000747 < 0.1`，通过，9.08 秒 |
| e2e 辅助调用编译 | [构建日志](streaming_key_generation_results/tests/e2e-build.txt)：`test_e2e_single_layer` / `test_e2e_multiple_layer` 已针对文件导入重新编译；未把这两个目标的编译报告成模型用例运行通过 |

初次组合 CTest 日志中的 MNIST 因缺少生成的 `task_signature.json` / `mega_ag.json` 在模型加载时失败。随后使用已有 `gen_mega_ag.py` 补齐 fixture 后单独重跑通过，见最终日志；生成任务文件未提交。另按两个任务的 `task_output_id` 核对日志中的输出键、加密/明文数量和完整比较表，确认 10 行数值均有限且满足容限。

`tests/` 下的控制台日志仅规范化了换行和行尾空白；`memory/` 保留测量输出原文。

文件读取成功不代表现有 Go parser 能可靠拒绝任意损坏 payload，本次保持设计规范限定的解析承诺。服务端仍全量读入文件，不把服务端内存计入客户端降峰结论。旧内存接口继续供完整评估键 context 使用，新的默认 setup 后使用文件接口。Python 绑定及 GPU 模型回归不在本次计划范围内。

重编译和验证使用现有工具（从顶层运行）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_SDK_ENABLE_GPU=OFF
cmake --build build --target go_libs fhe_ops_lib inference streaming_context_file_test -j 4
cmake --build build --target test_e2e_single_layer test_e2e_multiple_layer -j 4
python3 inference/interface/gen_mega_ag.py --task-dir examples/test_mnist/task
ctest --test-dir build -R '^streaming-context-file$' -V
ctest --test-dir build -R '^example-mnist-cpu$' -V
ctest --test-dir build -R '^example-cifar10-cpu$' -V

npm --prefix inference/interface/node run rebuild -- \
  --CDFHE_OPS_LIB="$PWD/build/inference/lattisense/fhe_ops_lib/libfhe_ops_lib.so"
LD_LIBRARY_PATH="$PWD/build/inference/lattisense/fhe_ops_lib:$PWD/inference/lattisense/fhe_ops_lib/lattigo/go_sdk${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  npm --prefix inference/interface/node test
LD_LIBRARY_PATH="$PWD/build/inference/lattisense/fhe_ops_lib:$PWD/inference/lattisense/fhe_ops_lib/lattigo/go_sdk${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  npm --prefix inference/interface/node test -- --btp

cmake -S inference/lattisense -B build/streaming-sdk-tests \
  -DCMAKE_BUILD_TYPE=Release -DLATTISENSE_BUILD_MODE=SHARED \
  -DLATTISENSE_BUILD_TESTS=ON -DLATTISENSE_BUILD_EXAMPLES=OFF \
  -DLATTISENSE_ENABLE_GPU=OFF -DLATTISENSE_ENABLE_FPGA=OFF
cmake --build build/streaming-sdk-tests --target test_lattigo -j 4
build/streaming-sdk-tests/unittests/test_lattigo '[streaming]'
```

Go SDK 测试需要既有 C++ bridge 的链接符号。使用单独静态归档后执行：

```bash
cc -c inference/lattisense/fhe_ops_lib/bridge.c -o /tmp/latti-streaming-bridge.o
ar rcs /tmp/liblatti-streaming-bridge.a /tmp/latti-streaming-bridge.o
cd inference/lattisense/fhe_ops_lib/lattigo
CGO_LDFLAGS='-L/tmp -llatti-streaming-bridge' go test ./go_sdk -run '^TestStreaming' -count=1 -v
```

## 后续远程发布与 CPU/GPU 构建验证

按后续要求，三个仓库分别执行 `git push -u origin HEAD`，远程均已创建 `feat/streaming-key-generation`，本地 upstream 均为各自的 `origin/feat/streaming-key-generation`。LattiSense 新增提交 `e4e17e2`，在 `.gitmodules` 中配置 Lattigo 的目标分支；顶层同步子模块指针和 LattiSense 的分支配置。

README 的两种构建只在 clone 命令中指定功能分支，保留原有 submodule 初始化与 CMake 编译流程。共用的 `cmake/BindSubmoduleBranch.cmake` 在配置阶段将已初始化的 LattiSense/Lattigo 附着到对应本地分支并设置 upstream，保留当前锁定的提交。已有同名分支若指向不同提交则报错；已经正确绑定时只读 Git 状态。临时 Git fixture 验证了 detached HEAD 自动绑定、远程分支领先时仍保留锁定提交、重复配置不写 index/config，以及不同提交的已有分支不会被重置。

当前服务器为 NVIDIA RTX 5880 Ada Generation，compute capability `8.9`，CUDA toolkit `12.2.140`、驱动 `580.76.05`。直接在主机上使用架构 `89` 完成 GPU 模式完整编译，实际生成 CUDA 对象、`libgpu_mega_ag_runner.so` 和推理示例；CPU 模式也重新配置并完整构建通过。最终两种配置日志都确认 LattiSense 为 `e4e17e2`、Lattigo 为 `cc39b5ae`，各自 tracking 远程同名分支。

| 验证 | 结果 / 日志 |
| :--- | :--- |
| CPU 配置及完整构建 | 通过：[配置](streaming_key_generation_results/tests/branch-cpu-configure.txt)、[构建](streaming_key_generation_results/tests/branch-cpu-build.txt) |
| GPU 配置及完整构建 | 通过：[配置](streaming_key_generation_results/tests/gpu-configure.txt)、[构建](streaming_key_generation_results/tests/gpu-build.txt) |
| MNIST GPU 端到端 | 通过：[CTest](streaming_key_generation_results/tests/e2e-mnist-gpu.txt)，实际选择 GPU，全部 10 个输出完整且有限，最大绝对误差 `0.00000753 < 0.1`，6.51 秒 |

GPU 构建使用单独输出目录，保留 CPU 构建产物。复现命令：

```bash
cmake -S . -B build/gpu -DINFERENCE_SDK_ENABLE_GPU=ON \
  -DLATTISENSE_CUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build/gpu -j 16
ctest --test-dir build/gpu -R '^example-mnist-gpu$' -V
```

## ResNet-20 GPU 端到端与密钥生成阶段峰值

2026-10-13，使用 `examples/test_cifar10/task` 的 ResNet-20、RTX 5880 Ada、架构 `89` 的 GPU 构建完成加密输入、服务端 GPU 推理和客户端解密验证。参数为默认 BTP `N16QP1546H192H32`、`N=65536`；客户端仍调用默认 `client.setup()`，逐键生成并发布 `eval_context.bin`，服务端通过文件导入。固定 `GOMAXPROCS=16`、`GOGC=100`、`GOMEMLIMIT=off`，未加入强制 GC。

| 指标 | 实测 |
| :--- | ---: |
| 密钥生成及落盘阶段峰值主机 RSS | **1,209,668 KiB = 1181.316 MiB = 1.154 GiB** |
| setup 前进程峰值 RSS | 14,752 KiB |
| 默认 setup 耗时（含密钥生成、序列化和文件发布） | 38.4647 秒 |
| evaluation context 文件大小 | 3,329,929,205 字节 |
| GPU 加密推理耗时（示例输出） | 6.608 秒 |
| 完整端到端进程耗时 | 60.07 秒 |
| 完整端到端进程峰值 RSS（含服务端导入和推理） | 22,549,752 KiB = 21.505 GiB |
| `output` 数量 | 10，全部完整且为有限数值 |
| 最大绝对误差 | **0.00042538 < 0.1**，通过 |

测量工具 [measure_example_key_memory.py](../examples/measure_example_key_memory.py) 临时复制已有 `examples/inference.cpp`，仅在 `client.setup()` 前后加入计时和 Linux `getrusage(RUSAGE_SELF).ru_maxrss`，沿用当前构建的编译/链接参数及真实 GPU SDK 库。峰值在 setup 返回后、加密输入及创建/导入服务端 context 前读取，不受后续服务端内存峰值影响；不会将整个端到端进程的峰值当作密钥生成峰值。`ru_maxrss` 是进程启动至该边界的高水位，包含 setup 前的少量加载和配置开销；本次 setup 前峰值仅 14,752 KiB。该指标是主机 RSS；密钥生成仍由 Go 在 CPU 上执行，GPU 用于服务端推理。

仓库中的原示例、默认接口和 SDK 实现未修改。测量时顶层为 `3512cfa`、LattiSense 为 `e4e17e2`、Lattigo 为 `cc39b5ae`。环境、源文件/临时测量源及实际 SDK 库 SHA256、编译/链接参数记录在 [environment.json](streaming_key_generation_results/resnet20-gpu/environment.json)；[阶段峰值](streaming_key_generation_results/resnet20-gpu/keygen-memory.json)、[完整推理日志](streaming_key_generation_results/resnet20-gpu/inference.txt)、[进程 GNU time 日志](streaming_key_generation_results/resnet20-gpu/process.time.txt) 和 [输出核对](streaming_key_generation_results/resnet20-gpu/verification.json) 已保存。原始日志确认测量标记先于 Step 2/3，服务端选择 GPU，所有 10 行输出满足容限。

复现需要 Linux 和已有 CMake Makefiles 构建（提供 `compile_commands.json` 及目标的 `link.txt`）：

```bash
python3 examples/measure_example_key_memory.py \
  --build-dir build/gpu \
  --task-dir examples/test_cifar10/task \
  --input examples/test_cifar10/task/client/img.csv \
  --gpu --output-dir /tmp/latti-resnet20-gpu-key-memory
```

临时测量源和二进制会自动清理。示例发布的评估键文件在本次记录完成后清理，未纳入提交。
