# TAA v6 实现验收记录

日期：2026-11-06。状态：六份规范的功能验收完成，六任务 14 配置全部通过；最终 clean revision 发布包及两任务客户端复测完成。Lattigo 全测试仍有已复现的基线 hash 断言失败，详见第 7 节。

实施要求见 [设计总则](../spec/taa_v6_adaptation_design.md)。当前测试均在 `feat/taa-adapt` 执行，任务生成、密钥、推理与浏览器产物写入临时目录，没有覆盖 `runs/`。工作区原有的 `docs/research/taa_wasm_pn15_memory_probe.md` 和根目录 profiling CSV 保留原状，不纳入本次实现提交。

## 1. 构建与证据边界

本轮使用 Go 1.27.3、Node v24.16.0、Python 3.10.14、CMake 3.31.10，GPU 为两张 NVIDIA RTX 5880 Ada（各 49,140 MiB）。WASM 跟随锁定 Lattigo 的 Go module，附带实际 Go toolchain 的 `wasm_exec.js`。测试用 Chromium 来自 `/tmp/taa-chromium/chrome-linux/chrome`。

长期测量与最初两任务端到端使用 `/tmp/taa-final-package-extracted/latti-client-wasm-0.2.0`，runtime 版本 `0.2.0`、Worker 协议 `2`、资源集 build ID 为 `6fa1bf4640914f05d17bf83b2403429ab5ed65d99c1e472eca835a6030a5ebec`。对应工作交付包 `/tmp/taa-final-delivery/latti-delivery-0.2.0.zip` 已逐项核对文件哈希并解压测试，其 manifest 如实记录当时的 dirty 源码。

最终包从独立 `/tmp/taa-release-source` checkout 构建，runtime、两任务和数据包位于 `/tmp/taa-release-0.2.0`，解压目录为 `/tmp/taa-release-0.2.0-extracted`。四仓库 manifest 的 `dirty` 均为 false：latti-ai `fc87f9afbf86327d803ea38b5b9cc1aacef845da`、LattiSense `77ab72657f6f282ee30987cfcf1b62105262f2d4`、Lattigo `11cd8312f91fb345040d8c5d5e911e6e88afe4f6`、GPU core `306f89fc7d453dba7195522fa9efdbee19ce3492`。子模块按 gitlink detached checkout；开发工作区仍在 `feat/taa-adapt`。后续验收文档提交不改变此发布源码版本。

最终 build ID 为 `81cffcdbdbf28bfa23aa198382d97c230899b84251bf80c6da26b6829a689418`。提交 LattiSense 后，默认 profile 的来源 revision 更新，因此 build ID 和嵌入它的 WASM 字节相应变化。构建检查逐项比较 profile 的参数/键能力与来源参数表哈希，并以旧 profile 字节重算全部运行时代码哈希，精确得到原 build ID；除来源信息及派生 build ID 外，密码学和客户端源代码与已测工作包一致。

两层 `SHA256SUMS`、runtime manifest 与独立 task/data content-manifest 全部校验通过。干净解压包的真实 Worker/OPFS 故障套件、四任务模块加载、9,216 个 CIFAR 预处理值及示例页面启动/关闭已通过；两任务完整客户端复测另行记录，复用本轮原生输出时明确标明，不计为新一轮 GPU 推理。构建/检查日志为 `/tmp/taa-clean-release-build.log`、`/tmp/taa-clean-delivery-build.log`、`/tmp/taa-clean-release-browser.log`、`/tmp/taa-clean-release-delivery-browser.log`。

最终 `WASM-0.2.0.zip` SHA256 为 `0fa1a800007cb226bbec4ba9be13e5f20a9f1980c6e1985dbee1c10e0d8919fe`；`latti-delivery-0.2.0.zip` SHA256 为 `0ecfc4b7b8c6f342d4bee5615c23713e486859e1e73dd4a1760f9fbe873e4a9b`。

## 2. 流式密钥与原生发布

生产参数来自 `inference/lattisense/frontend/parameter.json` 的 CKKS15 CI 条目，默认 rotations 与 bootstrapping 均开启。独立 native 进程固定 `GOMAXPROCS=8`、`GOGC=100`：

| 路径 | eval 文件字节 | eval 完成累计时间 | 进程 RSS 高水位 |
| --- | ---: | ---: | ---: |
| bulk | 12,069,339,185 | 128,876 ms | 11,934,716 KiB |
| stream | 12,069,339,185 | 118,248 ms | 730,772 KiB |

两条路径 full 均为 12,583,830 字节，恢复后逐项比较原 SK/PK。此表的 RSS 是独立进程高水位；阶段末 `HeapAlloc` 只是时点值，不能充当阶段峰值。

另一次使用同参、同线程/GC 的独立进程补测加入 10 ms 阶段采样，得到以下观测最大值。两进程同时执行，包含采样和磁盘竞争，耗时不与上表作性能对比；stream 的生成与序列化本来就是交错阶段。

| 阶段 | stream HeapAlloc 最大字节 / RSS KiB | bulk HeapAlloc 最大字节 / RSS KiB |
| --- | ---: | ---: |
| SK/PK | 67,159,968 / 79,416 | 67,154,480 / 79,420 |
| bulk keygen | 不适用 | 12,197,932,408 / 11,955,720 |
| eval 生成/写出 | 1,011,178,528 / 1,007,476 | 12,303,824,360 / 12,054,640 |
| full 写出 | 64,706,464 / 380,664 | 64,710,496 / 12,040,480 |
| restore | 122,083,632 / 347,900 | 122,046,200 / 12,040,524 |

RSS 是阶段内绝对占用，包含 Go 保留的前序堆空间；尤其 bulk full/restore 的高 RSS 不表示这两个阶段新分配了 12 GB。GC 后两路径活跃 HeapAlloc 均回到约 64.5 MB。日志为 `/tmp/taa-native-{bulk,stream}-phases.{log,time}`；这是观测最大值，保留两次 stream 测量差异，不把先前较小的 RSS 当作固定内存上限。

当前 `clientcontext`、无 cgo 的 WASM core、Go SDK、buffer 与流式 BTP 专项通过。BTP 覆盖 standard 同 N/跨 N、CI ring-swap、encapsulation 0/8。生产 eval 独立顺序解码了 53 把 bootstrapping 密钥，逐把核对 capability 的 level、degree、decomposition；最大单把序列化密钥为 207,621,536 字节。profile ID 绑定展开后的 residual/boot 键描述，Q/P 与全部 64 位字段按实际值匹配；CIFAR 和 Retina 当前生成器输出的绑定测试通过。

`test_streaming_context` 通过 106 项断言，覆盖文件默认 setup、原 v6 reader、恢复、发布第二文件/manifest 的链接失败和 `RLIMIT_FSIZE` 实际写入失败。相关三个 CTest 全部通过。Node addon 与 Python client 重新构建后，文件 eval、full、encrypt/decrypt、restore/release 测试通过；测试输入复制到自有临时目录。

提交前最终原生目标构建成功，日志为 `/tmp/taa-native-final-build.log`。正式 `build/examples/inference` 补跑 CIFAR 并比较全部 10 个输出通过，最大绝对误差 `4.171234974847948e-5`，证据为 `/tmp/taa-final-cli-smoke/{inference.log,comparison.json}`；这次生成的 profiling CSV 归入同一临时目录。

复现入口：Lattigo `clientcontext/{context_test,restore_stream_test,capabilities_acceptance_test,memory_acceptance_test}.go`、`circuits/ckks/bootstrapping/keys_stream_test.go`；native `inference/unittests/test_streaming_context.cpp`。生产 capability 测试需显式设置 `TAA_GENERATED_EVAL`；内存比较需 `TAA_MEMORY_MODE=bulk|stream` 和 `TAA_PARAMETER_JSON`。

证据：`/tmp/taa-native-{bulk,stream}-memory.{log,time}`、`/tmp/taa-production-capabilities.log`、`/tmp/taa-lattigo-final-focused.log`、`/tmp/taa-native-final-ctest.log`、`/tmp/taa-node-client-tests.log`、`/tmp/taa-python-client-tests.log`。

## 3. Worker、OPFS 与真实 ZIP64

Node 的 47 项快速测试、Go core 与实际浏览器套件通过。浏览器测试覆盖真实 Worker/OPFS、刷新恢复、任务绑定、单文件与批次、启动失败、混装 Worker/WASM、取消/关闭、锁冲突及 write/seek/close/cleanup 故障。配额故障使用浏览器原生 `DOMException`，保留其只读 `name/code` 和原始失败；清理失败附带残留路径。

正式解压包的真实 ZIP64 边界结果：

| 项目 | 实测 |
| --- | ---: |
| 单条目大小 | 4,294,967,313 字节 |
| 第二条目本地头偏移 | 4,294,967,371 字节 |
| 大归档大小 | 4,294,967,695 字节 |
| 独立数量用例 | 65,536 条目 |
| source 单块上限 | 4,194,304 字节 |
| JS heap 采样最大值 | 27,690,556 字节 |
| renderer RSS 采样最大值 | 640,488 KiB |
| Chromium 全进程 RSS 采样最大值 | 1,446,132 KiB |

浏览器读回大条目并核对尾部内容，Python `zipfile` 独立验证两归档的 CRC、大小、数量、偏移。JS/RSS 每 250 ms 采样，是观测最大值；一次 heap sample 失败已保留在报告，不能声称无采样缺口。该 ZIP 格式测试在页面执行，不含密码学 Worker，`workerJSHeapPeakBytes=0`。

完整生产 Worker setup 使用同一 12,069,339,185 字节 eval；setup+export 436,188.8 ms。线性内存观测最大 656,408,576 字节，Go HeapAlloc 观测最大 606,258,080 字节，JS heap 17,773,584 字节，renderer RSS 1,421,916 KiB，Chromium 全进程 RSS 2,062,156 KiB。OPFS 最终 12,081,945,181 字节，观测峰值 24,003,480,297 字节；复制导出会造成临时双份。块最大 65,536 字节、在途数 1，刷新后相同 profile 恢复成功。

证据：`/tmp/taa-final-browser.log`、`/tmp/taa-domexception-tests.log`、`/tmp/taa-final-zip64-browser.log`、`/tmp/taa-zip64-CFBQSV`、`/tmp/taa-capability-pregeneration/pregeneration.json`。对应测试为 WASM `tests/browser.mjs`、`tests/zip64-browser.mjs`、`tests/pregeneration.mjs`。

补测将固定 4 MiB source chunk 的 payload 从 64 MiB 增至 4 GiB+17（约 64 倍），分别采样读/写阶段，Python 独立 CRC 均通过：

| 阶段 | 64 MiB JS heap 字节 / renderer RSS KiB | 4 GiB+17 JS heap 字节 / renderer RSS KiB |
| --- | ---: | ---: |
| ZIP write | 2,578,804 / 301,468 | 3,516,148 / 326,328 |
| ZIP read/extract | 2,809,560 / 354,024 | 2,300,536 / 387,192 |

payload 内存没有随总字节数线性增长；65,536 条目的 metadata 用例另计，JS heap 观测最大 28,056,236 字节。大文件与解压副本同时存在时 OPFS 占用 8,589,935,864 字节，清理解压副本后为 4,305,093,217 字节。该次所有 heap sample 成功；完整数据见 `/tmp/taa-zip64-phase-browser.log`、`/tmp/taa-zip64-17ARSy`。

## 4. 任务与交付

CIFAR 端到端从正式解压 runtime、独立版本 task/data 包运行，3 例全部正确；本次 native 与 plaintext 最大绝对误差 `6.808099006150314e-5`，释放 device 后保留 host 并重载误差 `3.276006387764596e-5`。输入由 writer 强制 ZIP64，真实 native 服务接受其文件契约；服务端输出 ZIP 字节原样解密。目录模式、ZIP 模式分别报告 Go/WASM 与 OPFS 占用。

额外客户端阶段补测重用这次已完成的 native 输出，报告写明 `nativeArtifacts=/tmp/taa-final-cifar-workflow`；该补测不计为另一次 native 推理：

| 阶段 | JS heap 观测最大字节 | renderer RSS 观测最大 KiB |
| --- | ---: | ---: |
| restore/bind | 4,192,492 | 584,560 |
| directory encrypt | 5,701,144 | 619,224 |
| ZIP encrypt | 7,169,628 | 633,176 |
| ZIP decrypt | 4,046,492 | 595,280 |

Retina 当前 10 图像来自两位患者，7 个 DN、3 个 NDRD 标签；不能称为十位患者或泛化准确率队列。原离线分割/临床特征参考脚本保持不变，任务包附带同 SHA 来源与处理说明。10-case 本次真实 native/ZIP/Worker 流程通过，浏览器解密与独立 Python argmax/标签统计一致，5/10 正确（accuracy=0.5）；原生与明文的 10 个预测完全一致，不沿用历史 9/10。最大绝对误差 `9.13412863678964e-6`，保留 host 密钥后释放 device、重载模型并再次推理的误差 `3.9988777378319185e-6`。

Retina 三路共 30 个密文条目，强制 ZIP64 输入归档 2,181,248,778 字节，输入 ZIP 加密 182,645.4 ms。完整浏览器流程每 250 ms 采样，JS heap 观测最大 39,173,544 字节、renderer RSS 811,624 KiB、Chromium 全进程 RSS 1,381,064 KiB，无 heap sample 失败。输入加密后 OPFS 占用 2,362,394,093 字节，解密后为 2,372,894,879 字节。

Retina 客户端补测通过，明确复用 `/tmp/taa-final-retina-workflow` 的原生输出，仍得到相同 10 例预测；不计为新一轮 GPU 推理。每阶段 250 ms 的观测最大值如下：

| 阶段 | JS heap 字节 | renderer RSS KiB |
| --- | ---: | ---: |
| restore/bind | 18,442,920 | 684,560 |
| directory input_0 | 25,113,072 | 784,988 |
| directory input_1 | 5,614,856 | 696,128 |
| directory input_2 | 5,221,644 | 682,928 |
| ZIP encrypt | 37,202,760 | 756,724 |
| ZIP decrypt | 4,454,968 | 706,500 |

目录 input_0/1/2 的 Go HeapAlloc 观测最大分别为 197,614,688 / 162,325,176 / 161,153,496 字节，ZIP encrypt 为 195,529,520 字节；线性内存最大均为 212,860,928 字节，I/O 块最大 65,536 字节、在途 1。三目录加密后 OPFS 占用分别为 2,278,492,614 / 223,093,464 / 223,093,464 字节；每目录测完删除本次密文副本后再测下一目录。阶段数据见 `/tmp/taa-retina-client-phases/workflow.json`。

独立任务包的四个 ES Module 实际加载成功；CIFAR 3 张图的 9,216 个预处理值与交付的 Python 参考 CSV 完全一致。发布的 `example.html` 使用实际 Worker 完成初始化和关闭。构建器对 runtime/task/data 独立版本、哈希、私钥排除和 Retina 离线参考 provenance 做检查。

最终 clean revision 解压包的两任务客户端复测均通过，包含目录加密、ZIP64 加密、原生结果 ZIP 解密与独立标签统计。CIFAR 3/3、Retina 5/10，与本轮原生结果一致。该复测分别明确引用 `/tmp/taa-final-cifar-workflow`、`/tmp/taa-final-retina-workflow` 的原生输出，不计新 GPU 推理。CIFAR/Retina 的 JS heap 观测最大分别为 6,618,936 / 37,282,708 字节，renderer RSS 为 631,304 / 880,052 KiB；每次各有一次 heap sample 失败，不能称为连续无缺口采样。结果为 `/tmp/taa-clean-release-{cifar10,retina}/workflow.json`，日志为对应 `/tmp/taa-clean-release-{cifar10,retina}.log`。

证据：`/tmp/taa-final-cifar-workflow/workflow.json`、`/tmp/taa-final-retina-workflow/{workflow,native,reload,checked}.json`、`/tmp/taa-cifar-client-phases/workflow.json`、`/tmp/taa-delivery-browser.log`、`/tmp/taa-final-delivery.log`；测试入口 `tests/workflow.mjs`、`tests/delivery-browser.mjs` 与 `package_delivery.py`。

## 5. GPU 资源与故障传播

pool 环境预算/释放重建专项通过；CPU 故障 38 项断言，GPU owner/错误通道 50 项断言通过。新增 Linux CUDA interposition 测试通过 103 项断言，覆盖实际 stream create/sync/destroy、event create/record/sync/destroy、execute、生产 LOAD/STORE 的 H2D/D2H 失败。每个故障有有限超时、首错返回、handle 计数与下一次成功任务校验，早期失败不启动后继节点。

真实 CIFAR 流程暴露并修复了 context 析构晚于 stream 销毁的问题：所有引用 stream 的 device 对象在 StreamGuard 结束前析构。显式销毁可能同时消耗 handle 并返回先前异步错误，因此先清存储的 handle，避免再次销毁。没有增加生产故障注入开关。

两个真实 loaded CIFAR server 在同进程导入相同生产 eval，第一实例 release 返回 busy 后，两个实例均能继续推理。第二 owner 销毁后，第一实例释放 pool、幂等重复释放、使用保留 host 重载并再次推理全部通过，三次解密最大误差均为 `3.276006387764596e-5`。device pool 最大配置为 20 GiB、初始 1 GiB；该进程 GPU 占用从 18,552 MiB 降至 504 MiB，包含 CUDA 常驻开销，不能把它等同于 pool usage。证据：`/tmp/taa-server-resources.json`、`/tmp/taa-server-resources.log`，入口 `tests/server-resources.py`。Retina 对应保留/重载检查亦通过，见第 4 节。

临时验收程序曾因同目录 `queue.py` 遮蔽 Python 标准库而误启重复队列；已停止重复进程、更名并为后续 GPU 阶段加文件锁。受影响的两个 CIFAR 临时任务已隔离，必须重新验收，不能沿用其被覆盖的产物。重复进程在另一进程占满 GPU 时触发真实 pool OOM；CLI 原来没有顶层异常处理而退出 134。本次给 `examples/inference.cpp` 增加顶层错误返回，保持 SDK 原异常语义。已使用现有 CMake 编译/链接参数在独立路径重建 CLI，以生产 CIFAR 参数、256 MiB device pool 实际触发 GPU OOM，打印原异常并以 1 退出，没有 SIGABRT。证据：`/tmp/taa-cli-oom/{inference.log,result.json}`。

CIFAR 两配置已在重建目录重新生成并完成全部输出比较。BERT PDM 最初的锁拒绝来自仍存活的旧验收子进程，未抢占生产锁；该子进程随后因与 ViT 同时使用 GPU 0 而 OOM。保留其混写日志与失败目录，等待进程退出后，以同一完整密钥、模型和参数通过 GPU 独占锁重跑，原生推理和 ONNX 全输出比较均通过。后续进程使用含顶层错误处理的独立 CLI `/tmp/taa-inference-errors`；不在运行时覆盖已映射的库文件。

证据：`/tmp/taa-pool-final-tests.log`、`/tmp/taa-runtime-cpu-failures.log`、`/tmp/taa-gpu-final-failures.log`、`/tmp/taa-gpu-transfer-tests.log`；测试为 GPU core `test/test_memorypool.cpp`、LattiSense `unittests/{test_task_cancellation,test_gpu_failure}.cpp`。

<!-- regression-results:start -->
## 6. 六任务回归

输入加密使用 GPU lazy，参数加密使用 GPU runtime_lazy；BERT/ViT 另验原 PDM 输入链路。比较当前 ONNX 的全部输出，沿用既有绝对误差阈值 0.2，没有修改参数、模型或精度阈值。FHE 时间取 native 自报 GPU 时间；测试在同机并行生成任务，排队和磁盘竞争下的端到端耗时不作为性能基准。

| 任务 | 模式 / 变体 | 输出数 | 最大绝对误差 | FHE 秒 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| BERT | input / pcm | 2 | 0.0318609688 | 188.373 | 通过 |
| BERT | parameter / pcm | 2 | 0.0010390126 | 778.149 | 通过 |
| BERT | input / pdm | 2 | 0.00185270771 | 286.507 | 通过 |
| cifar10 | input / pcm | 10 | 6.44251692e-05 | 8.050 | 通过 |
| cifar10 | parameter / pcm | 10 | 0.000165912796 | 16.075 | 通过 |
| imagenet | input / pcm | 1000 | 0.000555142847 | 46.940 | 通过 |
| imagenet | parameter / pcm | 1000 | 0.00069846033 | 96.935 | 通过 |
| retina | input / pcm | 2 | 4.93858263e-06 | 305.899 | 通过 |
| retina | parameter / pcm | 2 | 5.46774734e-05 | 1209.140 | 通过 |
| ViT | input / pcm | 1970 | 0.0286334878 | 143.161 | 通过 |
| ViT | parameter / pcm | 1970 | 0.0402123692 | 294.728 | 通过 |
| ViT | input / pdm | 1970 | 0.000562338823 | 102.626 | 通过 |
| yolov5 | input / pcm | 129024 | 7.1708353e-05 | 50.206 | 通过 |
| yolov5 | parameter / pcm | 129024 | 0.000104615852 | 176.139 | 通过 |

每配置的命令、退出状态、当前比较结果和内存采样位于 `/tmp/taa-v6-final-regression/<task>/task_<mode>_encrypted/`，BERT/ViT 再加 `pcm/` 或 `pdm/`。汇总为同目录 `acceptance-summary.json`。CIFAR 使用重建后的正常目录；`*-interrupted` 与 BERT PDM 的 `*.overlap-failed` 均保留为失败诊断，不计通过。
<!-- regression-results:end -->

## 7. 验收结论与已知基线失败

六任务 14 配置均通过。最后完成的 Retina 参数加密重新生成图、参数包并执行真实 GPU 推理，全部 2 个输出最大绝对误差 `5.4677473379394215e-5`，FHE 时间 1209.14 秒；没有用其他 Retina 流程替代。clean revision 发布包、SHA、解压校验、真实浏览器故障套件及两任务客户端复测均完成，来源与证据见第 1、4 节。

Lattigo 全测试的 `TestRLWEConstSerialization` hash 断言在原始 `22bff324e146968c186106524c06cea607462b05` 独立 checkout 同样失败（expected `1MTZP69Y…`、actual `U+E7O5Qb…`）。保留 `/tmp/taa-lattigo-baseline-serialization.log`，没有修改预期 hash 来隐藏基线失败；本次功能验收通过不等于完整套件全部通过。

## 8. 专题实现提交

| 规范 | latti-ai 实现提交 | 子模块实现提交 |
| --- | --- | --- |
| 流式密钥与持久化 | `29413d1` | Lattigo `11cd8312`、LattiSense `fe4bf9d` |
| OPFS ZIP64 | `25475d7` | — |
| 浏览器客户端与 Worker | `3ba94e1` | 共用流式密钥提交 |
| 浏览器任务与交付 | `b87e55a` | — |
| 运行时资源与故障传播 | `9b01028`、`fc87f9a` | GPU core `306f89f`、LattiSense `7638337`、`77ab726` |

设计总则的整体验收以本记录与规范状态的后续文档提交固化；发布源码仍为第 1 节列出的精确四仓库 SHA。
