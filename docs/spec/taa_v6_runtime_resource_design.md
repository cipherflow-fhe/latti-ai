# TAA v6 运行时 GPU 资源与 CPU/GPU 故障传播设计

日期：2026-11-06。状态：已实现，资源释放与故障传播专项验收通过；六任务完整回归仍按[验收记录第 5–7 节](../research/taa_v6_acceptance_report.md)执行。遵循[总则](taa_v6_adaptation_design.md)，以其中锁定的 v6 源码为准。修改当前 `lattisense-gpu-core`、compound/ops runner 与统一 v6 context，不恢复旧 HEonGPU submodule 或旧单输出 executor。

## 1. 三个独立目标

1. `InferenceServer::release_device()` 释放模型执行 engine/device state 与可安全清理的全局 pool，保留 host 上已解析的 v6 eval context，之后无需重新读取大文件即可 load_model/evaluate。
2. 通过环境变量限制当前 GPU core 的初始/最大 device 与 pinned-host pool，保持现有显式 config 优先级。
3. CPU/GPU 任一任务及事件/ABI 失败都终止整个当前调度、排空后台工作并把原异常返回调用方，不挂在 completed==total 或缺失依赖上。

不建立通用 device service，不改变图执行算法和节点输出，不把资源释放扩成更换模型或跨进程恢复框架。

## 2. host 与 device 所有权

当前 server 将 unique_ptr context 移交 `InferenceProcess::ckks_contexts`，释放 engine 会丢失 host keys。本次只在这条共享需求上使用 `shared_ptr<CkksContext>`：server 持有 host context，engine 的 map 持有同一对象；更新直接受影响的赋值/构造点，原 unique_ptr 创建点可 move 转 shared_ptr，不复制密钥、不重新反序列化。

host context 的参数/键在 engine 使用期间保持不变；重新 import 必须先结束旧 engine，不能在运行中替换。已有 v6 线程安全 context 语义继续生效，不增加旧 `ShallowCopy`。CPU/其他 engine 若不需保留所有权，不额外改变其生命周期。

| server 状态 | 允许行为 |
| --- | --- |
| 无 context | import eval；load/evaluate 报缺 context |
| host-ready | load_model 构建 engine；release_device 幂等，不丢 host |
| loaded | evaluate；空闲时 release_device；重复 load 遵循当前模型检查 |
| running | 不允许 import/load/release 改变资源；返回 busy，调用方先等待/取消 |
| failed | 当前运行失败经 drain/cleanup 返回；host context 可保留，device 是否可重建取决于实际 CUDA 清理结果 |

同一 server 的状态修改由小范围互斥保护，不额外序列化不同 server 的正常 evaluate。`release_device()` 的严格全局 pool 前置条件在下一节定义。析构只能 best-effort 销毁本对象，不抛异常，也不能为了清理自己而强行释放其他 owner 的全局 pool。

现有 `InferenceRunner` 每次 run 内创建局部 `FheTaskGpu`，run 返回前同步的行为保留；它只接入同一个 pool owner 约束与错误返回，不为了本功能改成常驻 server 对象。

## 3. 进程全局 pool 的安全释放

pool 是进程全局资源，单个 server 的 shared_ptr 不代表其他 engine 已停止。采用一个 SDK 范围的 mutex + 活跃 GPU owner 计数 + 释放中标志，不新增调度服务：

- 每个能持有 pool 分配的 SDK GPU task/engine 在**构造、预热和首次分配之前**登记 lease，销毁所有 device 对象、stream/event 和 allocator 引用后注销；包括 `FheTaskGpu` 构造中的预热与 runner 局部 task。
- 创建/运行/释放的 admission 受同一个 gate 保护。正常操作不持有 gate 做长耗时计算；lease 覆盖整个资源生命周期，而不是只计 evaluate 时间。
- server release 首先在 gate 内验证自身空闲且没有其他 GPU owner。若其他 owner 存在，返回 `RESOURCE_BUSY`，本 server 仍保持原 loaded 状态；不能先销毁它再发现全局 pool 不可清。
- 验证后置 releasing 阻止新 owner，销毁本 server engine/device task，等待已提交 CUDA 工作完成并销毁其 event/stream，再移除本 lease。确认引用计数为 0 后调用 SDK pool release，最后解除 releasing。
- 所有退出路径解除 gate/标志。重复 release 无 owner 时成功。新增直接 SDK release 入口要求零 owner；CPU-only 实现为 no-op，不引入 CUDA 链接依赖。

GPU core 增加公开 `MemoryPool::release()`，在现有 pool mutex 保护下调用/复用 `clean_pool()`，按正确顺序清理 stats adaptor、pool、base resource 和 initialized 状态；后续 initialize 可重建。避免 public release 与 private clean_pool 重复加锁。SDK 负责 owner admission；直接绕过 SDK 的 GPU core 使用方也必须在无活跃分配/引用时调用，pool 方法不能凭空判断外部裸指针是否仍有效。

pool 当前是全局结构，本次不把它拆成 per-device allocator。所有可能持有它的 GPU owner 都纳入计数，不能只按 server 的 gpu_device 过滤；实际设备上下文按当前代码正确选择/同步。不得在仍有工作或 resource adaptor 引用时清理底层 RMM 资源。

同步/销毁失败时向显式 release 调用返回错误，不能报告已释放；保留 host，记录实际 device 清理状态。若 CUDA 本身已不可继续使用，原异常仍返回，不宣称一定可同进程重载。普通可恢复故障的释放/重载必须通过测试。

## 4. pool 环境预算

保留来源对外变量名，在当前 GPU core 的 `MemoryPoolConfig::Defaults()` 读入：

| 变量 | 作用 |
| --- | --- |
| HEONGPU_DEVICE_POOL_INITIAL_BYTES | device pool 初始字节 |
| HEONGPU_DEVICE_POOL_MAX_BYTES | device pool 最大字节 |
| HEONGPU_HOST_POOL_INITIAL_BYTES | pinned-host pool 初始字节 |
| HEONGPU_HOST_POOL_MAX_BYTES | pinned-host pool 最大字节 |

变量值为完整非负十进制整数；拒绝负数、溢出、尾随垃圾，max 必须有效且不小于 initial。遵循当前 256-byte 对齐、可用内存、bytes/fraction 和 initial≤max 检查，不通过截断使超预算值合法。无变量时完全沿用当前默认值。

显式 `initialize(config)` 的用户配置优先；`Defaults()` 由环境生成 bytes 字段，bytes 优先于默认 fraction。仅设置 device max 而未设 initial 时，initial 为 `min(max, 1 GiB)` 再按当前合法对齐规则处理，不能沿用较大的默认 fraction 导致启动超上限。host 未设 initial 时沿用现有逻辑，不另发明 host 预分配比例。

配置在 pool 初始化时生效，不在每次 allocate 重读环境，不热修改已初始化 pool。release 后下一次 initialize 读取新的 Defaults。预算是 pool 的实际配置范围，不把其他 CUDA/上下文占用算成“违反 pool max”；测量区分 pool usage 与 nvidia-smi/进程总占用。

## 5. 共享失败通道的最小修改

`mega_ag_runners/cpu_task_utils.h` 已有 first_exception/failed/清队列/通知，保留其语义，提取为当前一次 `run_tasks` 共用的小型 failure state，并经 `RunTasksOptions`/backend submitter 传给 GPU。只包含 first exception、必要锁/failed 标志和通知入口，不新建第二套 CPU runner。

第一异常用 `std::exception_ptr` 记录，后续清理异常只附加诊断，不覆盖首错。cancel 与 failure 分开：取消映射已有 TaskCancelled/取消返回；真实 compute 失败仍按错误返回，不能被并发 cancel 隐藏。所有 C/Go/Node/Python 边界沿当前错误规范返回/reject，禁止异常穿越 C ABI 或从 detached task 中重新抛出。

| 路径 | 本次需覆盖的异常范围 |
| --- | --- |
| CPU task | 输入 cache 查找、other_args/ABI 参数构造、compound `compute_node.execute`、多输出发布、purge/依赖调度、progress callback |
| GPU submit task | device/stream 选择、输入 cache/依赖 event 等待、LOAD/STORE、compound execute、event 创建/record、结果发布 |
| GPU completion task | event synchronize、输出引用 purge、完成计数及异常通知 |
| dispatcher/cleanup | submit/分配失败、wait 谓词、drain、stream synchronize/event destroy、用户取消 |

catch 必须包住整个后台任务体，而非只包 execute；当前 CPU 的若干准备/发布步骤在 try 外，GPU event 也在 execute catch 外，都需随本次故障通道局部调整。所有任务采用 catch-all 记录，不能只 catch std::exception 后 `throw`。

## 6. 终止、唤醒与清理顺序

1. 任一任务失败，原子/锁保护记录首错，设置 failed；在队列锁下停止新节点入队并清待调度队列，唤醒 dispatcher/completion wait。
2. CPU/GPU 结果发布前在相同调度同步边界检查 failed/cancelled；失败节点不发布输出/event，不安排依赖它的后继。已运行的独立任务可收尾，但失败后不能触发新后继。
3. 所有 wait 的谓词包含 `failed || cancelled || completed>=total`；改变谓词状态与 condition_variable 等待使用正确的同步/通知规则，修复当前“检查 !failed 后只等 completed”的竞态，不能只在 wait 外检查一次。
4. dispatcher 停止提交，drain 所有已提交 CPU/GPU worker 及 completion task；不要在持有队列锁时等待，也不要在失败后等永远不会生成的 event 或 total success count。
5. drain 后在正确 device 上同步仍在飞行的 stream，销毁已创建 event/临时 device 对象；部分初始化使用 scope cleanup，cleanup 自身失败也不得跳过其他 handle 的释放。
6. 当前 run 的后台任务都结束后，再向调用方 rethrow 首错或返回取消。共享 failure state 必须活到所有后台 callback 结束；下次运行创建新状态，不复用 failed 标志。

现有 `FheGpuTask::run` generic catch 也需等待本 run 的 device 工作安全终结，不只清 cancel_flag 后立即 throw。正常成功同步、已有 CPU exception 行为与 compound/多输出 ABI 保持不变。

## 7. 修改落点与 bindings

| 仓库 | 必要文件/职责 |
| --- | --- |
| GPU core | `src/include/heongpu/util/memorypool.cuh`、`src/lib/util/memorypool.cu`：Defaults 环境解析、release/reinitialize |
| LattiSense | `mega_ag_runners/cpu_task_utils.h`、`mega_ag_runners/gpu/gpu_wrapper.cu`：failure state、wait/drain、owner gate；当前 GPU/CPU 公共接口：安全 pool release |
| latti-ai | `inference/interface/inference_server.*`、`inference/inference_task/inference_process.*`：共享 host、release_device；直接受影响的 runner/provisioner 构造点与 Python/bindings |

只变更实际共享 context 的 CKKS 所有权链，不机械改变 BFV/其他 map。不复制来源旧 executor 参数列表，不改 compound node 算法或 runtime_lazy store/signature。Node 中不存在的 server 接口无需为对齐名称新增；实际暴露 server 的 Python/SDK 入口补 release，原 client release 继续只释放客户端内存。

## 8. 验收

- 单 server 多次 import/load/evaluate/release/load：解密结果与原路径一致；host context 对象身份/keys 不变，不重新读取数 GB eval；device/pinned pool 可观察下降，重建成功。
- 两个 engine 或 runner/server 同进程：一个 release 返回 busy 且原状态完整；另一个仍可推理；最后 owner 退出后清 pool。并发创建与 release 竞争不能越过 gate，构造/预热失败也不漏 lease。
- CPU-only 构建链接/调用 release 成功，无 CUDA 依赖；GPU core 不用新后端仓库替换，固定 gitlink/detached checkout 可构建。
- 四个变量分别/组合、仅 max、小预算、非法/溢出/initial>max、显式 config 优先、release 后再初始化；记录配置值、pool usage 与总设备内存差别。
- 有限超时的故障测试：CPU 准备/compute/多输出发布、GPU LOAD/STORE/execute、event create/record/sync、dispatcher submit、取消和 cleanup 注入错误；进程不 terminate、无永久 wait、首错返回，worker/stream/event 全部可收尾。
- 专门复现 failed 在 completion wait 预检查之后发生的竞态，验证 predicate 唤醒；故障后执行新的可恢复任务成功，不继承 failed/cancel_flag。
- 六任务、encparams/runtime_lazy、BERT/ViT PDM 与 metadata 回归按总则执行；正常推理结果/并发语义不因错误处理改动而退化。
