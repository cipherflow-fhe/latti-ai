# GPU runtime-lazy 验收记录

状态：两模型均通过。每个模型的 `[1,1]` 对照与 `[4,4]` 压缩组均完成 1 次预热和 3 次正式 GPU 运行。
以下数值来自模型 JSON、完整 logits CSV 与 monitor CSV；时间和峰值内存取三次正式运行的中位数。

环境：NVIDIA RTX 5880 Ada（GPU 0，49 GB），driver 535.230.02，CUDA toolkit 12.8；
GPU=ON、DEV=ON。原 ONNX 和输入 CSV 的 SHA-256、CKKS 参数、全部 argv/returncode、
源码逐文件 hash 和 binary hash 均已保留。两组每模型使用相同 CKKS 参数与输入。

## 实际存储量

| 模型 | 模式 | CT 数 | 参数 payload bytes | 完整 runner bytes | 完整 task bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| resnet20 | legacy | 2,096 | 4,588,616,340 | 7,927,478,957 | 17,947,838,825 |
| resnet20 | compressed | 252 | 993,679,628 | 4,347,337,596 | 14,367,516,710 |
| mobilenet | legacy | 19,757 | 49,947,018,835 | 53,400,328,219 | 63,475,739,451 |
| mobilenet | compressed | 1,518 | 4,883,697,767 | 8,434,630,328 | 18,508,245,595 |

payload 是真实 CT container 内各元素长度之和，不含 evaluation keys、container header 和 manifest。
runner/task 是目录中全部文件的逻辑字节数，不按硬盘 block 占用计费。task 大小包含 provisioner，
仅用于验收记录；验收生成的 task 已删除，未提交权重、ciphertext 或密钥。

- resnet20：payload 压缩 4.62 倍，减少 78.34%；完整 runner 缩小 1.82 倍。两组 evaluation context 均为 3,329,929,205 bytes。
- mobilenet：payload 压缩 10.23 倍，减少 90.22%；完整 runner 缩小 6.33 倍。两组 evaluation context 均为 3,329,929,205 bytes。

evaluation context 中的 keys 不随 seed 数同比缩小，因此完整 runner 的压缩倍数低于参数 payload。
`legacy` 是当前实现使用 `[1,1]` 的对照，已经包含 Runner-local public structural mask；
不是历史 revision 的二进制，以上差值不包含从旧 encrypted mask 改为 public PT 的独立收益。

## 逐类参数

| 模型 | 参数族 | legacy CT / bytes | compressed CT / bytes |
| --- | --- | ---: | ---: |
| resnet20 | Dense bias | 1 / 999,445 | 1 / 999,445 |
| resnet20 | Dense weight | 32 / 53,478,080 | 1 / 2,326,551 |
| resnet20 | mult-scalar | 7 / 14,336,158 | 7 / 17,645,731 |
| resnet20 | multiplexed bias | 21 / 26,296,769 | 21 / 49,463,780 |
| resnet20 | multiplexed weight | 1,940 / 4,209,749,112 | 127 / 579,718,423 |
| resnet20 | poly coefficient | 95 / 283,756,776 | 95 / 343,525,698 |
| mobilenet | Dense bias | 500 / 499,722,500 | 8 / 13,369,520 |
| mobilenet | Dense weight | 3,000 / 5,013,570,000 | 47 / 109,347,897 |
| mobilenet | depthwise bias | 95 / 364,939,619 | 95 / 381,487,484 |
| mobilenet | depthwise weight | 1,359 / 6,154,406,847 | 93 / 555,616,860 |
| mobilenet | multiplexed bias | 198 / 213,880,918 | 198 / 211,226,706 |
| mobilenet | multiplexed weight | 13,360 / 32,433,813,552 | 877 / 2,657,997,432 |
| mobilenet | poly coefficient | 1,245 / 5,266,685,399 | 200 / 954,651,868 |

逐层决策、候选 bytes、seed/use-site level、未压缩项和原因见 `layout_storage_audit.json`；
完整 geometry/scale/signature/manifest 见 `raw/*-layout-contract.json`。四组最终编译预测与
实际 payload 的每层、每 level、每参数类型均一致，各 level 元素长度与校准表一致，`unaccounted` 为空。

## 完整数值校验

| 模型 | 模式 | logits | max_abs | RMSE | cosine | top-1 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| resnet20 | legacy | 10 | 0.00017103216 | 8.5155285e-05 | 0.999999999957 | 一致 |
| resnet20 | compressed | 10 | 0.00016317196 | 0.00011455618 | 0.999999999915 | 一致 |
| mobilenet | legacy | 1000 | 0.0014371854 | 0.0004101852 | 0.999999990426 | 一致 |
| mobilenet | compressed | 1000 | 0.0015419532 | 0.00039857863 | 0.999999990809 | 一致 |

mean_abs、max_rel 和每次原始指标保留在模型 JSON。上表采用正式样本中 max_abs 最大的一次；
所有压缩样本（含预热）均满足 max_abs≤0.01、RMSE≤0.001、cosine≥0.9999、top-1 一致，
并满足 max_abs_compressed≤2×max_abs_legacy+1e-6；相对门槛的 legacy baseline 取该组四次样本的最大值。
ResNet20 解密最大 logit 约为 18.9296。MobileNet 比较全部 1,000 logits，未截取前十项代替校验。

## 时间与内存

| 模型 | 模式 | GPU core s | Runner wall s | decrypt s | Runner+decrypt s | CPU peak GiB | GPU pool used peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet20 | legacy | 25.482 | 41.666 | 15.375 | 57.063 | 17.616 | 11.311 |
| resnet20 | compressed | 9.384 | 25.970 | 15.411 | 41.568 | 18.562 | 11.357 |
| mobilenet | legacy | 223.232 | 248.473 | 20.377 | 268.851 | 26.960 | 13.556 |
| mobilenet | compressed | 64.896 | 93.744 | 20.485 | 114.138 | 34.926 | 17.653 |

GPU core 来自 Runner 的 `Run GPU time`；Runner wall 为独立进程 wall time，含
key/model 加载、CUDA 初始化、输入编码和输出写入。decrypt 含独立进程初始化、secret key 加载和完整 CSV 写入。
Runner+decrypt 先逐样本相加再取中位数，所以不保证等于两列中位数之和。编译与 Provisioner 不计入推理时间。
CPU peak 是进程 peak RSS；GPU used 为 100 ms 采样的内存池实际使用量，不是 CUDA 预留池容量。
CPU/GPU 峰值不随参数包大小同比下降，public PT、恢复临时 CT、graph executor 生命周期同样占内存。

- resnet20 原 ONNX CPU 推理中位数 1.196 ms，ORT preparation 2475.060 ms 单列记录。bootstrap 节点数为 17 → 12。
- mobilenet 原 ONNX CPU 推理中位数 13.084 ms，ORT preparation 3587.042 ms 单列记录。bootstrap 节点数为 192 → 189。

plaintext 在 CPUExecutionProvider 上执行，不含 session preparation；与包含进程初始化的 FHE wall
时间范围不同，不据此声称公平的加速倍数。两组独立 DP 因存储评分选择了不同 level/bootstrap 计划，
激活算子的 level_cost 未增加，但不能将两组耗时差全部归因于 seed 恢复。

## runtime-lazy、恢复与结构检查

四组均为 `server_provisioned_runner / plaintext / encrypted_offline / runtime_lazy`。
每个 canonical element 恰好一个 load 节点；所有实际 load 次数与导出 CT 数一致。public mask 未进入 manifest，
shape/level/scale/layout 在编译 metadata、signature、manifest 与实际密文之间交叉校验通过。

| 模型 | 模式 | canonical loads | cache hits | evictions（正式样本范围） | peak cached bytes（正式样本最大值） |
| --- | --- | ---: | ---: | --- | ---: |
| resnet20 | legacy | 2096 | 0 | 2062–2063 | 67,093,010 |
| resnet20 | compressed | 252 | 0 | 236–236 | 67,076,626 |
| mobilenet | legacy | 19757 | 0 | 19690–19692 | 67,093,575 |
| mobilenet | compressed | 1518 | 0 | 1487–1487 | 67,093,127 |

cache 上限 67,108,864 bytes，按序列化 CT payload 计费；不是总 CPU/GPU 内存限额。
零 cache hits 是因为 graph 已复用单个 canonical load 的结果，多个恢复分支不重复调用 store。
eviction 只移除 store 引用，不会销毁 executor 仍持有的 shared_ptr。

压缩组参数分支子图计数（工具字段 `recovery_operations`）：

| 模型 | encode_pt | mult | rescale | rotate_col | add |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet20 | 3880 | 3916 | 3881 | 8004 | 7795 |
| mobilenet | 19977 | 22551 | 21719 | 74619 | 64688 |

此统计从 packed load 出发，遇到包含其它 CT 输入的算子停止；它包括原有参数旋转，
并会包含明文输入首层中仅以参数 CT 与输入 PT 计算的部分乘加，因此不是新增恢复成本的净差。
完整 MegaAG 各类型节点总数另见 JSON 的 `graph_operations`。恢复分支的选择/边界 mask 均使用公开 PT，
按各自 level 的 q 编码并 rescale，seed 的 scale 保留对应权重、bias 或 degree 的真实目标 scale。

## 检查、失败与覆盖边界

- Python 45 tests、CKKS 7 cases / 1,919 assertions、store/manifest 2 cases / 54 assertions 通过；原始日志与命令见 `raw/validation_commands.md`。
- `raw/smoke/` 是早期 smoke，不作为正式验收。首次完整运行缺 mult-scalar signature scale，在 GPU 前失败，记录于 `raw/attempt-1-missing-scalar-scale/`。
- 下一次 ResNet20 compressed 因 JSON 对象键顺序比较误报布局不一致，在 GPU 前失败，记录于 `raw/attempt-2-layout-object-order/`。修复语义比较并增加回归测试后，使用最终 binary 重跑两组全部样本；编译/Provisioner 沿用未改变计算布局的隔离 task，准备命令与源码 hash 在 ResNet20 JSON 中单独记录。
- MobileNet 从 compile 到全部测量使用最终源码与 binary。最终证据复查见 `raw/final-verification.log`，存储复核见 `raw/layout-audit.log`。
- inverse canonical 有真实 CKKS 单元测试；两个固定 ONNX 未据此覆盖所有 inverse、client 模式或其它 shape。concat/upsample/padding 等 public mask 路径未扩展或验收。
- mult-scalar、普通/inverse bias 保留原密文布局；不跨层、跨 degree 或不兼容 scale 共享。部分层或 degree 因单请求、几何不适用、level 不可用或无正 bytes 收益保留原布局，逐层决策在 audit 中列出。
- lattisense 临时修改仅修复 DEV monitor 的一行调用；lattigo 两个 header 为构建自动生成差异。验收后均已撤销，未提交依赖文件或子模块指针；复现 DEV 构建需要该临时 monitor 修复，细节见 `code_change_summary.md`。
