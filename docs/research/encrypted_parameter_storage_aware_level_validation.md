# 加密参数存储感知 level 优化验收

2026-10-28，按 `docs/spec/encrypted_parameter_storage_aware_level_optimization.md`
验收 ResNet20/CIFAR10 的 periodic、Runner-local mask PT、`runtime_lazy`
`server_provisioned_runner` 路径。GPU 为 NVIDIA RTX 5880 Ada，未执行 CPU 或非 periodic 验收。

## 实现与计费口径

参数数量由 inference layer 的 `get_encrypted_param_count*()` 返回，编译器复用现有
layer 构造。覆盖当前链路的普通/multiplexed Conv2D、poly、dense、mult-scalar，
mask PT 不计费。本次报告的 `unaccounted` 为空。

`encrypted_parameter_storage_weight` 默认 0；正权重仅允许用于
`server_provisioned_runner`。目标为原 runtime、refresh 分数加上
`weight * storage_bytes / 2**30`。本次使用权重 **1.0**、`temperature=0`、单次编译。

采用固定校准表 `training/model_compiler/encrypted_parameter_size.json`，对应
`N16QP1546H192H32`、`CkksCiphertext.serialize`、参数 store v2。
验收时 lattisense commit 为 `7e7cf87fc8975cb97f7db2df2f8486e5ffc256ab`。
不支持的参数集或 level 明确报错，DP 内不执行加密/序列化。

参数计费使用实际 use-site level：

```text
min(input_level, output_level + level_cost)
```

这与最终插入的 drop-level 对齐，避免将 BTP 输出的多余 level 算入参数大小。
`recompute_final_level()` 后重新计数，若与 DP 候选不一致则报错。

## 回归验证

```bash
PYTHONPATH=inference/lattisense /home/linghm/pytorch_venv/bin/python -m unittest \
  inference.unittests.test_encrypted_param_count \
  training.model_compiler.unittests.test_storage_score \
  inference.unittests.test_server_provisioned_runner_generator
```

14 项测试通过：计数与真实参数节点的 shape/level 一致，覆盖 periodic 条件、
stride/skip、bias、poly、dense 和 scalar；零权重跳过存储路径；正权重精确加分；
use-site 与 drop-level 一致；缓存区分 successor level；不支持的校准及最终
level 改写触发错误。`git diff --check` 和修改模块的 Python 编译检查通过。

另外，在修改前 commit `376b4bb` 的隔离 worktree 和当前零权重实现上，分别重新
编译相同模型。归一化自动生成的时间戳和对象地址名称后，完整图的所有 feature/layer
属性及有序输入输出边一致，包含每个 feature level、BTP 和 drop-level 位置；
原始 score 均为 **4.618196680999998**。
证据为 `runs/storage_aware_validation/zero_weight_compatibility.json`，编译图分别在
`before/task/server/` 和 `weight0/task/server/`。

## GPU 完整链路

以下命令均从仓库根目录执行；本次产物位于独立目录，保留已有 baseline。

```bash
mkdir -p runs/storage_aware_validation
cat > runs/storage_aware_validation/config.json <<'JSON'
{"encrypted_parameter_storage_weight": 1.0}
JSON

PYTHONPATH=inference/lattisense /home/linghm/pytorch_venv/bin/python training/run_compile.py \
  --input runs/cifar10/model/trained_poly.onnx \
  --output runs/storage_aware_validation/optimized \
  --style multiplexed --deployment-mode server_provisioned_runner \
  --config runs/storage_aware_validation/config.json \
  --num_experiments 1 --num_workers 1 --temperature 0

PYTHONPATH=inference/lattisense /home/linghm/pytorch_venv/bin/python inference/interface/gen_mega_ag.py \
  --task-dir runs/storage_aware_validation/optimized/task \
  --deployment-mode server_provisioned_runner --parameter-loading-mode runtime_lazy

./build/tools/provision_encrypted_runner_bundle \
  --task-dir runs/storage_aware_validation/optimized/task \
  --out runs/storage_aware_validation/optimized/task/runner

/usr/bin/time -f '\nwall_seconds=%e' ./build/examples/inference \
  --task-dir runs/storage_aware_validation/optimized/task/runner \
  --deployment-mode server_provisioned_runner \
  --input examples/test_cifar10/task/client/img.csv \
  --output-cipher runs/storage_aware_validation/optimized/output_gpu.ct --gpu

./build/tools/decrypt_runner_output \
  --task-dir runs/storage_aware_validation/optimized/task/provisioner \
  --cipher runs/storage_aware_validation/optimized/output_gpu.ct
```

baseline 使用已有 `runs/cifar10/task/runner_periodic_mask_pt_runtime_lazy`，以相同
输入和 GPU 命令重跑，输出另存为 `runs/storage_aware_validation/baseline_output_gpu.ct`，
用 `runs/cifar10/task/provisioner` 解密。两次完整链路均成功，无 expected level、
scale/rescale 或在线 mask PT 错误。

使用 `train_poly.pth` 构造相同 ResNet20，严格加载参数，以 GPU float64 前向作为参考。
三者 top-1 均为 **3**；baseline 与参考最大绝对误差 **1.8750e-4**，优化版为
**1.9738e-4**，两份解密输出最大差 **2.2430e-4**。
这里只验证同一张输入的完整推理，不代表全数据集准确率验收。

## 大小与代价

大小为普通文件逻辑长度之和，GiB = `2**30` 字节。参数 payload 仅累加 manifest
`elements[].length`；参数文件还含 framing 和 manifest。完整 Runner 包括
密钥/评估上下文、图及配置。

| 项目 | periodic baseline，weight=0 | 存储感知，weight=1 |
| --- | ---: | ---: |
| 参数 payload，字节 | 6,703,138,582 | 4,588,616,340 |
| 参数 payload，GiB | 6.242784 | 4.273482 |
| 参数文件总和，GiB | 6.243106 | 4.273803 |
| 完整 Runner，GiB | 9.352308 | 7.383029 |
| 完整任务，GiB | 18.694632 | 16.715211 |
| 参数密文数 | 2,096 | 2,096 |
| BTP 节点/密文数 | 12 / 12 | 17 / 17 |
| drop-level 节点数 | 18 | 20 |
| runtime score | 1.018196681 | 0.741961551 |
| refresh score | 3.6 | 5.1 |
| storage score（以 weight=1 比较） | 6.242784282 | 4.273481984 |
| 总目标（以 weight=1 比较） | 10.860980963 | 10.115443535 |
| GPU 图执行进度计时，约秒 | 33 | 27 |
| 整个 Runner 进程 wall time，秒 | 50.45 | 44.77 |

baseline 完整任务只合计 `server`、`client`、`provisioner` 和
`runner_periodic_mask_pt_runtime_lazy`，排除同目录的其他版本 Runner；优化任务
合计独立 `optimized/task` 下全部文件。baseline 的 server 是保留的现有目录，
包含历史生成文件，因此完整任务大小还受这部分约 0.01 GiB 的差异影响。
表内 runtime/refresh 使用 DP 计费口径；零权重原始总分减去 refresh 得到 baseline
runtime，优化版来自最终、插入 drop-level 前的报告，不混用导出后图的重新计时。

2096 个参数密文始终为 weight 1972、bias 22、coefficient 95、scalar 7。
本次通过降低参数 level 缩小单元素，而非减少数量。level≤2 的密文由 1176 个增至
1928 个，level≥5 由 889 个降至 119 个。优化版在 layer1/2 多个 block 的 conv1
之后以及 Add 之后放置 BTP；逐项 BTP/drop-level 位置和 level 直方图保存在
`runs/storage_aware_validation/size_comparison.json`。

新 manifest 的每层、每类型、每 level 数量与报告完全一致；参数 payload 与估计
均为 **4,588,616,340 字节**，无估计偏差。参数 payload 下降 **31.55%**，完整
Runner 下降 **21.06%**。Runner 中约 3.11 GiB 的上下文等开销变化很小，因此整体
降幅较小；完整任务还保留约 9.33 GiB 的 provisioner 材料。

BTP 增多使 runtime+refresh 的估计总成本从 4.6182 增至 5.8420；实测 wall time
仍下降，包含低 level 运算和 lazy 参数读取等影响。上述为单次运行记录，不作为
稳定性能加速结论。

报告及日志位于 `runs/storage_aware_validation/`：
`optimized/encrypted_parameter_storage_report.json`、`size_comparison.json`、
`correctness.json`、`compile_optimized.log`、`gen_optimized.log`、
`provision_optimized.log`、`runner_{baseline,optimized}_gpu.log`、
`decrypt_{baseline,optimized}.log`。在线核对脚本为 `check_manifest.py` 和
`check_reference.py`。既有 periodic/nonperiodic bundle 与输出均保留。
