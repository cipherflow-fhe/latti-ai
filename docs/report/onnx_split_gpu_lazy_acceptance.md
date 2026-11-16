# ONNX prefix/suffix GPU lazy 验收

状态：六任务 × 输入加密/参数加密 **12/12 通过**；默认切分比例 `0.9`、PN15Default，所有原始输出的最大绝对误差严格小于 `0.1`。

环境：RTX 5880 Ada 48 GiB，GPU 0，单卡串行；GPU 推理设置 `OMP_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`。CNN 使用 multiplexed，BERT/ViT 使用 PCM。

输入加密使用 GPU lazy；参数加密使用 `server_provisioned_runner` 和 GPU `runtime_lazy`，均完成编译、计算图生成、密文推理和解密，参数加密另完成 provision。基准是原始完整 ONNX 接收原始 CSV 的全部明文输出；BERT/ViT 直接调用既有 NumPy float64 参考入口，其他任务使用 ONNX Runtime 与既有多项式适配。

## 切分与明文等价

节点计数不含 Constant，按稳定拓扑顺序取 `floor(0.9 × N)`。表中的节点数为实际提取结果；所有 crossing 均保留 batch 维并按 C-order 写入 CSV。

| 任务 | 原图计算节点 | prefix / suffix 节点 | 边界类型与逻辑 shape | 明文切分 max_abs |
| --- | ---: | ---: | --- | ---: |
| BERT | 99 | 89 / 10 | FeatureMat `[1, 44, 768]`；FeatureMat `[1, 44, 768]` | 5.43236567e-11 |
| cifar10 | 52 | 46 / 6 | Feature2D `[1, 64, 8, 8]`；Feature2D `[1, 64, 8, 8]` | 0 |
| imagenet | 100 | 90 / 10 | Feature2D `[1, 160, 8, 8]` | 0 |
| retina | 63 | 56 / 7 | Feature0D `[1, 512]`；Feature0D `[1, 32]` | 0 |
| ViT | 99 | 89 / 10 | FeatureMat `[1, 197, 192]` | 3.69095357e-08 |
| yolov5 | 142 | 127 / 15 | Feature2D `[1, 128, 16, 16]`；Feature2D `[1, 64, 64, 64]`；Feature2D `[1, 128, 32, 32]`；Feature2D `[1, 128, 16, 16]` | 0 |

prefix crossing 对照误差均为 0；完整切分的明文误差与 FHE 误差分开记录。六任务均用最终准备脚本重复验证，模型内容与边界 CSV 一致。YOLOv5 比较全部三个输出，共 129,024 个元素。

## GPU 性能和误差

| 任务 | 模式 | FHE 秒 | CPU RSS GiB | GPU 进程 GiB | GPU 池内 GiB | max_abs | 验收 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BERT | 输入加密 lazy | 7.717980 | 22.995 | 42.828 | 16.023 | 6.174119499e-05 | 通过 |
| BERT | 参数加密 runtime_lazy | 59.879400 | 22.617 | 42.828 | 16.170 | 0.0001690034672 | 通过 |
| cifar10 | 输入加密 lazy | 0.728499 | 2.097 | 42.828 | 0.987 | 2.620665882e-06 | 通过 |
| cifar10 | 参数加密 runtime_lazy | 3.037780 | 2.093 | 42.828 | 1.220 | 5.773795256e-06 | 通过 |
| imagenet | 输入加密 lazy | 4.525200 | 23.963 | 42.828 | 15.785 | 6.711755955e-06 | 通过 |
| imagenet | 参数加密 runtime_lazy | 9.630230 | 22.317 | 42.828 | 16.067 | 9.620824062e-05 | 通过 |
| retina | 输入加密 lazy | 0.636429 | 2.261 | 42.828 | 1.045 | 9.345691288e-08 | 通过 |
| retina | 参数加密 runtime_lazy | 3.888760 | 22.168 | 42.828 | 13.844 | 2.424625823e-06 | 通过 |
| ViT | 输入加密 lazy | 13.132600 | 22.454 | 42.828 | 16.204 | 0.01302569198 | 通过 |
| ViT | 参数加密 runtime_lazy | 22.842800 | 22.134 | 42.828 | 16.852 | 0.0030785627 | 通过 |
| yolov5 | 输入加密 lazy | 3.666520 | 22.659 | 42.828 | 14.397 | 1.15743152e-05 | 通过 |
| yolov5 | 参数加密 runtime_lazy | 7.459440 | 22.499 | 42.828 | 14.915 | 3.344760223e-05 | 通过 |

FHE 时间取推理日志 `Run GPU time`，不含编译、上下文/密钥准备、provision 和解密。CPU RSS 为 GNU time 峰值；GPU 进程显存每 0.5 秒采样，池内每 0.1 秒采样，GiB = 2³⁰ bytes。GPU 进程预留显存与池内实际峰值分别报告。每任务使用当前 CSV 样本，每模式实测一次；本表不代表完整数据集准确率或多轮性能统计。

## 参数加密准备

| 任务 | provision wall | 参数密文数 | 密文 payload bytes |
| --- | --- | ---: | ---: |
| BERT | 0:50.50 | 278 | 641,836,778 |
| cifar10 | 0:07.69 | 25 | 72,886,231 |
| imagenet | 0:57.40 | 141 | 269,537,235 |
| retina | 1:08.98 | 1,055 | 1,136,503,313 |
| ViT | 0:54.15 | 253 | 784,439,491 |
| yolov5 | 0:53.10 | 49 | 81,282,751 |

## 证据与复现

命令见六任务的 `runs/{BERT,cifar10,imagenet,retina,ViT,yolov5}/run_pipeline.md`。本机证据根目录为 `/tmp/onnx_split_validation/`，每个任务保留：

- `split/`：prefix/suffix 与 external data、边界 CSV、`split_manifest.json`、完整原图输出和 prefix 输出 NPZ。
- `{input,parameter}/task/input_contract.json`：模式对应的实际物理打包与 CKKS 上下文。
- `{input,parameter}/run/`：`inference.log`、`inference.metrics.json`、解密 CSV、汇合 NPZ 和逐输出 `comparison.json`；参数模式另保留 `output.ct`。
- `recheck_prepare.log`、`recheck_identity.json`：最终脚本的重复切分与明文验证；`parameter/provision.time`、`parameter/bundle.json`：provision 与密文存储记录。

CIFAR10/ImageNet 参数加密的最新证据位于 `parameter_fixed/`，其中 `run_threads8/` 保留推理指标、解密输出和 `comparison.json`，`provision.time`、`bundle.json` 保留准备耗时与存储指标。YOLOv5/Retina/BERT/ViT 参数加密的最新证据位于各任务的 `parameter_revalidation_20261116/`，推理指标与误差比较位于其 `run/`。

验收后仅清理本次生成的临时 CKKS context/key 文件以释放磁盘，删除路径记录在各任务 `transient*keys_removed.json`；模型、边界、参数密文、解密结果和比较报告保留。重新执行推理时需按流程重新准备上下文。

回归验证：CMake GPU 构建成功；35 项 Python 测试通过，覆盖稳定拓扑、残差/多输出/旁路、Constant/initializer、external data、CSV/shape/dtype、float64 原图基准、歧义/控制流拒绝、布局数量校验、多分辨率边界及矩阵入口 drop_level。
