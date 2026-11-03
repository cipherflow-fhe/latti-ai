# BERT / ViT 参数加密 GPU runtime_lazy 验收

验收标准按用户最新要求为：**全部原始 ONNX 输出的最大绝对误差严格小于 0.2**，等于 0.2 不通过。BERT 和 ViT 均已完成一次完整的实际 GPU 参数加密外包推理、输出解密及数值比较，均通过。

## 实测结果

| 指标 | BERT | ViT |
| --- | ---: | ---: |
| 原始输出形状 / 比较数量 | `logits [1,2]` / 2 | `output [1,197,10]` / 1,970 |
| 最大绝对误差 | 0.1090362757863583 | 0.0029372640703548697 |
| FHE GPU 推理时间 | 11,197.70 s（约 3 h 06 min 38 s） | 6,720.18 s（约 1 h 52 min） |
| Runner 总墙钟时间 | 11,561.05 s | 6,788.22 s |
| Runner 峰值 RSS | 36.581 GiB | 33.778 GiB |
| RMM 实际使用显存峰值 | 约 21.13 GiB | 约 35.53 GiB |
| 整卡显存采样峰值（含预留池及上下文） | 42.960 GiB | 42.958 GiB |
| 加密参数目录字节数 | 1,077,024,177,941 | 379,211,358,215 |
| 加密参数目录大小 | 1,003.057 GiB（约 1.077 TB） | 353.168 GiB（约 379.211 GB） |
| 完整 Runner 目录字节数 | 1,081,307,609,865 | 382,958,020,577 |
| 参数密文数量 | 306,921 | 78,159 |

硬件为 NVIDIA RTX 5880 Ada Generation（每卡 49,140 MiB），BERT 使用物理 GPU 1，ViT 使用物理 GPU 0。两者 RMM 预留池容量约 42.447 GiB；显存监控 CSV 中的 `used_gb` / `pool_gb` 实际按 1024³ 换算，本文统一标为 GiB。RSS 取 `/usr/bin/time` 的峰值，整卡显存为约 0.5 秒采样，RMM 为约 100 ms 采样；后两者是采样峰值。FHE 时间来自日志 `Run GPU time`，不包含离线参数加密，Runner 墙钟时间包含启动及加载等开销。目录大小包含目录内 manifest 等文件的逻辑字节数，不包含 provisioner 的密钥。以上为单次完整运行结果。

## 基准与误差来源

原始 ONNX 和输入 CSV 没有修改。基准由 `tools/evaluate_transformer_onnx_csv.py` 直接读取原始 ONNX 权重、属性及 CSV，以 NumPy float64 执行自定义算子，不读取编译器的推理输出。

- BERT：`runs/BERT/model/trained_poly.onnx` 和 `runs/BERT/input.csv`（44×768）。原始模型声明 128 个 token，mask 仅前 44 个有效；使用这些有效 token，并按模型的 CLS 选择语义输出全部 2 个 logits。Runner 已显式传递 `select_first_token`，验收没有对解密结果临时截取。
- ViT：`runs/ViT/model/trained_poly.onnx` 和实际流水线输入 `runs/ViT/patch_input.csv`（197×768），比较全部 1,970 个输出。ONNX 输出 batch 维声明为 `-1`，本次实际 batch 为 1，对应 `[1,197,10]`。

BERT 额外使用原始导出项目 `/home/linghm/clinical-demos/Retina-DKD/external_tools/nonlinear.py` 中的 PyTorch `PolySoftmax`、`PolyLayerNorm`、`PolyGELU`、`PolyTanh` 执行全部 99 个原始 ONNX 节点。其输出与 NumPy 基准最大差为 **2.5757e-14**。源码 SHA256：`444115cbd42e3e3ec4764717904769be30f21054f95241cb138f2ad78ecd430a`。

原始 softmax 的 `inverse_epsilon`、`inverse_alpha`、`final_inverse_alpha` 用于控制逆运算收敛，不能作为分母加数。该 BERT 导出配置使用 `use_asor=0` 和 15 次逆运算迭代，基准保留这一有限迭代语义。先前跟随编译器分母 guard 的基准不符合原始导出语义，曾得出约 0.00197 的误差；该结果已纠正。编译器现有的 `thor-wide` 稳定化会引入近似偏差，此次保留实测编译路径，未为新门槛修改其行为。

| BERT logits | 第 1 项 | 第 2 项 |
| --- | ---: | ---: |
| 原始模型 NumPy 基准 | -3.160952131436212 | 6.080518208917383 |
| GPU 解密输出 | -3.2699884072225704 | 6.173116736713766 |

BERT 满足新的 `< 0.2` 标准，但不满足此前的 `0.1` 标准。

提交前已复查：为旧 `0.1` 门槛临时尝试的分母 guard 移除及逆运算 bootstrap scale 调整均已撤回。当前修改仅纠正原始 ONNX 基准、增加对应回归测试、补充编译器说明注释并记录验收结果，不改变已实测的推理实现或配置。

## 执行配置与核验

两模型均使用 `deployment_mode=server_provisioned_runner`、`input_mode=plaintext`、`parameter_mode=encrypted_offline`、`parameter_loading_mode=runtime_lazy` 和 `par_diagonal_pack`。本次验收覆盖明文输入、加密参数、密文输出的外包推理路径。

BERT 当次验收在编译时显式传入 `training/config/config.json`，该文件现已原样重命名为 `training/config/BERT_base_config.json`。ViT 当次使用该配置的实验副本 `/tmp/transformer-param-gpu-validation/ViT/config.json`，设置 `n_heads=3`、`model_type=vit`；现在对应 `training/config/ViT_tiny_config.json`，省略不参与 ViT 编译的 BERT 专用字段。两份 `run_pipeline` 已分别引用对应配置。没有新增调度限制。

已核验完整原始输出的名称、形状、索引连续性、无重复及全部有限；Runner 返回码均为 0。逐参数核验 task signature 与 manifest 的 ID、level、size、类型及 layer，以及全部密文容器的数量、元素偏移、长度和文件大小。Runner 不携带明文模型 H5 或私钥文件。PCMM 权重、bias 及数值系数走离线参数密文，结构 mask 仍为公开明文。

两次运行的 `canonical_loads` 分别为 306,921 和 78,159，与对应离线参数数量一致；序列化密文缓存上限均为 64 MiB。这是参数加载缓存的计量，不代表完整进程或 GPU 内存上限。

相关 18 个 Python 单元测试通过，`git diff --check` 通过。推理二进制 SHA256 为 `3d274786476304e8da5e64520859144fdf84f1dd18790953a8d7a04a467a1128`；依赖 `inference/lattisense` 工作区未修改。

## 复核入口

完整实验根目录：`/tmp/transformer-param-gpu-validation`。每个 `<模型>-acceptance/` 包含 `report.json`、`runner.log`、`decrypted.csv`、`output.ct`、`runner-time.json` 及 CPU/GPU 内存监控文件。更新后的 JSON 明确记录严格 `< 0.2` 门槛、正确基准及审计结果。参考输出为 `<模型>-reference.npz`；BERT 独立复核记录为 `BERT-pytorch-source-crosscheck.json`。

重新计算原始模型基准（在仓库根目录，`MODEL` 为 BERT 或 ViT；ViT 输入改用 `patch_input.csv`）：

```bash
/home/linghm/pytorch_venv/bin/python tools/evaluate_transformer_onnx_csv.py \
  --onnx runs/BERT/model/trained_poly.onnx \
  --input-csv runs/BERT/input.csv \
  --output /tmp/transformer-param-gpu-validation/BERT-reference.npz
```

对已经完成 provision 的 Runner 执行 GPU 推理的原始命令如下；重新运行将再次产生数小时计算开销：

```bash
CUDA_VISIBLE_DEVICES=1 build/examples/inference \
  --task-dir /tmp/transformer-param-gpu-validation/BERT/task/runner \
  --input runs/BERT/input.csv --deployment-mode server_provisioned_runner \
  --output-cipher /tmp/transformer-param-gpu-validation/BERT-acceptance/output.ct --gpu

CUDA_VISIBLE_DEVICES=0 build/examples/inference \
  --task-dir /tmp/transformer-param-gpu-validation/ViT/task/runner \
  --input runs/ViT/patch_input.csv --deployment-mode server_provisioned_runner \
  --output-cipher /tmp/transformer-param-gpu-validation/ViT-acceptance/output.ct --gpu
```

解密使用 `build/tools/decrypt_runner_output --task-dir <模型>/task/provisioner --cipher <输出密文> --output-csv <解密CSV>`。精确完整运行命令、原始模型/输入/配置/二进制 hash 和字节计量保存在各自 `report.json` 中。大体积 task、密钥及实验 JSON 未纳入此次代码提交。
