# v6 语义下移植 encparams 功能与六任务 GPU lazy 验收规范

状态：设计规范
依据：v6 侧 feat/lattisense-adapt 代码、feat/linghm-encparams 功能代码、runs/{BERT,cifar10,imagenet,retina,ViT,yolov5}/run_pipeline.md（原研究文档已按要求删除）
目标：在 v6 语义和接口下重新移植 feat/linghm-encparams 的参数加密功能，并用 BERT、cifar10、imagenet、retina、ViT、yolov5 六个任务的输入加密 GPU lazy 和参数加密 GPU runtime_lazy 共 12 条主链路验收；BERT、ViT 还须复测原文档已有的 PDM 输入加密链路。

## 1. 范围与不可变原则

本规范只定义合并分支的代码修改和验证方法，不改变 v6 侧接口语义，也不恢复旧 SDK 的兼容层。合并分支仅保留 v6 语义，不为旧语义新增兼容类、适配类或 wrapper；所有 encparams 功能都必须按 v6 侧类型、metadata、level、scale 和 task graph 重新实现。

必须遵守：

1. v6 的 Lattigo/Lattisense ABI、RingType、plaintext metadata、MegaAG task node、序列化接口是唯一基准。
2. feat/linghm-encparams 只提供要移植的功能：encrypted parameter store、server_provisioned_runner、runtime-lazy、periodic/mask、seed、参数计数和 storage-aware level。
3. 共轭子环下容量统一命名为 n_real_num_slot，取值为 v6 parameter.max_slots()；latti-ai 层不得再用 N/2 作为实数容量。
4. C++ layer 的 ringt plaintext 必须通过 fhe_api::encode 或 fhe_api::encode_ringt 创建。
5. ringt_to_mult 加 mult_plain_ringt 合并为 v6 mult；add_plain_ringt 使用 v6 add overload。
6. parallel_for 共享 v6 线程安全 context，不创建 context copy；callback 的临时 vector 和索引保持线程私有。
7. 删除 data_compress 和 CkksCompressedCiphertext 路径；密文直接调用 serialize()/deserialize()。
8. Python layer 的每个 plaintext node 显式写出与 C++ 生成函数相同的 scale、level 和 ringt metadata。
9. MegaAG node id、task signature index、layer data source 和 feature input/output id 全局唯一。
10. 只把生成的 task、runner、manifest、性能采样和日志写入 /tmp/ 下的独立验收目录，不覆盖 runs/ 下任何已有模型、输入和产物。
11. 本轮 goal 固定使用现有同态参数；BERT 保持 PN15Default（共轭不变环 N=32768、默认 scale=2^40），不得以提高 N、scale 或模数精度替代算子正确性排查。

本规范不要求在本阶段重写模型结构、改变 activation packing、恢复 v6 已删除的 FPGA/HEonGPU 路径，也不把理论 slot 容量优化当作验收目标。

## 2. 分阶段实现

### 2.1 基线和 submodule

合并分支以 v6 侧 inference/lattisense tip、Lattigo v6 和 lattisense-gpu-core 为基线。encparams 侧 submodule 的 5 个协议行为单独移植：

| 行为 | 目的 |
| --- | --- |
| extra task signature fields | 保存 encrypted parameter、use-site level 和运行时触发信息 |
| custom compute wait inputs | 保证 lazy 参数在前驱输入就绪后加载 |
| online/offline argument validation | 防止两种部署模式混用 |
| inputless encode_pt | 生成 runner-local/public mask |
| GPU memory monitor path | 记录验收时的 GPU 内存证据 |

每个行为作为可独立回溯的 commit，不能把旧 submodule tip 直接覆盖 v6 tip。

#### 2.1.1 latti-ai 主仓库：切出分支和合并方向

主仓库从 `feat/linghm-encparams` 的 `02abe9e` 切出工作分支 `feat/v6-adapt-encparams`，将 `origin/feat/lattisense-adapt` 的 `d5fcca9` 合并到该工作分支。源分支均不直接修改。保留 encparams 分支历史只为保留待移植功能；发生语义或接口冲突时仍以 v6 为唯一基准，不能据此保留旧语义或添加兼容层。

以下是首次建立工作分支的命令；恢复执行时先检查 `git status` 和当前分支，不重复切出已存在的分支或启动第二次 merge：

    cd /home/linghm/project/latti-ai
    git fetch origin feat/linghm-encparams feat/lattisense-adapt
    git switch feat/linghm-encparams
    git rev-parse HEAD
    # 本次 encparams 基线必须为 02abe9e；不使用 reset 覆盖已有工作。
    git switch -c feat/v6-adapt-encparams
    git rev-parse origin/feat/lattisense-adapt
    # 本次 v6 基线必须为 d5fcca9；若远端已推进，仍使用下列固定提交。
    git merge --no-commit --no-ff d5fcca9603790768cff3aaccade588eff99f829a

解决主仓库冲突时，按第 2.2 节逐项把 encparams 功能改写为 v6 接口；不得对整个冲突文件统一选 ours/theirs 而丢失功能或带回旧接口。示例 task、compiled_mega_ag.json 和模型二进制不作为本次验收证据，验收必须在独立目录重新生成。merge 中的 `inference/lattisense` 冲突按下一节处理，不能合并旧 SDK tip。冲突全部解决并检查后，显式暂存相关源码并完成基线 merge：

    git diff --name-only --diff-filter=U
    # 上一命令必须无输出；逐项检查并 git add 实际解决的文件。
    git diff --cached --check
    git commit -m "Merge v6 adaptation baseline into encparams migration branch"

基线 merge 后，容量、序列化、各 FHE layer、generator/compiler、runner/store 的修改和修正分别提交；不得把验收目录加入任何 commit。记录方向始终为 `feat/lattisense-adapt -> feat/v6-adapt-encparams`，不向两个源分支反向合并。

#### 2.1.2 inference/lattisense：从 v6 基线切出并移植行为

主仓库 v6 提交绑定的 SDK 基线为 `898bc396cb247426e5c76ab2ae7702558b0c3766`，属于 SDK 的 `origin/feat/lattigo-v6-adaptation` 分支。SDK 的目标分支也命名为 `feat/v6-adapt-encparams`，但它与主仓库同名分支属于两个独立 Git 仓库。必须从该 v6 固定提交切出，不能从 SDK 的旧 `feat/linghm-encparams` tip `e0db939` 切出后覆盖 v6。

    cd /home/linghm/project/latti-ai/inference/lattisense
    git fetch origin feat/lattigo-v6-adaptation feat/linghm-encparams
    git switch -c feat/v6-adapt-encparams 898bc396cb247426e5c76ab2ae7702558b0c3766

SDK 不执行 `git merge feat/linghm-encparams`。从旧 SDK 功能分支向新 SDK 工作分支按以下顺序逐个移植提交，每次应用后都检查并按 v6 task node/ABI 语义调整；检查完成才提交，前一个行为未完成时不执行下一个：

| 源提交（旧 SDK 功能分支） | 移植行为 |
| --- | --- |
| `1f17d57` | extra task signature fields |
| `108f6ae` | custom compute wait inputs |
| `c1a248d` | online/offline argument validation |
| `7e7cf87` | inputless encode_pt |
| `e0db939` | GPU memory monitor path |

源补丁先用 `git show` 检查，再按 v6 的模块布局重写；旧版 `frontend/custom_task.py` 中的签名和 node 实现已拆分到 v6 的 `frontend/linker/` 与 `frontend/types.py`，因此本次不直接 cherry-pick 旧版文件。检查命令如下，`<source_commit>` 依次取表中值：

    git show --stat <source_commit>
    git show <source_commit>
    # 按 v6 语义编辑对应模块，显式 git add 实际修改的文件。
    git diff --cached --check
    git commit -m "<本次 v6 行为移植说明>"

这只是功能移植入口，不是允许保留旧接口。本次 SDK 行为提交为 `2bd6550`（离线参数和额外签名字段）、`3f9e241`（参数边界校验）、`f7857f3`（inputless encode_pt）、`387731e`（以 v6 普通 graph input 表达等待依赖）。`e0db939` 的内存监控路径修复在 v6 基线 `mega_ag_runners/gpu/gpu_wrapper.cu` 中已经存在，无须重复提交。后续修正仍按模块单独提交，不引入兼容类或 wrapper。

#### 2.1.3 SDK 嵌套子模块及主仓库指针

Lattigo 与 GPU core 使用 SDK v6 基线绑定的提交，不创建功能分支、不合并旧 HEonGPU/FPGA 分支：

| 嵌套子模块 | 固定提交 |
| --- | --- |
| `fhe_ops_lib/lattigo` | `22bff324e146968c186106524c06cea607462b05` |
| `backends/lattisense-gpu-core` | `aea1ef873aed3e199aa785a0e4f42d8a42b9351f` |

    cd /home/linghm/project/latti-ai/inference/lattisense
    git submodule sync --recursive
    git submodule update --init --recursive -- fhe_ops_lib/lattigo backends/lattisense-gpu-core
    git submodule status --recursive

`submodule update` 将嵌套仓库检出到 SDK gitlink 对应提交（detached HEAD），不执行嵌套仓库的 `git pull` 或分支 merge。旧工作区残留的 HEonGPU 目录不作为构建输入，也不重新加入 `.gitmodules`。SDK 行为提交完成后，在主仓库记录其最终指针；若主仓库 merge 尚未结束，先解决其他冲突再完成基线 merge，否则独立提交 SDK 指针：

    cd /home/linghm/project/latti-ai
    git -C inference/lattisense log --oneline -6
    git add inference/lattisense
    # 非 merge 状态下独立提交；merge 状态使用第 2.1.1 节的基线提交。
    git commit -m "Port encparams task protocols onto the v6 SDK baseline"

主仓库提交必须引用包含移植行为的 SDK commit，不能仅在 SDK 工作区保留未提交修改。两层仓库的源码提交与验收产物分开管理，任何 push 均不属于本规范的本地实现和验收步骤。

### 2.2 latti-ai v6 适配层

优先修改 data struct 和公共 API 使用点，再修改 layer。建议顺序：

1. 统一 plaintext/ciphertext 类型、参数访问和 fhe_api 调用。
2. 引入 n_real_num_slot，替换 Feature0D、Feature1D、Feature2D、FeatureMat、reshape、conv、poly、transpose、PCM 和 model compiler 中的实数容量计算。
3. 修改 parallel_for 和 parallel_for_with_extra_level_context，使所有线程调用同一个 context。
4. 删除 compressed ciphertext 字段、压缩 encrypt/decompress 和 compressed framing。
5. 逐类移植 dense、conv、inverse conv、poly、block-col-major 和 PCM。
6. 重新接入 encrypted parameter signature、materialize_encrypted_param、provisioner、runner 和 runtime-lazy executor。
7. 最后修改 model generator 和 MegaAG 生成，重新生成 task metadata。

每次修改只保留一个可验证的语义变化；不在同一 commit 中混入无关格式化或模型重构。

### 2.3 容量和布局

统一定义：

    n_real_num_slot = context.parameter().max_slots()

poly modulus degree 仍可在底层 v6 参数中使用 n 或 N，但不能作为 latti-ai feature 容量。所有依赖容量的计算必须显式区分：

| 概念 | 名称 | 用途 |
| --- | --- | --- |
| 多项式阶数 | poly_modulus_degree 或 n | 参数构造、底层环 |
| 实数槽容量 | n_real_num_slot | feature packing、mask、vector 长度、channel count |
| layer use-site level | use_site_level | 参数密文 level 和 graph metadata |
| plaintext scale | pt_scale | C++/Python encode metadata |

在 v6 conjugate-invariant 参数下，C++、Python、task JSON 和 manifest 对同一密文必须得到相同的 n_real_num_slot。不得在一侧使用 max_slots()、另一侧使用 N/2。

### 2.4 v6 运算接口

C++ layer 的目标调用约定：

| 功能 | 规范调用 |
| --- | --- |
| 普通编码 | fhe_api::encode(ctx, values, level, scale) |
| RingT 编码 | fhe_api::encode_ringt(ctx, values, scale) |
| 密文乘 RingT plaintext | ctx.mult(ciphertext, plaintext) |
| 密文加 RingT/plaintext | ctx.add(ciphertext, plaintext) |
| 密文序列化 | ciphertext.serialize() |
| 密文反序列化 | CkksCiphertext::deserialize(bytes) |
| 参数 q | parameter.q()[level] |
| 默认 scale | parameter.default_scale() |

seed 恢复必须保留 v6 plaintext metadata 的 ringt、level 和 scale；不能恢复旧的 ringt_to_mult 中间对象。结构 mask 也必须通过 fhe_api::encode_ringt 生成，并与 C++ layer 的实际 q[level] 或 q[level-1] 一致。

### 2.5 Python layer 与 C++ scale 合同

每个 Python CkksPlaintextNode 必须显式记录 scale。生成规则由 C++ layer 的实际生成函数决定：

| 参数 | C++ 来源 | Python 要求 |
| --- | --- | --- |
| weight | q[use_site_level] 及 residual/seed 因子 | 写相同 scale 和 level |
| mask/select | mask 所在 level，通常 q[level-1] | 写相同 scale 和 level |
| bias/add | default_scale 或 layer 明确的 D/q 组合 | 禁止省略 scale |
| polynomial coefficient | degree 对应的 cached coefficient scale | 每个 degree 独立记录 |
| seed restore | seed、mask、恢复后 plaintext metadata | 三者一致 |

task generator 在导出前检查 C++ 规则表、Python node metadata 和 task signature 的 level/scale；任一不一致直接失败。不得以 default scale 作为隐式补全。

### 2.6 encrypted parameter 与 runtime-lazy

保留 feat/linghm-encparams 的功能模型：

    materialize_encrypted_param()
      -> load_encrypted_param_ct(store, canonical_arg_id, flat_index)
      -> v6 mult/add 运算

runtime-lazy 只在第一次消费者需要时读取 canonical parameter。共享 seed 只加载一次；不同恢复分支不得原地修改共享 ciphertext。参数 store 必须记录 load、cache hit、eviction 和峰值字节数。

server_provisioned_runner 的 bundle 不包含明文模型参数、secret key 或 secret context。Provisioner 持有明文参数并生成加密参数；Runner 只持有 eval context、task graph 和参数密文；Decryptor 在 provisioner 侧解密输出。v6 context 序列化不包含 secret key；Provisioner 分别持久化 `eval_context.bin` 和原生 `SecretKey::serialize()` 的 `secret_key.bin`，Decryptor 使用原生 `set_secret_key()` 恢复解密能力，不解析旧 secret context 格式。

### 2.7 MegaAG 唯一性

生成节点的命名空间使用：

    <task>/<erg>/<layer_id>/<op_kind>/<parameter_kind>/<logical_index>

生成结束时检查：

* node id 到 producer 唯一；
* task signature index 到 node id 唯一；
* layer id 到 data source 唯一；
* feature id 的输入/输出节点数量与 task config 一致；
* eager、runtime-lazy、public mask 和 encrypted parameter 不重复占用 id。

compiled_mega_ag.json 必须由最终 v6 generator 重新生成；禁止手工合并远程生成文件。

## 3. 代码修改清单

### 3.1 C++ data struct 和 FHE layer

重点模块：

* inference/data_structs/feature.cpp：共享 context 的 parallel_for、序列化 helper。
* feature0d.cpp/h、feature1d.cpp/h、feature2d.cpp/h、feature_mat.cpp/h：n_real_num_slot、普通 ciphertext 容器、直接序列化。
* inference/fhe_layers/dense_packed_layer.*：v6 plaintext 返回类型、seed/lazy 参数、bias scale。
* multiplexed_conv2d_pack_layer.*、multiplexed_conv2d_pack_layer_depthwise.*：weight、mask、bias 的 v6 encode 和 mult/add。
* inverse_multiplexed_conv2d_layer.*：execution index、canonical seed、boundary mask。
* poly_relu*、par_block_col_major_*：逐 degree、逐 level、逐 scale 对齐。
* inference/inference_task/inference_process.cpp：v6 custom node 输入和 inputless encode_pt。
* inference/interface/encrypted_parameter_store.*、inference_provisioner.*、inference_runner.*：参数 signature、bundle、runtime-lazy executor。

删除内容：

* CkksCompressedCiphertext 成员和函数；
* encrypt_symmetric_compressed、compressed_ciphertext_to_ciphertext；
* 以 parameter 参数传入的旧 ciphertext serialize overload；
* resize_copies、get_copy 和每实例 context 副本；
* 旧 ringt_to_mult、mult_plain_ringt、add_plain_ringt 调用。

### 3.2 Python generator 和 compiler

重点模块：

* inference/model_generator/deploy_cmds.py：v6 参数构造、n_real_num_slot、部署模式、node 注册。
* inference/model_generator/layers/encrypted_param_ops.py：canonical 参数 materialization。
* multiplexed/inverse/dense/poly layer：显式 scale、level、ringt node 和 lazy signature。
* training/model_compiler/components.py、pipeline.py、score.py：v6 parameter name、容量、实际 use-site level 和 storage score。
* inference/interface/gen_mega_ag.py：两个部署模式和 runtime_lazy 参数 loading mode。

保留 storage-aware score 和 parameter count，但计数必须使用最终 v6 graph 的 use-site level、密文字节校准和 canonical parameter manifest；不能按旧 N/2 或旧 compressed type 计费。

### 3.3 BERT 在现有同态参数下的数值语义

BERT 固定使用现有 PN15Default，不提高环阶数、默认 scale 或模数精度。先以原始 ONNX 中间结果定位首次偏离，并逐槽检查有效位置与 padding；诊断程序、密钥、CSV 和日志均位于 `/tmp/`。padding mask 必须按实际 token、head 和列范围生成，CKKS 近似噪声不应被当作缺失 mask 的证据。

softmax 必须保留原导出器有限次 Newton 迭代的语义。epsilon/alpha 是收敛控制量，不能作为额外常数加进分母。若分母放大 s 倍，倒数状态须同步表示为 a/s，首步为 `2/s - b_scaled/s²`，后续仍执行 `a_scaled * (2 - b_scaled * a_scaled)`；不能只缩放分母、继续使用 `2-b_scaled` 初始化。初始信号放大折入指数多项式系数，再执行原有平方，避免用 `65536` 的大常数明文直接乘微小分母。后续归一化在平方前将概率乘以 `sqrt(s)`，平方结果和分母同时带有 s 倍信号；不能在微小平方结果产生后才放大分母、最后再放大概率，否则会同步放大 rescale 的绝对误差。bootstrap 前后仅做互逆的信号缩放，并按迭代次数约束倒数状态幅值；这不改变同态参数或参考模型。

须以 PCM/PDM 数值回归对比原始有限次迭代，并重新执行 BERT 三条完整 GPU 链路；中间层诊断不能替代全部 2 个 CLS logits 的正式验收。

### 3.4 Pooling 类内平均与 level/scale

Avgpool1DLayer/Avgpool2DLayer 的普通、adaptive 和 split 路径必须在类内部返回真正的平均值。对当前支持的非重叠窗口（kernel_shape 等于 stride），二维系数为 `1/(kernel_shape[0]*kernel_shape[1])`，一维系数为 `1/kernel_shape[0]`。旋转求和后，类内乘公开 RingT 明文并 rescale；不得把这一步移到类外，也不得把平均系数吸收到前后 Conv/FC 权重或多项式系数。ONNX 参数提取必须读取取消该吸收后的 weight_scale/coeff_scale，不再对相邻权重重复除以窗口面积。

若乘法输入密文处于 level L、scale S，平均明文的 scale 必须为 `q[L]`（RingT metadata level 为 0），乘法后 scale 为 `S*q[L]`，rescale 后输出 level 为 `L-1`、scale 恢复为 S。Python graph、C++ native、compiler level_cost、bootstrap 调度及算子计分必须一致：原先只求和的路径 level_cost 由 0 改为 1。inputless encode_pt 在输入加密执行器和参数加密 Runner 中均只依赖公开 divisor/level，不能读取明文模型或变成加密模型参数。

普通 multiplexed pooling 的 selection mask 已包含平均系数，保持原有一次乘法/rescale，禁止再次归一化。split repack 路径将平均系数并入类内现有 repack mask，仍只消耗一个 level；无 repack 路径在类内补乘平均明文。须验证矩形窗口、数值均值、scale 保持、level 消耗和不存在悬空 PT source。

回归范围以实际图为准：已验收的 YOLOv5、BERT、ViT 图不调用这些 pooling 类；CIFAR10、ImageNet 的 GlobalAveragePool 使用 adaptive 路径，不能仅因 pack_style=multiplexed 就认定不受影响。CIFAR10、ImageNet、Retina 的两种加密模式都必须重新编译、生成图并完整验收。旧 CNN 结果仅为历史证据，不可代表本次修改后的通过状态。Retina 还需验证 GAP 输出均值在进入 bootstrap 前已恢复到原模型幅值，避免求和放大 256 倍后超出 bootstrap 数值范围。

## 4. 端到端验收目录和环境

所有验收放入：

    /tmp/v6_six_task_validation/<TASK>/

建议环境变量：

    REPO="$(git rev-parse --show-toplevel)"
    PY=/home/linghm/pytorch_venv/bin/python3
    CMAKE=/home/linghm/pytorch_venv/bin/cmake
    TASK=cifar10  # 依次取 BERT、cifar10、imagenet、retina、ViT、yolov5
    DATA="$REPO/runs/$TASK"
    ROOT="/tmp/v6_six_task_validation/$TASK"
    MODEL="$DATA/model/trained_poly.onnx"
    INPUT="$DATA/img.csv"
    export PYTHONPATH="$REPO/inference/lattisense:$REPO"
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8

各任务必须保留原 run_pipeline.md 的配置和输入语义；下文命令以 cifar10 为模板，其余任务按此表替换，并以各自 run_pipeline.md 的完整命令为准：

| 任务 | 模型（相对 DATA） | 输入（相对 DATA） | 输入加密配置 | 参数加密配置 | 参考/输出规则 |
| --- | --- | --- | --- | --- | --- |
| BERT | model/trained_poly.onnx | input.csv | BERT_pcm_config.json | BERT_pcm_config.json | evaluate_transformer_onnx_csv.py，onnx.npz，全部 2 个 CLS logits |
| cifar10 | model/trained_poly.onnx | img.csv | 默认 | storage_aware_config.json | evaluate_onnx_csv.py，onnx.csv，全部 10 个 logits |
| imagenet | model/trained_poly.onnx | img.csv | 默认 | imagenet_storage_config.json | 全部输出 |
| retina | model/epoch041.onnx | input_0=img.csv、input_1=non_inv.csv、input_2=inv.csv | 默认 | storage_aware_config.json | 明文与两种 FHE 模式均显式绑定三个输入 |
| ViT | model/trained_poly.onnx | patch_input.csv | ViT_pcm_config.json | ViT_pcm_config.json | evaluate_transformer_onnx_csv.py，onnx.npz，全部 1970 个输出 |
| yolov5 | model/best_conv_outputs.onnx | img.csv | 默认 | storage_aware_config.json | 使用原文档三个 --output-map，三个输出全部比较 |

配置文件均位于 training/config/。BERT 和 ViT 的主验收使用 PCM，产物分别位于 task_input_encrypted/pcm 和 task_parameter_encrypted/pcm；两者附加 PDM 输入回归分别使用 BERT_base_config.json、ViT_tiny_config.json，产物位于 task_input_encrypted/pdm。BERT 自定义算子和 attention mask 语义从原图识别，不通过兼容类恢复旧 SDK 行为。

每个 GPU 同时只运行一个推理进程，BERT 等大内存输入加密任务应避免同时挤占主存。测量使用原文档的 OMP_NUM_THREADS=8、OPENBLAS_NUM_THREADS=8；记录具体 GPU、提交、配置、命令和返回码。FHE 时间不含编译、密钥准备、provision 和解密；CPU RSS 为推理进程生命周期最大 RSS，GPU 进程峰值包含预留，池内峰值另列。所有中间日志和采样文件的 cwd/output 必须在 /tmp/ 下。

GPU build 只执行一次：

    "$CMAKE" -S "$REPO" -B "$REPO/build" \
      -DINFERENCE_SDK_ENABLE_GPU=ON \
      -DLATTISENSE_DEV=ON \
      -DLATTISENSE_CUDA_ARCH=89 \
      -DCMAKE_BUILD_TYPE=Release
    "$CMAKE" --build "$REPO/build" -j8

先生成同一份明文参考：

    "$PY" "$REPO/tools/evaluate_onnx_csv.py" \
      --onnx "$MODEL" \
      --input-csv "$INPUT" \
      --output-csv "$ROOT/reference/onnx.csv" \
      --report "$ROOT/reference/onnx.json"

参考必须从各任务原始 ONNX 和原始 CSV 重新生成，保存全部输出；top-1 只能作为附加信息，不能作为唯一正确性标准。Retina 比较全部输出，YOLOv5 比较三个输出，BERT 比较 CLS 的全部 2 个 logits，ViT 比较全部 1970 个输出。不得截断输出、只比较单个输出或修改明文参考以迁就 FHE 结果。

## 5. 验收 A：六任务输入加密 GPU lazy

该模式使用 client_encrypted_input。模型参数保持明文/已有 server 路径，客户端加密输入，GPU server 执行 v6 MegaAG graph。lazy 指 graph 的 custom compute 和按需执行路径，不把它解释为 encrypted parameter runtime_lazy。

编译和生成：

    OUT="$ROOT/task_input_encrypted"
    mkdir -p "$OUT"
    "$PY" "$REPO/training/run_compile.py" \
      --input "$MODEL" --output "$OUT" --style multiplexed
    "$PY" "$REPO/inference/interface/gen_mega_ag.py" \
      --task-dir "$OUT/task"

GPU 执行：

    "$PY" "$REPO/tools/measure_gpu_command.py" \
      --output-dir "$OUT/run" -- \
      "$REPO/build/examples/inference" \
      --task-dir "$OUT/task" \
      --input "$INPUT" \
      --output-csv "$OUT/decrypted.csv" \
      --gpu

比较：

    "$PY" "$REPO/tools/compare_inference_csv.py" \
      --reference "$ROOT/reference/onnx.csv" \
      --actual "$OUT/decrypted.csv" \
      --report "$OUT/comparison.json"

验收条件：

* compile、MegaAG 生成和 GPU inference return code 都为 0；
* 输出 CSV 包含完整 logits，输出名和样本索引正确；
* comparison.json 的全部输出 max_abs 小于 0.2；六个任务及 PDM 回归均不放宽该阈值；
* task graph 中所有 plaintext node 使用 v6 metadata，node id 和 signature index 无重复；
* n_real_num_slot、feature packing、level 和 scale 在 task JSON 与 C++ runtime 一致；
* GPU 日志存在 FHE time 和无异常 context/thread 错误。

## 6. 验收 B：六任务参数加密 GPU runtime_lazy

该模式使用 server_provisioned_runner、encrypted_offline 和 runtime_lazy。输入保持明文；Provisioner 先加密模型参数并导出 Runner bundle；Runner GPU 按需从 encrypted parameter store 读取参数。

编译：

    OUT="$ROOT/task_parameter_encrypted"
    mkdir -p "$OUT"
    "$PY" "$REPO/training/run_compile.py" \
      --input "$MODEL" --output "$OUT" --style multiplexed \
      --config "$REPO/training/config/storage_aware_config.json" \
      --deployment-mode server_provisioned_runner

生成 runtime-lazy graph：

    "$PY" "$REPO/inference/interface/gen_mega_ag.py" \
      --task-dir "$OUT/task" \
      --deployment-mode server_provisioned_runner \
      --parameter-loading-mode runtime_lazy

Provision runner bundle：

    /usr/bin/time -v -o "$OUT/provision.time" \
      "$REPO/build/tools/provision_encrypted_runner_bundle" \
      --task-dir "$OUT/task" \
      --out "$OUT/task/runner"

GPU Runner：

    "$PY" "$REPO/tools/measure_gpu_command.py" \
      --output-dir "$OUT/run" -- \
      "$REPO/build/examples/inference" \
      --task-dir "$OUT/task/runner" \
      --deployment-mode server_provisioned_runner \
      --input "$INPUT" \
      --output-cipher "$OUT/output.ct" \
      --gpu

Decrypt 和比较：

    "$REPO/build/tools/decrypt_runner_output" \
      --task-dir "$OUT/task/provisioner" \
      --cipher "$OUT/output.ct" \
      --output-csv "$OUT/decrypted.csv"

    "$PY" "$REPO/tools/summarize_parameter_bundle.py" \
      --task-dir "$OUT/task" \
      --output "$OUT/bundle.json"

    "$PY" "$REPO/tools/compare_inference_csv.py" \
      --reference "$ROOT/reference/onnx.csv" \
      --actual "$OUT/decrypted.csv" \
      --report "$OUT/comparison.json"

验收条件：

* compile、MegaAG、provision、GPU Runner、decrypt 全部成功；
* task_config 明确包含 deployment_mode=server_provisioned_runner、parameter_mode=encrypted_offline、input_mode=plaintext、parameter_loading_mode=runtime_lazy；
* Runner 目录不包含 secret_key.bin、secret_context.bin 或 model_parameters.h5；
* manifest 的参数条目可被 runtime-lazy store 读取，日志显示 parameter cache 统计；
* public mask 通过 v6 encode_pt executor 生成，不进入 encrypted parameter payload；
* C++ weight/mask/bias scale 与 Python plaintext node scale 一致；
* comparison.json 的 max_abs 小于 0.2；
* bundle.json 能区分 canonical encrypted parameter payload、metadata、eval context 和完整 task 大小；
* provision.time、GPU metrics、cache statistics 和 comparison.json 均保存到 OUT。

## 7. 两种模式的共同回归

完成全部 12 条主验收和 2 条 PDM 输入回归后，执行最小单测集：

    PYTHONPATH="$REPO/inference/lattisense:$REPO" "$PY" -m unittest \
      inference.unittests.test_encrypted_param_count \
      inference.unittests.test_server_provisioned_runner_generator \
      inference.unittests.test_parameter_acceptance_tools \
      inference.unittests.test_interleaved_seed_weights \
      training.model_compiler.unittests.test_storage_score \
      training.model_compiler.unittests.test_pool_bootstrap_scale

C++/v6 侧执行已构建的 data struct、FHE layer、heterogeneous GPU 和 runner store tests。具体 test target 以合并分支的 CMake 输出为准，不为本规范新增平行测试框架。

静态检查：

    rg -n 'get_n\\(\\) / 2|N / 2|n // 2' inference training
    rg -n 'CkksCompressedCiphertext|encrypt_symmetric_compressed|data_compress|data_compressed' inference
    rg -n 'resize_copies|get_copy|get_extra_level_context' inference/data_structs inference/fhe_layers
    git diff --check

允许出现 polynomial degree 的明确底层适配，不允许把它用于 feature 实数容量。后两条若仍有结果，必须逐项说明为何不属于 v6 迁移残留；默认验收失败。

本轮主仓库 compressed/context 检索无执行路径残留。SDK 基线中的 `fhe_ops_lib/schemes/ckks/ckks.h`、`ckks.cpp` 仅保留已注释的 compressed API 文本，`doc/doc-en/4_API_Reference/4_API_Reference.md`、`doc/doc-zh/4_API参考/4_API参考.md` 为 SDK 历史接口文档；这些命中不参与编译，不作为旧语义兼容实现保留或调用。容量检索中的 `n_rotation // 2` 为卷积核中心，`span // 2` 及 polynomial 的 `n // 2` 为算法循环边界，均不用于 feature 容量。

Pooling 修改的验证证据：`/tmp/v6_six_task_validation/final_python_regressions.log`（41 个测试通过）、`pool_native_regressions.log`（3 个测试、145 个断言通过）、`pool_compile_export_audit.json`（两种模式的 level/scale 与 ONNX 权重导出审计）、`retina/probe_gap_bootstrap/comparison.json`（现有 PN15Default 下全部 512 个 GAP→bootstrap 输出，max_abs=0.000103199）。数据结构与参数 store 分别通过 1、60 个断言。额外执行整个 `[spatial]` 时，原有 `rectangular_polynomial_channel_layout` 在 PN13Default 下出现 `3.01779e-5 > 1e-5` 的严格单测失败，见 `final_spatial_regressions.log`；该多项式路径本次未修改，未调整该测试阈值或同态参数，也不宣称整个 C++ 测试集通过。

## 8. 证据文件和完成定义

每次验收在本地保留以下证据文件，不纳入 Git commit：

    /tmp/v6_six_task_validation/<TASK>/reference/onnx.csv
    /tmp/v6_six_task_validation/<TASK>/reference/onnx.json
    /tmp/v6_six_task_validation/<TASK>/task_input_encrypted/comparison.json
    /tmp/v6_six_task_validation/<TASK>/task_input_encrypted/run/inference.metrics.json
    /tmp/v6_six_task_validation/<TASK>/task_parameter_encrypted/comparison.json
    /tmp/v6_six_task_validation/<TASK>/task_parameter_encrypted/run/inference.metrics.json
    /tmp/v6_six_task_validation/<TASK>/task_parameter_encrypted/bundle.json
    /tmp/v6_six_task_validation/<TASK>/task_parameter_encrypted/provision.time

规范完成必须同时满足：

1. 六任务各自的输入加密和参数加密 GPU 链路全部成功（12/12），BERT、ViT 的 PDM 输入加密回归也全部成功（2/2）。
2. 每条链路的全部输出、全部元素 max_abs 都小于 0.2，输出名、数量和索引与各自原始 ONNX 参考完全对应。
3. v6 API、plaintext metadata、n_real_num_slot、序列化和共享 context 检查通过。
4. 参数加密模式确实使用 runtime-lazy，而不是 eager 参数加载。
5. Runner 安全边界和 public mask/secret material 检查通过。
6. 没有因 v6 适配而删除 encrypted parameter、seed、storage-aware 或 inputless encode_pt 功能。
7. 所有修改可由独立 commit 对应到本规范的一个阶段，且没有无关重构。
8. 六份 run_pipeline.md 的命令和性能表均对应本轮 v6 实测，包含 CPU RSS、GPU 进程/池内峰值、FHE 时间、误差，以及参数加密的 provision 时间和 bundle 字节数。历史旧 SDK 数据不得作为本轮通过证据。

若任一条件失败，保留失败日志和最小复现 task，先修复对应语义层，再重新执行受影响任务的两种完整流程，并复测共享模块影响的其他任务；不能仅通过放宽误差阈值或改写验收产物来“通过”。

### 8.1 六任务 v6 实测结果

本轮证据根目录为 `/tmp/v6_six_task_validation/`。旧 CIFAR10 单任务验证和六份原文档中的旧 SDK 性能不作为本轮验收结果。以下状态只在读取本轮 comparison.json（pass=true）、inference.metrics.json（exit_code=0）及参数模式 bundle/cache 证据后更新。

Pooling 修复源码提交为 `460fa5a60defe4c19ed9d6d4d1bb90634003ec82`，最终 SDK gitlink 为 `83e58218437eee1afb60f47f786e80a020eede63`。`pool_source_provenance.json` 保存受影响源码的 SHA256 和六条 CNN 重跑链路；最终 `acceptance_audit.json` 核对 14 条链路、完整输出数量、结果晚于本轮编译、runtime_lazy 配置、Runner 边界、manifest/signature metadata，以及 canonical_loads 与 bundle 密文数量一致。

本轮 12/12 条主链路、2/2 条 PDM 回归及上述证据审计全部通过，14 条链路最大误差为 `0.04015254092489218 < 0.2`。全程保持现有同态参数；Retina 参数加密重跑的完整输出误差为 `3.262896974964846e-5`。

| 任务 | 输入加密 GPU lazy | 参数加密 GPU runtime_lazy | PDM 输入回归 |
| --- | --- | --- | --- |
| BERT | 通过；max_abs=0.0330425，FHE=264.1700s | 通过；max_abs=0.000292991，FHE=809.6290s | 通过；max_abs=0.00418094，FHE=309.4750s |
| cifar10 | 通过；max_abs=6.39511e-05，FHE=7.7737s | 通过；max_abs=0.000208791，FHE=19.1589s | 不适用 |
| imagenet | 通过；max_abs=0.000548744，FHE=50.9462s | 通过；max_abs=0.000712488，FHE=77.5889s | 不适用 |
| retina | 通过；max_abs=3.7592e-06，FHE=276.7090s | 通过；max_abs=3.2629e-05，FHE=1241.0000s | 不适用 |
| ViT | 通过；max_abs=0.0287025，FHE=138.2260s | 通过；max_abs=0.0401525，FHE=330.0600s | 通过；max_abs=0.000584884，FHE=116.8200s |
| yolov5 | 通过；max_abs=6.29466e-05，FHE=44.2581s | 通过；max_abs=9.06926e-05，FHE=265.6730s | 不适用 |

## 9. 非目标和风险

* 不把输入加密模式误称为 encrypted parameter runtime-lazy；两者的 deployment/parameter/input mode 必须从 task_config 验证。
* 不恢复旧 N/2 容量口径，不增加兼容 wrapper 来隐藏旧接口。
* 不以默认 scale 补全 Python node。
* 不手工合并 compiled_mega_ag.json。
* 不把 GPU 内存或 wall time 的改善作为本次功能通过条件；正确性、metadata 一致性和 runtime-lazy 事实优先。
