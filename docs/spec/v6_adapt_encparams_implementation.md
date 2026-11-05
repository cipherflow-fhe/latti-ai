# v6 语义下移植 encparams 功能与 CIFAR10 GPU lazy 验收规范

状态：设计规范
依据：docs/research/v6_adapt_encparams_merge_research.md、v6 侧 feat/lattisense-adapt 代码、runs/cifar10/run_pipeline.md
目标：在 v6 语义和接口下重新移植 feat/linghm-encparams 的参数加密功能，并用 CIFAR10 的输入加密 GPU lazy 和参数加密 GPU runtime-lazy 两条端到端链路验收。

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
10. 只把最终生成的 task、runner、manifest 和日志写入独立验收目录，不覆盖已有 runs/cifar10 产物。

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

server_provisioned_runner 的 bundle 不包含明文模型参数、secret key 或 secret context。Provisioner 持有明文参数并生成加密参数；Runner 只持有 eval context、task graph 和参数密文；Decryptor 在 provisioner 侧解密输出。

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

## 4. 端到端验收目录和环境

所有验收放入：

    runs/cifar10/v6_adapt_validation/

建议环境变量：

    REPO="$(git rev-parse --show-toplevel)"
    PY=/home/linghm/pytorch_venv/bin/python3
    CMAKE=/home/linghm/pytorch_venv/bin/cmake
    ROOT="$REPO/runs/cifar10/v6_adapt_validation"
    MODEL="$REPO/runs/cifar10/model/trained_poly.onnx"
    INPUT="$REPO/runs/cifar10/img.csv"
    export PYTHONPATH="$REPO/inference/lattisense:$REPO"
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8

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

参考输出保存完整 logits；top-1 只能作为附加信息，不能作为唯一正确性标准。

## 5. 验收 A：CIFAR10 输入加密 GPU lazy

该模式使用 client_encrypted_input。模型参数保持明文/已有 server 路径，客户端加密输入，GPU server 执行 v6 MegaAG graph。lazy 指 graph 的 custom compute 和按需执行路径，不把它解释为 encrypted parameter runtime_lazy。

编译和生成：

    OUT="$ROOT/input_encrypted"
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
* comparison.json 的 max_abs 小于 0.2；若无历史阈值变化，目标值沿用已有 CIFAR10 流程的 0.2；
* task graph 中所有 plaintext node 使用 v6 metadata，node id 和 signature index 无重复；
* n_real_num_slot、feature packing、level 和 scale 在 task JSON 与 C++ runtime 一致；
* GPU 日志存在 FHE time 和无异常 context/thread 错误。

## 6. 验收 B：CIFAR10 参数加密 GPU runtime-lazy

该模式使用 server_provisioned_runner、encrypted_offline 和 runtime_lazy。输入保持明文；Provisioner 先加密模型参数并导出 Runner bundle；Runner GPU 按需从 encrypted parameter store 读取参数。

编译：

    OUT="$ROOT/parameter_encrypted"
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

完成两条主验收后，执行最小单测集：

    PYTHONPATH="$REPO/inference/lattisense:$REPO" "$PY" -m unittest \
      inference.unittests.test_encrypted_param_count \
      inference.unittests.test_server_provisioned_runner_generator \
      inference.unittests.test_parameter_acceptance_tools \
      inference.unittests.test_interleaved_seed_weights \
      training.model_compiler.unittests.test_storage_score

C++/v6 侧执行已构建的 data struct、FHE layer、heterogeneous GPU 和 runner store tests。具体 test target 以合并分支的 CMake 输出为准，不为本规范新增平行测试框架。

静态检查：

    rg -n 'get_n\\(\\) / 2|N / 2|n // 2' inference training
    rg -n 'CkksCompressedCiphertext|encrypt_symmetric_compressed|data_compress|data_compressed' inference
    rg -n 'resize_copies|get_copy|get_extra_level_context' inference/data_structs inference/fhe_layers
    git diff --check

允许出现 polynomial degree 的明确底层适配，不允许把它用于 feature 实数容量。后两条若仍有结果，必须逐项说明为何不属于 v6 迁移残留；默认验收失败。

## 8. 证据文件和完成定义

每次验收提交以下文件：

    runs/cifar10/v6_adapt_validation/reference/onnx.csv
    runs/cifar10/v6_adapt_validation/reference/onnx.json
    runs/cifar10/v6_adapt_validation/input_encrypted/comparison.json
    runs/cifar10/v6_adapt_validation/input_encrypted/run/inference.metrics.json
    runs/cifar10/v6_adapt_validation/parameter_encrypted/comparison.json
    runs/cifar10/v6_adapt_validation/parameter_encrypted/run/inference.metrics.json
    runs/cifar10/v6_adapt_validation/parameter_encrypted/bundle.json
    runs/cifar10/v6_adapt_validation/parameter_encrypted/provision.time

规范完成必须同时满足：

1. 两条 GPU 端到端命令全部成功。
2. 两种模式的完整 logits max_abs 都小于 0.2。
3. v6 API、plaintext metadata、n_real_num_slot、序列化和共享 context 检查通过。
4. 参数加密模式确实使用 runtime-lazy，而不是 eager 参数加载。
5. Runner 安全边界和 public mask/secret material 检查通过。
6. 没有因 v6 适配而删除 encrypted parameter、seed、storage-aware 或 inputless encode_pt 功能。
7. 所有修改可由独立 commit 对应到本规范的一个阶段，且没有无关重构。

若任一条件失败，保留失败日志和最小复现 task，先修复对应语义层，再重新执行两条完整流程；不能仅通过放宽误差阈值或改写验收产物来“通过”。

## 9. 非目标和风险

* 不把输入加密模式误称为 encrypted parameter runtime-lazy；两者的 deployment/parameter/input mode 必须从 task_config 验证。
* 不恢复旧 N/2 容量口径，不增加兼容 wrapper 来隐藏旧接口。
* 不以默认 scale 补全 Python node。
* 不手工合并 compiled_mega_ag.json。
* 不把 GPU 内存或 wall time 的改善作为本次功能通过条件；正确性、metadata 一致性和 runtime-lazy 事实优先。
