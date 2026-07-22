# BERT GPU 密文推理调试总结

本文记录针对 `runs/BERT/run_pipeline.md` 中 BERT layer1 GPU pipeline 的密文推理数值错误进行调试时，逐步发现的问题、定位依据、修复方式和最终验证结果。

## 背景

最初现象是：

- plaintext 推理结果准确。
- encrypted GPU 推理结果错误。
- debug manifest 显示第一个 CustomMultiHeadAttention 之后误差快速扩大。

本次调试始终使用 BERT layer1 GPU pipeline，不使用 MNIST。

## 1. 缺少密文逐节点可观测性

问题：

- 原 pipeline 只能看到最终 encrypted 输出错误，无法判断第一个出错节点。
- 需要比较每个计算节点的 plaintext expected 和 encrypted decrypted 输出。

解决：

- 在 `./build/examples/inference` 增加 `--debug` 模式。
- debug 模式下 client 将包含 secret key 的完整 context 发送给 server，仅用于本地调试。
- server 在执行 encrypted pass 前先运行 plaintext pass，随后将每个 layer 输出 feature 的 expected 和 encrypted 解密结果写入：
  - `runs/BERT/layer1/task/server/debug/*_expected.txt`
  - `runs/BERT/layer1/task/server/debug/*_encrypted.txt`
  - `runs/BERT/layer1/task/server/debug/debug_manifest.txt`

后续增强：

- txt 第一行写 shape。
- 数据按实际 shape 排布，矩阵按矩阵形式输出。
- `debug_manifest.txt` 增加最大绝对误差和最大误差坐标。

## 2. BERT 输入 shape 和 seq_len 处理错误

问题：

- `runs/BERT/input_layer1.csv` 原始输入是 `52 * 768`。
- 转置后进入 feature_mat 的真实 shape 应为 `768 * 52`。
- `52` 是真实 token 序列长度，不能写死，应从 BERT `CustomMultiHeadAttention.L_prepad` 读取。

解决：

- 在 compiler graph 准备阶段，从 `CustomMultiHeadAttention.L_prepad` 读取真实 seq_len。
- 将 BERT `feature_mat` 输入 shape 修正为真实序列长度对应的 shape。
- BERT/feature_mat 模式不再使用 `set_block_shape` 的空间分块逻辑，保持 `[1, 1]`。
- 修正 H5 导出中 `CustomPositionEmbedding` 的切片和转置，使其匹配 `[768, 52]`。

验证：

- 编译后的 graph 中 `inputs_embeds` 为 `[768, 52]`。
- per-head shape 为 `[64, 52]`。

## 3. compile 时找不到 `inference` 模块

问题：

运行：

```bash
python training/run_compile.py --input=./runs/BERT/model/trained_poly_layer1.onnx --output=./runs/BERT/layer1 --style=multiplexed --config training/config/config.json
```

报错：

```text
ModuleNotFoundError: No module named 'inference'
```

解决：

- 修正 `training/run_compile.py` 的导入路径处理，使从仓库根目录直接运行 compile 时可以找到本地 `inference` package。

## 4. 只打开 BTP scale 开关后，softmax 误差仍然发散

问题：

- C++ 单测 `inference/unittests/test_fhe_layers_hetero.cpp` 的 softmax 路径中，对 bootstrap 前后的 scale 有特殊处理。
- compiler 里虽然支持 `--btp_scale` / `--set_btp_scale`，但原先没有把 BERT softmax 的特殊 scale 从配置传入，并且部分值写死。

解决：

- 在 `training/config/config.json` 中加入 BERT softmax BTP scale 配置。
- `training/run_compile.py` 读取这些配置并传入 pipeline。
- `training/model_compiler/pipeline.py` 在插入 BTP 前后 gamma 时，按 BERT softmax 节点类型选择 scale。

当前配置入口包括：

```json
"btp_scale": 1.0,
"bert_softmax_denominator_btp_scale": 16.0,
"bert_softmax_inverse_btp_scale": 0.125
```

其中 `btp_scale` 控制普通 bootstrap 的默认保护缩放；BERT softmax 相关字段只在匹配到 softmax 子图的 bootstrap 节点时生效。

## 5. 当前 BERT softmax scale 处理

当前处理原则是：softmax 的数学缩放按 ONNX/C++ profile 的数值走，bootstrap 前后的保护缩放只用于改善 refresh 输入幅度，不改变 plaintext 语义。

BERT `CustomMultiHeadAttention` 展开时会读取 ONNX 中的 `range_min`、`range_max`、`exp_divisor`、`exp_coefficients`、`delta_1`、`delta_2` 和 `max_inverse_iterations`。其中 `thor-small` / `thor-wide` 这类 profile 名只作为标签保留；真正影响图结构和数值的是这些具体数值字段。

softmax normalize 使用和 C++ `par_upper_diagonal_polysoftmax_cpp` profile 一致的 denominator scale：

```json
"bert_softmax_initial_denominator_scale": 16.0,
"bert_softmax_first_refinement_denominator_scale": 2.0,
"bert_softmax_later_refinement_denominator_scale": 1.0
```

BTP 保护缩放当前为：

```json
"btp_scale": 1.0,
"bert_softmax_values_btp_scale": 1.0,
"bert_softmax_denominator_btp_scale": 16.0,
"bert_softmax_scaled_denominator_btp_scale": 16.0,
"bert_softmax_inverse_btp_scale": 0.125
```

对应逻辑：

- 普通 bootstrap 默认使用 `btp_scale`。
- softmax denominator bootstrap 前后使用 `16` 和 `1/16`。
- softmax inverse bootstrap 前后使用 `1/8` 和 `8`。
- 如果 bootstrap 发生在已经做过 denominator scale 的 denominator 上，仍只按 BTP 保护语义处理，不再把 normalize scale 和保护 scale 混在一起。
- BERT softmax values 当前不做额外保护缩放，避免把非常小的中间量和后续 denominator/inverse 误差耦合到一起。

## 6. BERT 和 ViT 的 H5 权重吸收差异

ViT 的 feature_mat attention 旧路径会在 H5 导出时对 attention polynomial coefficients 吸收 `1/seq_len`。BERT 不走这条语义：BERT `CustomMultiHeadAttention` 已经在 compiler 中展开为 `pdmupperaddpt`、`pdmgamma`、`pdmupperpoly`、`pdmheadcolsum`、`pdminv*` 等底层节点，softmax polynomial coefficients 直接来自 ONNX 的 `exp_coefficients`。

因此 `training/nn_tools/feature_mat_h5.py` 当前按 `model_type` 区分：

- `model_type="bert"` 时，不对 attention polynomial coefficients 吸收 `1/seq_len`。
- BERT Q projection 只吸收 ONNX `scaling` 对应的 `weight_multiplier` / `bias_multiplier`。
- BERT K/V projection 不额外吸收 `1/seq_len`。
- 非 BERT 的 ViT 路径继续保留原有 `1/seq_len` 吸收逻辑。

## 最终验证

使用当前配置重新 compile BERT layer1、生成 mega_ag 并运行 GPU verify：

```bash
python training/run_compile.py --input=./runs/BERT/model/trained_poly_layer1.onnx --output=/tmp/latti_bert_softmax_scale16 --style=multiplexed --config training/config/config.json --num_experiments 1 --num_workers 1
python inference/interface/gen_mega_ag.py --task-dir /tmp/latti_bert_softmax_scale16/task
./build/examples/inference --task-dir /tmp/latti_bert_softmax_scale16/task --input runs/BERT/input.csv --verify --gpu
```

结果：

```text
Max absolute error: 0.00740536
Avg absolute error: 0.00068327
Tolerance:          0.10000000
Result: PASS
```

## ONNX 统计量在 GPU 链路中的流通

结论先说清楚：当前代码对 BERT `CustomMultiHeadAttention` 的 softmax 统计量已经能做到“部分按 ONNX 数值属性展开”，但还不是“按 profile 名自动查表”。`CustomLayerNorm` 的 BERT 路径也已经按 `par_upper_diagonal_polylayernorm_ln3_cpp` 的公式读取 ONNX 数值属性，并把 runtime 需要的 `inv_var/inv_std/coeffs` 写到拆分后的 LN stage。只写 `softmax_profile`、`profile` 这样的名字本身不会改变划分；真正改变划分的是被展开逻辑读取的数值字段。

当前 `runs/BERT/run_pipeline.md` 的 GPU 链路是：

1. `training/run_compile.py` 读取 `training/config/config.json`，ONNX 输入会先调用 `onnx_to_json(...)` 生成 `pt.json`。
2. `LayerAbstractGraph.from_json(...)` 把 `pt.json` 读成 compiler graph。
3. `prepare_graph(...)` 先应用 BERT `L_prepad` shape，再展开 `CustomMultiHeadAttention`、`layernorm`、`CustomGELU/CustomTanh`。
4. DP/BTP 划分发生在展开后的底层节点上。后续输出 `task/server/nn_layers_ct_0.json`、`task/server/task_config.json` 和 `model_parameters.h5`。
5. `gen_mega_ag.py`/C++ runtime 初始化的是这些底层 `pdm*`/`pcm*` 层，不再初始化原始 `Custom*` 节点。

### CustomMultiHeadAttention

`onnx_to_json.py` 的 BERT 路径会把 ONNX attributes 原样写进 `pt.json`；`components.py` 读回 `CustomMultiHeadAttention` 时也会保留未识别字段。随后 `transforms._expand_bert_multi_head_attention()` 和 `_append_bert_softmax()` 会读取并使用这些字段：

- `L_prepad`：用于修正 BERT feature_mat 的真实 seq_len shape。
- `num_heads`：优先用节点属性，否则用 config。
- `scaling`：用于 Q projection 的 `weight_multiplier` / `bias_multiplier`。
- `range_min`、`range_max`：计算 softmax center 的 `mid`，展开为 `pdmupperaddpt.value = -mid`。
- `exp_divisor`：展开为 `pdmgamma.scalar_value = 1 / exp_divisor`。
- `exp_coefficients`：展开为 `pdmupperpoly.coefficients`；多项式阶数由 coefficient 数量决定，会影响 level cost。
- `delta_1`：决定 exp polynomial 后做多少次 square。
- `delta_2`：决定 softmax refinement 做多少轮。
- `max_inverse_iterations` / `max_inverse_iteration`：决定每个 normalize 中 `pdminviter` 的数量；复数形式优先，singular 作为兼容 fallback。
- `initial_denominator_scale`、`first_refinement_denominator_scale`、`later_refinement_denominator_scale`：如果 ONNX 节点上存在则使用；否则使用 `training/config/config.json` 中的 BERT softmax denominator scale。

这些字段会在展开后转成底层节点的 `value`、`scalar_value`、`coefficients`、节点数量和边结构。H5 export 会根据 `scalar_value` / `btp_scale` 生成 gamma 权重，根据 `coefficients` 生成 `pdmupperpoly` 权重；C++ runtime 再从 H5 和底层 JSON 初始化 `ParUpperDiagonalAddPt`、`ParUpperDiagonalPoly`、`ParUpperDiagonalMultipleSquare`、`ParUpperDiagonalHeadColSum`、`ParUpperDiagonalInverseInit/Iter`、`ParUpperDiagonalGELU` 等类。

当前没有真正读取或使用的 attention 属性包括：

- `softmax_profile`：只作为标签保留在 `pt.json`，不驱动查表或分支。
- `use_asor`：未参与展开或 runtime 初始化。
- `inverse_alpha`、`final_inverse_alpha`、`inverse_epsilon`：当前 inverse 初始化/迭代固定展开为 `pdminvinit` / `pdminviter` 语义，没有使用这些 ONNX 参数。

因此，`thor-small`、`thor-wide` 这类 profile 如果只改变 `softmax_profile` 字符串，当前划分不会变；只有当 exporter 同时把 `range_min/range_max/exp_divisor/exp_coefficients/delta_1/delta_2/max_inverse_iterations` 或 `max_inverse_iteration` 等数值写成不同值时，展开后的底层图和划分才会随之变化。

### CustomLayerNorm

`onnx_to_json.py` 遇到 `CustomLayerNorm` 时仍输出普通 `layernorm` 节点，但现在会把 ONNX attrs 保留在 `pt.json` 中。`components.py` 读回 `layernorm` 时也会保留这些 attrs，`expand_layer_norm()` 在 `model_type="bert"` 时按 C++ ln3 单测同款公式计算 runtime 参数：

```text
max_denominator = (max_var * w_buffer + eps) * input_scale * input_scale
normalized_epsilon = (min_var + eps) / max_denominator
inv_var = 1.0 / max_denominator
inv_std = sqrt(inv_var)
```

其中 `normalized_epsilon` 只是记录归一化后的 lower-bound；最终 `pdmstats.epsilon` 仍写原始 `eps`，和 C++ `ParUpperDiagonalLNStats(..., eps, inv_var)` 一致。

当前 BERT LayerNorm 会读取并使用：

- `eps` / `epsilon`
- `min_var`
- `max_var`
- `w_buffer`
- `input_scale`
- `max_inverse_sqrt_iterations` / `max_inverse_sqrt_iteration`
- `c0`、`c1`、`c2`
- `weight_path`、`bias_path`

展开和序列化后的流向是：

- `max_inverse_sqrt_iterations` 决定 `pdmgs_*` 节点数量，因此会影响 DP/BTP 划分。
- `pdmstats` 写入原始 `epsilon`、`inv_var`、`min_var/max_var/w_buffer/input_scale/max_denominator/normalized_epsilon/profile/use_asor`。
- `pdminit` 写入 `coeffs=[c0,c1,c2]`，避免被 `task_config.layernorm_param.minimax_init_coeffs` 的默认值覆盖。
- `pdmaffine` 写入 `inv_std`。
- `task_config.layernorm_param` 也同步写入第一组 LN 参数，便于调试；但 C++ runtime 对每个 stage 会优先读取 layer JSON 中的 per-layer 值。

`use_asor` 现在会被显式读取；当前实现只支持单测中的 `use_asor=0` 路径。如果 ONNX 写入非零 `use_asor`，compiler 会报错，避免静默使用错误逻辑。`profile` 仍主要是标签；比如当前 `runs/BERT/model/trained_poly_layer1.onnx` 实际包含 `ln1/ln2`，最终使用的是各节点的 `min_var/max_var/w_buffer/...` 数值，而不是 profile 名本身。

### CustomGELU 和 CustomTanh

`CustomGELU` / `CustomTanh` 当前会把 ONNX attributes 写进 `pt.json`，`components.py` 读回时保留所有非基础字段；在 `model_type=bert` 时由 `expand_bert_custom_poly_functions()` 展开成底层 `pdmgamma`、`pdmupperpoly` 和乘法节点。

`CustomGELU` 真正使用的属性：

- `scale`：展开为输入缩放 `1 / scale`。
- `f2_input_scale`、`f3_input_scale`：分别展开为 `1 / f2_input_scale`、`1 / f3_input_scale` 的 gamma。
- `p1_coefficients`、`p2_coefficients`、`p3_coefficients`：展开为三段 `pdmupperpoly.coefficients`。

`CustomTanh` 真正使用的属性：

- `scale`
- `f1_input_scale`
- `f2_input_scale`
- `p1_coefficients`
- `p2_coefficients`

`profile` 在这两类节点里也只是标签；真正驱动展开和 runtime 的是数值 scale 和 polynomial coefficients。

## 后续可改进方向

1. 统一 scale 管理

   当前 scale 主要通过 layer_id 模式匹配决定。短期可用，但长期应将 softmax 子图中的语义阶段显式标注为 metadata，例如 `softmax_stage=values/denominator/inverse/scaled_inverse`，避免依赖字符串命名。

2. 自动选择 BTP scale

   目前 BTP protection scale 仍是按节点语义配置。更好的方式是在 compile 或 profiling 阶段读取每个关键节点的统计量，自动将 bootstrap 输入映射到稳定区间，例如 `0.25` 到 `1.0`。

3. 区分 normalize scale 和 BTP protection scale

   `denominator_scale` 会改变 inverse 计算的数值范围，并在 inverse 末尾恢复；BTP protection scale 只用于降低 bootstrap 噪声。两者语义不同，建议在 config 和 graph metadata 中明确区分。

4. 将 C++ softmax 单测的 level/refresh 策略图化

   C++ 单测中有 `refresh_to_min_level`、`refresh_inverse_to_min_level`、`align_to_level` 等手写调度逻辑。compiler 目前依赖 DP 选择 bootstrap 位置，再用 scale 包裹。后续可以为 BERT softmax normalize 生成更接近单测语义的显式 refresh/align 子图。

5. 改进 debug 输出选择

   full debug 会把所有中间 feature 都作为输出，显存峰值高。可以增加 debug filter，例如只 dump 匹配 `softmax` 或某个 layer prefix 的节点，方便快速迭代。

6. 修正 `kt` debug shape mismatch

   manifest 中 `_attention_CustomMultiHeadAttention_kt` 仍显示 expected shape `[52, 768]`、encrypted shape `[52, 64]`。由于后续 `qkt` 和 softmax 已经正确收敛，这更像 debug unpack/metadata 问题，但仍建议单独修正，避免误导后续排查。
