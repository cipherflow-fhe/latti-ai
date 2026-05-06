# CKKS Softmax 开发实现细节

## 背景

本任务为 LattiAI 的 FHE 推理链路补充 CKKS softmax 支持，使模型生成器能够生成 softmax 对应的 hetero task，并在 C++ FHE runtime 中完成加密 softmax 推理验证。

当前实现面向 `Feature0DEncrypted` 场景，适用于分类 logits 已被打包到单个 CKKS ciphertext 中的情况。测试用例使用 32 类 softmax，CKKS 参数为 `PN15QP880`，多项式模数阶数 `N=32768`，输入 level 为 13。

## 改动范围

本任务主要涉及以下模块：

- `inference/fhe_layers/softmax_layer.h`
- `inference/fhe_layers/softmax_layer.cpp`
- `inference/fhe_layers/CMakeLists.txt`
- `inference/fhe_layers/fhe_layers.h`
- `inference/model_generator/layers/softmax_layer.py`
- `inference/model_generator/deploy_cmds.py`
- `inference/unittests/test_gen_layers.py`
- `inference/unittests/test_fhe_layers_hetero.cpp`

其中 C++ 侧负责 softmax layer 的离线参数准备、runtime 调用和明文对照实现；Python 侧负责生成 softmax 的抽象计算图与 hetero task；测试侧负责先生成任务文件，再执行 FHE 推理并和明文 softmax 对比。

## C++ FHE Layer 实现

新增 `SoftmaxLayer`，继承自现有 `Layer` 基类，主要职责包括：

- 管理 CKKS softmax 所需的 plaintext/ringt/mul plaintext 参数
- 根据 `n_classes` 和 `input_level` 预编码 softmax 多项式系数
- 暴露 `run()` 接口，接收 `Feature0DEncrypted` 并返回 softmax 后的 `Feature0DEncrypted`
- 提供 `run_plaintext()` 作为单元测试中的明文对照

核心接口如下：

```cpp
class SoftmaxLayer : public Layer {
public:
    explicit SoftmaxLayer(const ls::CkksParameter& param_in,
                          uint32_t n_classes = 0,
                          uint32_t input_level = 0);

    void prepare_offline_args(uint32_t n_classes, uint32_t input_level);
    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x) const;
    Array<double, 1> run_plaintext(const Array<double, 1>& x) const;
};
```

### 参数约束

当前 softmax 实现有以下约束：

- `n_classes > 0`
- `n_classes` 必须是 2 的幂，便于使用 rotation tree 做 repeated block sum
- `input_level >= 13`
- `input_level` 不能超过当前 CKKS 参数允许的最大 level
- 目前主要支持单 ciphertext 的 `Feature0D` 输入

如果输入 level 不足，C++ 侧会在 `prepare_offline_args()` 中抛出异常；Python 生成侧也会提前检查并提示需要增加 CKKS level budget 或在 softmax 前插入 bootstrap。

### 离线参数准备

`prepare_offline_args()` 会根据输入 level 计算 exp 多项式和 reciprocal 多项式各系数所在的 level，并将常量编码成 LattiSense runtime 可直接消费的 plaintext 参数。

exp 多项式使用 6 个系数：

```cpp
constexpr std::array<double, 6> kExpCoeffs = {
    1.0031377334916605,
    1.0026864218461349,
    0.4860498435309526,
    0.1624711376226941,
    0.05072464694309538,
    0.010053701974162384,
};
```

reciprocal 多项式使用 4 个系数：

```cpp
constexpr std::array<double, 4> kRecipCoeffs = {
    0.24885999074111392,
    -0.021622407621476325,
    0.0007824595670968044,
    -0.000010035965681343485,
};
```

为了适配不同分类数，reciprocal 多项式以 8 类 softmax 的 denominator 范围作为锚点，通过：

```text
alpha = 8 / n_classes
```

将 reciprocal 近似调整为 `alpha * P(alpha * x)`，使 32 类等更高分类数场景的 denominator 仍落在拟合范围内。

### Scale 与 Level 处理

softmax 近似链路包含多次 ciphertext-plaintext 乘法、ciphertext-ciphertext 乘法、relinearize 和 rescale，因此离线参数必须和执行图中的 level/scale 对齐。

实现中根据 `param_.get_default_scale()` 和每层 modulus `param_.get_q(level)` 推导 exp 与 reciprocal 各系数的编码 scale：

- exp 多项式从 `input_level - 2` 开始消耗 level
- reciprocal 多项式从 `input_level - 9` 开始消耗 level
- 最低系数 level 不能小于 0

这些计算保证生成的 plaintext 参数能和 model generator 生成的 DAG 指令对齐。

### Runtime 调用

`SoftmaxLayer::run()` 保留输入 feature 的基本元信息，并用 `run_core()` 执行核心 encrypted softmax：

```cpp
Feature0DEncrypted SoftmaxLayer::run(CkksContext& ctx, const Feature0DEncrypted& x) const {
    Feature0DEncrypted result(x.context, x.level);
    result.data = run_core(ctx, x.data);
    result.dim = x.dim;
    result.skip = x.skip;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = result.data.empty() ? x.level : result.data[0].get_level();
    result.ckks_scale = x.ckks_scale;
    result.multiplier = x.multiplier;
    return result;
}
```

测试中并不是直接调用 `run()`，而是通过 `FheTaskCpu` 执行由 Python 生成的 `mega_ag.json`，再将 C++ 侧预编码好的 softmax 离线参数传给 task signature。

## Python Model Generator 实现

新增 `inference/model_generator/layers/softmax_layer.py`，用于生成 softmax 的抽象计算图。

Python 侧 `SoftmaxLayer` 负责把 softmax 拆解为 FHE 可执行的 primitive DAG：

1. logits 乘以 `1/4`，将输入缩放到 exp 多项式更稳定的范围
2. 对缩放后的 logits 做 repeated block sum
3. 乘以 `1 / n_classes` 得到 mean
4. logits 减 mean，得到 centered logits
5. 使用 5 次多项式近似 exp
6. 对 exp logits 做 repeated block sum 得到 denominator
7. 使用 3 次多项式近似 reciprocal denominator
8. `exp_logits * inv_denom` 得到 softmax 输出

### Rotation Tree 求和

`_build_rotate_steps()` 根据 `n_classes` 生成旋转步长：

```text
1, 2, 4, ..., n_classes / 2
```

`_repeated_block_sum()` 依次执行 rotate + add，将每个重复 block 内的值求和。由于这种求和方式要求类别数为 2 的幂，因此 generator 和 C++ runtime 都对 `n_classes` 做了 power-of-two 检查。

### Exp 多项式

`_eval_exp_poly_v1()` 使用 Horner-like 结构计算 exp 多项式，并额外做两次平方：

```text
P(x) -> P(x)^2 -> P(x)^4
```

对应生成图中会显式插入 `drop_level`、`mult`、`mult_relin`、`add`、`rescale` 等节点，使每一步的 level 与 C++ 侧离线参数保持一致。

### Reciprocal 多项式

`_eval_recip_poly_v1()` 对 denominator 计算 3 次 reciprocal 近似，用于替代明文 softmax 中的除法：

```text
1 / sum(exp(logits))
```

最后通过：

```text
softmax = exp_logits * reciprocal(denominator)
```

得到每个类别的近似概率。

## Deploy 集成

`inference/model_generator/deploy_cmds.py` 中新增了 `layer_config['type'] == 'softmax'` 分支，使模型编译后的 softmax layer 能被转换成 hetero task。

该分支主要完成：

- 检查输入 feature 必须为 `dim == 0`
- 检查 softmax 目前只支持单 ciphertext 输入
- 检查输入输出 channel 一致
- 检查输入 level 至少为 13
- 根据输入 level 创建 exp 和 reciprocal 所需的 plaintext 节点
- 调用 Python `SoftmaxLayer.call()` 生成 softmax DAG
- 将所有 softmax 离线参数追加到 `input_args`

softmax 离线参数命名带有 layer id，例如：

```text
softmax_pt_quarter_<layer_id>
softmax_pt_inv_classes_<layer_id>
softmax_exp_c5_<layer_id>
softmax_exp_c4_<layer_id>
softmax_exp_c3_<layer_id>
softmax_exp_c2_<layer_id>
softmax_exp_c1_<layer_id>
softmax_exp_c0_<layer_id>
softmax_recip_c3_<layer_id>
softmax_recip_c2_<layer_id>
softmax_recip_c1_<layer_id>
softmax_recip_c0_<layer_id>
```

## 测试实现

### Python 任务生成测试

`test_gen_layers.py` 中新增 `test_softmax_feature0d()`，用于生成 softmax 的 hetero task。

测试配置：

- 参数集：`PN15QP880`
- `N = 32768`
- `n_classes = 32`
- `input_level = 13`
- 输出目录：`build/inference/hetero/CKKS_softmax/ch_32/level_13/server`

该测试会生成：

```text
mega_ag.json
task_signature.json
```

这些文件会被后续 C++ hetero 测试读取。

运行命令：

```bash
PYTHONPATH=. python -m unittest -v inference.unittests.test_gen_layers.TestLayerExport.test_softmax_feature0d
```

### C++ Hetero 执行测试

`test_fhe_layers_hetero.cpp` 中新增 `softmax_feature0d` 用例。

测试流程：

1. 使用 `N=32768` 创建 CKKS 参数和 context
2. 生成 rotation keys
3. 随机生成 32 类 logits
4. 将 logits 按类别循环 tiled 到所有 CKKS slots
5. 加密输入并构造 `Feature0DEncrypted`
6. 构造 `SoftmaxLayer` 并准备离线 plaintext 参数
7. 读取 Python 生成的 `task_signature.json`
8. 按 signature 名称绑定输入 ciphertext、输出 ciphertext 和 softmax 离线参数
9. 使用 `FheTaskCpu` 执行 `mega_ag.json`
10. 解密/解包输出，并和 `run_plaintext()` 的明文 softmax 结果对比

测试断言包括：

- `max_error < 2.5e-1`
- `rmse < 8.0e-2`
- FHE 输出 argmax 与明文输出 argmax 一致
- 输出概率和接近 1，容差为 `2.0e-1`

运行命令：

```bash
./build/inference/unittests/test_fhe_layers_hetero "*softmax_feature0d*" --success
```

本地验证结果：

```text
max_error = 0.0003867758
rmse = 0.000222177
argmax: 10 == 10
prob_sum = 1.0063581822
All tests passed (4 assertions in 1 test case)
```

## 构建与环境注意事项

本任务依赖 Lattigo 子模块的 Go toolchain。当前 Lattigo `go.mod` 指定：

```text
go 1.24.0
toolchain go1.24.11
```

如果系统默认 Go 版本较低，构建时会报 `unknown directive: toolchain` 或 `invalid go version`。本地验证时使用 Go 1.24.11，并在构建时将新版 Go 放到 `PATH` 前面：

```bash
export PATH=/usr/local/go/bin:$PATH
```

CMake 配置命令：

```bash
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=OFF -DGO_EXECUTABLE=/usr/local/go/bin/go
```

测试目标编译命令：

```bash
cmake --build build --target test_fhe_layers_hetero -j8
```

## 当前限制与后续方向

当前实现已经完成 32 类 `Feature0D` softmax 的端到端验证，但仍有一些限制：

- 类别数需要为 2 的幂
- 当前主要覆盖单 ciphertext 的 `Feature0D` 输入
- 输入 level 需要至少为 13
- reciprocal 近似系数基于当前 denominator 范围，后续可针对更多类别数或输入分布做进一步拟合优化
- GPU path 目前未在本地验证，本次测试使用 CPU hetero runtime

后续可以继续扩展：

- 支持非 2 的幂类别数
- 支持多 ciphertext logits
- 根据模型编译阶段自动规划 softmax 前的 bootstrap
- 扩展 E2E 模型级 softmax 输出验证
- 增加更多 `n_classes` 和输入范围的误差评估
