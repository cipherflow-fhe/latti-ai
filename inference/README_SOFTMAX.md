<br />

# FHE Softmax Layer

基于 CKKS 全同态加密方案的 Softmax 层实现，支持两种执行路径：直接 C++ API 调用与计算图生成推理。

## 目录

- [数学原理](#数学原理)
- [设计思路](#设计思路)
- [代码结构](#代码结构)
- [Level 预算](#level-预算)
- [实验结果](#实验结果)
- [测试结果](#测试结果)
- [未来规划](#未来规划)
- [已知问题与解决历程](#已知问题与解决历程)
***
***
## 算法设计
## 数学原理
### Softmax 函数

给定输入向量 $\mathbf{x} = (x\_1, x\_2, \ldots, x\_n)$，Softmax 函数定义为：
给定输入向量 $\mathbf{x} = (x\_1, x\_2, \ldots, x\_n)$，Softmax 函数定义为：
$$
\text{Softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_{j=1}^{n} e^{x\_j}}
\text{Softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_{j=1}^{n} e^{x\_j}}

在 FHE 环境下，指数函数 $e^x$ 和倒数 $1/x$ 均为非多项式运算，无法直接在密文上执行，需要通过多项式近似来替代。

### 整体架构

采用 Chebyshev 多项式插值对目标函数进行近似：

$$
f(x) \approx \sum\_{k=0}^{d} c\_k T\_k(x')
f(x) \approx \sum\_{k=0}^{d} c\_k T\_k(x')

其中 $T\_k$ 为第 $k$ 阶 Chebyshev 多项式，$d$ 为多项式阶数，$x'$ 为经过变量代换后的输入。
其中 $T\_k$ 为第 $k$ 阶 Chebyshev 多项式，$d$ 为多项式阶数，$x'$ 为经过变量代换后的输入。
### 变量代换

Chebyshev 多项式在 $\[-1, 1]$ 上具有最优的数值稳定性。对于定义在 $\[a, b]$ 上的目标函数，先做仿射变换将输入映射到 $\[-1, 1]$：
Chebyshev 多项式在 $\[-1, 1]$ 上具有最优的数值稳定性。对于定义在 $\[a, b]$ 上的目标函数，先做仿射变换将输入映射到 $\[-1, 1]$：
$$
x' = \alpha \cdot x + \beta, \quad \alpha = \frac{2}{b - a}, \quad \beta = -\frac{a + b}{b - a}
$$

映射关系：$x = a \Rightarrow x' = -1$，$x = b \Rightarrow x' = +1$。

变量代换后，在 $\[-1, 1]$ 上对复合函数 $g(x') = f\left(\frac{b-a}{2}x' + \frac{a+b}{2}\right)$ 进行 Chebyshev 插值，得到标准多项式系数 $c\_0, c\_1, \ldots, c\_d$，使得：
变量代换后，在 $\[-1, 1]$ 上对复合函数 $g(x') = f\left(\frac{b-a}{2}x' + \frac{a+b}{2}\right)$ 进行 Chebyshev 插值，得到标准多项式系数 $c\_0, c\_1, \ldots, c\_d$，使得：
$$
f(x) \approx \sum\_{k=0}^{d} c\_k (x')^k = \sum\_{k=0}^{d} c\_k (\alpha x + \beta)^k
f(x) \approx \sum\_{k=0}^{d} c\_k (x')^k = \sum\_{k=0}^{d} c\_k (\alpha x + \beta)^k

### Rotate-and-Sum 求和算法

CKKS 方案中，密文的多个 slot 存储不同的数据。求和操作通过旋转-累加实现：

1. 用明文 mask 将有效 slot 之外的值置零
2. 对 mask 后的密文执行 $\lceil\log\_2(N/2)\rceil$ 轮旋转-累加，其中 $N$ 为多项式模数次数
2. 对 mask 后的密文执行 $\lceil\log\_2(N/2)\rceil$ 轮旋转-累加，其中 $N$ 为多项式模数次数

$$
\text{sum} = x + \text{rot}(x, 1) + \text{rot}(x, 2) + \text{rot}(x, 4) + \cdots
$$

### 多项式求值的 Level 管理

CKKS 中每次乘法消耗一个 level。对于 $d$ 阶多项式，采用分治策略（Baby-Step Giant-Step / Paterson-Stockmeyer），乘法深度为 $\lceil\log\_2 d\rceil$：
CKKS 中每次乘法消耗一个 level。对于 $d$ 阶多项式，采用分治策略，乘法深度为 $\lceil\log\_2 d\rceil$：
```
P(x) = c_0 + c_1·x + c_2·x² + ... + c_d·x^d
      = (c_0 + c_1·x + ... + c_{mid-1}·x^{mid-1})
      + (c_{mid} + c_{mid+1}·x + ... + c_d·x^{d-mid}) · x^{mid}
```

其中 $x^{mid}$ 通过反复平方预计算：$x^2, x^4, x^8, \ldots$

### 两种执行路径
***

## 设计思路

### 整体架构

Softmax 计算被分解为四个 FHE 友好的子操作：

```
Input(x) → exp_poly(x) → sum_slots → inv_poly(sum) → exp(x) * inv(sum) → Output
```

1. **指数多项式近似** `exp_poly(x)`：在 $\[a\_{\text{exp}}, b\_{\text{exp}}]$ 上用 $d\_{\text{exp}}$ 阶多项式近似 $e^x$
2. **Slot 求和** `sum_slots`：通过 rotate-and-sum 算法计算 $\sum\_j e^{x\_j}$
3. **倒数多项式近似** `inv_poly(sum)`：在 $\[a\_{\text{inv}}, b\_{\text{inv}}]$ 上用 $d\_{\text{inv}}$ 阶多项式近似 $1/x$
4. **最终乘法**：计算 $e^{x\_i} \cdot (1 / \sum\_j e^{x\_j})$


| 特性    | 直接 C++ API 路径                          | 计算图生成路径                       |
| 特性    | 直接 C++ API 路径                          | 计算图生成路径                   |
| ----- | -------------------------------------- | ------------------------- |
| 实现文件  | `softmax_layer.cpp`                    | `softmax_layer.py`        |
| 变量代换  | Go 层自动完成                               | Python 层显式构建图节点               |
| 变量代换  | Go 层自动完成                               | Python 层显式构建图节点           |
| 执行方式  | 即时编译执行                                 | 生成计算图 → 图执行器调度            |
| 适用场景  | 单次推理、调试                                | 批量部署、服务端推理                |

### 变量代换的图构建

在计算图生成路径中，变量代换通过显式的图节点实现：

```python
# 变量代换: x' = alpha * x + beta
alpha_node = CkksPlaintextRingtNode(f'{prefix}_alpha')
beta_node = CkksPlaintextRingtNode(f'{prefix}_beta')

x_sub = mult(x, alpha_node)      # x' = alpha * x
x_sub = rescale(x_sub)           # 消耗 1 level
x_sub = add(x_sub, beta_node)    # x' = alpha * x + beta
```

变量代换消耗 1 个 level（mult + rescale），后续多项式求值在 $x'$ 上进行。
### 多项式系数计算

`get_plaintext_values` 方法计算映射域 $\[-1, 1]$ 上的多项式系数：
`get_plaintext_values` 方法计算映射域 $\[-1, 1]$ 上的多项式系数：
```python
exp_func_mapped = lambda x_prime: np.exp((b-a)/2 * x_prime + (a+b)/2)
# 复合函数: g(x') = f((b-a)/2 * x' + (a+b)/2)
exp_coeffs = compute_standard_poly_coeffs(exp_func_mapped, degree, -1.0, 1.0)
```

T≠1 时，返回的明文值顺序为：`[1/T, exp_alpha, exp_beta, exp_c0..exp_cd, mask, inv_alpha, inv_beta, inv_c0..inv_cd]`
返回的明文值顺序为：`[exp_alpha, exp_beta, exp_c0..exp_cd, mask, inv_alpha, inv_beta, inv_c0..inv_cd]`
T=1 时，返回的明文值顺序为：`[exp_alpha, exp_beta, exp_c0..exp_cd, mask, inv_alpha, inv_beta, inv_c0..inv_cd]`
***

| 配置名              | 输入范围         | T   | exp\_range   | inv\_range   | init\_level |
| ---------------- | ------------ | --- | ------------ | ------------ | ----------- |
| range\_0.5       | \[-0.5, 0.5] | 1.0 | \[-1.0, 1.0] | \[5.0, 30.0] | 12          |
| range\_1.0       | \[-1.0, 1.0] | 1.0 | \[-2.0, 2.0] | \[5.0, 50.0] | 12          |
| range\_1.5\_T1.5 | \[-1.5, 1.5] | 1.5 | \[-1.0, 1.0] | \[5.0, 30.0] | 12          |
| range\_2.0\_T2   | \[-2.0, 2.0] | 2.0 | \[-1.0, 1.0] | \[5.0, 30.0] | 12          |
| range\_8.0\_T8   | \[-8.0, 8.0] | 8.0 | \[-1.0, 1.0] | \[5.0, 30.0] | 12          |

**温度缩放的效果**：T>1 的配置将 exp\_range 和 inv\_range 固定为 \[-1, 1] 和 \[5, 30]，多项式近似始终在最精确的区间上工作。

### 测试输出格式

测试输出包含 True/Poly/FHE 三列概率对比：

```
  Cls     Input           True           Poly            FHE   Poly_Err%    FHE_Err%     True%     Poly%
----------------------------------------------------------------------------------------------------
    8    0.4068       0.088335       0.088172       0.088172       0.18%       0.18%     8.83%     8.82% <<<
```

| 字段         | 含义                                          |
| 类/方法                                  | 文件                    | 说明                      |
| ------------------------------------- | --------------------- | ----------------------- |
| `SoftmaxLayer`                        | `softmax_layer.h/cpp` | 直接 C++ API 的 Softmax 实现 |
| `SoftmaxLayer::run()`                 | `softmax_layer.cpp`   | 加密 Softmax 推理入口         |
| `SoftmaxLayer::sum_slots()`           | `softmax_layer.cpp`   | Rotate-and-Sum 求和       |
| `SoftmaxLayer` (Python)               | `softmax_layer.py`    | 计算图生成的 Softmax 实现       |
| `SoftmaxLayer._eval_poly()`           | `softmax_layer.py`    | 带变量代换的多项式求值图构建          |
| `SoftmaxLayer._sum_slots()`           | `softmax_layer.py`    | Rotate-and-Sum 求和图构建    |
| `SoftmaxLayer.get_plaintext_values()` | `softmax_layer.py`    | 返回所有明文节点数值              |

***

## Level 预算

以 `exp_degree=7, inv_degree=7` 为例：

| 操作                                    | 消耗 Level | 累计消耗 |
| ------------------------------------- | -------- | ---- |
| 初始 level                              | -        | 12   |
| exp 变量代换 (mult + rescale)             | 1        | 11   |
| exp 多项式求值 (degree=7, depth=⌈log₂7⌉=3) | 3        | 8    |
| sum\_slots (mult + rescale)           | 1        | 7    |
| inv 变量代换 (mult + rescale)             | 1        | 6    |
| inv 多项式求值 (degree=7, depth=⌈log₂7⌉=3) | 3        | 3    |
| 最终乘法 (mult\_relin + rescale)          | 1        | 2    |

**总消耗**: `exp_depth + inv_depth + 4 = 3 + 3 + 4 = 10` level

**最低输入 level**: 10（建议使用 12 以留有余量）

### 参数集支持

| 参数集       | LogN | MaxLevel | 支持的输入范围                    |
| --------- | ---- | -------- | -------------------------- |
| PN15QP880 | 15   | 17       | ±0.5 \~ ±2.0（level 10\~14） |
| PN14QP438 | 14   | 9        | 不支持（level 不足）              |

***

### 不同输入范围的精度对比

| 输入范围             | T       | 最大相对误差    | RMSE/RMS  | init\_level |
| ---------------- | ------- | --------- | --------- | ----------- |
| 测试名                             | 说明                   | 参数集                 |
| ------------------------------- | -------------------- | ------------------- |
| \[-1.5, 1.5]     | 1.5     | 0.23%     | 0.23%     | 12          |
| `test_softmax_graph_gen`        | 多输入范围计算图生成（4 组范围）    | PN15QP880, level=12 |
| **\[-8.0, 8.0]** | **8.0** | **0.05%** | **0.05%** | **12**      |

### 温度缩放前后精度对比
| 测试名                            | 说明                            | 执行路径   |
| ------------------------------ | ----------------------------- | ------ |
| `softmax_layer`                | 直接 C++ API 推理，4 组范围精度验证       | 直接 API |
| `softmax_layer_graph_gen`      | 计算图推理，单配置精度验证                 | 图生成    |
| `softmax_add_plain_ringt_test` | 多项式求值与 add\_plain\_ringt 操作验证 | 基础算子   |
| \[-8.0, 8.0] | 不可用   | 0.05% | —    |

**关键发现**：温度缩放后，输入范围越大，缩放后越接近 0，多项式近似反而更精确。range\_8.0 的精度（0.05%）甚至优于 range\_0.5（0.11%）。

### 直接 API vs 图生成路径

| 指标       | 直接 C++ API | 计算图生成 |
| -------- | ---------- | ----- |
| 最大相对误差   | 0.50%      | 0.50% |
| RMSE/RMS | 0.50%      | 0.50% |
| 执行时间     | \~2s       | \~2s  |

两种路径精度一致，验证了计算图生成的正确性。

| 配置名        | 输入范围         | exp\_range   | inv\_range    | init\_level |
| ---------- | ------------ | ------------ | ------------- | ----------- |
| range\_0.5 | \[-0.5, 0.5] | \[-1.0, 1.0] | \[5.0, 30.0]  | 10          |
| range\_1.0 | \[-1.0, 1.0] | \[-2.0, 2.0] | \[5.0, 50.0]  | 10          |
| range\_1.5 | \[-1.5, 1.5] | \[-2.5, 2.5] | \[3.0, 80.0]  | 12          |
| range\_2.0 | \[-2.0, 2.0] | \[-3.0, 3.0] | \[2.0, 130.0] | 14          |
bash bash.sh
```

脚本执行流程：

1. **编译 Go SDK 动态库** — 生成 `liblattigo.so`
2. **运行 Python 计算图生成测试** — 生成计算图到 `hetero/` 目录
3. **编译 C++ 测试** — `cmake` + `make test_fhe_layers_hetero`
4. **运行 C++ FHE 推理测试** — 验证计算图推理精度

也可单独运行特定测试：

```bash
# Python 端
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax_graph_gen_single -v
python3 -m unittest inference.unittests.test_gen_layers.TestLayerExport.test_softmax_graph_gen -v

# C++ 端
./build/inference/unittests/test_fhe_layers_hetero "*softmax_layer*"
./build/inference/unittests/test_fhe_layers_hetero "*softmax_layer_graph_gen*"
```

***

