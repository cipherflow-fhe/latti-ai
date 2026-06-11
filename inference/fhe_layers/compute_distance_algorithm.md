# 计算距离算法

## 目标

给定：

- 查询向量 `q = [q0, q1, ..., q{d-1}]`
- Gallery 向量 `g = [g0, g1, ..., g{d-1}]`
- 假设 `g` 已经做过 L2 归一化：

```text
||g|| = 1
```

目标是计算归一化后的平方 L2 距离：

```text
dist2(q, g) = || q / ||q|| - g ||^2
```

因为：

```text
||g|| = 1
```

所以可以展开为：

```text
dist2(q, g) = 2 - 2 * dot(q, g) / ||q||
```

密文算法实际计算的是下面这个近似值：

```text
2 - 2 * dot(q, g) * (1 / sqrt(dot(q, q)))
```

也就是用 Newton 迭代近似：

```text
1 / sqrt(dot(q, q))
```

## 输入

```text
q: 密文查询向量，长度为 d
g: 明文 Gallery 向量，长度为 d，假设已经归一化
norm2_min: dot(q, q) 的校准下界
norm2_max: dot(q, q) 的校准上界
nr_iterations: Newton 迭代轮数
```

## 第一步：计算缩放后的点积

先准备明文 Gallery：

```text
g_neg2 = -2 * g
```

然后计算：

```text
dot_neg2 = sum_i(q_i * g_neg2_i)
```

因此：

```text
dot_neg2 = -2 * dot(q, g)
```

## 第二步：计算查询向量的模长平方

计算：

```text
norm2 = sum_i(q_i * q_i)
```

也就是：

```text
norm2 = dot(q, q) = ||q||^2
```

## 第三步：近似计算平方根倒数

目标是计算：

```text
rsqrt = 1 / sqrt(norm2)
```

密文里不能直接开方，因此使用 Newton-Raphson 迭代近似平方根倒数。

初始值为：

```text
x0 = 0.5 * (1 / sqrt(norm2_min) + 1 / sqrt(norm2_max))
```

第一轮迭代可以写成关于 `norm2` 的一次表达式：

```text
x = 1.5 * x0 - 0.5 * norm2 * x0^3
```

后续每一轮迭代为：

```text
x = x * (1.5 - 0.5 * norm2 * x * x)
```

迭代 `nr_iterations` 轮后，得到：

```text
rsqrt ≈ x
```

## 第四步：计算归一化相似度项

计算：

```text
neg_cos2 = dot_neg2 * rsqrt
```

因为：

```text
dot_neg2 = -2 * dot(q, g)
rsqrt ≈ 1 / ||q||
```

所以：

```text
neg_cos2 ≈ -2 * dot(q, g) / ||q||
```

## 第五步：计算距离

最终计算：

```text
dist2 = 2 + neg_cos2
```

因此：

```text
dist2 ≈ 2 - 2 * dot(q, g) / ||q||
```

也就是：

```text
|| q / ||q|| - g ||^2
```

的近似结果。

## CKKS 密文计算结构

对应到 CKKS 密文计算，整体结构是：

```text
1. dot_neg2_ct = sum_slots(rescale(mult_plain(q_ct, -2 * g)))
2. norm2_ct    = sum_slots(rescale(mult_relin(q_ct, q_ct)))
3. rsqrt_ct    = inverse_sqrt_newton(norm2_ct)
4. neg_cos2_ct = rescale(mult_relin(dot_neg2_ct, rsqrt_ct))
5. dist2_ct    = 2 + neg_cos2_ct
```

其中：

```text
sum_slots(x):
    for step = 1, 2, 4, ..., < d:
        x = x + rotate(x, step)
```

`sum_slots` 的作用是把一个密文里前 `d` 个 slot 的值求和，并把求和结果放在目标 slot 中。

## F0D repack 说明

如果模型输出的 `Feature0DEncrypted` 不是顺序 packed 的单个 ciphertext，例如：

```text
data[0]: ch0 at slot0,     ch1 at slot16384
data[1]: ch2 at slot0,     ch3 at slot16384
...
```

则在计算距离前，需要先把它重排成：

```text
data[0]: ch0, ch1, ch2, ..., ch127 at slot0..slot127
```

重排流程为：

```text
for each channel:
    1. 用 plaintext mask 取出原 ciphertext 里的 source_slot
    2. mult_plain
    3. rescale
    4. rotate 到目标 target_slot
    5. add 到输出 ciphertext
```

这样输出就是一个顺序 packed 的 0D feature，可以直接作为 `ComputeDistanceLayer` 的输入。

## 注意事项

- Gallery 向量默认需要提前归一化。
- 如果输入 Gallery 是原始 embedding，则需要调用方先做 L2 归一化。
- `norm2_min` 和 `norm2_max` 应该来自训练集或验证集的统计范围。
- 密文结果对齐的是 Newton 近似结果，不是直接 `sqrt` 计算出来的精确明文结果。
- 精度主要受以下因素影响：
  - `norm2_min` / `norm2_max` 范围是否足够贴近真实数据；
  - Newton 迭代轮数；
  - CKKS 乘法、rescale、rotate 带来的近似误差。
