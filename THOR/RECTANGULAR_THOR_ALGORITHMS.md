# Rectangular THOR Algorithms 数学原理与公式推导

本文说明 `thor_ckks_matmul_primitives.py` 中新增的 rectangular 输入相关接口背后的数学依据，重点覆盖 Algorithm 1、Algorithm 2、Algorithm 4、Algorithm 5 以及相关辅助接口。

相关接口包括：

- `pack_repeated_upper_diagonals`
- `prop_3_8_rectangular_packed`
- `generate_algorithm_1_rectangular_plaintexts`
- `algorithm_1_rectangular_plaintext_ciphertext_matmul`
- `algorithm_4_replication_rectangular`
- `algorithm_5_matrix_transpose_rectangular`
- `algorithm_2_ciphertext_ciphertext_matmul_corollary_3_9`

本文只讨论 NumPy reference implementation 中的 slot 级数学语义，不讨论 CKKS scale、noise、relinearization、rescale 或 key-switching 成本。

---

## 1. 记号与基本定义

令矩阵 `A` 的 shape 为 `rows x cols`。THOR 使用 cyclic upper/lower diagonal 表示矩阵。

### 1.1 Upper diagonal

对 `A in R^{rows x cols}`，第 `i` 条 upper diagonal 定义为：

```text
U_i(A)[t] = A[t mod rows, (i + t) mod cols]
```

其中：

```text
0 <= i < min(rows, cols)
0 <= t < max(rows, cols)
```

所以 upper diagonal 的数量是 `min(rows, cols)`，每条 diagonal 的长度是 `max(rows, cols)`。

### 1.2 Lower diagonal

对 `A in R^{rows x cols}`，第 `i` 条 lower diagonal 定义为：

```text
L_i(A)[t] = A[(i + t) mod rows, t mod cols]
```

其中：

```text
0 <= i < min(rows, cols)
0 <= t < max(rows, cols)
```

同样，lower diagonal 的数量是 `min(rows, cols)`，每条 diagonal 的长度是 `max(rows, cols)`。

### 1.3 Slot rotation 约定

代码中的：

```python
rotate(x, step)
```

表示 CKKS SIMD slot 左旋 `step` 个 slot。因为 H-way batched diagonal encoding 采用 lane interlacing，每个 diagonal 坐标 `t` 占用连续的 `H` 个 lane：

```text
slot index = t * H + h
```

所以数学上的 diagonal rotation `rho^s` 在代码中对应：

```python
rotate(x, s * H)
```

为了简化公式，本文写作 `rho^s(v)`，其含义是：

```text
rho^s(v)[t] = v[(t + s) mod diag_len]
```

其中 `diag_len` 是当前 diagonal 的长度。

---

## 2. Multi-diagonal batched encoding

设同一 ciphertext 中并行打包 `H` 个矩阵。对 shape `(rows, cols)`：

```text
diag_len    = max(rows, cols)
n_diag      = min(rows, cols)
segment_len = H * diag_len
c           = n_slot / segment_len
```

其中 `c` 是每个 ciphertext 可以容纳的 diagonal segment 数量。第 `j` 个 ciphertext 包含 global diagonal index：

```text
c*j, c*j+1, ..., c*j+c-1
```

每个 segment 内部以 H-way interlacing 保存 `H` 个矩阵的同一条 diagonal：

```text
segment_i[t * H + h] = diagonal_i(matrix_h)[t]
```

因此：

- 对 shape `(m, n)` 且 `m < n`，有 `n_diag = m`、`diag_len = n`；
- 一个 packed lower/upper diagonal ciphertext 数量是 `m / c`；
- 若输出 shape 是 `(n, n)`，则 lower diagonal 数量是 `n`，输出 ciphertext 数量是 `n / c`。

本文所有 rectangular 推导默认：

```text
m < n,  m | n,  c | m,  n_slot = c * H * n
```

其中 Algorithm 4 的 rotate-and-sum replication 还要求：

```text
c 是 2 的幂
```

---

## 3. Rectangular Algorithm 1：plaintext-ciphertext matmul

### 3.1 问题形态

新增接口：

```python
algorithm_1_rectangular_plaintext_ciphertext_matmul(
    A_plain: np.ndarray,
    packed_lower_B: list[np.ndarray],
    m: int,
    n: int,
    n_slot: int,
    H: int,
) -> list[np.ndarray]
```

计算：

```text
C = A * B
```

其中：

```text
A in R^{d x d},  d = H*m
B in R^{d x n}
C in R^{d x n}
```

把 `A` 划分为 `H x H` 个 block：

```text
A_{p,q} in R^{m x m},  p,q in [H]
```

把 `B` 和 `C` 按行划分为 `H` 个 block：

```text
B_q in R^{m x n}
C_p in R^{m x n}
```

则：

```text
C_p = sum_{q in [H]} A_{p,q} B_q
```

代码为了适配 H-way batched encoding，按 block diagonal offset `i` 组织并行乘法：

```text
A^{(i)}_k = A_{(i+k) mod H, k},  k in [H]
```

也就是同一个 batch lane `k` 中计算：

```text
A^{(i)}_k * B_k -> contributes to C_{(i+k) mod H}
```

### 3.2 Block 级 rectangular 乘法公式

核心 block 乘法是：

```text
A in R^{m x m},  B in R^{m x n},  C = A B in R^{m x n}
```

因为 `B` 是 rectangular wide matrix，`L_ell(B)` 的长度是 `n`。而 `U_i(A)` 的自然长度只有 `m`。为了进行 slot-wise multiplication，需要把 `U_i(A)` 周期性扩展到长度 `n`：

```text
Urep_i(A)[t] = U_i(A)[t mod m],  0 <= t < n
```

由 upper diagonal 定义：

```text
U_i(A)[t mod m]
= A[t mod m, (i + t) mod m]
```

因此：

```text
Urep_i(A)[t]
= A[t mod m, (i + t) mod m]
```

新增 helper `pack_repeated_upper_diagonals(...)` 正是把 `U_i(A)` 重复到长度 `n` 后，按照 shape `(m, n)` 的 multi-diagonal format 打包。

### 3.3 Derived rectangular Proposition 3.8

对于：

```text
A in R^{m x m}
B in R^{m x n}
C = A B in R^{m x n}
```

有：

```text
L_r(C) = sum_{ell in [m]} rho^r(Urep_{ell-r}(A)) ⊙ L_ell(B),  r in [m]
```

其中 `ell-r` 对 `m` 取模。

#### 推导

取上式右侧第 `t` 个元素：

```text
RHS[t]
= sum_{ell in [m]} rho^r(Urep_{ell-r}(A))[t] * L_ell(B)[t]
```

根据 `rho` 定义：

```text
rho^r(Urep_{ell-r}(A))[t]
= Urep_{ell-r}(A)[t+r]
```

代入 `Urep`：

```text
Urep_{ell-r}(A)[t+r]
= A[(t+r) mod m, (ell-r+t+r) mod m]
= A[(t+r) mod m, (ell+t) mod m]
```

另一方面：

```text
L_ell(B)[t]
= B[(ell+t) mod m, t]
```

所以：

```text
RHS[t]
= sum_{ell in [m]} A[(t+r) mod m, (ell+t) mod m] * B[(ell+t) mod m, t]
```

令：

```text
s = (ell+t) mod m
```

当 `ell` 遍历 `[m]` 时，`s` 也遍历 `[m]`，因此：

```text
RHS[t]
= sum_{s in [m]} A[(t+r) mod m, s] * B[s, t]
= C[(t+r) mod m, t]
= L_r(C)[t]
```

故公式成立。

代码中的 `prop_3_8_rectangular_packed(...)` 实现的就是这个公式。

### 3.4 Algorithm 1 plaintext 生成公式

`generate_algorithm_1_rectangular_plaintexts(...)` 生成明文 mask/diagonal vectors `pt.A_{i,j,ell,r}`。设：

```text
m_c = m / c
ct.B_j contains L_{c*j}, ..., L_{c*j+c-1}(B)
```

为了枚举输入 ciphertext `ct.B_j` 的内部 rotation `ell in [c]`，以及输出 ciphertext `ct.C_r` 的 segment `tau in [c]`，定义：

```text
b_diag_idx   = c*j + ((tau + ell) mod c)
out_diag_idx = c*r + tau
```

对每个 block batch `A^{(i)}`，plaintext segment 为：

```text
pt.A_{i,j,ell,r}^{(tau)}
= rho^{out_diag_idx}(Urep_{b_diag_idx - out_diag_idx}(A^{(i)}))
```

其中 `b_diag_idx - out_diag_idx` 对 `m` 取模。完整 plaintext 是所有 `tau` segment 的 concatenation：

```text
pt.A_{i,j,ell,r}
= Concat_{tau in [c]} pt.A_{i,j,ell,r}^{(tau)}
```

随后 Algorithm 1 执行：

```text
ct.B_{j,ell} = Rot(ct.B_j, ell * H * n)
ct.i.r      = sum_{j in [m_c]} sum_{ell in [c]}
              PMult(ct.B_{j,ell}, pt.A_{i,j,ell,r})
```

这正是对每个 block product 应用上面的 rectangular Proposition 3.8。

最后，`algorithm_1_rectangular_plaintext_ciphertext_matmul(...)` 沿用 THOR Algorithm 1 的 H-lane internal rotation 聚合逻辑，把 block diagonal offset `i` 的结果从 lane `k` 调整到输出 block row `(i+k) mod H`，并累加得到最终 `C` 的 multi-lower-diagonal encoding。

---

## 4. Rectangular Algorithm 4：replication for B(m,n)

### 4.1 问题形态

新增接口：

```python
algorithm_4_replication_rectangular(
    packed_lower_B: list[np.ndarray],
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]
```

输入：

```text
B in R^{m x n},  m < n,  m | n
```

`packed_lower_B` 是普通 multi-lower-diagonal batched encoding：

```text
ct.B_j = Concat(
    L_{c*j}(B),
    L_{c*j+1}(B),
    ...,
    L_{c*j+c-1}(B)
)
```

输出是 replicated lower diagonals：

```text
ct.B'_ell,  ell in [m]
```

每个 `ct.B'_ell` 的所有 `c` 个 segment 都保存同一条 lower diagonal `L_ell(B)`。

### 4.2 Mask 选择

对指定 `ell in [m]`，令：

```text
j     = floor(ell / c)
local = ell mod c
```

`L_ell(B)` 位于 `ct.B_j` 的第 `local` 个 segment。定义 segment mask：

```text
v_ell[s] = 1, if local * H*n <= s < (local+1) * H*n
         = 0, otherwise
```

则：

```text
masked_ell = ct.B_j ⊙ v_ell
```

只保留第 `local` 个 segment。

### 4.3 Rotate-and-sum replication

因为一个 ciphertext 有 `c` 个 segment，而每个 segment 长度为：

```text
segment_len = H * n
```

把选中的 segment 复制到所有 segment，可写为：

```text
ct.B'_ell = sum_{q in [c]} rho^{q * n}(masked_ell)
```

在 slot 级实现中是：

```text
ct.B'_ell = sum_{q in [c]} Rot(masked_ell, q * H * n)
```

代码使用 BSGS 风格的 rotate-and-sum：

```python
repeated = masked_diag
for i in range(c.bit_length() - 1):
    repeated = add(repeated, rotate(repeated, segment_len * (1 << i)))
```

这要求 `c` 是 2 的幂。执行完后，每个 segment 都包含同一个 interlaced diagonal batch：

```text
segment_q(ct.B'_ell) = L_ell(B),  q in [c]
```

该 replicated form 是 rectangular Algorithm 2 的输入条件之一。

---

## 5. Rectangular Algorithm 5：B(n,m) upper-to-lower conversion

### 5.1 问题形态

新增接口：

```python
algorithm_5_matrix_transpose_rectangular(
    packed_upper_B: list[np.ndarray],
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]
```

输入：

```text
B in R^{n x m},  m < n,  m | n
```

`packed_upper_B` 保存：

```text
U_0(B), U_1(B), ..., U_{m-1}(B)
```

输出保存：

```text
L_0(B), L_1(B), ..., L_{m-1}(B)
```

注意：这里不是逻辑矩阵转置 `B^T`，而是同一个矩阵 `B` 的 upper-diagonal encoding 到 lower-diagonal encoding 的转换。输出应等价于：

```python
pack_lower_diagonals(B, n_slot, H)
```

### 5.2 Upper 与 lower diagonal 的关系

对 `B in R^{n x m}`，有：

```text
U_k(B)[t] = B[t mod n, (k+t) mod m]
L_r(B)[t] = B[(r+t) mod n, t mod m]
```

令：

```text
k = -r mod m
```

则：

```text
rho^r(U_k(B))[t]
= U_k(B)[t+r]
= B[(t+r) mod n, (k+t+r) mod m]
```

因为 `k = -r mod m`，所以：

```text
(k+t+r) mod m = t mod m
```

因此：

```text
rho^r(U_{-r}(B))[t]
= B[(t+r) mod n, t mod m]
= L_r(B)[t]
```

即：

```text
L_r(B) = rho^r(U_{-r mod m}(B)),  r in [m]
```

这就是 rectangular Algorithm 5 可用的核心 Lemma 3.4 关系。

### 5.3 Packed routing

输入 ciphertext `ct.B_j` 的第 `k_local` 个 segment 对应：

```text
source_diag_idx = c*j + k_local
```

它应该写入 output lower diagonal：

```text
out_diag_idx = -source_diag_idx mod m
```

进一步：

```text
ell           = floor(out_diag_idx / c)
out_local_idx = out_diag_idx mod c
```

也就是写入输出 ciphertext `ct.C_ell` 的第 `out_local_idx` 个 segment。

代码中完整 rotation 是：

```text
Rot(ct.B_j, (k_local - out_local_idx) * H*n + out_diag_idx * H)
```

其中：

- `(k_local - out_local_idx) * H*n`：把 source segment 对齐到 output segment；
- `out_diag_idx * H`：实现数学上的 `rho^{out_diag_idx}`。

### 5.4 Wrap mask

因为 `rho^{out_diag_idx}` 是长度 `n` 的 diagonal 内部 rotation，当 `out_diag_idx > 0` 时会在 segment 边界处产生 wrap。Algorithm 5 将 rotated result 分成两部分。令 `r = out_diag_idx`：

```text
mu_0: non-wrapped part，对应目标 L_r 的 t in [0, n-r)
mu_1: wrapped part，    对应目标 L_r 的 t in [n-r, n)
```

slot 级别上，`mu_0` 选择已经落在目标 output segment 内的 non-wrapped entries。`mu_1` 选择 rotation 后暂时落在目标 segment 前一个 segment 中的 wrapped entries；若目标 segment 是第 0 个 segment，则这些 entries 暂时落在 ciphertext 末尾。代码先分别累加：

```text
ct_ell_0 += PMult(ct_rot, mu_0)
ct_ell_1 += PMult(ct_rot, mu_1)
```

最后把 wrapped part 向右移动一个 segment，使它回到目标 output segment 的尾部：

```text
ct.C_ell = ct_ell_0 + Rot(ct_ell_1, -H*n)
```

因此输出 `ct.C_ell` 的各 segment 与 `pack_lower_diagonals(B, n_slot, H)` 完全一致。

---

## 6. Rectangular Algorithm 2：基于 Corollary 3.9 的 ciphertext-ciphertext matmul

### 6.1 问题形态

新增接口：

```python
algorithm_2_ciphertext_ciphertext_matmul_corollary_3_9(
    packed_lower_A: list[np.ndarray],
    packed_lower_B: list[np.ndarray],
    A_shape: Shape,
    B_shape: Shape,
    n_slot: int,
    H: int,
    B_is_replicated: bool = False,
) -> list[np.ndarray]
```

计算：

```text
C = A B
```

其中：

```text
A in R^{n x m}
B in R^{m x n}
C in R^{n x n}
m < n, m | n
```

输入 `A` 是普通 multi-lower-diagonal encoding：

```text
L_0(A), ..., L_{m-1}(A)
```

输入 `B` 支持两种形式：

1. `B_is_replicated=False`：`packed_lower_B` 是普通 multi-lower-diagonal encoding，函数内部调用 `algorithm_4_replication_rectangular`；
2. `B_is_replicated=True`：`packed_lower_B` 已经是 replicated lower diagonals `ct.B'_ell`。

输出是 `C in R^{n x n}` 的 multi-lower-diagonal encoding：

```text
L_0(C), ..., L_{n-1}(C)
```

因此输出 ciphertext 数量是：

```text
n_c = n / c
```

### 6.2 为什么不能用 Corollary 3.8

Corollary 3.8 对应：

```text
A in R^{m x n}, B in R^{n x n}, C in R^{m x n}
```

公式是：

```text
L_r(C) = sum_{ell in [n]} rho^ell(L_{r-ell}(A)) ⊙ L_ell(B),  r in [m]
```

它的输出 lower diagonal index `r` 只在 `[m]` 中，输出 shape 是 `(m,n)`。

新的 attention-score 类场景是：

```text
A in R^{n x m}, B in R^{m x n}, C in R^{n x n}
```

输出有 `n` 条 lower diagonals，且 `A` 只有 `m` 条 base lower diagonals。必须使用 Corollary 3.9 的 extended lower diagonal，而不能直接套用 Corollary 3.8。

### 6.3 Extended lower diagonal

对 `A in R^{n x m}`，普通 lower diagonal 只有：

```text
L_0(A), ..., L_{m-1}(A)
```

每条长度是 `n`。Corollary 3.9 定义扩展 lower diagonal：

```text
L'_k(A) = L_k(A),                                  if 0 <= k < m
        = rho^{m * floor(k / m)}(L_{k mod m}(A)),  if m <= k < n
```

等价地，令：

```text
k = q*m + a,  0 <= a < m
```

则：

```text
L'_k(A) = rho^{q*m}(L_a(A))
```

#### 推导扩展 diagonal 的元素形式

普通 lower diagonal：

```text
L_a(A)[t] = A[(a+t) mod n, t mod m]
```

所以：

```text
rho^{q*m}(L_a(A))[t]
= L_a(A)[t + q*m]
= A[(a+t+q*m) mod n, (t+q*m) mod m]
```

因为 `q*m mod m = 0`，所以：

```text
(t+q*m) mod m = t mod m
```

又因为 `k = q*m + a`，得到：

```text
L'_k(A)[t]
= A[(k+t) mod n, t mod m]
```

这说明 `L'_k(A)` 把 `A` 的 row offset 扩展到了 `[n]`，但 column index 仍然按 `m` cyclic。

### 6.4 Corollary 3.9 公式与证明

对：

```text
A in R^{n x m}
B in R^{m x n}
C = A B in R^{n x n}
```

有：

```text
L_r(C) = sum_{ell in [m]} rho^ell(L'_{r-ell}(A)) ⊙ L_ell(B),  r in [n]
```

其中 `r-ell` 对 `n` 取模。

#### 推导

取右侧第 `t` 个元素：

```text
RHS[t]
= sum_{ell in [m]} rho^ell(L'_{r-ell}(A))[t] * L_ell(B)[t]
```

根据 rotation：

```text
rho^ell(L'_{r-ell}(A))[t]
= L'_{r-ell}(A)[t+ell]
```

由扩展 lower diagonal 的元素形式：

```text
L'_{r-ell}(A)[t+ell]
= A[(r-ell+t+ell) mod n, (t+ell) mod m]
= A[(r+t) mod n, (t+ell) mod m]
```

而：

```text
L_ell(B)[t]
= B[(ell+t) mod m, t]
```

因此：

```text
RHS[t]
= sum_{ell in [m]}
  A[(r+t) mod n, (t+ell) mod m]
  B[(t+ell) mod m, t]
```

令：

```text
s = (t+ell) mod m
```

当 `ell` 遍历 `[m]` 时，`s` 也遍历 `[m]`，所以：

```text
RHS[t]
= sum_{s in [m]} A[(r+t) mod n, s] * B[s, t]
= C[(r+t) mod n, t]
= L_r(C)[t]
```

故 Corollary 3.9 公式成立。

### 6.5 修正后的 Eq. (10)：Corollary 3.9 的 batched 形式

原 Section 4.3.1 中基于 Corollary 3.8 的 Eq. (10) 适用于：

```text
A(m,n) @ B(n,n) -> C(m,n)
```

本节需要处理的是：

```text
A(n,m) @ B(m,n) -> C(n,n)
```

令：

```text
n_c = n / c
j in [n_c], r in [c]
out_diag = c*j + r
```

由 Corollary 3.9，输出第 `c*j+r` 条 lower diagonal 为：

```text
L_{c*j+r}(C)
= sum_{ell in [m]}
  rho^ell(L'_{c*j+r-ell}(A)) ⊙ L_ell(B)             (10)
```

其中 `c*j+r-ell` 对 `n` 取模。与论文原 Eq. (10) 相比，这里有三个变化：

1. `j` 的范围是 `[n_c]`，因为 `C in R^{n x n}` 有 `n` 条输出 lower diagonals；
2. 求和范围是 `[m]`，因为 `B in R^{m x n}` 只有 `m` 条 lower diagonals；
3. `A` 使用 Corollary 3.9 的 extended lower diagonal `L'_k(A)`，而不是普通 `L_k(A)`。

定义输出 ciphertext block：

```text
ct.C_j  <->  (L_{c*j}(C) | L_{c*j+1}(C) | ... | L_{c*(j+1)-1}(C))
```

则 Eq. (10) 的目标是同时生成 `ct.C_j` 中的 `c` 个 segment。

### 6.6 修正后的 Eq. (11)：定义 `ct.A_{j,ell}`

为了仿照论文 4.3.1/4.3.2，先把 `A` 的输入 ciphertext block 扩展到 `[n_c]` 个概念 block。对 `p in [n_c]`，定义：

```text
ct.A_p
<-> (L'_{c*p}(A) | L'_{c*p+1}(A) | ... | L'_{c*(p+1)-1}(A))
```

实际输入只有 `m_c = m/c` 个 base block。若：

```text
p = q*m_c + u,  u in [m_c]
```

则由 Corollary 3.9：

```text
ct.A_p
<-> (rho^{q*m}(L_{c*u}(A)) |
     rho^{q*m}(L_{c*u+1}(A)) |
     ... |
     rho^{q*m}(L_{c*u+c-1}(A)))
```

也就是说，`ct.A_p` 是 `A` 的第 `u` 个 base block 经 `rho^{q*m}` 后得到的扩展 block。

对固定的 `j in [n_c]` 和 `ell in [m]`，定义：

```text
ct.A_{j,ell}
<-> (rho^ell(L'_{c*j-ell}(A)) |
     rho^ell(L'_{c*j+1-ell}(A)) |
     ... |
     rho^ell(L'_{c*(j+1)-1-ell}(A)))
```

其中每个 `L'` 的下标都对 `n` 取模。于是 Eq. (10) 可以写成 ciphertext 级公式：

```text
ct.C_j
= sum_{ell in [m]} Mult(ct.A_{j,ell}, ct.B'_ell)     (11)
```

这里 `ct.B'_ell` 是 replicated lower diagonal ciphertext：它在 `ct.C_j` 的每个 local segment 中都提供同一条 `L_ell(B)`，因此可与 `ct.A_{j,ell}` 的 `c` 个 segment 一次相乘。

### 6.7 修正后的 Eq. (12)：生成 `ct.A_{j,ell}`

现在推导如何由扩展后的 `ct.A_p` 生成 `ct.A_{j,ell}`。

记：

```text
[ell]_m       = ell mod m
[[ell]_m]_c   = ([ell]_m) mod c
b_ell         = [[ell]_m]_c
a_ell         = ([ell]_m - [[ell]_m]_c) / c
d_ell         = -n*b_ell + ell
```

在本节的求和中 `ell in [m]`，所以 `[ell]_m = ell`。保留 `[ell]_m` 记号是为了强调：决定 source ciphertext block 的是 `ell` 在 `m` 条 `A` base diagonals 中的位置。

对任意 block index `u`，定义两个 rotated ciphertext：

```text
ct_{u,ell}  = Rot(ct.A_{u-a_ell}, d_ell * H)
ct'_{u,ell} = Rot(ct.A_{u-a_ell}, (d_ell - n) * H)
```

所有 `ct.A` 的 block index 都对 `n_c` 取模。特别地，对输出 block `j`，Eq. (12) 中真正使用的两个 source blocks 是：

```text
ct.A_{j-a_ell}
ct.A_{j-1-a_ell}
```

而不是未修正的 `ct.A_j` 和 `ct.A_{j-1}`。这正是从 `c*j+r-ell` 的 block borrow 推出来的修正项。

#### 为什么 source block 要减去 `a_ell`

把：

```text
ell = c*a_ell + b_ell,  0 <= b_ell < c
```

其中 `a_ell` 是 `ell` 跨过的完整 ciphertext-block 数，`b_ell` 是 block 内的 local diagonal 偏移。`Rot(..., -n*b_ell*H)` 处理 block 内的 local segment 对齐；剩余的完整 block 偏移必须体现在 source ciphertext index 上。

因此，生成 `ct.A_{j,ell}` 时，非边界部分来自：

```text
ct.A_{j-a_ell}
```

而跨过 block 边界的部分来自：

```text
ct.A_{j-1-a_ell}
```

也就是说，原论文 Eq. (12) 若直接从 `ct.A_j` / `ct.A_{j-1}` 生成 rotated ciphertext，会漏掉 `ell` 跨过 `c`-diagonal block 时产生的 `a_ell` 偏移；修正后必须使用 `ct.A_{j-a_ell}` / `ct.A_{j-1-a_ell}`。

#### 四个 mask

`Rot(..., d_ell * H)` 负责：

1. 把 local segment 按 `b_ell` 对齐到输出 segment；
2. 在 segment 内执行 `rho^ell` 的非 wrap 部分。

`Rot(..., (d_ell-n) * H)` 负责 `rho^ell` 的 wrap 部分。

因为本文的 slot rotation 约定是 `Rot(x, s)` 对 `s > 0` 执行左旋，并且上面的 packed rotation 已经包含 `-n*b_ell` 这一步 local segment 对齐，所以发生 block borrow 的 local segment 是：

```text
0 <= r < b_ell
```

而不是 `c-b_ell <= r < c`。因此四个 mask 均作用在输出 block `j` 的 slot 上。对每个 local segment `r in [c]` 和 diagonal position `t in [n]`：

```text
mu_{ell,0}: 1 iff 0 <= r < b_ell  and 0 <= t < n-ell
mu_{ell,1}: 1 iff b_ell <= r < c  and 0 <= t < n-ell
mu_{ell,2}: 1 iff 0 <= r < b_ell  and n-ell <= t < n
mu_{ell,3}: 1 iff b_ell <= r < c  and n-ell <= t < n
```

于是 corrected Eq. (12) 为：

```text
ct.A_{j,ell}
= ct_{j-1,ell}  * mu_{ell,0}
+ ct_{j,ell}    * mu_{ell,1}
+ ct'_{j-1,ell} * mu_{ell,2}
+ ct'_{j,ell}   * mu_{ell,3}                         (12)
```

当 `ell=0` 时，`a_ell=0`、`b_ell=0`，上式退化为：

```text
ct.A_{j,0} = ct.A_j
```

这与 Eq. (10) 中 `rho^0(L'_{c*j+r}(A))` 一致。

### 6.8 基于 Eq. (10)-(12) 的 BSGS 优化

由 6.7 的定义：

```text
ct'_{u,ell} = Rot(ct_{u,ell}, -n*H)
```

所以 Eq. (12) 可以改写为：

```text
ct.A_{j,ell}
= ct_{j-1,ell} * mu_{ell,0}
+ ct_{j,ell}   * mu_{ell,1}
+ Rot(
    ct_{j-1,ell} * rho^{n*H}(mu_{ell,2})
  + ct_{j,ell}   * rho^{n*H}(mu_{ell,3}),
    -n*H
  )
```

将它代入 Eq. (11)。因为 `ct.B'_ell` 是 replicated lower diagonal ciphertext，有：

```text
ct.B'_ell = Rot(ct.B'_ell, -n*H)
```

令：

```text
ct.C_{u,ell} = Mult(ct_{u,ell}, ct.B'_ell)
```

则非 wrap 部分为：

```text
sum_{1 <= ell < m}
  (ct.C_{j-1,ell} * mu_{ell,0}
 + ct.C_{j,ell}   * mu_{ell,1})
```

wrap 部分可先在未 rotate 状态下累加，再统一做一次 `Rot(..., -n*H)`：

```text
Rot(
  sum_{1 <= ell < m}
    (ct.C_{j-1,ell} * rho^{n*H}(mu_{ell,2})
   + ct.C_{j,ell}   * rho^{n*H}(mu_{ell,3})),
  -n*H
)
```

因此 Corollary 3.9 版本的 BSGS 聚合公式是：

```text
ct.C_{j,0} = Mult(ct.A_j, ct.B'_0)

ct.C'_j
= sum_{1 <= ell < m}
  (ct.C_{j-1,ell} * mu_{ell,0}
 + ct.C_{j,ell}   * mu_{ell,1})

ct.C''_j
= sum_{1 <= ell < m}
  (ct.C_{j-1,ell} * rho^{n*H}(mu_{ell,2})
 + ct.C_{j,ell}   * rho^{n*H}(mu_{ell,3}))

ct.C_j
= ct.C_{j,0} + ct.C'_j + Rot(ct.C''_j, -n*H)
```

这个推导与论文 4.3.2 的 BSGS 思路一致：不再为 `ct'_{u,ell}` 对每个输入 ciphertext 单独执行 `Rot(..., -n*H)`，而是先计算并累加乘法结果，再对聚合后的 wrap 部分执行一次 rotation。与论文原式相比，本节的关键差异是 source block 必须使用：

```text
j-a_ell      = j - ([ell]_m - [[ell]_m]_c) / c
j-1-a_ell    = j - 1 - ([ell]_m - [[ell]_m]_c) / c
```

这保证 Eq. (12) 与 Corollary 3.9 中的 `L'_{c*j+r-ell}(A)` 完全一致。

### 6.9 更激进的 direct-base BSGS：不显式生成 `ct.A_p`

6.6–6.8 的推导先引入概念上的 extended block：

```text
ct.A_p
<-> (L'_{c*p}(A) | L'_{c*p+1}(A) | ... | L'_{c*(p+1)-1}(A))
```

然后再从 `ct.A_p` 生成 `ct.A_{j,ell}`。这一节给出一个更激进、但等价的方案：**不显式计算任何 `ct.A_p`**，而是把 `ct.A_p` 的 extension rotation 直接融合进 Eq. (12) 和 BSGS。

#### 基本索引

设：

```text
m_c = m / c
n_c = n / c
j in [n_c]
ell in [m]
ell = c*a_ell + b_ell,  0 <= b_ell < c
```

对任意 conceptual extended block index：

```text
p in [n_c]
```

把它分解为：

```text
p = q_p*m_c + u_p,  u_p in [m_c]
```

则：

```text
ct.A_p
<-> internal_rotate((ct.A^{base}_{u_p}),q_p*m)
<-> (rho^{q_p*m}(L_{c*u_p}(A)) | rho^{q_p*m}(L_{c*u_p+1}(A)) | ... | rho^{q_p*m}(L_{c*(u_p+1)-1}(A)))
```

其中 `ct.A^{base}_{u_p}` 是原始输入 ciphertext block：

```text
ct.A^{base}_{u_p}
<-> (L_{c*u_p}(A) | L_{c*u_p+1}(A) | ... | L_{c*(u_p+1)-1}(A))
```

因此，`ct.A_p` 不需要 materialize；只要记录：

```text
p -> (u_p, q_p)
```

并把 `rho^{q_p*m}` 合并到后续 rotation 中即可。

#### 直接从 base block 生成 rotated input

在显式 `ct.A_p` 的版本中，对每个 `p, ell` 会使用：

```text
Rot(ct.A_p, (-n*b_ell + ell) * H)
```

而：

```text
ct.A_p <-> internal_rotate((ct.A^{base}_{u_p}),q_p*m)
```

所以可以直接定义 effective rotation：

```text
R_{p,ell} = q_p*m + ell
```

由于 `0 <= q_p*m + ell < n`，`R_{p,ell}` 是合法的 segment 内 rotation。这里的两个加数都表示作用在 length-`n` diagonal segment 内部的 rotation：

```text
rho^{R_{p,ell}}
= rho^{q_p*m + ell}
= rho^ell ∘ rho^{q_p*m}
```

其中：

1. `q_p*m` 来自 Corollary 3.9 的 extended diagonal，即 `ct.A_p` 相对 base block 的 internal rotation；
2. `ell` 来自 Eq. (10) 外层的 `rho^ell(L'_{...}(A))`。

因此，`R_{p,ell}` 决定的是 **segment 内部真实 rotation**，也决定 direct-base mask 的前后分界 `n-R_{p,ell}` / `R_{p,ell}`。

packed ciphertext 上实际执行的整体 slot rotation 还必须额外包含 local segment 对齐项。由于 `b_ell=[ell]_c` 表示 source local segment 到 output local segment 的 block 内偏移，整体 rotation amount 是：

```text
(R_{p,ell} - n*b_ell) * H
```

其中 `-n*b_ell` 负责 segment 对齐，`H` 是 SIMD batch 间隔因子。于是可以直接从原始 base block 生成：

```text
ctrot_{p,ell}
= Rot(ct.A^{base}_{u_p}, (R_{p,ell} - n*b_ell) * H)
```

对应的 wrap ciphertext 不再单独预先生成，而是保持为：

```text
ctrot'_{p,ell} = Rot(ctrot_{p,ell}, -n*H)
```

这一步等价于显式方案中的：

```text
Rot(ct.A_p, (-n*b_ell + ell - n) * H)
```

但没有先构造 `ct.A_p`。

#### 对输出 block `j` 的两个 source branches

对固定的 `j, ell`，Eq. (12) 需要两个 conceptual source blocks：

```text
p_prev = (j - 1 - a_ell) mod n_c
p_curr = (j     - a_ell) mod n_c
```

在本文使用的 `Rot(x, s)` / `rho^s` 约定下，`Rot(..., -n*b_ell*H)` 使 output local segment `r` 读取 source local segment `r-b_ell`。因此：

- `p_prev` 对应 local segment 范围 `0 <= r < b_ell`，即 `r-b_ell` 发生 block borrow 的部分；
- `p_curr` 对应 local segment 范围 `b_ell <= r < c`，即不发生 block borrow 的部分。

分别分解：

```text
p_prev = q_prev*m_c + u_prev
p_curr = q_curr*m_c + u_curr
```

并定义：

```text
R_prev = q_prev*m + ell
R_curr = q_curr*m + ell
```

则 direct-base 版本使用：

```text
ct_prev = Rot(ct.A^{base}_{u_prev}, (R_prev - n*b_ell) * H)
ct_curr = Rot(ct.A^{base}_{u_curr}, (R_curr - n*b_ell) * H)
```

这里的关键点是：`q_prev` 和 `q_curr` 可能不同，所以不能只使用一个全局的 `ell + q*m`。

#### direct-base 版本的 masks

显式 `ct.A_p` 的版本中，mask 的前后分界是 `n-ell` / `ell`，因为 `rho^{q_p*m}` 已经包含在 `ct.A_p` 中。

在 direct-base 版本中，`rho^{q_p*m}` 被融合进同一次 rotation，真正的 segment 内 rotation 是：

```text
R_prev = q_prev*m + ell
R_curr = q_curr*m + ell
```

因此 mask 的前后分界必须改成 branch-specific 的 `n-R_prev` / `R_prev` 和 `n-R_curr` / `R_curr`。

定义四个 branch-specific masks：

```text
mu^{prev}_{ell,0}: 1 iff 0 <= r < b_ell      and 0 <= t < n-R_prev
mu^{curr}_{ell,1}: 1 iff b_ell <= r < c      and 0 <= t < n-R_curr
mu^{prev}_{ell,2}: 1 iff 0 <= r < b_ell      and n-R_prev <= t < n
mu^{curr}_{ell,3}: 1 iff b_ell <= r < c      and n-R_curr <= t < n
```

于是 `ct.A_{j,ell}` 可直接写为：

```text
ct.A_{j,ell}
= ct_prev                * mu^{prev}_{ell,0}
+ ct_curr                * mu^{curr}_{ell,1}
+ Rot(ct_prev, -n*H)     * mu^{prev}_{ell,2}
+ Rot(ct_curr, -n*H)     * mu^{curr}_{ell,3}
```

这个公式完全绕过显式的 `ct.A_{p_prev}` 和 `ct.A_{p_curr}`。

#### direct-base BSGS 聚合

因为 `ct.B'_ell` 是 replicated lower diagonal ciphertext，有：

```text
ct.B'_ell = Rot(ct.B'_ell, -n*H)
```

所以仍然可以把 wrap branch 的 `Rot(..., -n*H)` 延迟到乘法和累加之后。

先定义 direct-base multiplication results：

```text
D_{p,ell} = Mult(ctrot_{p,ell}, ct.B'_ell)
```

对输出 block `j`，仍取：

```text
p_prev = (j - 1 - a_ell) mod n_c
p_curr = (j     - a_ell) mod n_c
```

则非 wrap 部分为：

```text
ct.C'_j
= sum_{ell in [m]}
  (D_{p_prev,ell} * mu^{prev}_{ell,0}
 + D_{p_curr,ell} * mu^{curr}_{ell,1})
```

wrap 部分先使用 rotated masks 聚合：

```text
ct.C''_j
= sum_{ell in [m]}
  (D_{p_prev,ell} * rho^{n*H}(mu^{prev}_{ell,2})
 + D_{p_curr,ell} * rho^{n*H}(mu^{curr}_{ell,3}))
```

最后统一执行一次 rotation：

```text
ct.C_j = ct.C'_j + Rot(ct.C''_j, -n*H)
```

注意这里的求和可以包含 `ell=0`。在显式 `ct.A_p` 的版本中，`ell=0` 通常退化为 `ct.A_j`，没有 wrap；但在 direct-base 版本中，如果 `p` 对应的 `q_p > 0`，则 effective rotation：

```text
R_{p,0} = q_p*m
```

仍可能产生 segment 内 wrap，因此 `ell=0` 也应由统一的 direct-base mask 和 BSGS 公式处理。

#### 该方案的约束

这个 aggressive 方案可行，但必须满足以下约束：

1. `q_p` 必须按 source block `p` 分别计算，不能对所有 branch 使用同一个 `q`；
2. mask 的前后分界必须使用 `R_{p,ell}=q_p*m+ell`，不能继续使用 `ell`；
3. `B` 侧仍然使用 `ct.B'_ell`，不能替换成 `ct.B'_{ell+q_p*m}`，因为 `B in R^{m x n}` 只有 `m` 条 lower diagonals；
4. local segment 的 block-boundary partition 仍然由 `b_ell=[ell]_c` 决定：在本文的 rotation 约定下，`0 <= r < b_ell` 使用 `p_prev`，`b_ell <= r < c` 使用 `p_curr`；`q_p*m` 是 `c` 的倍数，不改变 local segment 对齐。

因此，direct-base 方案本质上是把：

```text
ct.A_p + rho^ell
```

融合成：

```text
ct.A^{base}_{u_p} + rho^{q_p*m+ell}
```

同时保持 Eq. (12) 的 two-branch mask 结构和论文 4.3.2 的 BSGS 延迟 rotation 思路。

---

## 7. 接口与公式对应关系

| 接口 | 数学对象 | 核心公式 |
|---|---|---|
| `pack_repeated_upper_diagonals` | `A(m,m)` 的 upper diagonal 重复到长度 `n` | `Urep_i(A)[t] = U_i(A)[t mod m]` |
| `prop_3_8_rectangular_packed` | block 级 `A(m,m) @ B(m,n)` | `L_r(AB)=sum_ell rho^r(Urep_{ell-r}(A))⊙L_ell(B)` |
| `generate_algorithm_1_rectangular_plaintexts` | Algorithm 1 的 plaintext vectors | `rho^{out_diag_idx}(Urep_{b_diag_idx-out_diag_idx}(A))` |
| `algorithm_1_rectangular_plaintext_ciphertext_matmul` | `A(Hm,Hm) @ B(Hm,n)` | block diagonal batching + rectangular Proposition 3.8 |
| `algorithm_4_replication_rectangular` | `B(m,n)` lower diagonal replication | `ct.B'_ell=sum_q Rot((ct.B_j⊙v_ell), qHn)` |
| `algorithm_5_matrix_transpose_rectangular` | `B(n,m)` upper-to-lower conversion | `L_r(B)=rho^r(U_{-r mod m}(B))` |
| `algorithm_2_ciphertext_ciphertext_matmul_corollary_3_9` | `A(n,m) @ B(m,n)` | `L_r(AB)=sum_ell rho^ell(L'_{r-ell}(A))⊙L_ell(B)` |

---

## 8. 验证关系

当前 reference implementation 通过以下验证函数检查上述公式：

- `verify_prop_3_8_rectangular()`：验证 block 级 rectangular Proposition 3.8；
- `verify_algorithm_1_rectangular()`：验证 rectangular Algorithm 1 输出等价 NumPy `A @ B`；
- `verify_algorithm_2_corollary_3_9()`：验证 rectangular Algorithm 2 同时等价：
  - `corollary_3_9_packed(...)`；
  - NumPy `A @ B`；
  - packed `B` 输入与 replicated `B` 输入两种路径；
- `verify_algorithm_5_rectangular()`：验证 rectangular Algorithm 5 输出逐 ciphertext 等价 `pack_lower_diagonals(B, n_slot, H)`。

完整验证命令：

```bash
source /home/linghm/latti_venv/bin/activate && python thor_ckks_matmul_primitives.py
```

若输出：

```text
All THOR CKKS primitive verifications passed.
```

则说明 rectangular Algorithm 1、2、4、5 的 reference slot semantics 与上述公式一致。
