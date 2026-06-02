"""Strict NumPy reference for THOR paper matrix multiplication shapes.

This file models the rectangular matrix shapes and dimension assumptions used by
THOR Sections 4.2, 4.3, and 6.2. It is not a CKKS implementation: ciphertexts,
plaintexts, rotations, and multiplications are represented by NumPy vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PackedDiagonals:
    vectors: list[np.ndarray]
    matrix_shape: tuple[int, int]
    diagonal_count: int
    diagonal_length: int
    pack_capacity: int
    kind: str


# ---------------------------------------------------------------------------
# SIMD/HE primitive simulation
# ---------------------------------------------------------------------------


def rot_left(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    return np.roll(x, -(k % len(x)))


def rot_right(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    return np.roll(x, k % len(x))


def pmult(x: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    return x * p


def ct_mult(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def ct_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


# ---------------------------------------------------------------------------
# Shape checks and diagonal definitions from the paper
# ---------------------------------------------------------------------------


def _assert_pcmm_shape(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> tuple[int, int]:
    assert A.ndim == 2 and B.ndim == 2
    hidden_dim = A.shape[0]
    assert A.shape == (hidden_dim, hidden_dim), 'PCMM requires A in R^{d x d}'
    assert B.shape[0] == hidden_dim, 'PCMM requires B in R^{d x n}'
    seq_len = B.shape[1]
    assert seq_len <= hidden_dim, 'THOR PCMM assumes n <= d'
    assert hidden_dim % seq_len == 0, 'THOR PCMM assumes d is divisible by n'
    assert pack_capacity > 0
    assert seq_len % pack_capacity == 0, 'Algorithm 1 assumes c divides n'
    return hidden_dim, seq_len


def _assert_tall_by_wide_shape(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> tuple[int, int]:
    assert A.ndim == 2 and B.ndim == 2
    seq_len, short_dim = A.shape
    assert B.shape == (short_dim, seq_len), 'requires A in R^{n x m}, B in R^{m x n}'
    assert seq_len >= short_dim, 'THOR rectangular CCMM assumes n >= m'
    assert seq_len % short_dim == 0, 'THOR rectangular CCMM assumes n is divisible by m'
    assert pack_capacity > 0
    assert short_dim % pack_capacity == 0, 'batched diagonals assume c divides m'
    return seq_len, short_dim


def _assert_wide_by_square_shape(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> tuple[int, int]:
    assert A.ndim == 2 and B.ndim == 2
    short_dim, seq_len = A.shape
    assert B.shape == (seq_len, seq_len), 'requires A in R^{m x n}, B in R^{n x n}'
    assert seq_len >= short_dim, 'THOR rectangular CCMM assumes n >= m'
    assert seq_len % short_dim == 0, 'THOR rectangular CCMM assumes n is divisible by m'
    assert pack_capacity > 0
    assert short_dim % pack_capacity == 0, 'batched diagonals assume c divides m'
    return short_dim, seq_len


def upper_diag_square(A: np.ndarray, k: int) -> np.ndarray:
    """U_k(A) for A in R^{d x d}; vector length d."""
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    hidden_dim = A.shape[0]
    k %= hidden_dim
    return np.array([A[i, (i + k) % hidden_dim] for i in range(hidden_dim)], dtype=np.float64)


def upper_diag_tall(A: np.ndarray, k: int) -> np.ndarray:
    """U_k(A) for A in R^{n x m}, n >= m; vector length n."""
    assert A.ndim == 2
    seq_len, short_dim = A.shape
    assert seq_len >= short_dim
    k %= short_dim
    return np.array([A[i, (i + k) % short_dim] for i in range(seq_len)], dtype=np.float64)


def lower_diag_tall(A: np.ndarray, k: int) -> np.ndarray:
    """L_k(A) for A in R^{n x m}, n >= m; vector length n."""
    assert A.ndim == 2
    seq_len, short_dim = A.shape
    assert seq_len >= short_dim
    k %= short_dim
    return np.array([A[(i + k) % seq_len, i % short_dim] for i in range(seq_len)], dtype=np.float64)


def lower_diag_wide(A: np.ndarray, k: int) -> np.ndarray:
    """L_k(A) for A in R^{m x n}, n >= m; vector length n."""
    assert A.ndim == 2
    short_dim, seq_len = A.shape
    assert seq_len >= short_dim
    k %= short_dim
    return np.array([A[(i + k) % short_dim, i] for i in range(seq_len)], dtype=np.float64)


def lower_diag_square(A: np.ndarray, k: int) -> np.ndarray:
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    dim = A.shape[0]
    k %= dim
    return np.array([A[(i + k) % dim, i] for i in range(dim)], dtype=np.float64)


def matrix_from_lower_diags_tall(diags: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """Invert lower diagonals for a rows x cols matrix with rows >= cols and rows % cols == 0."""
    assert rows >= cols
    assert rows % cols == 0
    assert len(diags) == cols
    out = np.zeros((rows, cols), dtype=np.float64)
    seen = np.zeros((rows, cols), dtype=np.int64)
    for r, diag in enumerate(diags):
        assert len(diag) == rows
        for i, value in enumerate(diag):
            out[(i + r) % rows, i % cols] = value
            seen[(i + r) % rows, i % cols] += 1
    assert np.all(seen == 1)
    return out


def matrix_from_lower_diags_wide(diags: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """Invert lower diagonals for a rows x cols matrix with cols >= rows and cols % rows == 0."""
    assert cols >= rows
    assert cols % rows == 0
    assert len(diags) == rows
    out = np.zeros((rows, cols), dtype=np.float64)
    seen = np.zeros((rows, cols), dtype=np.int64)
    for r, diag in enumerate(diags):
        assert len(diag) == cols
        for i, value in enumerate(diag):
            out[(i + r) % rows, i] = value
            seen[(i + r) % rows, i] += 1
    assert np.all(seen == 1)
    return out


def matrix_from_lower_diags_square(diags: list[np.ndarray]) -> np.ndarray:
    dim = len(diags)
    out = np.zeros((dim, dim), dtype=np.float64)
    for r, diag in enumerate(diags):
        assert len(diag) == dim
        for i, value in enumerate(diag):
            out[(i + r) % dim, i] = value
    return out


# ---------------------------------------------------------------------------
# Multi-diagonal batched encodings
# ---------------------------------------------------------------------------


def _pack_diags(diags: list[np.ndarray], pack_capacity: int) -> list[np.ndarray]:
    assert pack_capacity > 0
    assert len(diags) % pack_capacity == 0
    vectors: list[np.ndarray] = []
    for start in range(0, len(diags), pack_capacity):
        vectors.append(np.concatenate(diags[start : start + pack_capacity]))
    return vectors


def _unpack_diags(
    vectors: list[np.ndarray], diagonal_count: int, diagonal_length: int, pack_capacity: int
) -> list[np.ndarray]:
    assert diagonal_count % pack_capacity == 0
    assert len(vectors) == diagonal_count // pack_capacity
    out: list[np.ndarray] = []
    for vec in vectors:
        assert len(vec) == diagonal_length * pack_capacity
        for local in range(pack_capacity):
            out.append(vec[local * diagonal_length : (local + 1) * diagonal_length].copy())
    assert len(out) == diagonal_count
    return out


def encode_pcmm_B(B: np.ndarray, pack_capacity: int) -> PackedDiagonals:
    """Multi-lower-diagonal batched encoding of B in R^{d x n} for Algorithm 1."""
    hidden_dim = B.shape[0]
    _assert_pcmm_shape(np.zeros((hidden_dim, hidden_dim)), B, pack_capacity)
    seq_len = B.shape[1]
    diags = [lower_diag_tall(B, r) for r in range(seq_len)]
    return PackedDiagonals(
        _pack_diags(diags, pack_capacity), B.shape, seq_len, hidden_dim, pack_capacity, 'pcmm_B_lower'
    )


def decode_pcmm_B(encoded: PackedDiagonals) -> np.ndarray:
    assert encoded.kind == 'pcmm_B_lower'
    hidden_dim, seq_len = encoded.matrix_shape
    diags = _unpack_diags(encoded.vectors, encoded.diagonal_count, encoded.diagonal_length, encoded.pack_capacity)
    return matrix_from_lower_diags_tall(diags, hidden_dim, seq_len)


def replicate_lower_diags(diags: list[np.ndarray], pack_capacity: int) -> list[np.ndarray]:
    assert pack_capacity > 0
    return [np.tile(diag, pack_capacity) for diag in diags]


def validated_replicated_diag(vec: np.ndarray, diagonal_length: int, pack_capacity: int) -> np.ndarray:
    assert len(vec) == diagonal_length * pack_capacity
    first = vec[:diagonal_length].copy()
    for local in range(1, pack_capacity):
        assert np.allclose(vec[local * diagonal_length : (local + 1) * diagonal_length], first)
    return first


# ---------------------------------------------------------------------------
# Strict paper algorithms
# ---------------------------------------------------------------------------


def extended_lower_diag_pcmm(B: np.ndarray, ell: int) -> np.ndarray:
    """L'_ell(B) = rho^{n floor(ell/n)}(L_{ell mod n}(B)) from Eq. (8)."""
    hidden_dim, seq_len = B.shape
    assert seq_len <= hidden_dim
    assert hidden_dim % seq_len == 0
    assert 0 <= ell < hidden_dim
    return rot_left(lower_diag_tall(B, ell % seq_len), seq_len * (ell // seq_len))


def pcmm_algorithm1_strict(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> np.ndarray:
    """Section 4.2 / Algorithm 1 semantics for A(d,d) @ B(d,n), n <= d."""
    hidden_dim, seq_len = _assert_pcmm_shape(A, B, pack_capacity)
    encoded_B = encode_pcmm_B(B, pack_capacity)
    assert np.allclose(decode_pcmm_B(encoded_B), B)

    out_diags: list[np.ndarray] = []
    for r in range(seq_len):
        acc = np.zeros(hidden_dim, dtype=np.float64)
        for ell in range(hidden_dim):
            plaintext = rot_left(upper_diag_square(A, ell - r), r)
            acc = ct_add(acc, pmult(extended_lower_diag_pcmm(B, ell), plaintext))
        out_diags.append(acc)
    return matrix_from_lower_diags_tall(out_diags, hidden_dim, seq_len)


def extended_lower_diag_tall(A: np.ndarray, k: int) -> np.ndarray:
    """L'_k(A) from Corollary 3.9 for A in R^{n x m}, n >= m.

    For k >= m, the paper uses rho^{m*floor(k/m)}(L_{k mod m}(A)).
    """
    seq_len, short_dim = A.shape
    assert seq_len >= short_dim
    assert seq_len % short_dim == 0
    k %= seq_len
    return rot_left(lower_diag_tall(A, k % short_dim), short_dim * (k // short_dim))


def ccmm_corollary_3_9_strict(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> np.ndarray:
    """Corollary 3.9 lower-lower CCMM for A(n,m) @ B(m,n), n >= m."""
    seq_len, short_dim = _assert_tall_by_wide_shape(A, B, pack_capacity)
    replicated_B = replicate_lower_diags([lower_diag_wide(B, ell) for ell in range(short_dim)], pack_capacity)
    lower_B = [validated_replicated_diag(vec, seq_len, pack_capacity) for vec in replicated_B]

    out_diags: list[np.ndarray] = []
    for r in range(seq_len):
        acc = np.zeros(seq_len, dtype=np.float64)
        for ell in range(short_dim):
            left = rot_left(extended_lower_diag_tall(A, r - ell), ell)
            acc = ct_add(acc, ct_mult(left, lower_B[ell]))
        out_diags.append(acc)
    return matrix_from_lower_diags_square(out_diags)


def ccmm_corollary_3_8_strict(A: np.ndarray, B: np.ndarray, pack_capacity: int) -> np.ndarray:
    """Corollary 3.8 lower-lower CCMM for A(m,n) @ B(n,n), n >= m.

    The paper rewrites Proposition 3.7's upper-lower form as:
    L_r(AB) = sum_ell rho^ell(L_{r-ell}(A)) * L_ell(B).
    """
    short_dim, seq_len = _assert_wide_by_square_shape(A, B, pack_capacity)
    lower_B = [lower_diag_square(B, ell) for ell in range(seq_len)]

    out_diags: list[np.ndarray] = []
    for r in range(short_dim):
        acc = np.zeros(seq_len, dtype=np.float64)
        for ell in range(seq_len):
            left = rot_left(lower_diag_wide(A, r - ell), ell)
            acc = ct_add(acc, ct_mult(left, lower_B[ell]))
        out_diags.append(acc)
    return matrix_from_lower_diags_wide(out_diags, short_dim, seq_len)


def ccmm_attention_score_strict(Q: np.ndarray, K: np.ndarray, pack_capacity: int) -> np.ndarray:
    """BERT-like attention score Q_h K_h^T with Q_h,K_h in R^{n x d_k}."""
    assert Q.shape == K.shape
    seq_len, head_dim = Q.shape
    assert seq_len == 2 * head_dim, 'THOR BERT comparison uses n = 2*d_k'
    return ccmm_corollary_3_9_strict(Q, K.T, pack_capacity)


def ccmm_attention_value_strict(alpha: np.ndarray, V: np.ndarray, pack_capacity: int) -> np.ndarray:
    """Attention head alpha_h V_h via transposed Corollary 3.8, matching THOR workflow."""
    seq_len = alpha.shape[0]
    assert alpha.shape == (seq_len, seq_len)
    assert V.shape[0] == seq_len
    head_dim = V.shape[1]
    assert seq_len == 2 * head_dim, 'THOR BERT comparison uses n = 2*d_k'
    head_t = ccmm_corollary_3_8_strict(V.T, alpha.T, pack_capacity)
    return head_t.T


@dataclass(frozen=True)
class ThorBertLayout:
    seq_len: int = 128
    head_dim: int = 64
    num_heads: int = 12
    diag_pack: int = 16
    lane_width: int = 16
    ffn_active_lanes: int = 6

    def __post_init__(self) -> None:
        assert self.seq_len == 2 * self.head_dim
        assert self.hidden_dim % self.seq_len == 0
        assert self.num_heads <= self.lane_width
        assert self.head_dim % self.diag_pack == 0
        assert self.ffn_active_lanes <= self.lane_width

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def hidden_blocks(self) -> int:
        return self.hidden_dim // self.seq_len

    @property
    def diag_ct_count(self) -> int:
        return self.head_dim // self.diag_pack


@dataclass
class ThorPackedEmbedding:
    values: np.ndarray
    layout: ThorBertLayout


def _lower_diags_square(A: np.ndarray) -> list[np.ndarray]:
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    return [lower_diag_square(A, k) for k in range(A.shape[0])]


def pack_embedding_thor_layout(embedding: np.ndarray, layout: ThorBertLayout = ThorBertLayout()) -> ThorPackedEmbedding:
    assert embedding.shape == (layout.seq_len, layout.hidden_dim)
    x_t = embedding.T
    x_blocks = np.vsplit(x_t, layout.hidden_blocks)
    values = np.zeros((layout.diag_ct_count, layout.diag_pack, layout.seq_len, layout.lane_width), dtype=np.complex128)
    for ct_idx in range(layout.diag_ct_count):
        for local_diag in range(layout.diag_pack):
            diag_idx = ct_idx * layout.diag_pack + local_diag
            for token_idx in range(layout.seq_len):
                for lane in range(layout.num_heads):
                    block = x_blocks[lane % layout.hidden_blocks]
                    values[ct_idx, local_diag, token_idx, lane] = complex(
                        lower_diag_square(block, diag_idx)[token_idx],
                        lower_diag_square(block, diag_idx + layout.head_dim)[token_idx],
                    )
    return ThorPackedEmbedding(values, layout)


def unpack_embedding_thor_layout(packed: ThorPackedEmbedding) -> np.ndarray:
    layout = packed.layout
    assert packed.values.shape == (layout.diag_ct_count, layout.diag_pack, layout.seq_len, layout.lane_width)
    blocks: list[np.ndarray] = []
    for block_idx in range(layout.hidden_blocks):
        diags = [np.zeros(layout.seq_len, dtype=np.float64) for _ in range(layout.seq_len)]
        for ct_idx in range(layout.diag_ct_count):
            for local_diag in range(layout.diag_pack):
                diag_idx = ct_idx * layout.diag_pack + local_diag
                vals = packed.values[ct_idx, local_diag, :, block_idx]
                diags[diag_idx] = vals.real.copy()
                diags[diag_idx + layout.head_dim] = vals.imag.copy()
        blocks.append(matrix_from_lower_diags_square(diags))
    return np.vstack(blocks).T


# ---------------------------------------------------------------------------
# Block-diagonal weight traversal and PCMM reference (THOR Section 6.2)
# ---------------------------------------------------------------------------


def thor_to_diag_blocks(matrix: np.ndarray, block_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    assert rows % block_rows == 0
    assert cols % block_cols == 0
    vertical = rows // block_rows
    horizontal = cols // block_cols
    blocks = np.empty((vertical, horizontal), dtype=object)
    row_blocks = np.vsplit(matrix, vertical)
    for row_idx, row_block in enumerate(row_blocks):
        blocks[row_idx] = np.hsplit(row_block, horizontal)
    diagonal_count = min(vertical, horizontal)
    diagonal_width = max(vertical, horizontal)
    diag_blocks = np.empty((diagonal_count, diagonal_width), dtype=object)
    for pos in range(diagonal_width):
        for diag_idx in range(diagonal_count):
            diag_blocks[diag_idx, pos] = blocks[(diag_idx + pos) % vertical, pos % horizontal]
    return diag_blocks, (diagonal_count, diagonal_width)


def pcmm_thor_block_layout_reference(weight: np.ndarray, x_t: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    block_rows, block_cols = block_shape
    assert weight.ndim == 2 and x_t.ndim == 2
    assert weight.shape[1] == x_t.shape[0]
    assert weight.shape[0] % block_rows == 0
    assert weight.shape[1] % block_cols == 0
    assert x_t.shape[0] % block_cols == 0
    diag_blocks, (diagonal_count, diagonal_width) = thor_to_diag_blocks(weight, block_shape)
    output_blocks = weight.shape[0] // block_rows
    input_blocks = weight.shape[1] // block_cols
    x_blocks = np.vsplit(x_t, input_blocks)
    out = np.zeros((weight.shape[0], x_t.shape[1]), dtype=np.float64)
    for out_block in range(output_blocks):
        acc = np.zeros((block_rows, x_t.shape[1]), dtype=np.float64)
        for pos in range(diagonal_width):
            diag_idx = (out_block - pos) % output_blocks
            if diag_idx < diagonal_count:
                acc += diag_blocks[diag_idx, pos] @ x_blocks[pos % input_blocks]
        out[out_block * block_rows : (out_block + 1) * block_rows] = acc
    return out


def make_rotated_copies_reference(diags: list[np.ndarray], rotations: int | None = None) -> list[list[np.ndarray]]:
    if rotations is None:
        rotations = len(diags)
    return [[diags[(start + offset) % len(diags)].copy() for offset in range(len(diags))] for start in range(rotations)]


def make_copies_reference(diags: list[np.ndarray], copy_count: int) -> list[np.ndarray]:
    assert copy_count > 0
    return [np.tile(diag, copy_count) for diag in diags]


def split_heads_token_major(x: np.ndarray, layout: ThorBertLayout) -> np.ndarray:
    assert x.shape == (layout.seq_len, layout.hidden_dim)
    return x.reshape(layout.seq_len, layout.num_heads, layout.head_dim).transpose(1, 0, 2).copy()


def merge_heads_token_major(heads: np.ndarray, layout: ThorBertLayout) -> np.ndarray:
    assert heads.shape == (layout.num_heads, layout.seq_len, layout.head_dim)
    return heads.transpose(1, 0, 2).reshape(layout.seq_len, layout.hidden_dim).copy()


def attention_score_thor_layout_reference(
    Q: np.ndarray, K: np.ndarray, layout: ThorBertLayout = ThorBertLayout()
) -> np.ndarray:
    q_heads = split_heads_token_major(Q, layout)
    k_heads = split_heads_token_major(K, layout)
    return np.stack(
        [
            ccmm_attention_score_strict(q_heads[head], k_heads[head], layout.diag_pack)
            for head in range(layout.num_heads)
        ]
    )


def attention_context_thor_layout_reference(
    alpha: np.ndarray, V: np.ndarray, layout: ThorBertLayout = ThorBertLayout()
) -> np.ndarray:
    assert alpha.shape == (layout.num_heads, layout.seq_len, layout.seq_len)
    v_heads = split_heads_token_major(V, layout)
    context_heads = np.stack(
        [ccmm_attention_value_strict(alpha[head], v_heads[head], layout.diag_pack) for head in range(layout.num_heads)]
    )
    return merge_heads_token_major(context_heads, layout)


def thor_layout_gap_report() -> str:
    return """当前 paper-level 复现和 THOR 代码实现的主要差距：
1. THOR 固定使用 BERT-base-like layout：seq_len=128, hidden_dim=768, num_heads=12, head_dim=64。
2. THOR 输入不是单个矩阵向量编码，而是 4 个 ciphertext，每个包含 16 个 lower diagonals；每个 token block 有 16 lanes，其中 12 lanes 有效。
3. THOR 把 diagonal l 和 l+64 分别放入同一 slot 的 real/imag，用 complex multiply 同时处理两条 diagonal。
4. THOR PCMM 先按 _encode_w_att/_encode_w_ff 的 block diagonal traversal 做 plaintext diagonal multiplication，再用 rotate_internal 对齐 block diagonals。
5. THOR attention CCMM 不只是 Corollary 3.8/3.9 的抽象公式；源码还包含 key upper-to-lower transpose、query/attention probability copied diagonals、mask split 和 fragment routing。
6. 本文件新增的 THOR-layout reference 复现这些算法/packing 原理，但仍故意不模拟 rescale、relinearize、level、bootstrap 或 CKKS 噪声。"""


# ---------------------------------------------------------------------------
# Verification entrypoint
# ---------------------------------------------------------------------------


def verify_pcmm_algorithm1() -> None:
    rng = np.random.default_rng(20260601)
    hidden_dim = 8
    seq_len = 4
    pack_capacity = 2
    A = rng.normal(size=(hidden_dim, hidden_dim))
    B = rng.normal(size=(hidden_dim, seq_len))

    C = pcmm_algorithm1_strict(A, B, pack_capacity)

    assert np.allclose(C, A @ B)


def verify_pcmm_square_bert_projection_shape() -> None:
    rng = np.random.default_rng(20260604)
    hidden_dim = 4
    seq_len = 4
    pack_capacity = 1
    A = rng.normal(size=(hidden_dim, hidden_dim))
    B = rng.normal(size=(hidden_dim, seq_len))

    C = pcmm_algorithm1_strict(A, B, pack_capacity)

    assert np.allclose(C, A @ B)


def verify_attention_score_case() -> None:
    rng = np.random.default_rng(20260602)
    seq_len = 4
    head_dim = 2
    pack_capacity = 1
    Q = rng.normal(size=(seq_len, head_dim))
    K = rng.normal(size=(seq_len, head_dim))

    score = ccmm_attention_score_strict(Q, K, pack_capacity)

    assert np.allclose(score, Q @ K.T)


def verify_attention_value_case() -> None:
    rng = np.random.default_rng(20260603)
    seq_len = 4
    head_dim = 2
    pack_capacity = 1
    alpha = rng.normal(size=(seq_len, seq_len))
    V = rng.normal(size=(seq_len, head_dim))

    head = ccmm_attention_value_strict(alpha, V, pack_capacity)

    assert np.allclose(head, alpha @ V)


def verify_corollary_variants() -> None:
    rng = np.random.default_rng(20260605)
    for seq_len, short_dim, pack_capacity in [(6, 3, 1), (9, 3, 1), (12, 3, 3)]:
        A_tall = rng.normal(size=(seq_len, short_dim))
        B_wide = rng.normal(size=(short_dim, seq_len))
        A_wide = rng.normal(size=(short_dim, seq_len))
        B_square = rng.normal(size=(seq_len, seq_len))

        assert np.allclose(ccmm_corollary_3_9_strict(A_tall, B_wide, pack_capacity), A_tall @ B_wide)
        assert np.allclose(ccmm_corollary_3_8_strict(A_wide, B_square, pack_capacity), A_wide @ B_square)


def verify_invalid_shapes() -> None:
    try:
        pcmm_algorithm1_strict(np.zeros((4, 4)), np.zeros((4, 5)), 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('PCMM must reject seq_len > hidden_dim')

    try:
        ccmm_attention_score_strict(np.zeros((5, 2)), np.zeros((5, 2)), 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('attention score must reject shapes outside n = 2*d_k')

    try:
        pcmm_algorithm1_strict(np.zeros((8, 8)), np.zeros((8, 4)), 3)
    except AssertionError:
        pass
    else:
        raise AssertionError('PCMM must reject pack_capacity that does not divide seq_len')


def verify_thor_embedding_layout_round_trip() -> None:
    rng = np.random.default_rng(20260606)
    layout = ThorBertLayout(seq_len=8, head_dim=4, num_heads=4, diag_pack=2, lane_width=4, ffn_active_lanes=2)
    embedding = rng.normal(size=(layout.seq_len, layout.hidden_dim))

    packed = pack_embedding_thor_layout(embedding, layout)
    decoded = unpack_embedding_thor_layout(packed)

    assert packed.values.shape == (2, 2, 8, 4)
    assert np.allclose(decoded, embedding)


def verify_thor_pcmm_block_layout_reference() -> None:
    rng = np.random.default_rng(20260607)
    layout = ThorBertLayout(seq_len=8, head_dim=4, num_heads=4, diag_pack=2, lane_width=4, ffn_active_lanes=2)
    W = rng.normal(size=(layout.hidden_dim, layout.hidden_dim))
    X = rng.normal(size=(layout.hidden_dim, layout.seq_len))

    out = pcmm_thor_block_layout_reference(W, X, block_shape=(layout.head_dim, layout.seq_len))

    assert np.allclose(out, W @ X)


def verify_thor_attention_layout_references() -> None:
    rng = np.random.default_rng(20260608)
    layout = ThorBertLayout(seq_len=8, head_dim=4, num_heads=4, diag_pack=2, lane_width=4, ffn_active_lanes=2)
    Q = rng.normal(size=(layout.seq_len, layout.hidden_dim))
    K = rng.normal(size=(layout.seq_len, layout.hidden_dim))
    V = rng.normal(size=(layout.seq_len, layout.hidden_dim))
    alpha = rng.normal(size=(layout.num_heads, layout.seq_len, layout.seq_len))

    copied = make_copies_reference(_lower_diags_square(Q[:, : layout.seq_len]), copy_count=layout.diag_pack)
    assert len(copied) == layout.seq_len
    assert np.allclose(copied[0][: layout.seq_len], lower_diag_square(Q[:, : layout.seq_len], 0))

    scores = attention_score_thor_layout_reference(Q, K, layout)
    context = attention_context_thor_layout_reference(alpha, V, layout)

    q_heads = split_heads_token_major(Q, layout)
    k_heads = split_heads_token_major(K, layout)
    v_heads = split_heads_token_major(V, layout)
    expected_scores = np.stack([q_heads[h] @ k_heads[h].T for h in range(layout.num_heads)])
    expected_context = merge_heads_token_major(
        np.stack([alpha[h] @ v_heads[h] for h in range(layout.num_heads)]), layout
    )

    assert np.allclose(scores, expected_scores)
    assert np.allclose(context, expected_context)


def verify_thor_layout_gap_report() -> None:
    report = thor_layout_gap_report()
    assert 'seq_len=128' in report
    assert 'real/imag' in report
    assert 'rotate_internal' in report


def verify_all() -> None:
    verify_pcmm_algorithm1()
    verify_pcmm_square_bert_projection_shape()
    verify_attention_score_case()
    verify_attention_value_case()
    verify_corollary_variants()
    verify_invalid_shapes()
    verify_thor_embedding_layout_round_trip()
    verify_thor_pcmm_block_layout_reference()
    verify_thor_attention_layout_references()
    verify_thor_layout_gap_report()
    print('All strict THOR paper checks passed.')


if __name__ == '__main__':
    verify_all()
