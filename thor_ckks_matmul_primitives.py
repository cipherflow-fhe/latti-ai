"""Packed CKKS-style THOR matrix multiplication primitives.

This file is a NumPy-only reference. A NumPy vector represents the slot
contents of one CKKS ciphertext; the operations below simulate CKKS SIMD
slot operations without encryption, rescaling, or noise modeling.
"""

from __future__ import annotations

import numpy as np


Shape = tuple[int, int]


# ---------------------------------------------------------------------------
# CKKS SIMD primitive simulation
# ---------------------------------------------------------------------------


def _as_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    assert arr.ndim == 1, f'{name} must be a 1D numpy vector'
    return arr


def rotate(x: np.ndarray, step: int) -> np.ndarray:
    """Rotate CKKS slots.

    `step > 0` rotates left by `step`; `step < 0` rotates right by `abs(step)`.
    """

    arr = _as_vector(x, 'x')
    if arr.size == 0:
        return arr.copy()
    return np.roll(arr, -step)


def multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise multiplication of two slot vectors."""

    lhs = _as_vector(x, 'x')
    rhs = _as_vector(y, 'y')
    assert lhs.shape == rhs.shape, f'shape mismatch: {lhs.shape} != {rhs.shape}'
    return lhs * rhs


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise addition of two slot vectors."""

    lhs = _as_vector(x, 'x')
    rhs = _as_vector(y, 'y')
    assert lhs.shape == rhs.shape, f'shape mismatch: {lhs.shape} != {rhs.shape}'
    return lhs + rhs


def subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pointwise subtraction of two slot vectors."""

    lhs = _as_vector(x, 'x')
    rhs = _as_vector(y, 'y')
    assert lhs.shape == rhs.shape, f'shape mismatch: {lhs.shape} != {rhs.shape}'
    return lhs - rhs


# ---------------------------------------------------------------------------
# THOR diagonal extraction
# ---------------------------------------------------------------------------


def _as_matrix(A: np.ndarray, name: str = 'A') -> np.ndarray:
    arr = np.asarray(A)
    assert arr.ndim == 2, f'{name} must be a 2D numpy matrix'
    assert arr.shape[0] > 0 and arr.shape[1] > 0, f'{name} must have non-empty dimensions'
    return arr


def _normalize_shape(shape: Shape) -> Shape:
    assert len(shape) == 2, 'shape must be a pair (rows, cols)'
    rows, cols = int(shape[0]), int(shape[1])
    assert rows > 0 and cols > 0, 'shape dimensions must be positive'
    return rows, cols


def upper_diagonal(A: np.ndarray, i: int) -> np.ndarray:
    """Return THOR's cyclic upper diagonal U_i(A).

    For A in R^{m x n}, Definition 3.1 gives:
        U_i(A)[t] = A[t mod m, (i + t) mod n]

    The output length is max(m, n). The index i must satisfy
    0 <= i < min(m, n).
    """

    matrix = _as_matrix(A)
    rows, cols = matrix.shape
    assert i >= 0, 'diagonal index i must be non-negative'
    assert i < min(rows, cols), 'diagonal index i must be smaller than min(A.shape)'
    diag_len = max(rows, cols)
    out = np.array([matrix[t % rows, (i + t) % cols] for t in range(diag_len)], dtype=matrix.dtype)
    assert out.shape == (diag_len,)
    return out


def lower_diagonal(A: np.ndarray, i: int) -> np.ndarray:
    """Return THOR's cyclic lower diagonal L_i(A).

    For A in R^{m x n}, Definition 3.1 gives:
        L_i(A)[t] = A[(i + t) mod m, t mod n]

    The output length is max(m, n). The index i must satisfy
    0 <= i < min(m, n).
    """

    matrix = _as_matrix(A)
    rows, cols = matrix.shape
    assert i >= 0, 'diagonal index i must be non-negative'
    assert i < min(rows, cols), 'diagonal index i must be smaller than min(A.shape)'
    diag_len = max(rows, cols)
    out = np.array([matrix[(i + t) % rows, t % cols] for t in range(diag_len)], dtype=matrix.dtype)
    assert out.shape == (diag_len,)
    return out


# ---------------------------------------------------------------------------
# Multi-diagonal batched packing and unpacking
# ---------------------------------------------------------------------------


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_H(H: int) -> int:
    H = int(H)
    assert _is_power_of_two(H), 'H must be a positive power of two'
    return H


def _as_matrix_list(matrices: list[np.ndarray], H: int) -> list[np.ndarray]:
    H = _validate_H(H)
    assert isinstance(matrices, list), 'matrices must be a list[np.ndarray]'
    assert len(matrices) == H, f'expected {H} matrices, got {len(matrices)}'
    matrix_list = [_as_matrix(matrix, f'matrices[{idx}]') for idx, matrix in enumerate(matrices)]
    shape = matrix_list[0].shape
    for idx, matrix in enumerate(matrix_list):
        assert matrix.shape == shape, f'matrices[{idx}] has shape {matrix.shape}, expected {shape}'
    return matrix_list


def _packing_params(shape: Shape, n_slot: int, H: int) -> tuple[int, int, int, int, int, int]:
    rows, cols = _normalize_shape(shape)
    H = _validate_H(H)
    assert n_slot > 0, 'n_slot must be positive'
    diag_len = max(rows, cols)
    n_diag = min(rows, cols)
    segment_len = H * diag_len
    assert n_slot % segment_len == 0, 'n_slot must be divisible by H * max(shape)'
    c = n_slot // segment_len
    assert c > 0, 'one ciphertext must contain at least one diagonal segment'
    assert n_diag % c == 0, 'min(shape) must be divisible by c = n_slot // (H * max(shape))'
    return rows, cols, n_diag, diag_len, c, segment_len


def _interlace_diagonal_batch(diags: list[np.ndarray], diag_len: int, H: int) -> np.ndarray:
    assert len(diags) == H, f'expected {H} diagonals, got {len(diags)}'
    dtype = np.result_type(*(diag.dtype for diag in diags))
    segment = np.zeros(H * diag_len, dtype=dtype)
    for h, diag in enumerate(diags):
        vec = _as_vector(diag, f'diags[{h}]')
        assert vec.shape == (diag_len,), f'expected diagonal length {diag_len}, got {vec.shape[0]}'
        segment[h::H] = vec
    return segment


def _deinterlace_diagonal_batch(segment: np.ndarray, diag_len: int, H: int) -> list[np.ndarray]:
    slots = _as_vector(segment, 'segment')
    assert slots.shape == (H * diag_len,), f'expected segment length {H * diag_len}, got {slots.shape[0]}'
    return [slots[h::H].copy() for h in range(H)]


def _pack_batched_diagonal_vectors(
    batched_diags: list[np.ndarray], shape: Shape, n_slot: int, H: int
) -> list[np.ndarray]:
    _, _, n_diag, _, c, segment_len = _packing_params(shape, n_slot, H)
    assert len(batched_diags) == n_diag, f'expected {n_diag} batched diagonals, got {len(batched_diags)}'

    normalized: list[np.ndarray] = []
    for idx, diag in enumerate(batched_diags):
        vec = _as_vector(diag, f'batched_diags[{idx}]')
        assert vec.shape == (segment_len,), f'expected batched diagonal length {segment_len}, got {vec.shape[0]}'
        normalized.append(vec)

    packed: list[np.ndarray] = []
    for start in range(0, n_diag, c):
        ciphertext_slots = np.concatenate(normalized[start : start + c])
        assert ciphertext_slots.shape == (n_slot,)
        packed.append(ciphertext_slots)
    return packed


def _unpack_batched_diagonal_vectors(vectors: list[np.ndarray], shape: Shape, n_slot: int, H: int) -> list[np.ndarray]:
    _, _, n_diag, _, c, segment_len = _packing_params(shape, n_slot, H)
    assert isinstance(vectors, list), 'packed vectors must be list[np.ndarray]'
    expected_vectors = n_diag // c
    assert len(vectors) == expected_vectors, f'expected {expected_vectors} packed vectors, got {len(vectors)}'

    batched_diags: list[np.ndarray] = []
    for vector in vectors:
        slots = _as_vector(vector, 'packed vector')
        assert slots.shape == (n_slot,), f'expected packed vector length {n_slot}, got {slots.shape[0]}'
        for local_idx in range(c):
            start = local_idx * segment_len
            batched_diags.append(slots[start : start + segment_len].copy())

    assert len(batched_diags) == n_diag
    return batched_diags


def pack_upper_diagonals(matrices: list[np.ndarray], n_slot: int, H: int) -> list[np.ndarray]:
    """Pack batched upper diagonals using THOR's multi-diagonal format.

    `matrices` contains H matrices of the same shape. Each ciphertext is split
    into c = n_slot // (H * max(shape)) consecutive segments. Segment j stores
    one diagonal index, interlaced as:
        segment[t * H + h] = U_i(matrices[h])[t]
    """

    matrix_list = _as_matrix_list(matrices, H)
    shape = matrix_list[0].shape
    _, _, n_diag, diag_len, _, _ = _packing_params(shape, n_slot, H)
    batched_diags = [
        _interlace_diagonal_batch([upper_diagonal(matrix, i) for matrix in matrix_list], diag_len, H)
        for i in range(n_diag)
    ]
    return _pack_batched_diagonal_vectors(batched_diags, shape, n_slot, H)


def pack_lower_diagonals(matrices: list[np.ndarray], n_slot: int, H: int) -> list[np.ndarray]:
    """Pack batched lower diagonals using THOR's multi-diagonal format.

    `matrices` contains H matrices of the same shape. Each ciphertext is split
    into c = n_slot // (H * max(shape)) consecutive segments. Segment j stores
    one diagonal index, interlaced as:
        segment[t * H + h] = L_i(matrices[h])[t]
    """

    matrix_list = _as_matrix_list(matrices, H)
    shape = matrix_list[0].shape
    _, _, n_diag, diag_len, _, _ = _packing_params(shape, n_slot, H)
    batched_diags = [
        _interlace_diagonal_batch([lower_diagonal(matrix, i) for matrix in matrix_list], diag_len, H)
        for i in range(n_diag)
    ]
    return _pack_batched_diagonal_vectors(batched_diags, shape, n_slot, H)


def unpack_upper_diagonals(vectors: list[np.ndarray], shape: Shape, n_slot: int, H: int) -> list[np.ndarray]:
    """Reconstruct H matrices from packed upper diagonals."""

    rows, cols, n_diag, diag_len, _, _ = _packing_params(shape, n_slot, H)
    batched_diags = _unpack_batched_diagonal_vectors(vectors, shape, n_slot, H)
    out = [np.zeros((rows, cols), dtype=batched_diags[0].dtype) for _ in range(H)]
    for i in range(n_diag):
        diags = _deinterlace_diagonal_batch(batched_diags[i], diag_len, H)
        for h, diag in enumerate(diags):
            for t, value in enumerate(diag):
                out[h][t % rows, (i + t) % cols] = value
    return out


def unpack_lower_diagonals(vectors: list[np.ndarray], shape: Shape, n_slot: int, H: int) -> list[np.ndarray]:
    """Reconstruct H matrices from packed lower diagonals."""

    rows, cols, n_diag, diag_len, _, _ = _packing_params(shape, n_slot, H)
    batched_diags = _unpack_batched_diagonal_vectors(vectors, shape, n_slot, H)
    out = [np.zeros((rows, cols), dtype=batched_diags[0].dtype) for _ in range(H)]
    for i in range(n_diag):
        diags = _deinterlace_diagonal_batch(batched_diags[i], diag_len, H)
        for h, diag in enumerate(diags):
            for t, value in enumerate(diag):
                out[h][(i + t) % rows, t % cols] = value
    return out


# ---------------------------------------------------------------------------
# THOR Propositions 3.6 and 3.7 on packed diagonal vectors
# ---------------------------------------------------------------------------


def prop_3_6_packed(
    packed_upper_A: list[np.ndarray],
    packed_lower_B: list[np.ndarray],
    A_shape: Shape,
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]:
    """Compute C = A @ B using THOR Proposition 3.6 on packed diagonals.

    公式（论文 Proposition 3.6）:
        L_r(AB) = sum_{ell in [m]} rho^r(U_{ell-r}(A)) ⊙ L_ell(B),  r in [n]

    变量解释：
    - A has shape (n, m), B has shape (m, n), with n >= m and n % m == 0.
    - packed_upper_A stores U_0(A), ..., U_{m-1}(A), using H-way batched encoding.
    - packed_lower_B stores L_0(B), ..., L_{m-1}(B), using H-way batched encoding.
    - rho^r is implemented as `rotate(x, r * H)` on interlaced batched slots.
    - ⊙ is pointwise slot multiplication.
    - The index ell-r of U is taken modulo m.

    Returns packed lower diagonals L_0(C), ..., L_{n-1}(C) of C = AB,
    where C has shape (n, n).
    """

    n, m = _normalize_shape(A_shape)
    b_rows, b_cols = _normalize_shape(B_shape)
    assert n >= m, 'Proposition 3.6 requires A shape (n, m) with n >= m'
    assert n % m == 0, 'Proposition 3.6 requires n divisible by m'
    assert (b_rows, b_cols) == (m, n), 'Proposition 3.6 requires B shape (m, n)'

    upper_A = _unpack_batched_diagonal_vectors(packed_upper_A, A_shape, n_slot, H)
    lower_B = _unpack_batched_diagonal_vectors(packed_lower_B, B_shape, n_slot, H)
    dtype = np.result_type(*(diag.dtype for diag in upper_A + lower_B))

    lower_C: list[np.ndarray] = []
    for r in range(n):
        acc = np.zeros(H * n, dtype=dtype)
        for ell in range(m):
            u_idx = (ell - r) % m
            acc = add(acc, multiply(rotate(upper_A[u_idx], r * H), lower_B[ell]))
        lower_C.append(acc)

    return _pack_batched_diagonal_vectors(lower_C, (n, n), n_slot, H)


def _extended_upper_diagonal_for_prop_3_7(upper_A: list[np.ndarray], m: int, n: int, H: int, k: int) -> np.ndarray:
    """Return U'_k(A) from Proposition 3.7 for 0 <= k < n."""

    assert 0 <= k < n
    assert len(upper_A) == m
    if k < m:
        return upper_A[k]
    return rotate(upper_A[k % m], m * (k // m) * H)


def prop_3_7_packed(
    packed_upper_A: list[np.ndarray],
    packed_lower_B: list[np.ndarray],
    A_shape: Shape,
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]:
    """Compute C = A @ B using THOR Proposition 3.7 on packed diagonals.

    公式（论文 Proposition 3.7）:
        L_r(AB) = sum_{ell in [n]} rho^r(U'_{ell-r}(A)) ⊙ L_ell(B),  r in [m]

    扩展上对角线：
        U'_k(A) = U_k(A),                              if k < m
        U'_k(A) = rho^{m * floor(k / m)}(U_{k mod m}(A)), if m <= k < n

    变量解释：
    - A has shape (m, n), B has shape (n, n), with n >= m and n % m == 0.
    - packed_upper_A stores the base diagonals U_0(A), ..., U_{m-1}(A).
    - packed_lower_B stores L_0(B), ..., L_{n-1}(B).
    - rho^r is implemented as `rotate(x, r * H)` on interlaced batched slots.
    - ⊙ is pointwise slot multiplication.
    - The index ell-r of U' is taken modulo n before applying the U' definition.

    Returns packed lower diagonals L_0(C), ..., L_{m-1}(C) of C = AB,
    where C has shape (m, n).
    """

    m, n = _normalize_shape(A_shape)
    b_rows, b_cols = _normalize_shape(B_shape)
    assert n >= m, 'Proposition 3.7 requires A shape (m, n) with n >= m'
    assert n % m == 0, 'Proposition 3.7 requires n divisible by m'
    assert (b_rows, b_cols) == (n, n), 'Proposition 3.7 requires B shape (n, n)'

    upper_A = _unpack_batched_diagonal_vectors(packed_upper_A, A_shape, n_slot, H)
    lower_B = _unpack_batched_diagonal_vectors(packed_lower_B, B_shape, n_slot, H)
    dtype = np.result_type(*(diag.dtype for diag in upper_A + lower_B))

    lower_C: list[np.ndarray] = []
    for r in range(m):
        acc = np.zeros(H * n, dtype=dtype)
        for ell in range(n):
            u_prime_idx = (ell - r) % n
            u_prime = _extended_upper_diagonal_for_prop_3_7(upper_A, m, n, H, u_prime_idx)
            acc = add(acc, multiply(rotate(u_prime, r * H), lower_B[ell]))
        lower_C.append(acc)

    return _pack_batched_diagonal_vectors(lower_C, (m, n), n_slot, H)


# ---------------------------------------------------------------------------
# THOR Corollaries 3.8 and 3.9 on packed lower diagonal vectors
# ---------------------------------------------------------------------------


def corollary_3_8_packed(
    packed_lower_A: list[np.ndarray],
    packed_lower_B: list[np.ndarray],
    A_shape: Shape,
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]:
    """Compute C = A @ B using THOR Corollary 3.8 on packed lower diagonals.

    公式（论文 Corollary 3.8）:
        L_r(AB) = sum_{ell in [n]} rho^ell(L_{r-ell}(A)) ⊙ L_ell(B),  r in [m]

    变量解释：
    - A has shape (m, n), B has shape (n, n), with n >= m and n % m == 0.
    - packed_lower_A stores L_0(A), ..., L_{m-1}(A), using H-way batched encoding.
    - packed_lower_B stores L_0(B), ..., L_{n-1}(B), using H-way batched encoding.
    - rho^ell is implemented as `rotate(x, ell * H)` on interlaced batched slots.
    - The index r-ell of L(A) is taken modulo m.

    Returns packed lower diagonals L_0(C), ..., L_{m-1}(C) of C = AB,
    where C has shape (m, n).
    """

    m, n = _normalize_shape(A_shape)
    b_rows, b_cols = _normalize_shape(B_shape)
    assert n >= m, 'Corollary 3.8 requires A shape (m, n) with n >= m'
    assert n % m == 0, 'Corollary 3.8 requires n divisible by m'
    assert (b_rows, b_cols) == (n, n), 'Corollary 3.8 requires B shape (n, n)'

    lower_A = _unpack_batched_diagonal_vectors(packed_lower_A, A_shape, n_slot, H)
    lower_B = _unpack_batched_diagonal_vectors(packed_lower_B, B_shape, n_slot, H)
    dtype = np.result_type(*(diag.dtype for diag in lower_A + lower_B))

    lower_C: list[np.ndarray] = []
    for r in range(m):
        acc = np.zeros(H * n, dtype=dtype)
        for ell in range(n):
            a_idx = (r - ell) % m
            acc = add(acc, multiply(rotate(lower_A[a_idx], ell * H), lower_B[ell]))
        lower_C.append(acc)

    return _pack_batched_diagonal_vectors(lower_C, (m, n), n_slot, H)


def _extended_lower_diagonal_for_corollary_3_9(lower_A: list[np.ndarray], m: int, n: int, H: int, k: int) -> np.ndarray:
    """Return L'_k(A) from Corollary 3.9 for 0 <= k < n."""

    assert 0 <= k < n
    assert len(lower_A) == m
    if k < m:
        return lower_A[k]
    return rotate(lower_A[k % m], m * (k // m) * H)


def corollary_3_9_packed(
    packed_lower_A: list[np.ndarray],
    packed_lower_B: list[np.ndarray],
    A_shape: Shape,
    B_shape: Shape,
    n_slot: int,
    H: int,
) -> list[np.ndarray]:
    """Compute C = A @ B using THOR Corollary 3.9 on packed lower diagonals.

    公式（论文 Corollary 3.9）:
        L_r(AB) = sum_{ell in [m]} rho^ell(L'_{r-ell}(A)) ⊙ L_ell(B),  r in [n]

    扩展下对角线：
        L'_k(A) = L_k(A),                              if k < m
        L'_k(A) = rho^{m * floor(k / m)}(L_{k mod m}(A)), if m <= k < n

    变量解释：
    - A has shape (n, m), B has shape (m, n), with n >= m and n % m == 0.
    - packed_lower_A stores the base diagonals L_0(A), ..., L_{m-1}(A).
    - packed_lower_B stores L_0(B), ..., L_{m-1}(B).
    - rho^ell is implemented as `rotate(x, ell * H)` on interlaced batched slots.
    - The index r-ell of L' is taken modulo n before applying the L' definition.

    Returns packed lower diagonals L_0(C), ..., L_{n-1}(C) of C = AB,
    where C has shape (n, n).
    """

    n, m = _normalize_shape(A_shape)
    b_rows, b_cols = _normalize_shape(B_shape)
    assert n >= m, 'Corollary 3.9 requires A shape (n, m) with n >= m'
    assert n % m == 0, 'Corollary 3.9 requires n divisible by m'
    assert (b_rows, b_cols) == (m, n), 'Corollary 3.9 requires B shape (m, n)'

    lower_A = _unpack_batched_diagonal_vectors(packed_lower_A, A_shape, n_slot, H)
    lower_B = _unpack_batched_diagonal_vectors(packed_lower_B, B_shape, n_slot, H)
    dtype = np.result_type(*(diag.dtype for diag in lower_A + lower_B))

    lower_C: list[np.ndarray] = []
    for r in range(n):
        acc = np.zeros(H * n, dtype=dtype)
        for ell in range(m):
            l_prime_idx = (r - ell) % n
            l_prime = _extended_lower_diagonal_for_corollary_3_9(lower_A, m, n, H, l_prime_idx)
            acc = add(acc, multiply(rotate(l_prime, ell * H), lower_B[ell]))
        lower_C.append(acc)

    return _pack_batched_diagonal_vectors(lower_C, (n, n), n_slot, H)


# ---------------------------------------------------------------------------
# THOR Section 4.2.2 Algorithm 1 plaintext-ciphertext matrix multiplication
# ---------------------------------------------------------------------------


def generate_algorithm_1_plaintexts(A_plain: np.ndarray, n: int, n_slot: int, H: int) -> list[list[np.ndarray]]:
    """Generate plaintext vectors pt.A_{i,j,ell,r} for Algorithm 1.

    A_plain has shape (d, d), d = H * n, and is partitioned into n x n
    blocks A_{row_block, col_block}. For every block-diagonal offset i,
    Algorithm 1 uses the aligned block batch
        A^(i) = (A_{i+k, k})_{k in [H]}
    with block-row index i+k taken modulo H.

    Return layout:
        pt_A[i][((j * c + ell) * n_c + r)] == pt.A_{i,j,ell,r}

    where c = n_slot / (H*n), n_c = n/c, i in [H], j,r in [n_c],
    and ell in [c]. Each vector has length n_slot and matches the existing
    H-way interlaced multi-diagonal batched encoding.
    """

    A = _as_matrix(A_plain, 'A_plain')
    H = _validate_H(H)
    assert n > 0, 'n must be positive'
    d = H * n
    assert A.shape == (d, d), f'A_plain must have shape {(d, d)}, got {A.shape}'
    _, _, _, diag_len, c, segment_len = _packing_params((n, n), n_slot, H)
    assert diag_len == n
    assert segment_len == d
    n_c = n // c

    pt_A: list[list[np.ndarray]] = []
    for i in range(H):
        pt_i: list[np.ndarray] = []
        blocks = [A[((i + k) % H) * n : ((i + k) % H + 1) * n, k * n : (k + 1) * n] for k in range(H)]
        for j in range(n_c):
            for ell in range(c):
                for r in range(n_c):
                    segments: list[np.ndarray] = []
                    # Then use the equation of proposition 3.6: L_{out_diag_idx}(block0,block1)=\sum_{b_diag_idx in [n]} \rho^{out_diag_idx}(U_{b_diag_idx-out_diag_idx}(block0)) \odot
                    # L_{b_diag_idx}(block1)
                    for tau in range(c):
                        b_diag_idx = c * j + (
                            (tau + ell) % c
                        )  # global input diag idx for the tau-th segment of the j-th input ciphertext ct.B_j
                        # each segment has H interleaved diagonals, each segment has total length H * n = d, tau-th segmetn corresponds tau-th diagonal within the ciphertext
                        out_diag_idx = (
                            c * r + tau
                        )  # global output diag idx for the tau-th segment of the r-th output ciphertext ct.C_r
                        interlaced = _interlace_diagonal_batch(
                            [upper_diagonal(block, (b_diag_idx - out_diag_idx) % n) for block in blocks],
                            n,
                            H,
                        )
                        segments.append(rotate(interlaced, out_diag_idx * H))
                    plaintext = np.concatenate(segments)
                    assert plaintext.shape == (n_slot,)
                    pt_i.append(plaintext)
        assert len(pt_i) == n_c * c * n_c
        pt_A.append(pt_i)
    return pt_A


def algorithm_1_plaintext_ciphertext_matmul(
    A_plain: np.ndarray,
    packed_lower_B: list[np.ndarray],
    n: int,
    n_slot: int,
    H: int,
) -> list[np.ndarray]:
    """Implement THOR Section 4.2.2 Algorithm 1.

    This computes C = A * B where A is plaintext with shape (d, d), B is
    encrypted/packed as H blocks B_k in R^{n x n}, d = H*n, and the output is
    the packed lower-diagonal encoding of H output blocks C_k in R^{n x n}.

    The implementation follows Algorithm 1 directly:
    - lines 2-5: rotate input ciphertexts ct.B_j by d*ell;
    - lines 6-10: compute block-diagonal partial products via Proposition 3.6
      using the generated plaintext vectors pt.A_{i,j,ell,r};
    - lines 11-16: internally rotate partial products into output-block order
      and aggregate them into ct.C_r.
    """

    A = _as_matrix(A_plain, 'A_plain')
    H = _validate_H(H)
    assert n > 0, 'n must be positive'
    d = H * n
    assert A.shape == (d, d), f'A_plain must have shape {(d, d)}, got {A.shape}'
    _, _, _, _, c, segment_len = _packing_params((n, n), n_slot, H)
    assert segment_len == d
    n_c = n // c
    assert isinstance(packed_lower_B, list), 'packed_lower_B must be list[np.ndarray]'
    assert len(packed_lower_B) == n_c, f'expected {n_c} input ciphertexts, got {len(packed_lower_B)}'
    for idx, vector in enumerate(packed_lower_B):
        slots = _as_vector(vector, f'packed_lower_B[{idx}]')
        assert slots.shape == (n_slot,), f'packed_lower_B[{idx}] must have length {n_slot}'

    pt_A = generate_algorithm_1_plaintexts(A, n, n_slot, H)

    ct_Br: list[list[np.ndarray]] = []
    for j in range(n_c):
        rotations: list[np.ndarray] = []
        for ell in range(c):
            rotations.append(packed_lower_B[j].copy() if ell == 0 else rotate(packed_lower_B[j], d * ell))
        ct_Br.append(rotations)

    ct_ir: list[list[np.ndarray]] = []
    for i in range(H):
        row: list[np.ndarray] = []
        for r in range(n_c):
            acc = np.zeros(n_slot, dtype=np.result_type(A.dtype, *(vector.dtype for vector in packed_lower_B)))
            for j in range(n_c):
                for ell in range(c):
                    pt_idx = (j * c + ell) * n_c + r
                    acc = add(acc, multiply(ct_Br[j][ell], pt_A[i][pt_idx]))
            row.append(acc)
        ct_ir.append(row)

    ct_C: list[np.ndarray] = []
    for r in range(n_c):
        acc = ct_ir[0][r].copy()
        for i in range(1, H):
            mask_wrap = np.zeros(n_slot, dtype=ct_ir[i][r].dtype)
            for segment_start in range(0, n_slot, segment_len):
                for t in range(n):
                    group_start = segment_start + t * H
                    mask_wrap[group_start + H - i : group_start + H] = 1
            ct_i_r_R = multiply(ct_ir[i][r], mask_wrap)
            ct_i_r_L = subtract(ct_ir[i][r], ct_i_r_R)
            ct_i_r_prime = add(rotate(ct_i_r_R, H - i), rotate(ct_i_r_L, -i))
            acc = add(acc, ct_i_r_prime)
        ct_C.append(acc)

    return ct_C


# ---------------------------------------------------------------------------
# Broad verification helpers
# ---------------------------------------------------------------------------


def verify_pack_unpack() -> None:
    """Verify upper/lower pack+unpack round trips for H=1 and H=2."""

    H1 = 1
    A1 = np.arange(8, dtype=np.float64).reshape(2, 4)
    n_slot1 = 8
    assert np.allclose(unpack_upper_diagonals(pack_upper_diagonals([A1], n_slot1, H1), A1.shape, n_slot1, H1)[0], A1)
    assert np.allclose(unpack_lower_diagonals(pack_lower_diagonals([A1], n_slot1, H1), A1.shape, n_slot1, H1)[0], A1)

    H2 = 2
    matrices = [
        np.arange(8, dtype=np.float64).reshape(2, 4),
        np.arange(8, dtype=np.float64).reshape(2, 4) + 100,
    ]
    n_slot2 = 16
    upper_roundtrip = unpack_upper_diagonals(
        pack_upper_diagonals(matrices, n_slot2, H2), matrices[0].shape, n_slot2, H2
    )
    lower_roundtrip = unpack_lower_diagonals(
        pack_lower_diagonals(matrices, n_slot2, H2), matrices[0].shape, n_slot2, H2
    )
    for expected, actual_upper, actual_lower in zip(matrices, upper_roundtrip, lower_roundtrip):
        assert np.allclose(actual_upper, expected)
        assert np.allclose(actual_lower, expected)


def verify_propositions_3_6_3_7() -> None:
    """Verify packed Proposition 3.6 and 3.7 interfaces using H=2 batched inputs."""

    H = 2
    n_slot = 16

    A36 = [
        np.arange(8, dtype=np.float64).reshape(4, 2) + 1,
        np.arange(8, dtype=np.float64).reshape(4, 2) + 11,
    ]
    B36 = [
        np.arange(8, dtype=np.float64).reshape(2, 4) + 1,
        np.arange(8, dtype=np.float64).reshape(2, 4) + 21,
    ]
    C36 = unpack_lower_diagonals(
        prop_3_6_packed(
            pack_upper_diagonals(A36, n_slot, H),
            pack_lower_diagonals(B36, n_slot, H),
            A36[0].shape,
            B36[0].shape,
            n_slot,
            H,
        ),
        (A36[0].shape[0], B36[0].shape[1]),
        n_slot,
        H,
    )
    for A, B, C in zip(A36, B36, C36):
        assert np.allclose(C, A @ B)

    A37 = [
        np.arange(8, dtype=np.float64).reshape(2, 4) + 1,
        np.arange(8, dtype=np.float64).reshape(2, 4) + 31,
    ]
    B37 = [
        np.arange(16, dtype=np.float64).reshape(4, 4) + 1,
        np.arange(16, dtype=np.float64).reshape(4, 4) + 41,
    ]
    C37 = unpack_lower_diagonals(
        prop_3_7_packed(
            pack_upper_diagonals(A37, n_slot, H),
            pack_lower_diagonals(B37, n_slot, H),
            A37[0].shape,
            B37[0].shape,
            n_slot,
            H,
        ),
        (A37[0].shape[0], B37[0].shape[1]),
        n_slot,
        H,
    )
    for A, B, C in zip(A37, B37, C37):
        assert np.allclose(C, A @ B)


def verify_corollaries_3_8_3_9() -> None:
    """Verify Corollary 3.8 and 3.9 together using H=2 batched inputs."""

    H = 2
    n_slot = 16

    A38 = [
        np.arange(8, dtype=np.float64).reshape(2, 4) + 1,
        np.arange(8, dtype=np.float64).reshape(2, 4) + 11,
    ]
    B38 = [
        np.arange(16, dtype=np.float64).reshape(4, 4) + 1,
        np.arange(16, dtype=np.float64).reshape(4, 4) + 21,
    ]
    C38 = unpack_lower_diagonals(
        corollary_3_8_packed(
            pack_lower_diagonals(A38, n_slot, H),
            pack_lower_diagonals(B38, n_slot, H),
            A38[0].shape,
            B38[0].shape,
            n_slot,
            H,
        ),
        (A38[0].shape[0], B38[0].shape[1]),
        n_slot,
        H,
    )
    for A, B, C in zip(A38, B38, C38):
        assert np.allclose(C, A @ B)

    A39 = [
        np.arange(8, dtype=np.float64).reshape(4, 2) + 1,
        np.arange(8, dtype=np.float64).reshape(4, 2) + 31,
    ]
    B39 = [
        np.arange(8, dtype=np.float64).reshape(2, 4) + 1,
        np.arange(8, dtype=np.float64).reshape(2, 4) + 41,
    ]
    C39 = unpack_lower_diagonals(
        corollary_3_9_packed(
            pack_lower_diagonals(A39, n_slot, H),
            pack_lower_diagonals(B39, n_slot, H),
            A39[0].shape,
            B39[0].shape,
            n_slot,
            H,
        ),
        (A39[0].shape[0], B39[0].shape[1]),
        n_slot,
        H,
    )
    for A, B, C in zip(A39, B39, C39):
        assert np.allclose(C, A @ B)


def verify_algorithm_1() -> None:
    """Verify Section 4.2.2 Algorithm 1 against NumPy plaintext matmul."""

    H = 2
    n = 4
    d = H * n
    n_slot = 16
    A = (np.arange(d * d, dtype=np.float64).reshape(d, d) % 17) + 1
    B = (np.arange(d * n, dtype=np.float64).reshape(d, n) % 13) + 1
    B_blocks = [B[k * n : (k + 1) * n, :] for k in range(H)]

    packed_C = algorithm_1_plaintext_ciphertext_matmul(
        A,
        pack_lower_diagonals(B_blocks, n_slot, H),
        n,
        n_slot,
        H,
    )
    C_blocks = unpack_lower_diagonals(packed_C, (n, n), n_slot, H)
    C = np.vstack(C_blocks)
    assert np.allclose(C, A @ B)


def verify_all() -> None:
    verify_pack_unpack()
    verify_propositions_3_6_3_7()
    verify_corollaries_3_8_3_9()
    verify_algorithm_1()
    print('All THOR CKKS primitive verifications passed.')


if __name__ == '__main__':
    verify_all()
