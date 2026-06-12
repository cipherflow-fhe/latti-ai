/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "feature_mat.h"

using namespace std;
using namespace lattisense;

FeatureMatEncrypted::FeatureMatEncrypted(CkksContext* context_in, int ct_level) {
    dim = 2;
    context = context_in;
    level = ct_level;
}

FeatureMatEncrypted FeatureMatEncrypted::refresh_ciphertext() const {
    CkksBtpContext* ctx = dynamic_cast<CkksBtpContext*>(context);
    if (ctx == nullptr) {
        throw std::runtime_error("refresh_ciphertext() requires CkksBtpContext");
    }
    int new_level = 9;
    FeatureMatEncrypted result(ctx, new_level);
    result.data.resize(data.size());
    parallel_for(data.size(), th_nums, *ctx, [&](CkksBtpContext& ctx_copy, int ct_idx) {
        result.data[ct_idx] = ctx_copy.bootstrap(data[ct_idx]);
        assert(new_level == result.data[ct_idx].get_level());
    });
    result.shape = shape;
    result.head_shape = head_shape;
    result.matmul_block_size = matmul_block_size;
    result.n_channel = n_channel;
    result.n_channel_per_ct = n_channel_per_ct;
    return result;
}

void FeatureMatEncrypted::block_col_major_pack(const Array<double, 2>& matrix,
                                               uint32_t d,
                                               bool is_symmetric,
                                               double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t n_cols = matrix.get_shape()[1];
    shape = {m, n_cols};
    head_shape = {m, n_cols};
    matmul_block_size = d;
    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n_cols, d);
    int n_slot = context->get_parameter().get_n() / 2;
    uint32_t chunk_size = d * d;
    const int N_THREAD = 4;

    uint32_t total_blocks = num_block_rows * num_block_cols;
    vector<vector<double>> block_vecs(total_blocks);

    // Column-major block order: for bj in [0, num_block_cols), for bi in [0, num_block_rows)
    for (uint32_t bj = 0; bj < num_block_cols; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows; bi++) {
            uint32_t block_idx = bi + num_block_rows * bj;
            vector<double> vec(n_slot, 0.0);
            uint32_t num_chunks = n_slot / chunk_size;
            for (uint32_t c = 0; c < num_chunks; c++) {
                for (uint32_t col = 0; col < d; col++) {
                    for (uint32_t row = 0; row < d; row++) {
                        uint32_t r = bi * d + row;
                        uint32_t c_col = bj * d + col;
                        if (r < m && c_col < n_cols) {
                            vec[c * chunk_size + row + d * col] = matrix.get(r, c_col);
                        }
                    }
                }
            }
            block_vecs[block_idx] = move(vec);
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_blocks);
    } else {
        data.resize(total_blocks);
    }

    parallel_for(total_blocks, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(block_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2> FeatureMatEncrypted::block_col_major_unpack(uint32_t m, uint32_t n, uint32_t d) const {
    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n, d);
    const int N_THREAD = 4;
    uint32_t total_blocks = num_block_rows * num_block_cols;

    Array<double, 2> result({(uint64_t)m, (uint64_t)n});

    parallel_for(total_blocks, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        // Recover bi, bj from column-major block index
        uint32_t bi = idx % num_block_rows;
        uint32_t bj = idx / num_block_rows;

        CkksPlaintext x_pt = ctx_copy.decrypt(data[idx]);
        Array1D x_mg = ctx_copy.decode(x_pt);
        // Extract first d*d elements (column-major within block)
        for (uint32_t col = 0; col < d; col++) {
            for (uint32_t row = 0; row < d; row++) {
                uint32_t r = bi * d + row;
                uint32_t c_col = bj * d + col;
                if (r < m && c_col < n) {
                    result.set(r, c_col, x_mg[row + d * col]);
                }
            }
        }
    });
    return result;
}

static uint32_t next_power_of_2(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

void FeatureMatEncrypted::par_block_col_major_pack(const Array<double, 2>& matrix,
                                                   uint32_t d,
                                                   uint32_t n_heads,
                                                   uint32_t head_dim,
                                                   bool is_symmetric,
                                                   double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t total_cols = matrix.get_shape()[1];
    shape = {m, total_cols};
    head_shape = {m, head_dim};
    matmul_block_size = d;
    uint32_t cols_per_head = head_dim;
    uint32_t n = cols_per_head * n_heads;  // columns per megablock
    uint32_t K_col = div_ceil(total_cols, n);
    uint32_t n_h_padded = next_power_of_2(n_heads);
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    // Determine chunk sizing and n_heads_per_chunk(S)
    uint32_t S, chunk_size, n_cts_per_block_idx;
    if ((uint32_t)n_slot >= n_h_padded * d * d) {
        S = n_h_padded;
        chunk_size = n_h_padded * d * d;
        n_cts_per_block_idx = 1;
    } else {
        S = n_slot / (d * d);
        chunk_size = n_slot;
        if (S == 1) {
            n_h_padded = n_heads;
        }
        n_cts_per_block_idx = n_h_padded / S;
    }
    uint32_t num_chunks = n_slot / chunk_size;

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(cols_per_head, d);
    uint32_t cts_per_mb = num_block_rows * num_block_cols * n_cts_per_block_idx;
    uint32_t total_vecs = K_col * cts_per_mb;

    vector<vector<double>> block_vecs(total_vecs);

    for (uint32_t col_mb = 0; col_mb < K_col; col_mb++) {
        // Column-major block order: for bj, for bi, for g
        for (uint32_t bj = 0; bj < num_block_cols; bj++) {
            for (uint32_t bi = 0; bi < num_block_rows; bi++) {
                for (uint32_t g = 0; g < n_cts_per_block_idx; g++) {
                    uint32_t local_idx = (bi + num_block_rows * bj) * n_cts_per_block_idx + g;
                    uint32_t vec_idx = col_mb * cts_per_mb + local_idx;
                    vector<double> vec(n_slot, 0.0);

                    for (uint32_t h_local = 0; h_local < S; h_local++) {
                        uint32_t h = g * S + h_local;  // global head index
                        for (uint32_t col = 0; col < d; col++) {
                            for (uint32_t row = 0; row < d; row++) {
                                uint32_t r = bi * d + row;
                                uint32_t c = bj * d + col;
                                double val = 0.0;
                                if (h < n_heads && r < m && c < cols_per_head) {
                                    uint32_t global_col = col_mb * n + h * cols_per_head + c;
                                    if (global_col < total_cols)
                                        val = matrix.get(r, global_col);
                                }
                                uint32_t base_slot = (row + d * col) * S + h_local;
                                for (uint32_t ci = 0; ci < num_chunks; ci++) {
                                    vec[ci * chunk_size + base_slot] = val;
                                }
                            }
                        }
                    }
                    block_vecs[vec_idx] = move(vec);
                }
            }
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_vecs);
    } else {
        data.resize(total_vecs);
    }

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(block_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2>
FeatureMatEncrypted::par_block_col_major_unpack(uint32_t m, uint32_t n_per_head, uint32_t d, uint32_t n_heads) const {
    uint32_t total_cols = shape[1];
    uint32_t n_h_padded = next_power_of_2(n_heads);
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    uint32_t S, chunk_size, n_cts_per_block_idx;
    if ((uint32_t)n_slot >= n_h_padded * d * d) {
        S = n_h_padded;
        chunk_size = n_h_padded * d * d;
        n_cts_per_block_idx = 1;
    } else {
        S = n_slot / (d * d);
        chunk_size = n_slot;
        if (S == 1) {
            n_h_padded = n_heads;
        }
        n_cts_per_block_idx = n_h_padded / S;
    }

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n_per_head, d);
    uint32_t cts_per_mb = num_block_rows * num_block_cols * n_cts_per_block_idx;

    // Infer K_col (number of output megablocks) from ciphertext count
    uint32_t K_col = data.size() / cts_per_mb;
    assert(K_col * cts_per_mb == data.size());

    uint32_t n = n_heads * n_per_head;  // columns per megablock
    Array<double, 2> result({(uint64_t)m, (uint64_t)total_cols});

    for (uint32_t col_mb = 0; col_mb < K_col; col_mb++) {
        uint32_t ct_offset = col_mb * cts_per_mb;

        parallel_for(cts_per_mb, N_THREAD, *context, [&](CkksContext& ctx_copy, int local_idx) {
            uint32_t vec_idx = ct_offset + local_idx;
            // Recover bi, bj, g from local_idx
            uint32_t block_idx = local_idx / n_cts_per_block_idx;
            uint32_t g = local_idx % n_cts_per_block_idx;
            uint32_t bi = block_idx % num_block_rows;
            uint32_t bj = block_idx / num_block_rows;

            CkksPlaintext x_pt = ctx_copy.decrypt(data[vec_idx]);
            Array1D x_mg = ctx_copy.decode(x_pt);

            for (uint32_t h_local = 0; h_local < S; h_local++) {
                uint32_t h = g * S + h_local;
                if (h >= n_heads)
                    continue;
                for (uint32_t col = 0; col < d; col++) {
                    for (uint32_t row = 0; row < d; row++) {
                        uint32_t r = bi * d + row;
                        uint32_t c = bj * d + col;
                        if (r < m && c < n_per_head) {
                            uint32_t global_col = col_mb * n + h * n_per_head + c;
                            if (global_col < total_cols) {
                                uint32_t slot = (row + d * col) * S + h_local;
                                result.set(r, global_col, x_mg[slot]);
                            }
                        }
                    }
                }
            }
        });
    }
    return result;
}

void FeatureMatEncrypted::par_lower_diagonal_pack(const Array<double, 2>& matrix,
                                                  uint32_t n_heads,
                                                  const Duo& head_shape_in,
                                                  bool is_symmetric,
                                                  double scale_in) {
    uint32_t total_rows = matrix.get_shape()[0];
    uint32_t n_prepad = head_shape_in[1];
    uint32_t H_prepad = n_heads;
    uint32_t m_prepad = head_shape_in[0];
    uint32_t m = next_power_of_2(m_prepad);
    assert(H_prepad > 0);
    assert(m_prepad > 0);
    assert(matrix.get_shape()[1] == n_prepad);

    shape = {total_rows, n_prepad};
    head_shape = head_shape_in;
    matmul_block_size = m;

    uint32_t rows_per_mb = H_prepad * m_prepad;
    uint32_t n_mb = div_ceil(total_rows, rows_per_mb);
    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    assert(n % m == 0);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_mb = m / c;
    uint32_t total_vecs = n_mb * cts_per_mb;
    const int N_THREAD = 4;

    vector<vector<double>> packed_vecs(total_vecs);

    for (uint32_t mb = 0; mb < n_mb; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb; ct_local++) {
            vector<double> vec(n_slot, 0.0);
            for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
                uint32_t diag_idx = ct_local * c + local_diag;
                uint32_t segment_base = local_diag * H * n;
                for (uint32_t t = 0; t < n; t++) {
                    for (uint32_t h = 0; h < H; h++) {
                        double val = 0.0;
                        uint32_t local_row = (diag_idx + t) % m;
                        uint32_t transposed_col = t;
                        if (h < H_prepad && local_row < m_prepad && transposed_col < n_prepad) {
                            uint32_t global_row = mb * rows_per_mb + h * m_prepad + local_row;
                            if (global_row < total_rows) {
                                val = matrix.get(global_row, transposed_col);
                            }
                        }
                        vec[segment_base + t * H + h] = val;
                    }
                }
            }
            packed_vecs[mb * cts_per_mb + ct_local] = move(vec);
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_vecs);
    } else {
        data.resize(total_vecs);
    }

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(packed_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2> FeatureMatEncrypted::par_lower_diagonal_unpack(uint32_t n_heads, const Duo& head_shape_in) const {
    uint32_t n_prepad = head_shape_in[1];
    assert(shape[1] == n_prepad);
    uint32_t total_rows = shape[0];
    uint32_t H_prepad = n_heads;
    uint32_t m_prepad = head_shape_in[0];
    uint32_t m = next_power_of_2(m_prepad);
    assert(H_prepad > 0);
    assert(m_prepad > 0);

    uint32_t rows_per_mb = H_prepad * m_prepad;
    uint32_t n_mb = div_ceil(total_rows, rows_per_mb);
    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    assert(n % m == 0);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_mb = m / c;
    assert(data.size() == n_mb * cts_per_mb);
    const int N_THREAD = 4;

    Array<double, 2> result({(uint64_t)total_rows, (uint64_t)n_prepad});

    for (uint32_t mb = 0; mb < n_mb; mb++) {
        uint32_t ct_offset = mb * cts_per_mb;
        parallel_for(cts_per_mb, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_local) {
            CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_offset + ct_local]);
            Array1D slots = ctx_copy.decode(x_pt);
            for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
                uint32_t diag_idx = ct_local * c + local_diag;
                uint32_t segment_base = local_diag * H * n;
                for (uint32_t t = 0; t < n; t++) {
                    for (uint32_t h = 0; h < H_prepad; h++) {
                        uint32_t local_row = (diag_idx + t) % m;
                        uint32_t transposed_col = t;
                        if (transposed_col < n_prepad && local_row < m_prepad) {
                            uint32_t global_row = mb * rows_per_mb + h * m_prepad + local_row;
                            if (global_row < total_rows) {
                                result.set(global_row, transposed_col, slots[segment_base + t * H + h]);
                            }
                        }
                    }
                }
            }
        });
    }

    return result;
}

void FeatureMatEncrypted::par_lower_diagonal_transpose_pack(const Array<double, 2>& matrix,
                                                            uint32_t n_heads,
                                                            uint32_t head_dim,
                                                            bool is_symmetric,
                                                            double scale_in) {
    uint32_t n_prepad = matrix.get_shape()[0];
    uint32_t total_cols = matrix.get_shape()[1];
    uint32_t H_prepad = n_heads;
    uint32_t m = head_dim;
    assert(H_prepad > 0);
    assert(m > 0 && (m & (m - 1)) == 0);
    assert(total_cols == H_prepad * m);

    shape = {n_prepad, total_cols};
    head_shape = {n_prepad, m};
    matmul_block_size = m;

    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_head_block = m / c;
    const int N_THREAD = 4;

    vector<vector<double>> packed_vecs(cts_per_head_block);

    for (uint32_t ct_local = 0; ct_local < cts_per_head_block; ct_local++) {
        vector<double> vec(n_slot, 0.0);
        for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
            uint32_t diag_idx = ct_local * c + local_diag;
            uint32_t segment_base = local_diag * H * n;
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H; h++) {
                    double val = 0.0;
                    uint32_t row = (diag_idx + t) % n;
                    uint32_t col = t % m;
                    if (h < H_prepad && row < n_prepad) {
                        val = matrix.get(row, h * m + col);
                    }
                    vec[segment_base + t * H + h] = val;
                }
            }
        }
        packed_vecs[ct_local] = move(vec);
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(cts_per_head_block);
    } else {
        data.resize(cts_per_head_block);
    }

    parallel_for(cts_per_head_block, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(packed_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2>
FeatureMatEncrypted::par_lower_diagonal_transpose_unpack(uint32_t n_prepad, uint32_t n_heads, uint32_t head_dim) const {
    uint32_t H_prepad = n_heads;
    uint32_t m = head_dim;
    assert(H_prepad > 0);
    assert(m > 0 && (m & (m - 1)) == 0);
    assert(shape[0] == n_prepad);
    assert(shape[1] == H_prepad * m);
    assert(head_shape[0] == n_prepad);
    assert(head_shape[1] == m);

    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_head_block = m / c;
    assert(data.size() == cts_per_head_block);
    const int N_THREAD = 4;

    Array<double, 2> result({(uint64_t)n_prepad, (uint64_t)(H_prepad * m)});

    parallel_for(cts_per_head_block, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_local) {
        CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_local]);
        Array1D slots = ctx_copy.decode(x_pt);
        for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
            uint32_t diag_idx = ct_local * c + local_diag;
            uint32_t segment_base = local_diag * H * n;
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H_prepad; h++) {
                    uint32_t row = (diag_idx + t) % n;
                    uint32_t col = t % m;
                    if (row < n_prepad) {
                        result.set(row, h * m + col, slots[segment_base + t * H + h]);
                    }
                }
            }
        }
    });

    return result;
}

void FeatureMatEncrypted::par_upper_diagonal_pack(const Array<double, 2>& matrix,
                                                  uint32_t n_heads,
                                                  uint32_t head_dim,
                                                  bool is_symmetric,
                                                  double scale_in) {
    uint32_t total_cols = matrix.get_shape()[0];
    uint32_t n_prepad = matrix.get_shape()[1];
    uint32_t H_prepad = n_heads;
    uint32_t m = head_dim;
    assert(H_prepad > 0);
    assert(m > 0 && (m & (m - 1)) == 0);

    shape = {total_cols, n_prepad};
    head_shape = {head_dim, n_prepad};
    matmul_block_size = head_dim;

    uint32_t d_prepad = H_prepad * m;
    uint32_t n_mb = div_ceil(total_cols, d_prepad);
    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_mb = m / c;
    uint32_t total_vecs = n_mb * cts_per_mb;
    const int N_THREAD = 4;

    vector<vector<double>> packed_vecs(total_vecs);

    for (uint32_t mb = 0; mb < n_mb; mb++) {
        for (uint32_t ct_local = 0; ct_local < cts_per_mb; ct_local++) {
            vector<double> vec(n_slot, 0.0);
            for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
                uint32_t diag_idx = ct_local * c + local_diag;
                uint32_t segment_base = local_diag * H * n;
                for (uint32_t t = 0; t < n; t++) {
                    for (uint32_t h = 0; h < H; h++) {
                        double val = 0.0;
                        uint32_t transposed_row = h * m + (t % m);
                        uint32_t transposed_col = (diag_idx + t) % n;
                        if (h < H_prepad && transposed_col < n_prepad) {
                            uint32_t global_row = mb * d_prepad + transposed_row;
                            if (transposed_row < d_prepad && global_row < total_cols) {
                                val = matrix.get(global_row, transposed_col);
                            }
                        }
                        vec[segment_base + t * H + h] = val;
                    }
                }
            }
            packed_vecs[mb * cts_per_mb + ct_local] = move(vec);
        }
    }

    data.clear();
    data_compress.clear();
    if (is_symmetric) {
        data_compress.resize(total_vecs);
    } else {
        data.resize(total_vecs);
    }

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int idx) {
        auto enc = ctx_copy.encode(packed_vecs[idx], level, scale_in);
        if (is_symmetric) {
            data_compress[idx] = ctx_copy.encrypt_symmetric_compressed(enc);
        } else {
            data[idx] = ctx_copy.encrypt_symmetric(enc);
        }
    });
}

Array<double, 2>
FeatureMatEncrypted::par_upper_diagonal_unpack(uint32_t n_prepad, uint32_t n_heads, uint32_t head_dim) const {
    assert(shape[1] == n_prepad);
    uint32_t total_cols = shape[0];
    uint32_t H_prepad = n_heads;
    uint32_t m = head_dim;
    assert(H_prepad > 0);
    assert(m > 0 && (m & (m - 1)) == 0);

    uint32_t d_prepad = H_prepad * m;
    uint32_t n_mb = div_ceil(total_cols, d_prepad);
    uint32_t H = next_power_of_2(H_prepad);
    uint32_t n = next_power_of_2(n_prepad);
    assert(n >= m);
    uint32_t n_slot = context->get_parameter().get_n() / 2;
    assert(n_slot % (H * n) == 0);
    uint32_t c = n_slot / (H * n);
    assert(m % c == 0);
    uint32_t cts_per_mb = m / c;
    assert(data.size() == n_mb * cts_per_mb);
    const int N_THREAD = 4;

    Array<double, 2> result({(uint64_t)total_cols, (uint64_t)n_prepad});

    for (uint32_t mb = 0; mb < n_mb; mb++) {
        uint32_t ct_offset = mb * cts_per_mb;
        parallel_for(cts_per_mb, N_THREAD, *context, [&](CkksContext& ctx_copy, int ct_local) {
            CkksPlaintext x_pt = ctx_copy.decrypt(data[ct_offset + ct_local]);
            Array1D slots = ctx_copy.decode(x_pt);
            for (uint32_t local_diag = 0; local_diag < c; local_diag++) {
                uint32_t diag_idx = ct_local * c + local_diag;
                uint32_t segment_base = local_diag * H * n;
                for (uint32_t t = 0; t < n; t++) {
                    for (uint32_t h = 0; h < H_prepad; h++) {
                        uint32_t transposed_row = h * m + (t % m);
                        uint32_t transposed_col = (diag_idx + t) % n;
                        uint32_t padded_row = transposed_col;
                        uint32_t padded_col = transposed_row;
                        if (padded_row < n_prepad && padded_col < d_prepad) {
                            uint32_t global_col = mb * d_prepad + padded_col;
                            if (global_col < total_cols) {
                                result.set(global_col, padded_row, slots[segment_base + t * H + h]);
                            }
                        }
                    }
                }
            }
        });
    }

    return result;
}

void FeatureMatEncrypted::decompress() {
    assert(data.size() == 0 && data_compress.size() > 0);
    size_t n_ct = data_compress.size();
    for (size_t i = 0; i < n_ct; i++) {
        data.push_back(context->compressed_ciphertext_to_ciphertext(data_compress[i]));
    }
    data_compress.clear();
}

Bytes FeatureMatEncrypted::serialize() const {
    stringstream ss;
    ss_write(ss, dim);
    ss_write(ss, n_channel);
    ss_write(ss, n_channel_per_ct);
    ss_write(ss, level);
    for (int i = 0; i < 2; i++) {
        ss_write(ss, shape[i]);
    }
    ss_write(ss, matmul_block_size);
    uint32_t n_ct = data.size();
    ss_write(ss, n_ct);
    for (const CkksCiphertext& ct : data) {
        Bytes ct_data = ct.serialize(context->get_parameter());
        ss_write_vector(ss, ct_data);
    }
    uint32_t n_cct = data_compress.size();
    ss_write(ss, n_cct);
    for (const CkksCompressedCiphertext& cct : data_compress) {
        Bytes cct_data = cct.serialize(context->get_parameter());
        ss_write_vector(ss, cct_data);
    }
    return ss_to_bytes(ss);
}

void FeatureMatEncrypted::deserialize(const Bytes& bytes) {
    stringstream ss;
    bytes_to_ss(bytes, ss);
    ss_read(ss, &dim);
    ss_read(ss, &n_channel);
    ss_read(ss, &n_channel_per_ct);
    ss_read(ss, &level);
    for (int i = 0; i < 2; i++) {
        ss_read(ss, &shape[i]);
    }
    ss_read(ss, &matmul_block_size);
    uint32_t n_ct;
    ss_read(ss, &n_ct);
    for (uint32_t i = 0; i < n_ct; i++) {
        Bytes ct_data;
        ss_read_vector(ss, &ct_data);
        data.push_back(CkksCiphertext::deserialize(ct_data));
    }
    uint32_t n_cct;
    ss_read(ss, &n_cct);
    for (uint32_t i = 0; i < n_cct; i++) {
        Bytes cct_data;
        ss_read_vector(ss, &cct_data);
        data_compress.push_back(CkksCompressedCiphertext::deserialize(cct_data));
    }
}

FeatureMatEncrypted FeatureMatEncrypted::drop_level(int n_level_to_drop) const {
    int new_level = level - n_level_to_drop;
    FeatureMatEncrypted result(context, new_level);
    result.shape = shape;
    result.head_shape = head_shape;
    result.matmul_block_size = matmul_block_size;
    result.data.resize(data.size());
    parallel_for(data.size(), th_nums, *context, [&](CkksContext& ctx_copy, int ct_idx) {
        auto ct_tmp = data[ct_idx].copy();
        for (int j = 0; j < n_level_to_drop; j++) {
            ct_tmp = ctx_copy.drop_level(ct_tmp);
        }
        result.data[ct_idx] = move(ct_tmp);
        assert(new_level == result.data[ct_idx].get_level());
    });
    return result;
}
