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
using namespace cxx_sdk_v2;

FeatureMatEncrypted::FeatureMatEncrypted(CkksContext* context_in, int ct_level) {
    dim = 2;
    context = context_in;
    level = ct_level;
}

void FeatureMatEncrypted::block_col_major_pack(const Array<double, 2>& matrix,
                                               uint32_t d,
                                               bool is_symmetric,
                                               double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t n_cols = matrix.get_shape()[1];
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
                                                   bool is_symmetric,
                                                   double scale_in) {
    uint32_t m = matrix.get_shape()[0];
    uint32_t total_cols = matrix.get_shape()[1];
    uint32_t cols_per_head = total_cols / n_heads;
    uint32_t n_h_padded = next_power_of_2(n_heads);
    int n_slot = context->get_parameter().get_n() / 2;
    const int N_THREAD = 4;

    // Determine chunk sizing and n_blocks_per_chunk(S)
    uint32_t S, chunk_size, n_cts_per_block_idx;
    if ((uint32_t)n_slot >= n_h_padded * d * d) {
        S = n_h_padded;
        chunk_size = n_h_padded * d * d;
        n_cts_per_block_idx = 1;
    } else {
        S = n_slot / (d * d);
        chunk_size = n_slot;
        n_cts_per_block_idx = n_h_padded / S;
    }
    uint32_t num_chunks = n_slot / chunk_size;

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(cols_per_head, d);
    uint32_t total_vecs = num_block_rows * num_block_cols * n_cts_per_block_idx;

    vector<vector<double>> block_vecs(total_vecs);

    // Column-major block order: for bj, for bi, for g (group number of cts for the same block idx)
    for (uint32_t bj = 0; bj < num_block_cols; bj++) {
        for (uint32_t bi = 0; bi < num_block_rows; bi++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx; g++) {
                uint32_t vec_idx = (bi + num_block_rows * bj) * n_cts_per_block_idx + g;
                vector<double> vec(n_slot, 0.0);

                for (uint32_t h_local = 0; h_local < S; h_local++) {
                    uint32_t h = g * S + h_local;  // global head index
                    for (uint32_t col = 0; col < d; col++) {
                        for (uint32_t row = 0; row < d; row++) {
                            uint32_t r = bi * d + row;
                            uint32_t c = bj * d + col;
                            double val = 0.0;
                            if (h < n_heads && r < m && c < cols_per_head) {
                                val = matrix.get(r, h * cols_per_head + c);
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
        n_cts_per_block_idx = n_h_padded / S;
    }

    uint32_t num_block_rows = div_ceil(m, d);
    uint32_t num_block_cols = div_ceil(n_per_head, d);
    uint32_t total_vecs = num_block_rows * num_block_cols * n_cts_per_block_idx;
    uint32_t total_cols = n_heads * n_per_head;

    Array<double, 2> result({(uint64_t)m, (uint64_t)total_cols});

    parallel_for(total_vecs, N_THREAD, *context, [&](CkksContext& ctx_copy, int vec_idx) {
        // Recover bi, bj, g from vec_idx
        uint32_t block_idx = vec_idx / n_cts_per_block_idx;
        uint32_t g = vec_idx % n_cts_per_block_idx;
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
                        uint32_t slot = (row + d * col) * S + h_local;
                        result.set(r, h * n_per_head + c, x_mg[slot]);
                    }
                }
            }
        }
    });
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

FeatureMatEncrypted FeatureMatEncrypted::drop_level(int n_level_to_drop) const {
    int new_level = level - n_level_to_drop;
    FeatureMatEncrypted result(context, new_level);
    result.shape = shape;
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
