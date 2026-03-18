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

#include "par_block_col_major_transpose.h"
#include <cassert>
#include <cmath>

using namespace std;

static uint32_t next_pow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

ParBlockColMajorTranspose::ParBlockColMajorTranspose(const CkksParameter& param_in,
                                                     const Duo& shape,
                                                     uint32_t block_size,
                                                     uint32_t n_heads,
                                                     uint32_t level)
    : param_(param_in.copy()) {
    level_ = level;
    d_ = block_size;
    n_heads_ = n_heads;

    uint32_t m = shape[0];
    uint32_t n = shape[1];
    m_ = m;
    n_ = n;

    n_slot_ = param_.get_n() / 2;
    n_h_padded_ = next_pow2(n_heads);

    // Determine chunk sizing
    if (n_slot_ >= n_h_padded_ * d_ * d_) {
        n_blocks_per_chunk_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        n_blocks_per_chunk_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        n_cts_per_block_idx_ = n_h_padded_ / n_blocks_per_chunk_;
    }

    assert(n_slot_ % chunk_size_ == 0 && "n_slot must be divisible by chunk_size");
    num_chunks_ = n_slot_ / chunk_size_;

    num_block_rows_ = div_ceil(m, d_);
    num_block_cols_ = div_ceil(n, d_);
}

ParBlockColMajorTranspose::~ParBlockColMajorTranspose() {}

int ParBlockColMajorTranspose::get_block_index(int bi, int bj, int num_block_rows) {
    return bi + num_block_rows * bj;
}

// Build the k-th non-zero diagonal of U^t for interleaved format.
// Base diagonal on d² elements, then expand by n_blocks_per_chunk_.
std::vector<double> ParBlockColMajorTranspose::build_transpose_diagonal(int k) const {
    uint32_t d_sq = d_ * d_;
    uint32_t S = n_blocks_per_chunk_;

    // Build base diagonal on d² elements (same as non-interleaved)
    vector<double> t_base(d_sq, 0.0);
    int diag = (((int)(d_ - 1) * k) % (int)d_sq + (int)d_sq) % (int)d_sq;

    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            int idx = i + d_ * j;
            int col_from_diag = (idx + diag) % (int)d_sq;
            int col_required = j + d_ * i;  // transpose: (i,j) -> (j,i)
            if (col_from_diag == col_required) {
                t_base[idx] = 1.0;
            }
        }
    }

    // Expand by S: each base value replicated to S consecutive slots
    vector<double> expanded_chunk(chunk_size_, 0.0);
    for (uint32_t idx = 0; idx < d_sq; idx++) {
        for (uint32_t h = 0; h < S; h++) {
            expanded_chunk[idx * S + h] = t_base[idx];
        }
    }

    // Tile to n_slot_ length
    vector<double> t(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            t[c * chunk_size_ + s] = expanded_chunk[s];
        }
    }
    return t;
}

void ParBlockColMajorTranspose::precompute_diagonals() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    double transpose_scale = param_.get_q(level_);

    // Transpose: (2d-1) diagonal vectors, k from -(d-1) to (d-1)
    transpose_diag_pt_.clear();
    transpose_diag_pt_.reserve(2 * d_ - 1);
    for (int k = -(int)(d_ - 1); k <= (int)(d_ - 1); k++) {
        auto diag_vec = build_transpose_diagonal(k);
        transpose_diag_pt_.push_back(ctx.encode_ringt(diag_vec, transpose_scale));
    }
}

// parallelly transpose each block in the interleaved block ciphertext
// transpose_on_ct: (2d-1) rotations + (2d-1) pt_muls + (2d-2) adds + 1 rescale
// Input level L -> Output level L-1
// Rotation amounts scaled by n_blocks_per_chunk_ compared to non-interleaved version.
CkksCiphertext ParBlockColMajorTranspose::transpose_on_ct(CkksContext& ctx, const CkksCiphertext& ct) const {
    double default_scale = param_.get_default_scale();
    uint32_t S = n_blocks_per_chunk_;
    uint32_t d_sq = d_ * d_;
    CkksCiphertext result(0);

    int diag_idx = 0;
    for (int k = -(int)(d_ - 1); k <= (int)(d_ - 1); k++) {
        // Scaled rotation: ((d-1)*k * S) % chunk_size_
        int base_diag = (((int)(d_ - 1) * k) % (int)d_sq + (int)d_sq) % (int)d_sq;
        int rot_amount = (base_diag * (int)S) % (int)chunk_size_;
        CkksCiphertext rotated = (rot_amount == 0) ? ct.copy() : ctx.rotate(ct, rot_amount);

        auto diag_mul = ctx.ringt_to_mul(transpose_diag_pt_[diag_idx], level_);
        auto product = ctx.mult_plain_mul(rotated, diag_mul);

        if (diag_idx == 0) {
            result = move(product);
        } else {
            result = ctx.add(result, product);
        }
        diag_idx++;
    }
    return ctx.rescale(result, default_scale);
}

// perform block-based transpose for each head parallelly in the multi-head attention module
std::vector<CkksCiphertext> ParBlockColMajorTranspose::run_core(CkksContext& ctx,
                                                                const std::vector<CkksCiphertext>& cts) {
    // Result: per-head shape (n, m), so ceil(n/d) x ceil(m/d) blocks
    uint32_t num_block_rows_T = num_block_cols_;  // ceil(n/d)
    uint32_t num_block_cols_T = num_block_rows_;  // ceil(m/d)
    uint32_t num_result_vecs = num_block_rows_T * num_block_cols_T * n_cts_per_block_idx_;
    vector<CkksCiphertext> result_cts;
    result_cts.resize(num_result_vecs);

    parallel_for(num_result_vecs, th_nums, ctx, [&](CkksContext& ctx_copy, int dst_vec_idx) {
        uint32_t dst_block_idx = dst_vec_idx / n_cts_per_block_idx_;
        uint32_t g = dst_vec_idx % n_cts_per_block_idx_;

        // Column-major index in result: dst_block_idx = row_t + num_block_rows_T * col_t
        int row_t = dst_block_idx % num_block_rows_T;  // row in transposed = col in original
        int col_t = dst_block_idx / num_block_rows_T;  // col in transposed = row in original

        // Source: block (col_t, row_t) in original, same group g
        int src_block_idx = get_block_index(col_t, row_t, num_block_rows_);
        int src_vec_idx = src_block_idx * n_cts_per_block_idx_ + g;

        result_cts[dst_vec_idx] = transpose_on_ct(ctx_copy, cts[src_vec_idx]);
    });

    return result_cts;
}

Feature2DEncrypted ParBlockColMajorTranspose::run(CkksContext& ctx, const Feature2DEncrypted& input) {
    Feature2DEncrypted result(&ctx, input.level);
    result.data = run_core(ctx, input.data);
    result.level = input.level - 1;  // transpose consumes 1 level
    result.shape = {n_, m_};         // transposed per-head shape
    result.matmul_block_size = d_;
    return result;
}
