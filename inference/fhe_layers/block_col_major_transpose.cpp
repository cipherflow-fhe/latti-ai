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

#include "block_col_major_transpose.h"
#include <cassert>
#include <cmath>

using namespace std;

BlockColMajorTranspose::BlockColMajorTranspose(const CkksParameter& param_in,
                                               const Duo& shape,
                                               uint32_t block_size,
                                               uint32_t level)
    : param_(param_in.copy()) {
    level_ = level;
    d_ = block_size;

    uint32_t m = shape[0];
    uint32_t n = shape[1];
    m_ = m;
    n_ = n;

    assert(m % d_ == 0 && "m must be divisible by block_size");
    assert(n % d_ == 0 && "n must be divisible by block_size");

    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
    assert(n_slot_ % chunk_size_ == 0 && "n_slot must be divisible by d^2");
    num_chunks_ = n_slot_ / chunk_size_;

    num_block_rows_ = m / d_;
    num_block_cols_ = n / d_;
}

BlockColMajorTranspose::~BlockColMajorTranspose() {}

int BlockColMajorTranspose::get_block_index(int bi, int bj, int num_block_rows) {
    return bi + num_block_rows * bj;
}

// Build the k-th non-zero diagonal of U^t (transpose permutation matrix).
// The (2d-1) non-zero diagonals are at positions (d-1)*k for k in -(d-1) to (d-1).
std::vector<double> BlockColMajorTranspose::build_transpose_diagonal(int k) const {
    vector<double> t_base(chunk_size_, 0.0);
    int diag = (((int)(d_ - 1) * k) % (int)chunk_size_ + (int)chunk_size_) % (int)chunk_size_;

    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            int idx = i + d_ * j;
            int col_from_diag = (idx + diag) % (int)chunk_size_;
            int col_required = j + d_ * i;  // U^t: (M^T)_{i,j} = M_{j,i}
            if (col_from_diag == col_required) {
                t_base[idx] = 1.0;
            }
        }
    }

    // Tile to n_slot_ length
    vector<double> t(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            t[c * chunk_size_ + s] = t_base[s];
        }
    }
    return t;
}

void BlockColMajorTranspose::precompute_diagonals() {
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

// transpose_on_ct: (2d-1) rotations + (2d-1) pt_muls + (2d-2) adds + 1 rescale
// Input level L -> Output level L-1
// Same rotate->multiply->accumulate->rescale pattern as BlockCCMM::sigma_on_ct
CkksCiphertext BlockColMajorTranspose::transpose_on_ct(CkksContext& ctx, const CkksCiphertext& ct) const {
    double default_scale = param_.get_default_scale();
    CkksCiphertext result(0);

    int diag_idx = 0;
    for (int k = -(int)(d_ - 1); k <= (int)(d_ - 1); k++) {
        int rot_amount = (((int)(d_ - 1) * k) % (int)chunk_size_ + (int)chunk_size_) % (int)chunk_size_;
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

std::vector<CkksCiphertext> BlockColMajorTranspose::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& cts) {
    // Result has shape (n, m), so (n/d) x (m/d) blocks
    uint32_t num_block_rows_T = num_block_cols_;                      // n/d
    uint32_t num_result_blocks = num_block_rows_T * num_block_rows_;  // (n/d) * (m/d)
    vector<CkksCiphertext> result_cts;
    result_cts.resize(num_result_blocks);

    parallel_for(num_result_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int dst_idx) {
        // Column-major index in result: dst_idx = row_t + num_block_rows_T * col_t
        int row_t = dst_idx % num_block_rows_T;  // row in transposed = col in original
        int col_t = dst_idx / num_block_rows_T;  // col in transposed = row in original

        // Source: block (col_t, row_t) in original
        int src_idx = get_block_index(col_t, row_t, num_block_rows_);

        result_cts[dst_idx] = transpose_on_ct(ctx_copy, cts[src_idx]);
    });

    return result_cts;
}

Feature2DEncrypted BlockColMajorTranspose::run(CkksContext& ctx, const Feature2DEncrypted& input) {
    Feature2DEncrypted result(&ctx, input.level);
    result.data = run_core(ctx, input.data);
    result.level = input.level - 1;  // transpose consumes 1 level
    result.shape = {n_, m_};         // transposed shape
    result.matmul_block_size = d_;
    return result;
}
