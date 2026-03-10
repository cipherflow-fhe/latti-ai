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

#include "block_col_major_cpmm.h"
#include <cassert>
#include <cmath>

using namespace std;

BlockColMajorCPMM::BlockColMajorCPMM(const CkksParameter& param_in,
                                     const Duo& shape_A,
                                     const Duo& shape_B,
                                     const Array<double, 2>& B_mat_in,
                                     uint32_t block_size,
                                     uint32_t level_A)
    : param_(param_in.copy()), B_mat_(B_mat_in.copy()) {
    assert(shape_A[1] == shape_B[0] && "inner dimensions must match: shape_A[1] != shape_B[0]");

    level_ = level_A;
    d_ = block_size;

    uint32_t m = shape_A[0];
    uint32_t n = shape_A[1];
    uint32_t p = shape_B[1];
    m_ = m;
    n_ = n;
    p_ = p;

    assert(m % d_ == 0 && "m must be divisible by block_size");
    assert(n % d_ == 0 && "n must be divisible by block_size");
    assert(p % d_ == 0 && "p must be divisible by block_size");

    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
    assert(n_slot_ % chunk_size_ == 0 && "n_slot must be divisible by block_size^2");
    num_chunks_ = n_slot_ / chunk_size_;

    num_block_rows_A_ = m / d_;
    num_block_cols_A_ = n / d_;
    num_block_rows_B_ = n / d_;
    num_block_cols_B_ = p / d_;
}

BlockColMajorCPMM::~BlockColMajorCPMM() {}

int BlockColMajorCPMM::get_block_index(int bi, int bj, int num_block_rows) {
    return bi + num_block_rows * bj;
}

// Build diagonal k for B block (bj, bp):
// diag_k[i + d*j] = B[(bj*d + (j+k)%d), (bp*d + j)]
// Same value replicated across all rows i, tiled across num_chunks_
std::vector<double> BlockColMajorCPMM::build_block_diagonal(int bj, int bp, int k) const {
    vector<double> diag_base(chunk_size_, 0.0);
    for (uint32_t j = 0; j < d_; j++) {
        double val = B_mat_.get(bj * d_ + (j + k) % d_, bp * d_ + j);
        for (uint32_t i = 0; i < d_; i++) {
            diag_base[i + d_ * j] = val;
        }
    }
    // Tile to n_slot_ length
    vector<double> diag(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            diag[c * chunk_size_ + s] = diag_base[s];
        }
    }
    return diag;
}

void BlockColMajorCPMM::precompute_diagonals() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    double scale = param_.get_q(level_);

    uint32_t total_b_blocks = num_block_rows_B_ * num_block_cols_B_;
    diag_pt_.clear();
    diag_pt_.resize(total_b_blocks);

    for (uint32_t bp = 0; bp < num_block_cols_B_; bp++) {
        for (uint32_t bj = 0; bj < num_block_rows_B_; bj++) {
            int b_idx = get_block_index(bj, bp, num_block_rows_B_);
            diag_pt_[b_idx].reserve(d_);
            for (uint32_t k = 0; k < d_; k++) {
                auto diag_vec = build_block_diagonal(bj, bp, k);
                diag_pt_[b_idx].push_back(ctx.encode_ringt(diag_vec, scale));
            }
        }
    }
}

// block_mult_cpmm: C_block = Σ_{k=0}^{d-1} rot(a, k*d) ⊙ diag_k(B_block)
// Input level L -> Output level L-1
CkksCiphertext BlockColMajorCPMM::block_mult_cpmm(CkksContext& ctx, const CkksCiphertext& a, int bj, int bp) const {
    double default_scale = param_.get_default_scale();
    int b_idx = get_block_index(bj, bp, num_block_rows_B_);
    CkksCiphertext result(0);

    for (uint32_t k = 0; k < d_; k++) {
        int rot_amount = (k * d_) % chunk_size_;
        CkksCiphertext rotated = (rot_amount == 0) ? a.copy() : ctx.rotate(a, rot_amount);

        auto diag_mul = ctx.ringt_to_mul(diag_pt_[b_idx][k], level_);
        auto product = ctx.mult_plain_mul(rotated, diag_mul);

        if (k == 0) {
            result = move(product);
        } else {
            result = ctx.add(result, product);
        }
    }
    return ctx.rescale(result, default_scale);
}

std::vector<CkksCiphertext> BlockColMajorCPMM::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& A_cts) {
    uint32_t num_result_blocks = num_block_rows_A_ * num_block_cols_B_;
    vector<CkksCiphertext> C_cts;
    C_cts.resize(num_result_blocks);

    parallel_for(num_result_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int c_idx) {
        int bi = c_idx % num_block_rows_A_;
        int bp = c_idx / num_block_rows_A_;

        for (uint32_t bj = 0; bj < num_block_cols_A_; bj++) {
            int a_idx = get_block_index(bi, bj, num_block_rows_A_);

            auto product = block_mult_cpmm(ctx_copy, A_cts[a_idx], bj, bp);

            if (bj == 0) {
                C_cts[c_idx] = move(product);
            } else {
                C_cts[c_idx] = ctx_copy.add(C_cts[c_idx], product);
            }
        }
    });

    return C_cts;
}

Feature2DEncrypted BlockColMajorCPMM::run(CkksContext& ctx, const Feature2DEncrypted& A) {
    Feature2DEncrypted result(&ctx, A.level);
    result.data = run_core(ctx, A.data);
    result.level = A.level - 1;  // block_mult_cpmm consumes 1 level
    result.shape = {m_, p_};
    result.matmul_block_size = d_;
    return result;
}
