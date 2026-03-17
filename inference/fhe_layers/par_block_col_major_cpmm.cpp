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

#include "par_block_col_major_cpmm.h"
#include <cassert>
#include <cmath>

using namespace std;

static uint32_t next_pow2_cpmm(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

ParBlockColMajorCPMM::ParBlockColMajorCPMM(const CkksParameter& param_in,
                                           const Duo& shape_A,
                                           const Array<double, 2>& W_mat,
                                           uint32_t block_size,
                                           uint32_t n_heads,
                                           uint32_t level_A)
    : param_(param_in.copy()) {
    level_ = level_A;
    d_ = block_size;
    n_heads_ = n_heads;

    m_ = shape_A[0];
    n_per_head_ = shape_A[1];
    n_total_per_mb_ = n_heads_ * n_per_head_;

    assert(n_per_head_ <= d_ && "per-head width must fit in one block column");

    n_slot_ = param_.get_n() / 2;
    n_h_padded_ = next_pow2_cpmm(n_heads);

    // Determine chunk sizing (same logic as ParBlockColMajorTranspose/CCMM)
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

    num_block_rows_A_ = div_ceil(m_, d_);

    // Auto-detect mode from W dimensions
    uint32_t W_rows = W_mat.get_shape()[0];
    uint32_t W_cols = W_mat.get_shape()[1];

    if (W_rows == W_cols) {
        mode_ = Mode::SQUARE;
        K_ = 1;
        assert(W_rows == n_total_per_mb_);
    } else if (W_cols > W_rows) {
        mode_ = Mode::EXPAND;
        assert(W_rows == n_total_per_mb_ && W_cols % W_rows == 0);
        K_ = W_cols / W_rows;
    } else {
        mode_ = Mode::REDUCE;
        assert(W_cols == n_total_per_mb_ && W_rows % W_cols == 0);
        K_ = W_rows / W_cols;
    }

    // Split and pad K sub-weights
    uint32_t padded_dim = n_h_padded_ * d_;
    W_padded_.resize(K_);
    for (uint32_t mb = 0; mb < K_; mb++) {
        Array<double, 2> W_sub({padded_dim, padded_dim});
        for (uint32_t i = 0; i < n_total_per_mb_; i++) {
            for (uint32_t j = 0; j < n_total_per_mb_; j++) {
                double val;
                if (mode_ == Mode::EXPAND)
                    val = W_mat.get(i, mb * n_total_per_mb_ + j);
                else if (mode_ == Mode::REDUCE)
                    val = W_mat.get(mb * n_total_per_mb_ + i, j);
                else
                    val = W_mat.get(i, j);
                W_sub.set(i, j, val);
            }
        }
        W_padded_[mb] = std::move(W_sub);
    }
}

ParBlockColMajorCPMM::~ParBlockColMajorCPMM() {}

int ParBlockColMajorCPMM::get_block_index(int bi, int bj, int num_block_rows) {
    return bi + num_block_rows * bj;
}

// Build per-head diagonal for output block column bp, rotation index k, and megablock.
//
// In the interleaved layout, slot[(i + d*j) * S + h] holds head h's block
// element at (row=i, col=j).  For the CPMM diagonal multiply, head h uses
// weight block W_{h, bp}, so:
//
//   diag[(i + d*j) * S + h] = W_padded[megablock][h*d + (j+k)%d,  bp*d + j]
//
// The value is constant across row index i (same as standard CPMM) but
// differs across head index h (unlike standard CPMM which replicates).
std::vector<double>
ParBlockColMajorCPMM::build_block_diagonal(uint32_t megablock, uint32_t g_input, int bp, int k) const {
    uint32_t S = n_blocks_per_chunk_;

    vector<double> diag_chunk(chunk_size_, 0.0);
    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t h = 0; h < S; h++) {
            uint32_t row = (g_input * S + h) * d_ + (j + k) % d_;
            uint32_t col = bp * d_ + j;
            double val = W_padded_[megablock].get(row, col);
            for (uint32_t i = 0; i < d_; i++) {
                diag_chunk[(i + d_ * j) * S + h] = val;
            }
        }
    }

    // Tile to n_slot_
    vector<double> diag(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            diag[c * chunk_size_ + s] = diag_chunk[s];
        }
    }
    return diag;
}

// Build mask selecting h=0 positions: mask[(i+d*j)*S + 0] = 1, rest = 0.
std::vector<double> ParBlockColMajorCPMM::build_head0_mask() const {
    uint32_t S = n_blocks_per_chunk_;

    vector<double> mask_chunk(chunk_size_, 0.0);
    for (uint32_t j = 0; j < d_; j++) {
        for (uint32_t i = 0; i < d_; i++) {
            mask_chunk[(i + d_ * j) * S + 0] = 1.0;
        }
    }

    vector<double> mask(n_slot_, 0.0);
    for (uint32_t c = 0; c < num_chunks_; c++) {
        for (uint32_t s = 0; s < chunk_size_; s++) {
            mask[c * chunk_size_ + s] = mask_chunk[s];
        }
    }
    return mask;
}

void ParBlockColMajorCPMM::precompute_diagonals() {
    CkksContext ctx = CkksContext::create_empty_context(param_);

    double diag_scale = param_.get_q(level_);
    double mask_scale = param_.get_q(level_ - 1);

    // Diagonals: for each megablock, input group g, output block column bp, and rotation k
    diag_pt_.clear();
    diag_pt_.resize(K_);
    for (uint32_t mb = 0; mb < K_; mb++) {
        diag_pt_[mb].resize(n_cts_per_block_idx_);
        for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
            diag_pt_[mb][g].resize(n_heads_);
            for (uint32_t bp = 0; bp < n_heads_; bp++) {
                diag_pt_[mb][g][bp].reserve(d_);
                for (uint32_t k = 0; k < d_; k++) {
                    auto diag_vec = build_block_diagonal(mb, g, bp, k);
                    diag_pt_[mb][g][bp].push_back(ctx.encode_ringt(diag_vec, diag_scale));
                }
            }
        }
    }

    // Mask for selecting h=0 after cross-head sum
    auto mask_vec = build_head0_mask();
    mask_h0_pt_ = ctx.encode_ringt(mask_vec, mask_scale);
}

// block_mult_cpmm: d rotations + d pt_muls + (d-1) adds + 1 rescale
// Computes a ciphertext's all interleaved heads' contributions to output block column bp in parallel.
// Input level L -> Output level L-1
CkksCiphertext ParBlockColMajorCPMM::block_mult_cpmm(CkksContext& ctx,
                                                     const CkksCiphertext& a,
                                                     uint32_t megablock,
                                                     uint32_t g_input,
                                                     int bp) const {
    double default_scale = param_.get_default_scale();
    uint32_t S = n_blocks_per_chunk_;
    CkksCiphertext result(0);

    for (uint32_t k = 0; k < d_; k++) {
        // Rotation scaled by S (same as par CCMM sigma)
        int rot_amount = ((int)(d_ * k) * (int)S) % (int)chunk_size_;
        CkksCiphertext rotated = (rot_amount == 0) ? a.copy() : ctx.rotate(a, rot_amount);

        auto diag_mul = ctx.ringt_to_mul(diag_pt_[megablock][g_input][bp][k], level_);
        auto product = ctx.mult_plain_mul(rotated, diag_mul);

        if (k == 0) {
            result = move(product);
        } else {
            result = ctx.add(result, product);
        }
    }
    return ctx.rescale(result, default_scale);
}

// head_sum: tree-reduction sum across S head slots with fixed megablock idx and fixed heads group idx g.
// log2(S) rotations + log2(S) additions, no level consumed.
// After this, position h=0 at every (i,j) holds the correct sum over all heads.
CkksCiphertext ParBlockColMajorCPMM::head_sum(CkksContext& ctx, const CkksCiphertext& ct) const {
    uint32_t S = n_blocks_per_chunk_;
    if (S == 1)
        return ct.copy();

    CkksCiphertext result = ct.copy();
    uint32_t step = 1;
    while (step < S) {
        auto rotated = ctx.rotate(result, step);
        result = ctx.add(result, rotated);
        step *= 2;
    }
    return result;
}

// run_core: unified core processing mb_indices megablocks.
// Supports n_cts_per_block_idx > 1 (G > 1).
//
// Per output ct (indexed by bi, g_out):
//   for bp in [0, n_heads):
//     1. For each megablock in mb_indices and each g in [0, G):
//        block_mult_cpmm + head_sum, sum all contributions
//     2. mask h=0          -> zero out h>0                  (L-1 -> L-2)
//     3. rotate            -> place sum at output head pos  (L-2    )
//     4. accumulate        -> add into result ct            (L-2    )
std::vector<CkksCiphertext> ParBlockColMajorCPMM::run_core(CkksContext& ctx,
                                                           const std::vector<CkksCiphertext>& A_cts,
                                                           const std::vector<uint32_t>& mb_indices) {
    uint32_t K_eff = mb_indices.size();
    uint32_t cts_per_mb = num_block_rows_A_ * n_cts_per_block_idx_;
    uint32_t num_result_cts = num_block_rows_A_ * n_cts_per_block_idx_;
    vector<CkksCiphertext> C_cts(num_result_cts);
    double default_scale = param_.get_default_scale();

    parallel_for(num_block_rows_A_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        for (uint32_t bp = 0; bp < n_heads_; bp++) {
            // Step 1+2: sum block_mult + head_sum across megablocks and g groups
            CkksCiphertext full_sum(0);
            bool first = true;
            for (uint32_t mb_i = 0; mb_i < K_eff; mb_i++) {
                uint32_t mb = mb_indices[mb_i];
                for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                    uint32_t a_idx = mb_i * cts_per_mb + bi * n_cts_per_block_idx_ + g;
                    auto tmp = block_mult_cpmm(ctx_copy, A_cts[a_idx], mb, g, bp);
                    tmp = head_sum(ctx_copy, tmp);
                    if (first) {
                        full_sum = move(tmp);
                        first = false;
                    } else {
                        full_sum = ctx_copy.add(full_sum, tmp);
                    }
                }
            }

            // Step 3: mask h=0 (L-1 -> L-2)
            auto mask_mul = ctx_copy.ringt_to_mul(mask_h0_pt_, level_ - 1);
            full_sum = ctx_copy.rescale(ctx_copy.mult_plain_mul(full_sum, mask_mul), default_scale);

            // Step 4: rotate to output head position (generalized for G > 1)
            uint32_t g_out = bp / n_blocks_per_chunk_;
            uint32_t h_local = bp % n_blocks_per_chunk_;
            if (h_local > 0) {
                full_sum = ctx_copy.rotate(full_sum, (int)chunk_size_ - (int)h_local);
            }

            // Step 5: accumulate into output
            uint32_t out_idx = bi * n_cts_per_block_idx_ + g_out;
            if (bp == g_out * n_blocks_per_chunk_) {
                C_cts[out_idx] = move(full_sum);
            } else {
                C_cts[out_idx] = ctx_copy.add(C_cts[out_idx], full_sum);
            }
        }
    });
    return C_cts;
}

Feature2DEncrypted ParBlockColMajorCPMM::run(CkksContext& ctx, const Feature2DEncrypted& A) {
    Feature2DEncrypted result(&ctx, A.level);
    result.level = A.level - 2;  // block_mult (1 level) + mask (1 level)
    result.shape = {m_, n_per_head_};
    result.matmul_block_size = d_;

    if (mode_ == Mode::EXPAND) {
        for (uint32_t mb = 0; mb < K_; mb++) {
            auto mb_cts = run_core(ctx, A.data, {mb});
            for (auto& ct : mb_cts)
                result.data.push_back(std::move(ct));
        }
    } else {
        // Square (K=1) or Reduce (K>1)
        vector<uint32_t> all_mbs(K_);
        for (uint32_t i = 0; i < K_; i++)
            all_mbs[i] = i;
        result.data = run_core(ctx, A.data, all_mbs);
    }
    return result;
}
