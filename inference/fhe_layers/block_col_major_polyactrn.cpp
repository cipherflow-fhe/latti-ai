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

#include "block_col_major_polyactrn.h"
#include <cassert>

using namespace std;
using namespace lattisense;

// ============================================================
// BlockColMajorPolyActRNGamma
// ============================================================

BlockColMajorPolyActRNGamma::BlockColMajorPolyActRNGamma(const CkksParameter& param,
                                                         Duo shape,
                                                         uint32_t block_size,
                                                         uint32_t init_level,
                                                         Array<double, 1>&& gamma)
    : Layer(param), gamma_vals_(move(gamma)) {
    level_ = init_level;
    m_ = shape[0];
    n_ = shape[1];
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(n_, d_);
}

void BlockColMajorPolyActRNGamma::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double q_L = param_.get_q(level_);

    gamma_pt_.resize(num_block_cols_);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        vector<double> gamma_vec(n_slot_, 0.0);
        for (uint32_t c = 0; c < num_chunks_; c++) {
            for (uint32_t col = 0; col < d_; col++) {
                uint32_t actual_col = bj * d_ + col;
                for (uint32_t row = 0; row < d_; row++) {
                    uint32_t slot = c * chunk_size_ + row + d_ * col;
                    if (actual_col < n_) {
                        gamma_vec[slot] = gamma_vals_.get(actual_col);
                    }
                }
            }
        }
        gamma_pt_[bj] = ctx.encode_ringt(gamma_vec, q_L);
    }
}

FeatureMatEncrypted BlockColMajorPolyActRNGamma::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_blocks = num_block_rows_ * num_block_cols_;

    FeatureMatEncrypted result(&ctx, level_ - 1);
    result.shape = {m_, n_};
    result.matmul_block_size = d_;
    result.data.resize(total_blocks);

    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int block_idx) {
        uint32_t bj = block_idx / num_block_rows_;

        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[bj], level_);
        auto product = ctx_copy.mult_plain_mul(x.data[block_idx], gamma_mul);
        result.data[block_idx] = ctx_copy.rescale(product, D);
    });

    result.level = level_ - 1;
    return result;
}

// Helper: build per-column coefficient vector for a block-column bj
static vector<double> build_block_coeff_vec(uint32_t n_slot,
                                            uint32_t chunk_size,
                                            uint32_t num_chunks,
                                            uint32_t d,
                                            uint32_t n,
                                            uint32_t bj,
                                            const Array<double, 2>& coeffs,
                                            uint32_t coeff_row) {
    vector<double> vec(n_slot, 0.0);
    for (uint32_t c = 0; c < num_chunks; c++) {
        for (uint32_t col = 0; col < d; col++) {
            uint32_t actual_col = bj * d + col;
            for (uint32_t row = 0; row < d; row++) {
                uint32_t slot = c * chunk_size + row + d * col;
                if (actual_col < n) {
                    vec[slot] = coeffs.get(coeff_row, actual_col);
                }
            }
        }
    }
    return vec;
}

// ============================================================
// BlockColMajorPolyActRNPoly
// ============================================================

BlockColMajorPolyActRNPoly::BlockColMajorPolyActRNPoly(const CkksParameter& param,
                                                       Duo shape,
                                                       uint32_t block_size,
                                                       uint32_t init_level,
                                                       Array<double, 2>&& coeffs,
                                                       uint32_t degree)
    : Layer(param), degree_(degree), coeffs_(move(coeffs)) {
    assert(degree_ == 2 || degree_ == 4);
    assert(coeffs_.get_shape()[0] == degree_ + 1);
    assert(coeffs_.get_shape()[1] >= shape[1]);
    level_ = init_level;
    m_ = shape[0];
    n_ = shape[1];
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(n_, d_);
}

void BlockColMajorPolyActRNPoly::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);

    // --- c2: per bj ---
    double c2_scale = q_L / D * q_L1;
    c2_pt_.resize(num_block_cols_);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        auto vec = build_block_coeff_vec(n_slot_, chunk_size_, num_chunks_, d_, n_, bj, coeffs_, 2);
        c2_pt_[bj] = ctx.encode_ringt(vec, c2_scale);
    }

    // --- c1: per bj ---
    c1_pt_.resize(num_block_cols_);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        auto vec = build_block_coeff_vec(n_slot_, chunk_size_, num_chunks_, d_, n_, bj, coeffs_, 1);
        c1_pt_[bj] = ctx.encode_ringt(vec, q_L);
    }

    // --- c0: per (bi,bj), scale D, with 0-padding ---
    c0_add_pt_.resize(num_block_rows_ * num_block_cols_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            vector<double> vec(n_slot_, 0.0);
            for (uint32_t c = 0; c < num_chunks_; c++) {
                for (uint32_t col = 0; col < d_; col++) {
                    uint32_t actual_col = bj * d_ + col;
                    for (uint32_t row = 0; row < d_; row++) {
                        uint32_t actual_row = bi * d_ + row;
                        uint32_t slot = c * chunk_size_ + row + d_ * col;
                        if (actual_row < m_ && actual_col < n_) {
                            vec[slot] = coeffs_.get(0, actual_col);
                        }
                    }
                }
            }
            uint32_t idx = bi + num_block_rows_ * bj;
            c0_add_pt_[idx] = ctx.encode_ringt(vec, D);
        }
    }

    // --- Degree-4 additional: c3 and c4 ---
    if (degree_ == 4) {
        double q_L2 = param_.get_q(level_ - 2);

        double c4_scale = q_L / D * q_L / D * q_L1 / D * q_L2;
        c4_pt_.resize(num_block_cols_);
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            auto vec = build_block_coeff_vec(n_slot_, chunk_size_, num_chunks_, d_, n_, bj, coeffs_, 4);
            c4_pt_[bj] = ctx.encode_ringt(vec, c4_scale);
        }

        double c3_scale = q_L / D * q_L / D * q_L2;
        c3_pt_.resize(num_block_cols_);
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            auto vec = build_block_coeff_vec(n_slot_, chunk_size_, num_chunks_, d_, n_, bj, coeffs_, 3);
            c3_pt_[bj] = ctx.encode_ringt(vec, c3_scale);
        }
    }
}

FeatureMatEncrypted BlockColMajorPolyActRNPoly::run(CkksContext& ctx, const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_blocks = num_block_rows_ * num_block_cols_;
    uint32_t out_level = (degree_ == 4) ? level_ - 3 : level_ - 2;

    FeatureMatEncrypted result(&ctx, out_level);
    result.shape = {m_, n_};
    result.matmul_block_size = d_;
    result.data.resize(total_blocks);

    if (degree_ == 2) {
        parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int block_idx) {
            uint32_t bj = block_idx / num_block_rows_;

            // x^2 = x * x -> level L-1
            auto x_sq_raw = ctx_copy.mult(x.data[block_idx], x.data[block_idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);

            // c2*x^2 -> level L-2, scale D
            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[bj], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);

            // c1*x -> level L-1, scale D
            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[bj], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[block_idx], c1_mul), D);
            // Drop c1*x to level L-2
            auto c1x_drop = ctx_copy.drop_level(c1x);

            // y = c0 + c1*x + c2*x^2
            auto y = ctx_copy.add(c1x_drop, c2x2);
            result.data[block_idx] = ctx_copy.add_plain_ringt(y, c0_add_pt_[block_idx]);
        });
    } else {
        // degree == 4
        double q_L2 = param_.get_q(level_ - 2);
        double S_high = param_.get_q(level_) / D * q_L2;

        parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int block_idx) {
            uint32_t bj = block_idx / num_block_rows_;

            // === x^2 ===
            auto x_sq_raw = ctx_copy.mult(x.data[block_idx], x.data[block_idx]);
            auto x_sq = ctx_copy.rescale(ctx_copy.relinearize(x_sq_raw), D / param_.get_q(level_) * D);
            // x_sq at level L-1

            // === Low part: c0 + c1*x + c2*x^2 -> level L-2, scale D ===
            auto c2_mul = ctx_copy.ringt_to_mul(c2_pt_[bj], level_ - 1);
            auto c2x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c2_mul), D);

            auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_[bj], level_);
            auto c1x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[block_idx], c1_mul), D);
            auto c1x_drop = ctx_copy.drop_level(c1x);

            auto low = ctx_copy.add(c1x_drop, c2x2);
            low = ctx_copy.add_plain_ringt(low, c0_add_pt_[block_idx]);
            // low at level L-2, scale D

            // === High part: c3*x + c4*x^2 -> level L-2, scale S_high ===
            auto c4_mul = ctx_copy.ringt_to_mul(c4_pt_[bj], level_ - 1);
            auto c4x2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(x_sq, c4_mul), S_high);

            auto c3_mul = ctx_copy.ringt_to_mul(c3_pt_[bj], level_);
            auto c3x = ctx_copy.rescale(ctx_copy.mult_plain_mul(x.data[block_idx], c3_mul), S_high);
            auto c3x_drop = ctx_copy.drop_level(c3x);

            auto high = ctx_copy.add(c3x_drop, c4x2);
            // high at level L-2

            // === Final: drop(low) + x^2 * high -> level L-3, scale D ===
            auto x_sq_drop = ctx_copy.drop_level(x_sq);  // L-2

            auto product = ctx_copy.mult(x_sq_drop, high);
            auto x2_high = ctx_copy.rescale(ctx_copy.relinearize(product), D);
            // x2_high at level L-3, scale D

            auto low_drop = ctx_copy.drop_level(low);  // L-3, scale D

            result.data[block_idx] = ctx_copy.add(low_drop, x2_high);
        });
    }

    result.level = out_level;
    return result;
}
