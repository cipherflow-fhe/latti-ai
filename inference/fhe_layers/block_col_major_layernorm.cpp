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

#include "block_col_major_layernorm.h"
#include <cassert>
#include <cmath>

using namespace std;
using namespace lattisense;

// ============================================================
// BlockColMajorLNStats
// ============================================================

BlockColMajorLNStats::BlockColMajorLNStats(const CkksParameter& param,
                                           Duo shape,
                                           uint32_t block_size,
                                           uint32_t init_level,
                                           double eps,
                                           double inv_var)
    : Layer(param), eps_(eps), inv_var_(inv_var) {
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

void BlockColMajorLNStats::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);

    double n_cols = static_cast<double>(n_);

    // 1/n_cols for both E_x_sq and mean paths (D-preserving pt*ct):
    //   scale = q_L
    //   proof: D * q_L / q_L = D  (for mean path at level L)
    //   also:  (D²/q_L) * q_L / q_{L-1} = D²/q_{L-1}  (for E_x_sq path at level L-1)
    vector<double> inv_n_vec(n_slot_, 1.0 / n_cols);
    inv_n_pt_ = ctx.encode_ringt(inv_n_vec, q_L);

    // inv_var normalizing pt*ct: var at scale D²/q_{L-1} -> D
    //   encode scale = q_{L-1}/D * q_{L-2}
    //   proof: (D²/q_{L-1}) * (q_{L-1}/D * q_{L-2}) / q_{L-2} = D
    double iv_scale = q_L1 / D * q_L2;
    vector<double> iv_vec(n_slot_, inv_var_);
    iv_pt_ = ctx.encode_ringt(iv_vec, iv_scale);

    // eps * inv_var for add_plain:
    //   scale = D (matching ciphertext after normalizing pt*ct)
    vector<double> eps_vec(n_slot_, eps_ * inv_var_);
    eps_add_pt_ = ctx.encode_ringt(eps_vec, D);
}

CkksCiphertext BlockColMajorLNStats::intra_block_row_sum(CkksContext& ctx, const CkksCiphertext& ct) const {
    // Sum across d columns within a d×d block.
    // Slot layout: slot[row + d*col]. Stride between columns = d.
    // Rotate by d, 2d, 4d, ..., (d/2)*d and accumulate.
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < d_; step *= 2) {
        int rot = step * d_;
        auto rotated = ctx.rotate(result, rot);
        result = ctx.add(result, rotated);
    }
    return result;
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>> BlockColMajorLNStats::run(CkksContext& ctx,
                                                                               const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_blocks = num_block_rows_ * num_block_cols_;

    // --- Step 1: sum_x per block-row (rotate-and-add within block, then add across block-cols) ---
    // Also compute x_sq = x*x for each block
    vector<CkksCiphertext> sum_x_per_block(total_blocks);
    vector<CkksCiphertext> x_sq(total_blocks);

    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        // Intra-block row sum
        sum_x_per_block[idx] = intra_block_row_sum(ctx_copy, x.data[idx]);
        // x² = x * x -> level L-1, scale D²/q_L
        auto prod = ctx_copy.mult(x.data[idx], x.data[idx]);
        auto relin = ctx_copy.relinearize(prod);
        x_sq[idx] = ctx_copy.rescale(relin, D / param_.get_q(level_) * D);
    });

    // Cross-block-column row sum: for each block-row bi, sum across all bj
    vector<CkksCiphertext> sum_x(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        sum_x[bi] = sum_x_per_block[bi].copy();  // bj=0: block index = bi + num_block_rows*0 = bi
        for (uint32_t bj = 1; bj < num_block_cols_; bj++) {
            uint32_t block_idx = bi + num_block_rows_ * bj;
            sum_x[bi] = ctx.add(sum_x[bi], sum_x_per_block[block_idx]);
        }
    }

    // --- Step 2: sum_x_sq per block-row ---
    vector<CkksCiphertext> sum_x_sq_per_block(total_blocks);
    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        sum_x_sq_per_block[idx] = intra_block_row_sum(ctx_copy, x_sq[idx]);
    });

    vector<CkksCiphertext> sum_x_sq_row(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        sum_x_sq_row[bi] = sum_x_sq_per_block[bi].copy();
        for (uint32_t bj = 1; bj < num_block_cols_; bj++) {
            uint32_t block_idx = bi + num_block_rows_ * bj;
            sum_x_sq_row[bi] = ctx.add(sum_x_sq_row[bi], sum_x_sq_per_block[block_idx]);
        }
    }

    // --- Step 3: mean = sum_x * pt(1/n_cols) -> level L-1, exact D ---
    vector<CkksCiphertext> mean_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_, level_);
        auto p = ctx_copy.mult_plain_mul(sum_x[bi], pt_inv_n);
        mean_cts[bi] = ctx_copy.rescale(p, D);
    });

    // --- Step 4: x_centered = drop(x) - mean, per block -> level L-1, scale D ---
    vector<CkksCiphertext> x_centered(total_blocks);
    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        uint32_t bi = idx % num_block_rows_;
        auto x_drop = ctx_copy.drop_level(x.data[idx]);
        x_centered[idx] = ctx_copy.sub(x_drop, mean_cts[bi]);
    });

    // --- Step 5: mean_sq and E_x_sq -> level L-2, scale D²/q_{L-1} ---
    vector<CkksCiphertext> E_x_sq(num_block_rows_);
    vector<CkksCiphertext> mean_sq(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto prod = ctx_copy.mult(mean_cts[bi], mean_cts[bi]);
        mean_sq[bi] = ctx_copy.rescale(ctx_copy.relinearize(prod), D / param_.get_q(level_ - 1) * D);

        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_pt_, level_ - 1);
        auto p = ctx_copy.mult_plain_mul(sum_x_sq_row[bi], pt_inv_n);
        E_x_sq[bi] = ctx_copy.rescale(p, D / param_.get_q(level_ - 1) * D);
    });

    // --- Step 6: var = E_x_sq - mean_sq ---
    vector<CkksCiphertext> var_cts(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        var_cts[bi] = ctx.sub(E_x_sq[bi], mean_sq[bi]);
    }

    // --- Step 7: a = var * inv_var + eps*inv_var ---
    vector<CkksCiphertext> a_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_iv = ctx_copy.ringt_to_mul(iv_pt_, level_ - 2);
        auto p = ctx_copy.mult_plain_mul(var_cts[bi], pt_iv);
        a_cts[bi] = ctx_copy.rescale(p, D);
        a_cts[bi] = ctx_copy.add_plain_ringt(a_cts[bi], eps_add_pt_);
    });

    return {move(a_cts), move(x_centered)};
}

// ============================================================
// BlockColMajorLNMinimaxInit
// ============================================================

BlockColMajorLNMinimaxInit::BlockColMajorLNMinimaxInit(const CkksParameter& param,
                                                       uint32_t block_size,
                                                       uint32_t input_level,
                                                       double c0,
                                                       double c1,
                                                       double c2)
    : Layer(param), c0_(c0), c1_(c1), c2_(c2) {
    level_ = input_level;
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
}

void BlockColMajorLNMinimaxInit::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);

    double c2_scale = q_L / D * q_L1;
    vector<double> c2_vec(n_slot_, c2_);
    c2_norm_pt_ = ctx.encode_ringt(c2_vec, c2_scale);

    vector<double> c1_vec(n_slot_, c1_);
    c1_pt_ = ctx.encode_ringt(c1_vec, q_L);

    // c0 for add_plain at scale D
    vector<double> c0_vec(n_slot_, c0_);
    c0_add_pt_ = ctx.encode_ringt(c0_vec, D);
}

vector<CkksCiphertext> BlockColMajorLNMinimaxInit::run(CkksContext& ctx, const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n_rows = a_cts.size();
    vector<CkksCiphertext> y_cts(n_rows);

    parallel_for(n_rows, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // a² = a * a -> level_-1, scale D²/q_{level_}
        auto a_sq_raw = ctx_copy.mult(a_cts[bi], a_cts[bi]);
        auto a_sq_relin = ctx_copy.relinearize(a_sq_raw);
        auto a_sq = ctx_copy.rescale(a_sq_relin, D / param_.get_q(level_) * D);
        // a_sq now at level_-1, scale D²/q_{level_}

        // c2*a² = normalizing pt*ct -> level_-2, scale D exact
        auto c2_mul = ctx_copy.ringt_to_mul(c2_norm_pt_, level_ - 1);
        auto c2a2_raw = ctx_copy.mult_plain_mul(a_sq, c2_mul);
        auto c2a2 = ctx_copy.rescale(c2a2_raw, D);
        // c2a2 at level_-2, scale D exact

        // c1*a = pt*ct preserving D -> level_-1, scale D exact
        auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_, level_);
        auto c1a_raw = ctx_copy.mult_plain_mul(a_cts[bi], c1_mul);
        auto c1a = ctx_copy.rescale(c1a_raw, D);
        // c1a at level_-1, scale D exact

        // Drop c1a to level_-2 to match c2a2
        auto c1a_drop = ctx_copy.drop_level(c1a);
        // c1a_drop at level_-2, scale D

        // y0 = c0 + c1*a + c2*a²  (all at level_-2, scale D)
        auto y0 = ctx_copy.add(c1a_drop, c2a2);
        y0 = ctx_copy.add_plain_ringt(y0, c0_add_pt_);

        y_cts[bi] = move(y0);
    });

    return y_cts;
}

// ============================================================
// BlockColMajorLNGoldschmidt
// ============================================================

BlockColMajorLNGoldschmidt::BlockColMajorLNGoldschmidt(const CkksParameter& param,
                                                       uint32_t block_size,
                                                       uint32_t input_level)
    : Layer(param) {
    level_ = input_level;
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
}

void BlockColMajorLNGoldschmidt::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);

    double three_scale = D / q_L * D / q_L1 * D;
    vector<double> three_vec(n_slot_, 3.0);
    three_pt_ = ctx.encode_ringt(three_vec, three_scale);

    double half_scale = q_L / D * q_L / D * q_L1 / D * q_L2;
    vector<double> half_vec(n_slot_, 0.5);
    half_norm_pt_ = ctx.encode_ringt(half_vec, half_scale);
}

vector<CkksCiphertext> BlockColMajorLNGoldschmidt::run(CkksContext& ctx,
                                                       const vector<CkksCiphertext>& y_cts,
                                                       const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n_rows = y_cts.size();
    vector<CkksCiphertext> y_new(n_rows);

    parallel_for(n_rows, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // Drop a to level_ to match y
        auto a_drop = a_cts[bi].copy();
        if (a_drop.get_level() > (int)level_) {
            a_drop = ctx_copy.drop_level(a_drop, a_drop.get_level() - (int)level_);
        }

        // Step 1 (parallel): y*a and y*y -> level_-1, scale S1 = D²/q_{L}
        auto ya_raw = ctx_copy.mult(y_cts[bi], a_drop);
        auto ya = ctx_copy.rescale(ctx_copy.relinearize(ya_raw), D / param_.get_q(level_) * D);

        auto yy_raw = ctx_copy.mult(y_cts[bi], y_cts[bi]);
        auto yy = ctx_copy.rescale(ctx_copy.relinearize(yy_raw), D / param_.get_q(level_) * D);

        // Step 2: (y*a)*(y*y) -> level_-2, scale S_prod = D⁴/(q_{L}²·q_{L-1})
        auto ya_yy_raw = ctx_copy.mult(ya, yy);
        double S_prod = D / param_.get_q(level_) * D / param_.get_q(level_) * D / param_.get_q(level_ - 1) * D;
        auto ya_yy = ctx_copy.rescale(ctx_copy.relinearize(ya_yy_raw), S_prod);

        // Step 3: 3*y (pt*ct) -> level_-1, scale S_prod
        auto three_mul = ctx_copy.ringt_to_mul(three_pt_, level_);
        auto three_y_raw = ctx_copy.mult_plain_mul(y_cts[bi], three_mul);
        auto three_y = ctx_copy.rescale(three_y_raw, S_prod);
        // Drop 3*y from level_-1 to level_-2 to match (y*a)*(y*y)
        auto three_y_drop = ctx_copy.drop_level(three_y);

        // Step 4: 3*y - (y*a)*(y*y) -> level_-2, scale S_prod
        auto diff = ctx_copy.sub(three_y_drop, ya_yy);

        // Step 5: 0.5 * diff (normalizing pt*ct -> exact D at level_-3)
        auto half_mul = ctx_copy.ringt_to_mul(half_norm_pt_, level_ - 2);
        auto half_raw = ctx_copy.mult_plain_mul(diff, half_mul);
        y_new[bi] = ctx_copy.rescale(half_raw, D);
    });

    return y_new;
}

// ============================================================
// BlockColMajorLNAffine
// ============================================================

BlockColMajorLNAffine::BlockColMajorLNAffine(const CkksParameter& param,
                                             Duo shape,
                                             uint32_t block_size,
                                             uint32_t y_level,
                                             double inv_std,
                                             Array<double, 1>&& gamma,
                                             Array<double, 1>&& beta)
    : Layer(param), y_level_(y_level), inv_std_(inv_std), gamma_vals_(move(gamma)), beta_vals_(move(beta)) {
    level_ = y_level;
    m_ = shape[0];
    n_ = shape[1];
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(n_, d_);
}

void BlockColMajorLNAffine::prepare_weight() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(y_level_);
    double q_L1 = param_.get_q(y_level_ - 1);

    gamma_pt_.resize(num_block_cols_);
    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        vector<double> gamma_vec(n_slot_, 0.0);
        for (uint32_t c = 0; c < num_chunks_; c++) {
            for (uint32_t col = 0; col < d_; col++) {
                uint32_t actual_col = bj * d_ + col;
                for (uint32_t row = 0; row < d_; row++) {
                    uint32_t slot = c * chunk_size_ + row + d_ * col;
                    if (actual_col < n_) {
                        gamma_vec[slot] = inv_std_ * gamma_vals_.get(actual_col);
                    }
                }
            }
        }
        gamma_pt_[bj] = ctx.encode_ringt(gamma_vec, q_L / D * q_L1);
    }

    beta_add_pt_.resize(num_block_rows_ * num_block_cols_);
    double beta_scale = D;
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            vector<double> beta_vec(n_slot_, 0.0);
            for (uint32_t c = 0; c < num_chunks_; c++) {
                for (uint32_t col = 0; col < d_; col++) {
                    uint32_t actual_col = bj * d_ + col;
                    for (uint32_t row = 0; row < d_; row++) {
                        uint32_t actual_row = bi * d_ + row;
                        uint32_t slot = c * chunk_size_ + row + d_ * col;
                        if (actual_row < m_ && actual_col < n_) {
                            beta_vec[slot] = beta_vals_.get(actual_col);
                        }
                    }
                }
            }
            uint32_t idx = bi + num_block_rows_ * bj;
            beta_add_pt_[idx] = ctx.encode_ringt(beta_vec, beta_scale);
        }
    }
}

FeatureMatEncrypted BlockColMajorLNAffine::run(CkksContext& ctx,
                                               const vector<CkksCiphertext>& x_centered,
                                               const vector<CkksCiphertext>& y_cts) {
    double D = param_.get_default_scale();
    uint32_t total_blocks = num_block_rows_ * num_block_cols_;

    FeatureMatEncrypted result(&ctx, y_level_ - 2);
    result.shape = {m_, n_};
    result.matmul_block_size = d_;
    result.data.resize(total_blocks);

    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int block_idx) {
        uint32_t bi = block_idx % num_block_rows_;
        uint32_t bj = block_idx / num_block_rows_;

        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[bj], y_level_);
        auto yw = ctx_copy.rescale(ctx_copy.mult_plain_mul(y_cts[bi], gamma_mul), param_.get_q(y_level_ - 1));

        auto xc = x_centered[block_idx].copy();
        if (xc.get_level() > (int)(y_level_ - 1)) {
            xc = ctx_copy.drop_level(xc, xc.get_level() - (int)(y_level_ - 1));
        }

        auto out = ctx_copy.rescale(ctx_copy.relinearize(ctx_copy.mult(xc, yw)), D);

        uint32_t beta_idx = bi + num_block_rows_ * bj;
        result.data[block_idx] = ctx_copy.add_plain_ringt(out, beta_add_pt_[beta_idx]);
    });

    result.level = y_level_ - 2;
    return result;
}
