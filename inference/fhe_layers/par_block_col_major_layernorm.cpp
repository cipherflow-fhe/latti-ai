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

#include "par_block_col_major_layernorm.h"
#include <cassert>
#include <cmath>

using namespace std;
using namespace lattisense;

static uint32_t next_power_of_2(uint32_t x) {
    uint32_t p = 1;
    while (p < x)
        p *= 2;
    return p;
}

// ============================================================
// ParBlockColMajorLNStats
// ============================================================

ParBlockColMajorLNStats::ParBlockColMajorLNStats(const CkksParameter& param,
                                                 Duo shape,
                                                 uint32_t block_size,
                                                 uint32_t n_heads,
                                                 uint32_t init_level,
                                                 double eps,
                                                 double inv_var_scale)
    : Layer(param), eps_(eps), inv_var_scale_(inv_var_scale) {
    level_ = init_level;
    m_ = shape[0];
    cols_per_head_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    n_h_padded_ = next_power_of_2(n_heads);
    n_slot_ = param_.get_n() / 2;
    total_dim_ = n_heads_ * cols_per_head_;

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

void ParBlockColMajorLNStats::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);
    double q_L3 = param_.get_q(level_ - 3);
    double N = static_cast<double>(total_dim_);

    // h=0 mask for cross-head sum cleanup (applied to sum_x at level L)
    // Encode at q_L so that pt*ct preserves scale: D * q_L / q_L = D
    vector<double> h0_mask(n_slot_, 0.0);
    for (uint32_t i = 0; i < (uint32_t)n_slot_; i++) {
        // h_local = i % S_ (within each chunk)
        uint32_t pos_in_chunk = i % chunk_size_;
        uint32_t h_local = pos_in_chunk % S_;
        if (h_local == 0)
            h0_mask[i] = 1.0;
    }
    h0_mask_pt_ = ctx.encode_ringt(h0_mask, q_L);

    // Same mask but for sum_x_sq at level L-1 (scale D^2/q_L after squaring)
    // Encode at q_{L-1} so that pt*ct preserves the input scale:
    //   (D^2/q_L) * q_{L-1} / q_{L-1} = D^2/q_L
    h0_mask_sq_pt_ = ctx.encode_ringt(h0_mask, q_L1);

    // Normalizing pt*ct for E_x_sq: input at level L-2, scale D^2/q_L
    // Need output D: encode scale = q_L/D * q_{L-2}
    //   proof: (D^2/q_L) * (q_L/D * q_{L-2}) / q_{L-2} = D
    double norm_scale = q_L / D * q_L2;
    vector<double> inv_n_vec(n_slot_, 1.0 / N);
    inv_n_norm_pt_ = ctx.encode_ringt(inv_n_vec, norm_scale);

    // Normalizing pt*ct for mean_sq: input at level L-2, scale D^2/q_{L-1}
    // Need output D: encode scale = q_{L-1}/D * q_{L-2}
    double norm_sq_scale = q_L1 / D * q_L2;
    vector<double> inv_n_sq_vec(n_slot_, 1.0 / (N * N));
    inv_n_sq_norm_pt_ = ctx.encode_ringt(inv_n_sq_vec, norm_sq_scale);

    // mean = sum_x * pt(1/N): sum_x at level L-1, scale D
    // Encode at q_{L-1} to preserve D
    inv_n_mean_pt_ = ctx.encode_ringt(inv_n_vec, q_L1);

    // inv_var_scale: var at level L-3, scale D
    // Encode at q_{L-3}
    vector<double> ivs_vec(n_slot_, inv_var_scale_);
    inv_var_scale_pt_ = ctx.encode_ringt(ivs_vec, q_L3);

    // eps add: a at level L-4, scale D
    vector<double> eps_vec(n_slot_, eps_ * inv_var_scale_);
    eps_add_pt_ = ctx.encode_ringt(eps_vec, D);
}

CkksCiphertext ParBlockColMajorLNStats::intra_block_col_sum(CkksContext& ctx, const CkksCiphertext& ct) const {
    // Par format slot: (row + d*col)*S + h_local
    // Column sum only — rotate by d*S, 2*d*S, ..., (d/2)*d*S
    CkksCiphertext result = ct.copy();
    for (uint32_t step = 1; step < d_; step *= 2) {
        int rot = step * d_ * S_;
        auto rotated = ctx.rotate(result, rot);
        result = ctx.add(result, rotated);
    }
    return result;
}

CkksCiphertext ParBlockColMajorLNStats::cross_head_sum_masked(CkksContext& ctx,
                                                              const CkksCiphertext& col_summed,
                                                              const CkksPlaintextRingt& mask_pt) const {
    // After column sum, all col positions for same (row, head) are identical.
    // To sum across S interleaved heads: rotate by +h for h=1..S-1 and add.
    // rotate(ct, +h): output[i] = input[i+h], so h=0 gets head h's value.
    // Only h=0 gets the correct total (h!=0 corrupted by cross-row wrap).
    CkksCiphertext summed = col_summed.copy();
    for (uint32_t h = 1; h < S_; h++) {
        auto rotated = ctx.rotate(col_summed, (int)h);
        summed = ctx.add(summed, rotated);
    }
    // Now h=0 has correct total, h!=0 has garbage.
    // Mask: keep only h=0 (pt*ct, costs 1 level)
    auto mask_mul = ctx.ringt_to_mul(mask_pt, summed.get_level());
    auto masked = ctx.mult_plain_mul(summed, mask_mul);
    double D = param_.get_default_scale();
    auto masked_rescaled = ctx.rescale(masked, D);
    // Replicate h=0 to all h positions: rotate ORIGINAL masked ct, not accumulated
    CkksCiphertext result = masked_rescaled.copy();
    for (uint32_t offset = 1; offset < S_; offset++) {
        auto rotated = ctx.rotate(masked_rescaled, -(int)offset);
        result = ctx.add(result, rotated);
    }
    return result;
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>> ParBlockColMajorLNStats::run(CkksContext& ctx,
                                                                                  const FeatureMatEncrypted& x) {
    double D = param_.get_default_scale();
    uint32_t total_vecs = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;

    // Step 1: Column sum + x_sq for all ciphertexts
    vector<CkksCiphertext> col_sum_per_vec(total_vecs);
    vector<CkksCiphertext> x_sq(total_vecs);

    parallel_for(total_vecs, th_nums, ctx, [&](CkksContext& ctx_copy, int vec_idx) {
        col_sum_per_vec[vec_idx] = intra_block_col_sum(ctx_copy, x.data[vec_idx]);
        auto prod = ctx_copy.mult(x.data[vec_idx], x.data[vec_idx]);
        auto relin = ctx_copy.relinearize(prod);
        x_sq[vec_idx] = ctx_copy.rescale(relin, D / param_.get_q(level_) * D);
    });

    // Step 2: Cross-block-col and cross-group sums of column-summed data
    vector<CkksCiphertext> col_sum_x(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        bool first = true;
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t vec_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                if (first) {
                    col_sum_x[bi] = col_sum_per_vec[vec_idx].copy();
                    first = false;
                } else {
                    col_sum_x[bi] = ctx.add(col_sum_x[bi], col_sum_per_vec[vec_idx]);
                }
            }
        }
    }

    // Step 3: Cross-head sum + mask + replicate for sum_x
    // col_sum_x[bi] at level L, scale D -> after mask: level L-1, scale D
    vector<CkksCiphertext> sum_x(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        sum_x[bi] = cross_head_sum_masked(ctx_copy, col_sum_x[bi], h0_mask_pt_);
    });
    // sum_x at level L-1, scale D

    // Step 4: Column sum + cross-block/cross-group sum of x_sq
    vector<CkksCiphertext> col_sum_x_sq_per_vec(total_vecs);
    parallel_for(total_vecs, th_nums, ctx, [&](CkksContext& ctx_copy, int vec_idx) {
        col_sum_x_sq_per_vec[vec_idx] = intra_block_col_sum(ctx_copy, x_sq[vec_idx]);
    });

    vector<CkksCiphertext> col_sum_x_sq(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        bool first = true;
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                uint32_t vec_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                if (first) {
                    col_sum_x_sq[bi] = col_sum_x_sq_per_vec[vec_idx].copy();
                    first = false;
                } else {
                    col_sum_x_sq[bi] = ctx.add(col_sum_x_sq[bi], col_sum_x_sq_per_vec[vec_idx]);
                }
            }
        }
    }

    // Step 5: Cross-head sum + mask + replicate for sum_x_sq
    // col_sum_x_sq[bi] at level L-1, scale D^2/q_L -> after mask: level L-2, scale D^2/q_L
    vector<CkksCiphertext> sum_x_sq_row(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        sum_x_sq_row[bi] = cross_head_sum_masked(ctx_copy, col_sum_x_sq[bi], h0_mask_sq_pt_);
    });
    // sum_x_sq_row at level L-2, scale D^2/q_L

    // Step 6: sum_x_sq2 = sum_x * sum_x
    // sum_x at level L-1, scale D. ct*ct -> level L-2, scale D^2/q_{L-1}
    vector<CkksCiphertext> sum_x_sq2(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto prod = ctx_copy.mult(sum_x[bi], sum_x[bi]);
        auto relin = ctx_copy.relinearize(prod);
        sum_x_sq2[bi] = ctx_copy.rescale(relin, D / param_.get_q(level_ - 1) * D);
    });
    // sum_x_sq2 at level L-2, scale D^2/q_{L-1}

    // Step 7: Normalizing pt*ct
    // E_x_sq: sum_x_sq_row(L-2, D^2/q_L) * pt -> L-3, D exact
    // mean_sq: sum_x_sq2(L-2, D^2/q_{L-1}) * pt -> L-3, D exact
    vector<CkksCiphertext> E_x_sq(num_block_rows_);
    vector<CkksCiphertext> mean_sq(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_norm_pt_, level_ - 2);
        auto p1 = ctx_copy.mult_plain_mul(sum_x_sq_row[bi], pt_inv_n);
        E_x_sq[bi] = ctx_copy.rescale(p1, D);

        auto pt_inv_n_sq = ctx_copy.ringt_to_mul(inv_n_sq_norm_pt_, level_ - 2);
        auto p2 = ctx_copy.mult_plain_mul(sum_x_sq2[bi], pt_inv_n_sq);
        mean_sq[bi] = ctx_copy.rescale(p2, D);
    });

    // Step 8: var = E_x_sq - mean_sq (both at L-3, exact D)
    vector<CkksCiphertext> var_cts(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        var_cts[bi] = ctx.sub(E_x_sq[bi], mean_sq[bi]);
    }

    // Step 9: mean = sum_x * pt(1/N)
    // sum_x at L-1, scale D -> L-2, exact D
    vector<CkksCiphertext> mean_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n_mean = ctx_copy.ringt_to_mul(inv_n_mean_pt_, level_ - 1);
        auto p = ctx_copy.mult_plain_mul(sum_x[bi], pt_inv_n_mean);
        mean_cts[bi] = ctx_copy.rescale(p, D);
    });
    // mean at L-2, exact D

    // Step 10: a = var * inv_var_scale + eps
    // var at L-3, scale D -> L-4, exact D
    vector<CkksCiphertext> a_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_ivs = ctx_copy.ringt_to_mul(inv_var_scale_pt_, level_ - 3);
        auto p = ctx_copy.mult_plain_mul(var_cts[bi], pt_ivs);
        a_cts[bi] = ctx_copy.rescale(p, D);
        a_cts[bi] = ctx_copy.add_plain_ringt(a_cts[bi], eps_add_pt_);
    });
    // a at L-4, exact D

    return {move(a_cts), move(mean_cts)};
}

// ============================================================
// ParBlockColMajorLNMinimaxInit (same logic as block version)
// ============================================================

ParBlockColMajorLNMinimaxInit::ParBlockColMajorLNMinimaxInit(const CkksParameter& param,
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

void ParBlockColMajorLNMinimaxInit::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_inp = param_.get_q(level_);
    double q_inp_m1 = param_.get_q(level_ - 1);

    double c2_scale = q_inp / D * q_inp_m1;
    vector<double> c2_vec(n_slot_, c2_);
    c2_norm_pt_ = ctx.encode_ringt(c2_vec, c2_scale);

    vector<double> c1_vec(n_slot_, c1_);
    c1_pt_ = ctx.encode_ringt(c1_vec, q_inp);

    vector<double> c0_vec(n_slot_, c0_);
    c0_add_pt_ = ctx.encode_ringt(c0_vec, D);
}

vector<CkksCiphertext> ParBlockColMajorLNMinimaxInit::run(CkksContext& ctx, const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n = a_cts.size();
    vector<CkksCiphertext> y_cts(n);

    parallel_for(n, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto a_sq_raw = ctx_copy.mult(a_cts[bi], a_cts[bi]);
        auto a_sq = ctx_copy.rescale(ctx_copy.relinearize(a_sq_raw), D / param_.get_q(level_) * D);

        auto c2_mul = ctx_copy.ringt_to_mul(c2_norm_pt_, level_ - 1);
        auto c2a2 = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_sq, c2_mul), D);

        auto c1_mul = ctx_copy.ringt_to_mul(c1_pt_, level_);
        auto c1a = ctx_copy.rescale(ctx_copy.mult_plain_mul(a_cts[bi], c1_mul), D);

        auto c1a_drop = ctx_copy.drop_level(c1a);

        auto y0 = ctx_copy.add(c1a_drop, c2a2);
        y0 = ctx_copy.add_plain_ringt(y0, c0_add_pt_);

        y_cts[bi] = move(y0);
    });

    return y_cts;
}

// ============================================================
// ParBlockColMajorLNGoldschmidt (same logic as block version)
// ============================================================

ParBlockColMajorLNGoldschmidt::ParBlockColMajorLNGoldschmidt(const CkksParameter& param,
                                                             uint32_t block_size,
                                                             uint32_t input_level)
    : Layer(param) {
    level_ = input_level;
    d_ = block_size;
    n_slot_ = param_.get_n() / 2;
    chunk_size_ = d_ * d_;
}

void ParBlockColMajorLNGoldschmidt::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_Ly = param_.get_q(level_);
    double q_Ly1 = param_.get_q(level_ - 1);
    double q_Ly2 = param_.get_q(level_ - 2);
    double q_Ly3 = param_.get_q(level_ - 3);

    double S2 = D / q_Ly * D / q_Ly1 * D;
    vector<double> three_vec(n_slot_, 3.0);
    three_add_pt_ = ctx.encode_ringt(three_vec, S2);

    double half_scale = q_Ly / D * q_Ly1 / D * q_Ly2 / D * q_Ly3;
    vector<double> half_vec(n_slot_, 0.5);
    half_norm_pt_ = ctx.encode_ringt(half_vec, half_scale);
}

vector<CkksCiphertext> ParBlockColMajorLNGoldschmidt::run(CkksContext& ctx,
                                                          const vector<CkksCiphertext>& y_cts,
                                                          const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n = y_cts.size();
    vector<CkksCiphertext> y_new(n);

    parallel_for(n, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto y_sq_raw = ctx_copy.mult(y_cts[bi], y_cts[bi]);
        auto y_sq = ctx_copy.rescale(ctx_copy.relinearize(y_sq_raw), D / param_.get_q(level_) * D);

        auto a_drop = a_cts[bi].copy();
        while (a_drop.get_level() > (int)(level_ - 1)) {
            a_drop = ctx_copy.drop_level(a_drop);
        }
        auto ay_sq_raw = ctx_copy.mult(a_drop, y_sq);
        auto ay_sq = ctx_copy.rescale(ctx_copy.relinearize(ay_sq_raw), D / param_.get_q(level_ - 1) * D);

        auto neg_ay_sq = ctx_copy.negate(ay_sq);
        auto three_minus = ctx_copy.add_plain_ringt(neg_ay_sq, three_add_pt_);

        auto y_drop = y_cts[bi].copy();
        while (y_drop.get_level() > (int)(level_ - 2)) {
            y_drop = ctx_copy.drop_level(y_drop);
        }
        auto prod_raw = ctx_copy.mult(y_drop, three_minus);
        auto prod = ctx_copy.rescale(ctx_copy.relinearize(prod_raw), D / param_.get_q(level_ - 2) * D);

        auto half_mul = ctx_copy.ringt_to_mul(half_norm_pt_, level_ - 3);
        auto half_raw = ctx_copy.mult_plain_mul(prod, half_mul);
        y_new[bi] = ctx_copy.rescale(half_raw, D);
    });

    return y_new;
}

// ============================================================
// ParBlockColMajorLNAffine
// ============================================================

ParBlockColMajorLNAffine::ParBlockColMajorLNAffine(const CkksParameter& param,
                                                   Duo shape,
                                                   uint32_t block_size,
                                                   uint32_t n_heads,
                                                   uint32_t init_level,
                                                   uint32_t y_level,
                                                   double inv_std_scale,
                                                   const vector<double>& gamma,
                                                   const vector<double>& beta)
    : Layer(param), init_level_(init_level), y_level_(y_level), inv_std_scale_(inv_std_scale), gamma_vals_(gamma),
      beta_vals_(beta) {
    level_ = y_level;
    m_ = shape[0];
    cols_per_head_ = shape[1];
    d_ = block_size;
    n_heads_ = n_heads;
    n_h_padded_ = next_power_of_2(n_heads);
    n_slot_ = param_.get_n() / 2;

    if ((uint32_t)n_slot_ >= n_h_padded_ * d_ * d_) {
        S_ = n_h_padded_;
        chunk_size_ = n_h_padded_ * d_ * d_;
        n_cts_per_block_idx_ = 1;
    } else {
        S_ = n_slot_ / (d_ * d_);
        chunk_size_ = n_slot_;
        n_cts_per_block_idx_ = n_h_padded_ / S_;
    }
    num_chunks_ = n_slot_ / chunk_size_;
    num_block_rows_ = div_ceil(m_, d_);
    num_block_cols_ = div_ceil(cols_per_head_, d_);
}

void ParBlockColMajorLNAffine::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_Lout = param_.get_q(y_level_);
    double q_Lout_m1 = param_.get_q(y_level_ - 1);
    double beta_scale = D / q_Lout_m1 * D;

    uint32_t n_gamma_vecs = num_block_cols_ * n_cts_per_block_idx_;
    gamma_pt_.resize(n_gamma_vecs);

    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
            vector<double> gamma_vec(n_slot_, 0.0);
            for (uint32_t h_local = 0; h_local < S_; h_local++) {
                uint32_t h = g * S_ + h_local;
                for (uint32_t col = 0; col < d_; col++) {
                    uint32_t actual_col = bj * d_ + col;
                    for (uint32_t row = 0; row < d_; row++) {
                        uint32_t base_slot = (row + d_ * col) * S_ + h_local;
                        for (uint32_t ci = 0; ci < num_chunks_; ci++) {
                            uint32_t slot = ci * chunk_size_ + base_slot;
                            if (h < n_heads_ && actual_col < cols_per_head_) {
                                gamma_vec[slot] = inv_std_scale_ * gamma_vals_[h * cols_per_head_ + actual_col];
                            }
                        }
                    }
                }
            }
            uint32_t idx = bj * n_cts_per_block_idx_ + g;
            gamma_pt_[idx] = ctx.encode_ringt(gamma_vec, q_Lout);
        }
    }

    // beta: per (bi, bj, g) to handle row masking
    uint32_t total_beta = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;
    beta_add_pt_.resize(total_beta);

    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
            for (uint32_t g = 0; g < n_cts_per_block_idx_; g++) {
                vector<double> beta_vec(n_slot_, 0.0);
                for (uint32_t h_local = 0; h_local < S_; h_local++) {
                    uint32_t h = g * S_ + h_local;
                    for (uint32_t col = 0; col < d_; col++) {
                        uint32_t actual_col = bj * d_ + col;
                        for (uint32_t row = 0; row < d_; row++) {
                            uint32_t actual_row = bi * d_ + row;
                            uint32_t base_slot = (row + d_ * col) * S_ + h_local;
                            for (uint32_t ci = 0; ci < num_chunks_; ci++) {
                                uint32_t slot = ci * chunk_size_ + base_slot;
                                if (actual_row < m_ && h < n_heads_ && actual_col < cols_per_head_) {
                                    beta_vec[slot] = beta_vals_[h * cols_per_head_ + actual_col];
                                }
                            }
                        }
                    }
                }
                uint32_t beta_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
                beta_add_pt_[beta_idx] = ctx.encode_ringt(beta_vec, beta_scale);
            }
        }
    }
}

FeatureMatEncrypted ParBlockColMajorLNAffine::run(CkksContext& ctx,
                                                  const FeatureMatEncrypted& x,
                                                  const vector<CkksCiphertext>& mean_cts,
                                                  const vector<CkksCiphertext>& y_cts) {
    double D = param_.get_default_scale();
    uint32_t total_vecs = num_block_rows_ * num_block_cols_ * n_cts_per_block_idx_;

    FeatureMatEncrypted result(&ctx, y_level_ - 2);
    result.shape = {m_, cols_per_head_};
    result.matmul_block_size = d_;
    result.data.resize(total_vecs);

    parallel_for(total_vecs, th_nums, ctx, [&](CkksContext& ctx_copy, int vec_idx) {
        uint32_t block_idx = vec_idx / n_cts_per_block_idx_;
        uint32_t g = vec_idx % n_cts_per_block_idx_;
        uint32_t bi = block_idx % num_block_rows_;
        uint32_t bj = block_idx / num_block_rows_;

        // y_weighted = y[bi] * gamma_pt[bj, g]
        uint32_t gamma_idx = bj * n_cts_per_block_idx_ + g;
        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[gamma_idx], y_level_);
        auto yw_raw = ctx_copy.mult_plain_mul(y_cts[bi], gamma_mul);
        auto yw = ctx_copy.rescale(yw_raw, D);

        // x_centered = drop(x) - drop(mean)
        auto x_drop = x.data[vec_idx].copy();
        while (x_drop.get_level() > (int)(y_level_ - 1)) {
            x_drop = ctx_copy.drop_level(x_drop);
        }
        auto mean_drop = mean_cts[bi].copy();
        while (mean_drop.get_level() > (int)(y_level_ - 1)) {
            mean_drop = ctx_copy.drop_level(mean_drop);
        }
        auto x_centered = ctx_copy.sub(x_drop, mean_drop);

        // output = x_centered * y_weighted
        auto out_raw = ctx_copy.mult(x_centered, yw);
        auto out = ctx_copy.rescale(ctx_copy.relinearize(out_raw), D / param_.get_q(y_level_ - 1) * D);

        // + beta
        uint32_t beta_idx = (bi + num_block_rows_ * bj) * n_cts_per_block_idx_ + g;
        result.data[vec_idx] = ctx_copy.add_plain_ringt(out, beta_add_pt_[beta_idx]);
    });

    result.level = y_level_ - 2;
    return result;
}
