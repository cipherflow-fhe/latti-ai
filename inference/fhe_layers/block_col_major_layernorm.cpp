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
                                           double inv_var_scale)
    : Layer(param), eps_(eps), inv_var_scale_(inv_var_scale) {
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

void BlockColMajorLNStats::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_L = param_.get_q(level_);
    double q_L1 = param_.get_q(level_ - 1);
    double q_L2 = param_.get_q(level_ - 2);

    double N = static_cast<double>(n_);

    // 1/N for normalizing ct*ct result (D²/q_L -> D):
    //   scale = q_L / D * q_{L-1}
    //   proof: (D²/q_L) * (q_L/D * q_{L-1}) / q_{L-1} = D
    double norm_scale = q_L / D * q_L1;
    vector<double> inv_n_vec(n_slot_, 1.0 / N);
    inv_n_norm_pt_ = ctx.encode_ringt(inv_n_vec, norm_scale);

    vector<double> inv_n_sq_vec(n_slot_, 1.0 / (N * N));
    inv_n_sq_norm_pt_ = ctx.encode_ringt(inv_n_sq_vec, norm_scale);

    // 1/N for mean (preserving D):
    //   scale = q_L
    //   proof: D * q_L / q_L = D
    inv_n_mean_pt_ = ctx.encode_ringt(inv_n_vec, q_L);

    // inv_var_scale for a = var * inv_var_scale:
    //   scale = q_{L-2}
    //   proof: D * q_{L-2} / q_{L-2} = D
    vector<double> ivs_vec(n_slot_, inv_var_scale_);
    inv_var_scale_pt_ = ctx.encode_ringt(ivs_vec, q_L2);

    // eps * inv_var_scale for add_plain:
    //   scale = D (matching ciphertext after pt*ct)
    vector<double> eps_vec(n_slot_, eps_ * inv_var_scale_);
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

    // --- Step 3: sum_x_sq2 = sum_x * sum_x (ct*ct, same chain as x_sq -> same actual scale) ---
    vector<CkksCiphertext> sum_x_sq2(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto prod = ctx_copy.mult(sum_x[bi], sum_x[bi]);
        auto relin = ctx_copy.relinearize(prod);
        sum_x_sq2[bi] = ctx_copy.rescale(relin, D / param_.get_q(level_) * D);
    });

    // --- Step 4: Normalizing pt*ct to get E_x_sq and mean_sq at exact scale D ---
    // E_x_sq = sum_x_sq_row * pt(1/N, normalizing)     -> level L-2, scale D exact
    // mean_sq = sum_x_sq2 * pt(1/N², normalizing)       -> level L-2, scale D exact
    vector<CkksCiphertext> E_x_sq(num_block_rows_);
    vector<CkksCiphertext> mean_sq(num_block_rows_);

    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // E[x²] = sum_x_sq / N
        auto pt_inv_n = ctx_copy.ringt_to_mul(inv_n_norm_pt_, level_ - 1);
        auto p1 = ctx_copy.mult_plain_mul(sum_x_sq_row[bi], pt_inv_n);
        E_x_sq[bi] = ctx_copy.rescale(p1, D);

        // E[x]² = (sum_x)² / N²
        auto pt_inv_n_sq = ctx_copy.ringt_to_mul(inv_n_sq_norm_pt_, level_ - 1);
        auto p2 = ctx_copy.mult_plain_mul(sum_x_sq2[bi], pt_inv_n_sq);
        mean_sq[bi] = ctx_copy.rescale(p2, D);
    });

    // --- Step 5: var = E_x_sq - mean_sq (both at level L-2, exact scale D) ---
    vector<CkksCiphertext> var_cts(num_block_rows_);
    for (uint32_t bi = 0; bi < num_block_rows_; bi++) {
        var_cts[bi] = ctx.sub(E_x_sq[bi], mean_sq[bi]);
    }

    // --- Step 6: mean = sum_x * pt(1/N, scale=q_L) -> level L-1, exact D ---
    vector<CkksCiphertext> mean_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_inv_n_mean = ctx_copy.ringt_to_mul(inv_n_mean_pt_, level_);
        auto p = ctx_copy.mult_plain_mul(sum_x[bi], pt_inv_n_mean);
        mean_cts[bi] = ctx_copy.rescale(p, D);
    });

    // --- Step 7: a = var * inv_var_scale + eps*inv_var_scale ---
    //   pt*ct on var(L-2, D): encode at q_{L-2} -> exact D at L-3
    vector<CkksCiphertext> a_cts(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        auto pt_ivs = ctx_copy.ringt_to_mul(inv_var_scale_pt_, level_ - 2);
        auto p = ctx_copy.mult_plain_mul(var_cts[bi], pt_ivs);
        a_cts[bi] = ctx_copy.rescale(p, D);
        // add eps * inv_var_scale
        a_cts[bi] = ctx_copy.add_plain_ringt(a_cts[bi], eps_add_pt_);
    });

    return {move(a_cts), move(mean_cts)};
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

void BlockColMajorLNMinimaxInit::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_inp = param_.get_q(level_);         // q at input_level (= L-3)
    double q_inp_m1 = param_.get_q(level_ - 1);  // q at input_level-1 (= L-4)

    // c2 normalizing: a² at scale D²/q_inp, pt*ct to get D at level_-2
    //   encode scale = q_inp / D * q_inp_m1
    //   proof: (D²/q_inp) * (q_inp/D * q_inp_m1) / q_inp_m1 = D
    double c2_scale = q_inp / D * q_inp_m1;
    vector<double> c2_vec(n_slot_, c2_);
    c2_norm_pt_ = ctx.encode_ringt(c2_vec, c2_scale);

    // c1 preserving D: encode at q_inp
    //   proof: D * q_inp / q_inp = D
    vector<double> c1_vec(n_slot_, c1_);
    c1_pt_ = ctx.encode_ringt(c1_vec, q_inp);

    // c0 for add_plain at scale D
    vector<double> c0_vec(n_slot_, c0_);
    c0_add_pt_ = ctx.encode_ringt(c0_vec, D);
}

vector<CkksCiphertext> BlockColMajorLNMinimaxInit::run(CkksContext& ctx, const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n = a_cts.size();
    vector<CkksCiphertext> y_cts(n);

    parallel_for(n, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
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

void BlockColMajorLNGoldschmidt::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_Ly = param_.get_q(level_);
    double q_Ly1 = param_.get_q(level_ - 1);
    double q_Ly2 = param_.get_q(level_ - 2);
    double q_Ly3 = param_.get_q(level_ - 3);

    // "3" for add_plain at scale S2 = D³/(q_{L_y}·q_{L_y-1})
    //   divide-first: S2 = D/q_{L_y} * D/q_{L_y-1} * D
    double S2 = D / q_Ly * D / q_Ly1 * D;
    vector<double> three_vec(n_slot_, 3.0);
    three_add_pt_ = ctx.encode_ringt(three_vec, S2);

    // 0.5 normalizing: convert S3 back to D
    //   S3 = D⁴/(q_{L_y}·q_{L_y-1}·q_{L_y-2})
    //   encode scale = D·q_{L_y-3}/S3 = q_{L_y}/D * q_{L_y-1}/D * q_{L_y-2}/D * q_{L_y-3}
    double half_scale = q_Ly / D * q_Ly1 / D * q_Ly2 / D * q_Ly3;
    vector<double> half_vec(n_slot_, 0.5);
    half_norm_pt_ = ctx.encode_ringt(half_vec, half_scale);
}

vector<CkksCiphertext> BlockColMajorLNGoldschmidt::run(CkksContext& ctx,
                                                       const vector<CkksCiphertext>& y_cts,
                                                       const vector<CkksCiphertext>& a_cts) {
    double D = param_.get_default_scale();
    uint32_t n = y_cts.size();
    vector<CkksCiphertext> y_new(n);

    parallel_for(n, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // Step 1: y² = y*y -> level_-1, scale S1 = D²/q_{L_y}
        auto y_sq_raw = ctx_copy.mult(y_cts[bi], y_cts[bi]);
        auto y_sq = ctx_copy.rescale(ctx_copy.relinearize(y_sq_raw), D / param_.get_q(level_) * D);

        // Step 2: a*y² -> level_-2, scale S2 = D·S1/q_{L_y-1} = D³/(q_{L_y}·q_{L_y-1})
        // Drop a to level_-1
        auto a_drop = a_cts[bi].copy();
        while (a_drop.get_level() > (int)(level_ - 1)) {
            a_drop = ctx_copy.drop_level(a_drop);
        }
        auto ay_sq_raw = ctx_copy.mult(a_drop, y_sq);
        auto ay_sq = ctx_copy.rescale(ctx_copy.relinearize(ay_sq_raw), D / param_.get_q(level_ - 1) * D);

        // Step 3: 3 - a*y²
        // negate(ay_sq) + add_plain(3, scale=S2)
        auto neg_ay_sq = ctx_copy.negate(ay_sq);
        auto three_minus = ctx_copy.add_plain_ringt(neg_ay_sq, three_add_pt_);

        // Step 4: y * (3 - a*y²) -> level_-3, scale S3
        // Drop y to level_-2
        auto y_drop = y_cts[bi].copy();
        while (y_drop.get_level() > (int)(level_ - 2)) {
            y_drop = ctx_copy.drop_level(y_drop);
        }
        auto prod_raw = ctx_copy.mult(y_drop, three_minus);
        auto prod = ctx_copy.rescale(ctx_copy.relinearize(prod_raw), D / param_.get_q(level_ - 2) * D);

        // Step 5: 0.5 * result (normalizing pt*ct -> exact D)
        auto half_mul = ctx_copy.ringt_to_mul(half_norm_pt_, level_ - 3);
        auto half_raw = ctx_copy.mult_plain_mul(prod, half_mul);
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
                                             uint32_t init_level,
                                             uint32_t y_level,
                                             double inv_std_scale,
                                             const vector<double>& gamma,
                                             const vector<double>& beta)
    : Layer(param), init_level_(init_level), y_level_(y_level), inv_std_scale_(inv_std_scale), gamma_vals_(gamma),
      beta_vals_(beta) {
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

void BlockColMajorLNAffine::precompute_plaintexts() {
    CkksContext ctx = CkksContext::create_empty_context(param_);
    double D = param_.get_default_scale();
    double q_Lout = param_.get_q(y_level_);
    double q_Lout_m1 = param_.get_q(y_level_ - 1);

    gamma_pt_.resize(num_block_cols_);
    beta_add_pt_.resize(num_block_cols_);

    for (uint32_t bj = 0; bj < num_block_cols_; bj++) {
        // Build gamma * inv_std_scale vector for this block-column
        // Slot layout: slot[row + d*col] corresponds to matrix[bi*d+row, bj*d+col]
        // gamma is per-column, same for all rows. 0 at invalid positions.
        vector<double> gamma_vec(n_slot_, 0.0);
        vector<double> beta_vec(n_slot_, 0.0);

        for (uint32_t c = 0; c < num_chunks_; c++) {
            for (uint32_t col = 0; col < d_; col++) {
                uint32_t actual_col = bj * d_ + col;
                for (uint32_t row = 0; row < d_; row++) {
                    uint32_t slot = c * chunk_size_ + row + d_ * col;
                    // For gamma_pt: need to check ALL block-rows for this block-col.
                    // Since gamma is the same for all rows, we only check column validity.
                    // Row validity is checked per block-row at runtime... but since
                    // gamma_pt is broadcast to all block-rows, we need to handle row
                    // masking differently.
                    //
                    // Actually, for block format, each y_cts[bi] is multiplied by
                    // gamma_pt[bj]. The y_cts[bi] already has a per-row value.
                    // gamma_pt[bj] is the same plaintext used for all bi.
                    // So we need gamma_pt to be 0 only for invalid COLUMNS.
                    // Row masking is handled by gamma=0 for invalid rows within
                    // each bi block when we construct the plaintext per (bi, bj).
                    //
                    // Wait - gamma_pt[bj] is used for ALL bi. So we can't mask
                    // per-row here. Instead, we'll rely on beta_add_pt to be per
                    // (bi, bj). But that changes the interface...
                    //
                    // Simpler approach: since padded rows have mean=0 and x=0,
                    // x_centered=0 for padded rows. So output = 0 * y_weighted + beta.
                    // We need beta=0 at padded rows.
                    //
                    // gamma_pt[bj]: 0 at invalid columns only (row masking not needed
                    // because x_centered=0 at padded rows, so 0*gamma=0).
                    // beta_add_pt: must be 0 at both invalid rows AND columns.
                    // But beta_add_pt[bj] is used for all bi... so we need per-(bi,bj).
                    //
                    // Let's use per (bi, bj) for beta instead.
                    if (actual_col < n_) {
                        gamma_vec[slot] = inv_std_scale_ * gamma_vals_[actual_col];
                    }
                }
            }
        }

        // gamma_pt: encode at q_{L_out} (pt*ct preserving D)
        gamma_pt_[bj] = ctx.encode_ringt(gamma_vec, q_Lout);
    }

    // beta: per (bi, bj) to handle row masking
    // Resize to num_block_rows * num_block_cols
    beta_add_pt_.resize(num_block_rows_ * num_block_cols_);
    double beta_scale = D / q_Lout_m1 * D;

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
                            beta_vec[slot] = beta_vals_[actual_col];
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
                                               const FeatureMatEncrypted& x,
                                               const vector<CkksCiphertext>& mean_cts,
                                               const vector<CkksCiphertext>& y_cts) {
    double D = param_.get_default_scale();
    uint32_t total_blocks = num_block_rows_ * num_block_cols_;

    FeatureMatEncrypted result(&ctx, y_level_ - 2);
    result.shape = {m_, n_};
    result.matmul_block_size = d_;
    result.data.resize(total_blocks);

    // Step 1: y_weighted[bi] = y[bi] * pt(gamma*inv_std_scale) -> level y_level_-1, exact D
    vector<CkksCiphertext> y_weighted(num_block_rows_);
    parallel_for(num_block_rows_, th_nums, ctx, [&](CkksContext& ctx_copy, int bi) {
        // All block-cols share the same y_weighted per block-row.
        // We'll compute y * gamma per (bi, bj) below.
        // Actually, y is per block-row (scalar per row), gamma is per block-col.
        // So y_weighted must be per (bi, bj). But we can save by computing
        // y * gamma_pt[bj] for each bj separately.
        // For now, just drop y to correct level.
        // (y_weighted computed per block in the main loop below)
    });

    // Main computation: per (bi, bj) block
    parallel_for(total_blocks, th_nums, ctx, [&](CkksContext& ctx_copy, int block_idx) {
        uint32_t bi = block_idx % num_block_rows_;
        uint32_t bj = block_idx / num_block_rows_;

        // y_weighted = y[bi] * pt(gamma[bj]) -> level y_level_-1, exact D
        auto gamma_mul = ctx_copy.ringt_to_mul(gamma_pt_[bj], y_level_);
        auto yw_raw = ctx_copy.mult_plain_mul(y_cts[bi], gamma_mul);
        auto yw = ctx_copy.rescale(yw_raw, D);
        // yw at y_level_-1, scale D exact

        // x_centered = drop(x[bi,bj]) - drop(mean[bi])
        auto x_drop = x.data[block_idx].copy();
        while (x_drop.get_level() > (int)(y_level_ - 1)) {
            x_drop = ctx_copy.drop_level(x_drop);
        }
        auto mean_drop = mean_cts[bi].copy();
        while (mean_drop.get_level() > (int)(y_level_ - 1)) {
            mean_drop = ctx_copy.drop_level(mean_drop);
        }
        auto x_centered = ctx_copy.sub(x_drop, mean_drop);

        // output = x_centered * y_weighted -> level y_level_-2, scale D²/q_{y_level_-1}
        auto out_raw = ctx_copy.mult(x_centered, yw);
        auto out_relin = ctx_copy.relinearize(out_raw);
        auto out = ctx_copy.rescale(out_relin, D / param_.get_q(y_level_ - 1) * D);

        // + beta (add_plain at scale D²/q_{y_level_-1} = D/q_{y_level_-1}*D)
        uint32_t beta_idx = bi + num_block_rows_ * bj;
        result.data[block_idx] = ctx_copy.add_plain_ringt(out, beta_add_pt_[beta_idx]);
    });

    result.level = y_level_ - 2;
    return result;
}
