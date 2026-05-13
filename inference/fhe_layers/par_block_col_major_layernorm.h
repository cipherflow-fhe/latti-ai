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

#pragma once
#include "layer.h"
#include "../data_structs/feature_mat.h"

// ============================================================
// ParBlockColMajorLNStats — Phase 1: compute scaled variance (a) and mean
// Full-dimension normalization: N = n_heads * cols_per_head
// Levels consumed: 4 (L -> L-4)
//   Extra level vs block format: cross-head mask pt*ct to fix rotate wrap
// ============================================================
class ParBlockColMajorLNStats : public Layer {
public:
    ParBlockColMajorLNStats(const ls::CkksParameter& param,
                            Duo shape,  // per-head: {M, cols_per_head}
                            uint32_t block_size,
                            uint32_t n_heads,
                            uint32_t init_level,
                            double eps,
                            double inv_var_scale);
    void precompute_plaintexts();

    // Returns: {a_cts, mean_cts}, each has num_block_rows elements
    std::pair<std::vector<ls::CkksCiphertext>, std::vector<ls::CkksCiphertext>> run(ls::CkksContext& ctx,
                                                                                    const FeatureMatEncrypted& x);

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t total_dim_;  // N = n_heads * cols_per_head
    double eps_, inv_var_scale_;

    // Column sum only (no cross-head sum)
    ls::CkksCiphertext intra_block_col_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;
    // Cross-head sum + mask + replicate (costs 1 level via mask pt*ct)
    ls::CkksCiphertext cross_head_sum_masked(ls::CkksContext& ctx,
                                             const ls::CkksCiphertext& col_summed,
                                             const ls::CkksPlaintextRingt& mask_pt) const;

    ls::CkksPlaintextRingt h0_mask_pt_;     // 1 at h=0, 0 at h!=0 (for sum_x mask)
    ls::CkksPlaintextRingt h0_mask_sq_pt_;  // same mask, different encode scale (for sum_x_sq mask)
    ls::CkksPlaintextRingt inv_n_norm_pt_;
    ls::CkksPlaintextRingt inv_n_sq_norm_pt_;
    ls::CkksPlaintextRingt inv_n_mean_pt_;
    ls::CkksPlaintextRingt inv_var_scale_pt_;
    ls::CkksPlaintextRingt eps_add_pt_;
};

// ============================================================
// ParBlockColMajorLNMinimaxInit — Phase 2: y0 = c0 + c1*a + c2*a²
// Levels consumed: 2
// ============================================================
class ParBlockColMajorLNMinimaxInit : public Layer {
public:
    ParBlockColMajorLNMinimaxInit(const ls::CkksParameter& param,
                                  uint32_t block_size,
                                  uint32_t input_level,
                                  double c0,
                                  double c1,
                                  double c2);
    void precompute_plaintexts();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& a_cts);

private:
    uint32_t d_, n_slot_, chunk_size_;
    double c0_, c1_, c2_;

    ls::CkksPlaintextRingt c2_norm_pt_;
    ls::CkksPlaintextRingt c1_pt_;
    ls::CkksPlaintextRingt c0_add_pt_;
};

// ============================================================
// ParBlockColMajorLNGoldschmidt — Phase 3: one Goldschmidt iteration
// Levels consumed: 4
// ============================================================
class ParBlockColMajorLNGoldschmidt : public Layer {
public:
    ParBlockColMajorLNGoldschmidt(const ls::CkksParameter& param, uint32_t block_size, uint32_t input_level);
    void precompute_plaintexts();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx,
                                        const std::vector<ls::CkksCiphertext>& y_cts,
                                        const std::vector<ls::CkksCiphertext>& a_cts);

private:
    uint32_t d_, n_slot_, chunk_size_;

    ls::CkksPlaintextRingt three_add_pt_;
    ls::CkksPlaintextRingt half_norm_pt_;
};

// ============================================================
// ParBlockColMajorLNAffine — Phase 4: output = (x - mean) * y * gamma + beta
// Full-dimension normalization
// Levels consumed: 2
// ============================================================
class ParBlockColMajorLNAffine : public Layer {
public:
    ParBlockColMajorLNAffine(const ls::CkksParameter& param,
                             Duo shape,  // per-head: {M, cols_per_head}
                             uint32_t block_size,
                             uint32_t n_heads,
                             uint32_t init_level,
                             uint32_t y_level,
                             double inv_std_scale,
                             const std::vector<double>& gamma,  // length = n_heads * cols_per_head
                             const std::vector<double>& beta);
    void precompute_plaintexts();

    FeatureMatEncrypted run(ls::CkksContext& ctx,
                            const FeatureMatEncrypted& x,
                            const std::vector<ls::CkksCiphertext>& mean_cts,
                            const std::vector<ls::CkksCiphertext>& y_cts);

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t init_level_, y_level_;
    double inv_std_scale_;
    std::vector<double> gamma_vals_, beta_vals_;

    // gamma_pt and beta_pt: per (bj, g) for par format
    // indexed as bj * n_cts_per_block_idx_ + g
    std::vector<ls::CkksPlaintextRingt> gamma_pt_;
    // beta_pt: per (bi, bj, g) to handle row masking
    std::vector<ls::CkksPlaintextRingt> beta_add_pt_;
};
