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
// ParBlockColMajorLNStats — Phase 1: compute scaled variance (a)
// Levels consumed: 4 (L -> L-4)
// ============================================================
class ParBlockColMajorLNStats : public Layer {
public:
    ParBlockColMajorLNStats(const ls::CkksParameter& param,
                            Duo shape,  // full matrix: {M, n_heads * cols_per_head}
                            uint32_t block_size,
                            uint32_t n_heads,
                            uint32_t init_level,
                            double eps,
                            double inv_var);
    void prepare_weight();

    // Returns a_cts[num_block_rows]
    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t bi = 0, uint32_t bj = 0, uint32_t g = 0) const;

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t total_dim_;
    double eps_, inv_var_;

    ls::CkksCiphertext intra_block_col_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt h0_mask_pt_;
    ls::CkksPlaintextRingt inv_n_pt_;  // 1/N
    ls::CkksPlaintextRingt iv_pt_;     // inv_var
    ls::CkksPlaintextRingt eps_add_pt_;
};

// ============================================================
// ParBlockColMajorLNXCentered — compute x_centered = x - mean(x)
// Levels consumed: 2 (L -> L-2)
// ============================================================
class ParBlockColMajorLNXCentered : public Layer {
public:
    ParBlockColMajorLNXCentered(const ls::CkksParameter& param,
                                Duo shape,  // full matrix: {M, n_heads * cols_per_head}
                                uint32_t block_size,
                                uint32_t n_heads,
                                uint32_t init_level);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t bi = 0, uint32_t bj = 0, uint32_t g = 0) const;

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t total_dim_;

    ls::CkksCiphertext intra_block_col_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt h0_mask_pt_;
    ls::CkksPlaintextRingt inv_n_pt_;  // 1/N
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
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& a_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& a) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t bi = 0, uint32_t bj = 0, uint32_t g = 0) const;

private:
    uint32_t d_, n_slot_, chunk_size_;
    double c0_, c1_, c2_;

    ls::CkksPlaintextRingt c2_norm_pt_;
    ls::CkksPlaintextRingt c1_pt_;
    ls::CkksPlaintextRingt c0_add_pt_;
};

// ============================================================
// ParBlockColMajorLNGoldschmidt — Phase 3: one Goldschmidt iteration
//   y_new = 0.5 * (3*y - (a*y)*(y²))
// Levels consumed: 3 (L_y -> L_y-3)
// ============================================================
class ParBlockColMajorLNGoldschmidt : public Layer {
public:
    ParBlockColMajorLNGoldschmidt(const ls::CkksParameter& param, uint32_t block_size, uint32_t input_level);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx,
                                        const std::vector<ls::CkksCiphertext>& y_cts,
                                        const std::vector<ls::CkksCiphertext>& a_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& y, const Array<double, 2>& a) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t bi = 0, uint32_t bj = 0, uint32_t g = 0) const;

private:
    uint32_t d_, n_slot_, chunk_size_;

    ls::CkksPlaintextRingt three_pt_;      // 3.0, for pt*ct with y
    ls::CkksPlaintextRingt half_norm_pt_;  // 0.5, normalizing scale
};

// ============================================================
// ParBlockColMajorLNAffine — Phase 4: output = x_centered * y * gamma*inv_std + beta
// Levels consumed: 2
// ============================================================
class ParBlockColMajorLNAffine : public Layer {
public:
    ParBlockColMajorLNAffine(const ls::CkksParameter& param,
                             Duo shape,  // full matrix: {M, n_heads * cols_per_head}
                             uint32_t block_size,
                             uint32_t n_heads,
                             uint32_t y_level,
                             double inv_std,
                             Array<double, 1>&& gamma,
                             Array<double, 1>&& beta);
    void prepare_weight();

    FeatureMatEncrypted run(ls::CkksContext& ctx,
                            const std::vector<ls::CkksCiphertext>& x_centered,
                            const std::vector<ls::CkksCiphertext>& y_cts);
    Array<double, 2> run_plaintext(const Array<double, 2>& x_centered, const Array<double, 2>& y) const;

    ls::CkksPlaintextRingt
    generate_pt(ls::CkksContext& ctx, uint32_t pt_idx, uint32_t bi = 0, uint32_t bj = 0, uint32_t g = 0) const;

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t y_level_;
    double inv_std_;
    Array<double, 1> gamma_vals_, beta_vals_;

    std::vector<ls::CkksPlaintextRingt> gamma_pt_;
    std::vector<ls::CkksPlaintextRingt> beta_add_pt_;
};
