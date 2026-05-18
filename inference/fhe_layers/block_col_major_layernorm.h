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
// BlockColMajorLNStats — Phase 1: compute scaled variance (a)
// ============================================================
class BlockColMajorLNStats : public Layer {
public:
    BlockColMajorLNStats(const ls::CkksParameter& param,
                         Duo shape,
                         uint32_t block_size,
                         uint32_t init_level,
                         double eps,
                         double inv_var);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);

private:
    uint32_t m_, n_, d_, n_slot_, chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    double eps_, inv_var_;

    ls::CkksCiphertext intra_block_row_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt inv_n_pt_;    // 1/N
    ls::CkksPlaintextRingt iv_pt_;       // inv_var
    ls::CkksPlaintextRingt eps_add_pt_;  // eps*inv_var
};

// ============================================================
// BlockColMajorLNXCentered — compute x_centered = x - mean(x)
// ============================================================
class BlockColMajorLNXCentered : public Layer {
public:
    BlockColMajorLNXCentered(const ls::CkksParameter& param, Duo shape, uint32_t block_size, uint32_t init_level);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);

private:
    uint32_t m_, n_, d_, n_slot_, chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;

    ls::CkksCiphertext intra_block_row_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    ls::CkksPlaintextRingt inv_n_pt_;  // 1/N
};

// ============================================================
// BlockColMajorLNMinimaxInit — Phase 2: y0 = c0 + c1*a + c2*a² (non-Horner)
// Levels consumed: 2 (input_level -> input_level-2)
// Input:  a_cts (per block-row, level input_level, scale D)
// Output: y_cts (per block-row, level input_level-2, scale D exact)
// ============================================================
class BlockColMajorLNMinimaxInit : public Layer {
public:
    BlockColMajorLNMinimaxInit(const ls::CkksParameter& param,
                               uint32_t block_size,
                               uint32_t input_level,
                               double c0,
                               double c1,
                               double c2);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& a_cts);

private:
    uint32_t d_, n_slot_, chunk_size_;
    double c0_, c1_, c2_;

    ls::CkksPlaintextRingt c2_norm_pt_;  // c2
    ls::CkksPlaintextRingt c1_pt_;       // c1
    ls::CkksPlaintextRingt c0_add_pt_;   // c0
};

// ============================================================
// BlockColMajorLNGoldschmidt — Phase 3: one Goldschmidt iteration
// ============================================================
class BlockColMajorLNGoldschmidt : public Layer {
public:
    BlockColMajorLNGoldschmidt(const ls::CkksParameter& param, uint32_t block_size, uint32_t input_level);
    void prepare_weight();

    std::vector<ls::CkksCiphertext> run(ls::CkksContext& ctx,
                                        const std::vector<ls::CkksCiphertext>& y_cts,
                                        const std::vector<ls::CkksCiphertext>& a_cts);

private:
    uint32_t d_, n_slot_, chunk_size_;

    ls::CkksPlaintextRingt three_pt_;      // 3.0, for pt*ct with y
    ls::CkksPlaintextRingt half_norm_pt_;  // 0.5, normalizing scale
};

// ============================================================
// BlockColMajorLNAffine — Phase 4: output = x_centered * y * gamma*inv_std + beta
// Levels consumed: 2 (L_out -> L_out-2)
// ============================================================
class BlockColMajorLNAffine : public Layer {
public:
    BlockColMajorLNAffine(const ls::CkksParameter& param,
                          Duo shape,
                          uint32_t block_size,
                          uint32_t y_level,
                          double inv_std,
                          Array<double, 1>&& gamma,
                          Array<double, 1>&& beta);
    void prepare_weight();

    FeatureMatEncrypted run(ls::CkksContext& ctx,
                            const std::vector<ls::CkksCiphertext>& x_centered,
                            const std::vector<ls::CkksCiphertext>& y_cts);

private:
    uint32_t m_, n_, d_, n_slot_, chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t y_level_;
    double inv_std_;
    Array<double, 1> gamma_vals_, beta_vals_;

    std::vector<ls::CkksPlaintextRingt> gamma_pt_;
    std::vector<ls::CkksPlaintextRingt> beta_add_pt_;
};
