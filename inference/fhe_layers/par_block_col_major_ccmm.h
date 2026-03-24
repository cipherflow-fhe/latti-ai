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

class ParBlockColMajorCCMM : public Layer {
public:
    ParBlockColMajorCCMM(const ls::CkksParameter& param_in,
                         const Duo& shape_A,  // per-head shape of A (m, n)
                         const Duo& shape_B,  // per-head shape of B (n, p)
                         uint32_t block_size,
                         uint32_t n_heads,
                         uint32_t level);
    void precompute_diagonals();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& A, const FeatureMatEncrypted& B);

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx,
                                             const std::vector<ls::CkksCiphertext>& A_cts,
                                             const std::vector<ls::CkksCiphertext>& B_cts);

    ls::CkksCiphertext sigma_on_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& a) const;
    ls::CkksCiphertext tau_on_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& b) const;
    ls::CkksCiphertext phi_on_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& a_sigma, int i) const;
    ls::CkksCiphertext psi_on_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& b_tau, int i) const;
    ls::CkksCiphertext
    block_mult_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& a, const ls::CkksCiphertext& b) const;

    std::vector<double> build_sigma_diagonal(int k_idx) const;
    std::vector<double> build_tau_diagonal(int offset) const;
    std::pair<std::vector<double>, std::vector<double>> build_psi_diagonals(int k_val) const;
    std::vector<double> build_psi_k_equal_0_diagonals() const;

    static int get_block_index(int bi, int bj, int num_block_rows);

    uint32_t m_, n_, p_;  // per-head matrix dimensions
    uint32_t d_;          // block size
    uint32_t n_slot_;
    uint32_t chunk_size_;  // S * d²
    uint32_t num_chunks_;
    uint32_t num_block_rows_A_, num_block_cols_A_;
    uint32_t num_block_rows_B_, num_block_cols_B_;

    // Multi-head interleaving parameters
    uint32_t n_heads_;
    uint32_t n_h_padded_;
    uint32_t n_blocks_per_chunk_;  // S
    uint32_t n_cts_per_block_idx_;

    // Precomputed diagonal plaintexts
    std::vector<ls::CkksPlaintextRingt> sigma_diag_pt_;       // d vectors
    std::vector<ls::CkksPlaintextRingt> tau_diag_pt_;         // 2d-1 vectors
    ls::CkksPlaintextRingt psi_k0_pt_;                        // all-ones for i=0
    std::vector<ls::CkksPlaintextRingt> psi_w_k_pt_;          // d-1 vectors (i=1..d-1)
    std::vector<ls::CkksPlaintextRingt> psi_w_k_minus_d_pt_;  // d-1 vectors (i=1..d-1)
};
