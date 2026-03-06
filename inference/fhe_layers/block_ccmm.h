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
#include "../data_structs/feature.h"

using namespace cxx_sdk_v2;

class BlockCCMM {
public:
    BlockCCMM(const CkksParameter& param_in,
              const Duo& shape_A,
              const Duo& shape_B,
              uint32_t block_size_A,
              uint32_t block_size_B,
              uint32_t level_A,
              uint32_t level_B);
    ~BlockCCMM();

    void precompute_diagonals();

    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& A, const Feature2DEncrypted& B);

private:
    std::vector<CkksCiphertext>
    run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& A_cts, const std::vector<CkksCiphertext>& B_cts);

    CkksCiphertext sigma_on_ct(CkksContext& ctx, const CkksCiphertext& a) const;
    CkksCiphertext tau_on_ct(CkksContext& ctx, const CkksCiphertext& b) const;
    CkksCiphertext phi_on_ct(CkksContext& ctx, const CkksCiphertext& a_sigma, int i) const;
    CkksCiphertext psi_on_ct(CkksContext& ctx, const CkksCiphertext& b_tau, int i) const;
    CkksCiphertext block_mult_ct(CkksContext& ctx, const CkksCiphertext& a, const CkksCiphertext& b) const;

    std::vector<double> build_sigma_diagonal(int k_idx) const;
    std::vector<double> build_tau_diagonal(int offset) const;
    std::pair<std::vector<double>, std::vector<double>> build_psi_diagonals(int k_val) const;
    std::vector<double> build_psi_k_equal_0_diagonals() const;

    static int get_block_index(int bi, int bj, int num_block_rows);

    CkksParameter param_;
    uint32_t m_, n_, p_;
    uint32_t d_;           // block size (matrix dimension within block)
    uint32_t n_slot_;      // N/2
    uint32_t chunk_size_;  // k*k
    uint32_t num_chunks_;  // n_slot_ / chunk_size_
    uint32_t num_block_rows_A_, num_block_cols_A_;
    uint32_t num_block_rows_B_, num_block_cols_B_;
    uint32_t level_;

    // Precomputed diagonal plaintexts
    std::vector<CkksPlaintextRingt> sigma_diag_pt_;       // d vectors
    std::vector<CkksPlaintextRingt> tau_diag_pt_;         // 2d-1 vectors
    CkksPlaintextRingt psi_k0_pt_;                        // all-ones for i=0
    std::vector<CkksPlaintextRingt> psi_w_k_pt_;          // d-1 vectors (i=1..d-1)
    std::vector<CkksPlaintextRingt> psi_w_k_minus_d_pt_;  // d-1 vectors (i=1..d-1)
};
