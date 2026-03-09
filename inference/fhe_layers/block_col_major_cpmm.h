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

class BlockColMajorCPMM {
public:
    BlockColMajorCPMM(const CkksParameter& param_in,
                      const Duo& shape_A,
                      const Duo& shape_B,
                      const Array<double, 2>& B_mat_in,
                      uint32_t block_size,
                      uint32_t level_A);
    ~BlockColMajorCPMM();

    void precompute_diagonals();
    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& A);

private:
    std::vector<CkksCiphertext> run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& A_cts);
    CkksCiphertext block_mult_cpmm(CkksContext& ctx, const CkksCiphertext& a, int bj, int bp) const;
    std::vector<double> build_block_diagonal(int bj, int bp, int k) const;
    static int get_block_index(int bi, int bj, int num_block_rows);

    CkksParameter param_;
    Array<double, 2> B_mat_;
    uint32_t m_, n_, p_;
    uint32_t d_;  // block size (d×d blocks)
    uint32_t n_slot_;
    uint32_t chunk_size_;  // d*d
    uint32_t num_chunks_;  // n_slot_ / chunk_size_
    uint32_t num_block_rows_A_, num_block_cols_A_;
    uint32_t num_block_rows_B_, num_block_cols_B_;
    uint32_t level_;

    // diag_pt_[b_block_idx][k], k=0..d-1
    std::vector<std::vector<CkksPlaintextRingt>> diag_pt_;
};
