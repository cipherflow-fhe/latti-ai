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

/**
 * ParBlockColMajorAddPt: Add a plaintext matrix B to a ciphertext matrix A
 * in par_block_col_major format.  Result = A + B.
 *
 * B is Array<double, 2> of shape (m, total_dim) where total_dim = n_heads * cols_per_head.
 * The plaintext is encoded per output CT at (bi, bj, g) with proper slot mapping
 * and zero-padding for out-of-bounds positions.
 *
 * Level consumption: 0 (plaintext addition does not consume a level in CKKS).
 */
class ParBlockColMajorAddPt : public Layer {
public:
    ParBlockColMajorAddPt(const ls::CkksParameter& param_in,
                          Duo shape,  // full matrix shape: {m, total_dim}
                          uint32_t block_size,
                          uint32_t n_heads,
                          uint32_t level,
                          Array<double, 2>&& B);
    void precompute_pts();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& A);
    Array<double, 2> run_plaintext(const Array<double, 2>& A) const;

    ls::CkksPlaintextRingt generate_pt(ls::CkksContext& ctx, uint32_t bi, uint32_t bj, uint32_t g) const;
    std::vector<double> generate_values(uint32_t bi, uint32_t bj, uint32_t g) const;
    uint32_t num_block_rows() const {
        return num_block_rows_;
    }
    uint32_t num_block_cols() const {
        return num_block_cols_;
    }
    uint32_t num_cts_per_block_idx() const {
        return n_cts_per_block_idx_;
    }

private:
    std::vector<double> build_pt_vec(uint32_t bi, uint32_t bj, uint32_t g) const;

    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    Array<double, 2> B_vals_;

    // pt_[(bi + num_block_rows_ * bj) * G + g]
    std::vector<ls::CkksPlaintextRingt> pt_;
};
