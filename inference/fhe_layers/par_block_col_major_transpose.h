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

class ParBlockColMajorTranspose : public Layer {
public:
    ParBlockColMajorTranspose(const ls::CkksParameter& param_in,
                              const Duo& shape,  // per-head matrix shape (m, n_per_head)
                              uint32_t block_size,
                              uint32_t n_heads,
                              uint32_t level);
    void precompute_diagonals();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& input);

private:
    std::vector<ls::CkksCiphertext> run_core(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& cts);

    ls::CkksCiphertext transpose_on_ct(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;

    std::vector<double> build_transpose_diagonal(int k) const;

    static int get_block_index(int bi, int bj, int num_block_rows);

    uint32_t m_, n_;  // per-head matrix dimensions
    uint32_t d_;      // block size
    uint32_t n_slot_;
    uint32_t chunk_size_;  // S * d²
    uint32_t num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;

    // Multi-head interleaving parameters
    uint32_t n_heads_;              // actual heads
    uint32_t n_h_padded_;           // padded to power of 2
    uint32_t n_blocks_per_chunk_;   // S: heads per chunk
    uint32_t n_cts_per_block_idx_;  // CTs per block position

    // Precomputed: (2d-1) transpose diagonal plaintexts
    std::vector<ls::CkksPlaintextRingt> transpose_diag_pt_;
};
