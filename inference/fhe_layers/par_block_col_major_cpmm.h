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

/**
 * ParBlockColMajorCPMM: Parallel (multi-head interleaved) ciphertext-plaintext
 * matrix multiplication for right-multiplying by a plaintext matrix W.
 *
 * Supports three modes (auto-detected from W dimensions):
 *   - Square:  W is n × n           → output has R*G cts (same as input)
 *   - Expand:  W is n × (K*n)       → output has K*R*G cts (K megablocks)
 *   - Reduce:  W is (K*n) × n       → output has R*G cts (summed across K megablocks)
 * where n = n_heads * n_per_head, R = num_block_rows_A, G = n_cts_per_block_idx.
 *
 * Input A is in par_block_col_major format where different column blocks are
 * different heads. Different row blocks of W are treated as different heads
 * for interleaved diagonal encoding.
 *
 * Level consumption: 2 levels (1 for block_mult_cpmm + 1 for head-sum mask).
 */
class ParBlockColMajorCPMM {
public:
    ParBlockColMajorCPMM(const CkksParameter& param_in,
                         const Duo& shape_A,             // per-head shape of A (m, n_per_head)
                         const Array<double, 2>& W_mat,  // weight matrix: n×n, n×Kn, or Kn×n
                         uint32_t block_size,
                         uint32_t n_heads,
                         uint32_t level_A);
    ~ParBlockColMajorCPMM();

    void precompute_diagonals();
    Feature2DEncrypted run(CkksContext& ctx, const Feature2DEncrypted& A);

private:
    enum class Mode { SQUARE, EXPAND, REDUCE };

    // Unified core: processes mb_indices megablocks, sums before mask
    std::vector<CkksCiphertext>
    run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& A_cts, const std::vector<uint32_t>& mb_indices);

    // Block CPMM: d rotations + d pt_muls + 1 rescale, Level L -> L-1
    CkksCiphertext
    block_mult_cpmm(CkksContext& ctx, const CkksCiphertext& a, uint32_t megablock, uint32_t g_input, int bp) const;

    // Cross-head sum: (S-1) rotations + (S-1) additions, no level consumed
    CkksCiphertext head_sum(CkksContext& ctx, const CkksCiphertext& ct) const;

    // Build per-head diagonal for megablock, input group g_input, output block column bp, rotation k
    std::vector<double> build_block_diagonal(uint32_t megablock, uint32_t g_input, int bp, int k) const;

    // Build mask that selects h=0 positions only
    std::vector<double> build_head0_mask() const;

    static int get_block_index(int bi, int bj, int num_block_rows);

    Mode mode_;
    uint32_t K_;                              // megablock count (1 for square)
    std::vector<Array<double, 2>> W_padded_;  // K padded sub-weights

    CkksParameter param_;
    uint32_t m_;               // rows of A (and result)
    uint32_t n_per_head_;      // columns per head in A
    uint32_t n_total_per_mb_;  // total columns per megablock = n_heads * n_per_head
    uint32_t d_;               // block size
    uint32_t n_slot_;
    uint32_t chunk_size_;  // S * d^2
    uint32_t num_chunks_;
    uint32_t num_block_rows_A_;  // ceil(m / d)
    uint32_t level_;

    // Multi-head interleaving parameters
    uint32_t n_heads_;
    uint32_t n_h_padded_;
    uint32_t n_blocks_per_chunk_;  // S
    uint32_t n_cts_per_block_idx_;

    // diag_pt_[mb][g][bp][k]: per-head diagonal for megablock mb, input group g, output block column bp, rotation k
    std::vector<std::vector<std::vector<std::vector<CkksPlaintextRingt>>>> diag_pt_;

    // Mask plaintext for selecting h=0 after cross-head sum
    CkksPlaintextRingt mask_h0_pt_;
};
