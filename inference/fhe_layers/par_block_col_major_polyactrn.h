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
// ParBlockColMajorPolyActRNGamma — element-wise gamma scaling (no beta)
// Levels consumed: 1 (L -> L-1)
// Input:  FeatureMatEncrypted at level L, scale D
// Output: FeatureMatEncrypted at level L-1, scale D
// ============================================================
class ParBlockColMajorPolyActRNGamma : public Layer {
public:
    ParBlockColMajorPolyActRNGamma(const ls::CkksParameter& param,
                                   Duo shape,
                                   uint32_t block_size,
                                   uint32_t n_heads,
                                   uint32_t K,
                                   uint32_t init_level,
                                   Array<double, 1>&& gamma);
    void prepare_weight();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t K_, total_dim_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    Array<double, 1> gamma_vals_;

    std::vector<ls::CkksPlaintextRingt> gamma_pt_;  // per (mb, bj, g)
};

// ============================================================
// ParBlockColMajorPolyActRNPoly — per-column polynomial evaluation
// p_j(x) = coeffs(0,j) + coeffs(1,j)*x + coeffs(2,j)*x^2
//         [+ coeffs(3,j)*x^3 + coeffs(4,j)*x^4]
// Levels consumed: 2 (degree=2) or 3 (degree=4)
// Input:  FeatureMatEncrypted at level L, scale D
// Output: FeatureMatEncrypted at level L-2 or L-3, scale D
// ============================================================
class ParBlockColMajorPolyActRNPoly : public Layer {
public:
    ParBlockColMajorPolyActRNPoly(const ls::CkksParameter& param,
                                  Duo shape,
                                  uint32_t block_size,
                                  uint32_t n_heads,
                                  uint32_t K,
                                  uint32_t init_level,
                                  Array<double, 2>&& coeffs,
                                  uint32_t degree);
    void prepare_weight();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);

    ls::CkksPlaintextRingt generate_coeff_pt(ls::CkksContext& ctx,
                                             uint32_t coeff_idx,
                                             uint32_t mb,
                                             uint32_t bi,
                                             uint32_t bj,
                                             uint32_t g) const;

private:
    uint32_t m_, cols_per_head_, d_, n_slot_;
    uint32_t K_, total_dim_;
    uint32_t n_heads_, n_h_padded_, S_, n_cts_per_block_idx_;
    uint32_t chunk_size_, num_chunks_;
    uint32_t num_block_rows_, num_block_cols_;
    uint32_t degree_;
    Array<double, 2> coeffs_;

    // Degree >= 2
    std::vector<ls::CkksPlaintextRingt> c2_pt_;      // per (mb, bj, g)
    std::vector<ls::CkksPlaintextRingt> c1_pt_;      // per (mb, bj, g)
    std::vector<ls::CkksPlaintextRingt> c0_add_pt_;  // per (mb, bi, bj, g)

    // Degree 4 only
    std::vector<ls::CkksPlaintextRingt> c4_pt_;  // per (mb, bj, g)
    std::vector<ls::CkksPlaintextRingt> c3_pt_;  // per (mb, bj, g)
};
