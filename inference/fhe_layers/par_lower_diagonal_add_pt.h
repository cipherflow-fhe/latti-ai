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
 * ParLowerDiagonalAddPt: add a plaintext matrix B to a ciphertext matrix A
 * packed by FeatureMatEncrypted::par_diagonal_pack(..., is_lower=true, is_transposed=true).
 *
 * shape is the full matrix shape {total_rows, n_prepad}; head_shape is {m_prepad, n_prepad}.
 * The generated plaintext mirrors the diagonal slot layout and keeps all row/column/head/
 * megablock padding slots at zero.
 */
class ParLowerDiagonalAddPt : public Layer {
public:
    ParLowerDiagonalAddPt(const ls::CkksParameter& param_in,
                          Duo shape,
                          Duo head_shape,
                          uint32_t n_heads,
                          uint32_t level,
                          Array<double, 2>&& B);
    void precompute_pts();

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& A);
    Array<double, 2> run_plaintext(const Array<double, 2>& A) const;

    ls::CkksPlaintextRingt generate_pt(ls::CkksContext& ctx, uint32_t mb, uint32_t ct_local) const;

private:
    std::vector<double> build_pt_vec(uint32_t mb, uint32_t ct_local) const;

    uint32_t total_rows_ = 0, n_prepad_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, packed_extent_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;
    Array<double, 2> B_vals_;

    std::vector<ls::CkksPlaintextRingt> pt_;
};
