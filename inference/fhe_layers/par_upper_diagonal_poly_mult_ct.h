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
#include <vector>

class ParUpperDiagonalPolyMultCt : public Layer {
public:
    ParUpperDiagonalPolyMultCt(const ls::CkksParameter& param,
                               Duo shape,
                               Duo head_shape,
                               uint32_t n_heads,
                               uint32_t init_level);

    void prepare_weight() override;

    std::vector<ls::CkksPlaintextRingt> one_pt_;
    std::vector<ls::CkksPlaintextRingt> half_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& half_tanh, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& half_tanh, const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_one_pt(ls::CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t g = 0) const;
    ls::CkksPlaintextRingt generate_half_pt(ls::CkksContext& ctx, uint32_t mb, uint32_t ct_local, uint32_t g = 0) const;

private:
    uint32_t n_prepad_ = 0, total_cols_ = 0, m_prepad_ = 0;
    uint32_t H_prepad_ = 0, H_ = 0, m_ = 0, n_ = 0, packed_extent_ = 0;
    uint32_t n_slot_ = 0, segment_len_ = 0, c_ = 0, cts_per_mb_ = 0, n_mb_ = 0;

    uint32_t ct_index(uint32_t mb, uint32_t ct_local) const;
    uint32_t total_cts() const;
    std::vector<double> build_constant_vec(double value, uint32_t mb, uint32_t ct_local) const;
};
