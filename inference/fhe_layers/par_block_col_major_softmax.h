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
#include "par_block_col_major_layout.h"
#include "../data_structs/feature_mat.h"
#include <vector>

class ParBlockColMajorMultipleSquare : public Layer {
public:
    ParBlockColMajorMultipleSquare(const ls::CkksParameter& param,
                                   Duo shape,
                                   Duo head_shape,
                                   uint32_t n_heads,
                                   uint32_t init_level,
                                   uint32_t block_size);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    ParBlockColMajorLayout layout_;
    uint32_t n_prepad_, total_cols_, m_prepad_, H_prepad_, m_;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParBlockColMajorHeadColSum : public Layer {
public:
    ParBlockColMajorHeadColSum(const ls::CkksParameter& param,
                               Duo shape,
                               Duo head_shape,
                               uint32_t n_heads,
                               uint32_t init_level,
                               uint32_t block_size);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_, input_mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& x);
    Array<double, 2> run_plaintext(const Array<double, 2>& x) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

    ls::CkksPlaintextRingt generate_input_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    ParBlockColMajorLayout layout_;
    uint32_t n_prepad_, total_cols_, m_prepad_, H_prepad_, m_;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
    ls::CkksCiphertext reduce_local_columns(ls::CkksContext& ctx, const ls::CkksCiphertext& ct) const;
};

class ParBlockColMajorInverseInit : public Layer {
public:
    ParBlockColMajorInverseInit(const ls::CkksParameter& param,
                                Duo shape,
                                Duo head_shape,
                                uint32_t n_heads,
                                uint32_t init_level,
                                uint32_t block_size);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> two_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_two_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    ParBlockColMajorLayout layout_;
    uint32_t n_prepad_, total_cols_, m_prepad_, H_prepad_, m_;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParBlockColMajorInverseIter : public Layer {
public:
    ParBlockColMajorInverseIter(const ls::CkksParameter& param,
                                Duo shape,
                                Duo head_shape,
                                uint32_t n_heads,
                                uint32_t init_level,
                                uint32_t block_size);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> one_pt_;
    std::vector<ls::CkksPlaintextRingt> two_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_one_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;
    ls::CkksPlaintextRingt generate_two_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    ParBlockColMajorLayout layout_;
    uint32_t n_prepad_, total_cols_, m_prepad_, H_prepad_, m_;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};

class ParBlockColMajorCtMul : public Layer {
public:
    ParBlockColMajorCtMul(const ls::CkksParameter& param,
                          Duo shape,
                          Duo head_shape,
                          uint32_t n_heads,
                          uint32_t init_level,
                          uint32_t block_size);
    void prepare_weight() override;
    std::vector<ls::CkksPlaintextRingt> mask_pt_;

    FeatureMatEncrypted run(ls::CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b);
    Array<double, 2> run_plaintext(const Array<double, 2>& a, const Array<double, 2>& b) const;

    ls::CkksPlaintextRingt generate_mask_pt(ls::CkksContext& ctx, uint32_t ct_idx) const;

private:
    ParBlockColMajorLayout layout_;
    uint32_t n_prepad_, total_cols_, m_prepad_, H_prepad_, m_;

    uint32_t total_cts() const;
    std::vector<double> build_valid_mask(uint32_t ct_idx, double value) const;
};
