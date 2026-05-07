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
#include "../data_structs/feature.h"

/**
 * @brief Softmax layer implementation using FHE-friendly polynomial approximation.
 * 
 * Computes Softmax(x)_i = exp(x_i) / sum(exp(x_j)).
 * The implementation uses:
 * 1. Polynomial approximation for exp(x).
 * 2. Summation via rotations and additions.
 * 3. Polynomial approximation for 1/sum.
 * 4. Multiplication to get the final result.
 */
class SoftmaxLayer : public Layer {
public:
    /**
     * @param exp_range Range [a, b] for exp(x) approximation.
     * @param exp_degree Degree of the polynomial for exp(x).
     * @param inv_range Range [min, max] for 1/x approximation.
     * @param inv_degree Degree of the polynomial for 1/x.
     * @param n_channel Number of channels in the input.
     * @param n_channel_per_ct Number of channels packed per ciphertext.
     * @param temperature Temperature scaling factor (1.0 = no scaling).
     */
    SoftmaxLayer(const std::array<double, 2>& exp_range, int exp_degree,
                 const std::array<double, 2>& inv_range, int inv_degree,
                 int n_channel, int n_channel_per_ct, double temperature = 1.0);

    virtual ~SoftmaxLayer();

    /**
     * @brief Run encrypted Softmax on a 0D feature (vector).
     * @param ctx CKKS context.
     * @param x Input encrypted feature.
     * @return Encrypted Softmax result.
     */
    Feature0DEncrypted run(ls::CkksContext& ctx, const Feature0DEncrypted& x);

    /**
     * @brief Run plaintext Softmax for verification (1D).
     */
    Array<double, 1> run_plaintext(const Array<double, 1>& x);

private:
    std::array<double, 2> exp_range_;
    int exp_degree_;
    std::array<double, 2> inv_range_;
    int inv_degree_;
    int n_channel_;
    int n_channel_per_ct_;
    int n_slots_;
    double temperature_;

    /**
     * @brief Sum elements in a ciphertext across slots.
     */
    ls::CkksCiphertext sum_slots(ls::CkksContext& ctx, const ls::CkksCiphertext& x);
};
