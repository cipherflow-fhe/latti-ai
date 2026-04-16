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

#include "conv2d_layer.h"
#include "../util/types.h"

#include <array>
#include <cmath>
#include <sstream>

using namespace std;
using namespace lattisense;

#ifdef _OPENMP
#    include <omp.h>
#endif

// ============================================================================
// Constructor and Destructor
// ============================================================================

Conv2DLayer::Conv2DLayer(const CkksParameter& param,
                         const Duo& input_shape,
                         Array<double, 4>&& weight,
                         Array<double, 1>&& bias,
                         const Duo& stride,
                         const Duo& skip)
    : Layer(param), input_shape_(input_shape), stride_(stride), skip_(skip), weight_(move(weight)), bias_(move(bias)) {
    const auto weight_shape = weight_.get_shape();
    n_out_channel_ = weight_shape[0];
    n_in_channel_ = weight_shape[1];
    kernel_shape_ = {static_cast<uint32_t>(weight_shape[2]), static_cast<uint32_t>(weight_shape[3])};

    if ((input_shape_[0] & (input_shape_[0] - 1)) != 0 || (input_shape_[1] & (input_shape_[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: " + str(input_shape_));
    }
    if ((stride_[0] & (stride_[0] - 1)) != 0 || (stride_[1] & (stride_[1] - 1)) != 0) {
        throw std::invalid_argument("stride must be powers of 2, got: " + str(stride_));
    }
    if ((skip_[0] & (skip_[0] - 1)) != 0 || (skip_[1] & (skip_[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: " + str(skip_));
    }
}

// ============================================================================
// Plaintext Convolution
// ============================================================================

double Conv2DLayer::compute_output_element(uint32_t out_ch,
                                           const Duo& output_pos,
                                           const Array<double, 3>& padded_input,
                                           double weight_scale) const {
    double sum = bias_.get(out_ch);
    const Duo input_base = output_pos * stride_;

    for (uint32_t in_ch = 0; in_ch < n_in_channel_; ++in_ch) {
        for (const Duo& kernel_pos : duo_range(kernel_shape_)) {
            const Duo input_pos = input_base + kernel_pos;
            const double input_val = padded_input.get(in_ch, input_pos[0], input_pos[1]);
            const double weight_val = weight_.get(out_ch, in_ch, kernel_pos[0], kernel_pos[1]) * weight_scale;

            sum += input_val * weight_val;
        }
    }

    return sum;
}

Array<double, 3> Conv2DLayer::run_plaintext(const Array<double, 3>& x, double multiplier) {
    const auto x_shape = x.get_shape();
    const Duo actual_input_shape = {static_cast<uint32_t>(x_shape[1]), static_cast<uint32_t>(x_shape[2])};

    if (x_shape[0] != n_in_channel_) {
        std::ostringstream oss;
        oss << "Input channels mismatch: expected " << n_in_channel_ << ", got " << x_shape[0];
        throw std::invalid_argument(oss.str());
    }

    const Duo padding = kernel_shape_ / 2;
    const Duo output_shape = actual_input_shape / stride_;
    const Duo padded_shape = actual_input_shape + padding * 2;
    const double weight_scale = 1.0 / multiplier;

    Array<double, 3> padded_input({n_in_channel_, padded_shape[0], padded_shape[1]}, 0.0);

    for (uint32_t ch = 0; ch < n_in_channel_; ++ch) {
        for (const Duo& input_pos : duo_range(actual_input_shape)) {
            const Duo padded_pos = input_pos + padding;
            padded_input.set(ch, padded_pos[0], padded_pos[1], x.get(ch, input_pos[0], input_pos[1]));
        }
    }

    Array<double, 3> result({n_out_channel_, output_shape[0], output_shape[1]});

#ifdef _OPENMP
#    pragma omp parallel for collapse(3) schedule(static)
#endif
    for (uint32_t out_ch = 0; out_ch < n_out_channel_; ++out_ch) {
        for (uint32_t out_i = 0; out_i < output_shape[0]; ++out_i) {
            for (uint32_t out_j = 0; out_j < output_shape[1]; ++out_j) {
                result.set(out_ch, out_i, out_j,
                           compute_output_element(out_ch, Duo{out_i, out_j}, padded_input, weight_scale));
            }
        }
    }

    return result;
}
