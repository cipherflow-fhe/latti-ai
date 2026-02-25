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

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <sstream>

#ifdef _OPENMP
#    include <omp.h>
#endif

// ============================================================================
// Constructor and Destructor
// ============================================================================

Conv2DLayer::Conv2DLayer(const CkksParameter& param,
                         const Duo& input_shape,
                         const Array<double, 4>& weight,
                         const Array<double, 1>& bias,
                         const Duo& stride,
                         const Duo& skip)
    : param_(param.copy()), input_shape_{input_shape[0], input_shape[1]}, stride_{stride[0], stride[1]},
      skip_{skip[0], skip[1]}, weight_(weight.copy()), bias_(bias.copy()) {
    const auto weight_shape = weight.get_shape();
    n_slot_ = param_.get_n() / 2;
    n_out_channel_ = weight_shape[0];
    n_in_channel_ = weight_shape[1];
    kernel_shape_[0] = weight_shape[2];
    kernel_shape_[1] = weight_shape[3];

    if ((input_shape_[0] & (input_shape_[0] - 1)) != 0 || (input_shape_[1] & (input_shape_[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: ["
                                    + std::to_string(input_shape_[0]) + ", " + std::to_string(input_shape_[1]) + "]");
    }
    if ((stride_[0] & (stride_[0] - 1)) != 0 || (stride_[1] & (stride_[1] - 1)) != 0) {
        throw std::invalid_argument("stride must be powers of 2, got: ["
                                    + std::to_string(stride_[0]) + ", " + std::to_string(stride_[1]) + "]");
    }
    if ((skip_[0] & (skip_[0] - 1)) != 0 || (skip_[1] & (skip_[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: ["
                                    + std::to_string(skip_[0]) + ", " + std::to_string(skip_[1]) + "]");
    }
}

Conv2DLayer::~Conv2DLayer() = default;

// ============================================================================
// Rotation Generation
// ============================================================================

std::vector<CkksCiphertext>
Conv2DLayer::populate_rotations_1_side(CkksContext& ctx, const CkksCiphertext& x, int n_rotation, int unit) {
    std::vector<CkksCiphertext> result;
    result.reserve(1 + n_rotation);

    result.push_back(x.copy());

    std::vector<int32_t> rotation_steps;
    rotation_steps.reserve(n_rotation);
    for (int i = 1; i <= n_rotation; i++) {
        rotation_steps.push_back(i * unit);
    }

    std::map<int32_t, CkksCiphertext> rotated_map = ctx.rotate(x, rotation_steps);
    for (auto& [step, ct] : rotated_map) {
        result.push_back(std::move(ct));
    }

    return result;
}

std::vector<CkksCiphertext>
Conv2DLayer::populate_rotations_2_sides(CkksContext& ctx, const CkksCiphertext& x, int n_rotation, int unit) {
    std::vector<CkksCiphertext> result;
    result.reserve(n_rotation);

    const int filter_center = n_rotation / 2;

    std::vector<int32_t> rotation_steps;
    for (int i = -filter_center; i < n_rotation - filter_center; i++) {
        if (i != 0) {
            rotation_steps.push_back(i * unit);
        }
    }

    std::map<int32_t, CkksCiphertext> rotated_map = ctx.rotate(x, rotation_steps);

    if (-filter_center < 0) {
        for (int i = -filter_center; i < 0; i++) {
            result.push_back(std::move(rotated_map.at(i * unit)));
        }
    }

    result.push_back(x.copy());

    if (n_rotation - filter_center > 1) {
        for (int i = 1; i < n_rotation - filter_center; i++) {
            result.push_back(std::move(rotated_map.at(i * unit)));
        }
    }

    return result;
}

// ============================================================================
// Plaintext Convolution
// ============================================================================

void Conv2DLayer::compute_output_element(Array<double, 3>& result,
                                         uint32_t out_ch,
                                         uint32_t out_i,
                                         uint32_t out_j,
                                         const std::vector<double>& padded_input,
                                         uint32_t padded_h,
                                         uint32_t padded_w,
                                         const std::array<uint32_t, 2>& output_shape,
                                         double weight_scale) const {
    double sum = bias_.get(out_ch);

    const uint32_t input_base_i = out_i * stride_[0];
    const uint32_t input_base_j = out_j * stride_[1];

    for (uint32_t in_ch = 0; in_ch < n_in_channel_; ++in_ch) {
        const uint32_t channel_offset = in_ch * padded_h * padded_w;

        for (uint32_t ki = 0; ki < kernel_shape_[0]; ++ki) {
            const uint32_t input_i = input_base_i + ki;
            const uint32_t row_offset = channel_offset + input_i * padded_w;

            for (uint32_t kj = 0; kj < kernel_shape_[1]; ++kj) {
                const uint32_t input_j = input_base_j + kj;
                const double input_val = padded_input[row_offset + input_j];
                const double weight_val = weight_.get(out_ch, in_ch, ki, kj) * weight_scale;

                sum += input_val * weight_val;
            }
        }
    }

    result.set(out_ch, out_i, out_j, sum);
}

Array<double, 3> Conv2DLayer::run_plaintext(const Array<double, 3>& x, double multiplier) {
    const auto x_shape = x.get_shape();
    const uint32_t actual_input_h = x_shape[1];
    const uint32_t actual_input_w = x_shape[2];

    if (x_shape[0] != n_in_channel_) {
        std::ostringstream oss;
        oss << "Input channels mismatch: expected " << n_in_channel_ << ", got " << x_shape[0];
        throw std::invalid_argument(oss.str());
    }

    const std::array<uint32_t, 2> padding{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    const std::array<uint32_t, 2> output_shape{actual_input_h / stride_[0], actual_input_w / stride_[1]};

    const uint32_t padded_h = actual_input_h + padding[0] * 2;
    const uint32_t padded_w = actual_input_w + padding[1] * 2;
    const double weight_scale = 1.0 / multiplier;

    std::vector<double> padded_input(n_in_channel_ * padded_h * padded_w, 0.0);

    auto padded_at = [&](uint32_t ch, uint32_t i, uint32_t j) -> double& {
        return padded_input[ch * (padded_h * padded_w) + i * padded_w + j];
    };

    for (uint32_t ch = 0; ch < n_in_channel_; ++ch) {
        for (uint32_t i = 0; i < actual_input_h; ++i) {
            for (uint32_t j = 0; j < actual_input_w; ++j) {
                padded_at(ch, i + padding[0], j + padding[1]) = x.get(ch, i, j);
            }
        }
    }

    Array<double, 3> result({n_out_channel_, output_shape[0], output_shape[1]});

#ifdef _OPENMP
#    pragma omp parallel for collapse(3) schedule(static)
#endif
    for (uint32_t out_ch = 0; out_ch < n_out_channel_; ++out_ch) {
        for (uint32_t out_i = 0; out_i < output_shape[0]; ++out_i) {
            for (uint32_t out_j = 0; out_j < output_shape[1]; ++out_j) {
                compute_output_element(result, out_ch, out_i, out_j, padded_input, padded_h, padded_w, output_shape,
                                       weight_scale);
            }
        }
    }

    return result;
}
