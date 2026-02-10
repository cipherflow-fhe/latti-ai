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

#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <thread>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "conv2d_depthwise.h"
#include "util.h"

Conv2DPackedDepthwiseLayer::Conv2DPackedDepthwiseLayer(const CkksParameter& param,
                                                       const Duo& input_shape,
                                                       const Array<double, 4>& weight,
                                                       const Array<double, 1>& bias,
                                                       const Duo& stride,
                                                       const Duo& skip,
                                                       uint32_t n_channel_per_ct,
                                                       uint32_t level,
                                                       double residual_scale)
    : Conv2DLayer(param, input_shape, weight, bias, stride, skip), n_channel_per_ct_(n_channel_per_ct),
      n_packed_in_ct_(div_ceil(n_in_channel_, n_channel_per_ct)),
      n_packed_out_ct_(div_ceil(n_out_channel_, n_channel_per_ct)), level_(level),
      modified_scale_(param_.get_q(level) * residual_scale) {}

void Conv2DPackedDepthwiseLayer::prepare_weight() {
    const std::array<uint32_t, 2> padding{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    const double encode_pt_scale = modified_scale_;
    const double bias_scale = param_.get_default_scale();

    kernel_masks_.clear();
    for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
        for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
            std::vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);

            for (uint32_t i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (uint32_t j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    const bool valid_i =
                        (ki * skip_[0] + i_s >= padding[0]) && (ki * skip_[0] + i_s - padding[0] < input_shape_ct[0]);
                    const bool valid_j =
                        (kj * skip_[1] + j_s >= padding[1]) && (kj * skip_[1] + j_s - padding[1] < input_shape_ct[1]);
                    const bool aligned_stride =
                        (i_s % (skip_[0] * stride_[0]) == 0) && (j_s % (skip_[1] * stride_[1]) == 0);

                    if (valid_i && valid_j && aligned_stride) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
            kernel_masks_.push_back(std::move(mask));
        }
    }

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[0] * 1);

    input_rotate_ranges_.clear();
    input_rotate_ranges_.push_back(padding[1]);
    input_rotate_ranges_.push_back(padding[0]);

    weight_pt_.clear();
    bias_pt_.clear();
    weight_pt_.resize(n_packed_out_ct_);
    bias_pt_.resize(n_packed_out_ct_);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    ctx.resize_copies(n_packed_out_ct_);

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
#endif
    for (int packed_out_ct_idx = 0; packed_out_ct_idx < static_cast<int>(n_packed_out_ct_); packed_out_ct_idx++) {
        auto ctx_copy = ctx.make_public_context();

        std::vector<CkksPlaintextRingt> encoded_kernels;
        for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
            for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
                const uint32_t mask_idx = ki * kernel_shape_[1] + kj;
                const auto& mask = kernel_masks_[mask_idx];

                std::vector<double> packed_weights;
                packed_weights.reserve(n_channel_per_ct_ * input_shape_ct[0] * input_shape_ct[1]);

                for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
                    const uint32_t out_ch_idx = packed_out_ct_idx * n_channel_per_ct_ + pack_idx;
                    if (out_ch_idx < n_out_channel_) {
                        const double weight_val = weight_.get(out_ch_idx, 0, ki, kj);
                        for (uint32_t slot = 0; slot < input_shape_ct[0] * input_shape_ct[1]; slot++) {
                            packed_weights.push_back(weight_val * mask[slot]);
                        }
                    } else {
                        packed_weights.insert(packed_weights.end(), input_shape_ct[0] * input_shape_ct[1], 0.0);
                    }
                }

                auto encoded = ctx_copy.encode_ringt(packed_weights, encode_pt_scale);
                encoded_kernels.push_back(std::move(encoded));
            }
        }
        weight_pt_[packed_out_ct_idx] = std::move(encoded_kernels);

        std::vector<double> packed_bias;
        for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
            const uint32_t out_ch_idx = packed_out_ct_idx * n_channel_per_ct_ + pack_idx;

            for (uint32_t i = 0; i < input_shape_ct[0]; i++) {
                for (uint32_t j = 0; j < input_shape_ct[1]; j++) {
                    const bool is_output_position =
                        (i % (skip_[0] * stride_[0]) == 0) && (j % (skip_[1] * stride_[1]) == 0);
                    if (is_output_position && out_ch_idx < n_out_channel_) {
                        packed_bias.push_back(bias_.get(out_ch_idx));
                    } else {
                        packed_bias.push_back(0.0);
                    }
                }
            }
        }

        auto encoded_bias = ctx_copy.encode_ringt(packed_bias, bias_scale);
        bias_pt_[packed_out_ct_idx] = std::move(encoded_bias);
    }
}

void Conv2DPackedDepthwiseLayer::mult_add(CkksContext* ctx,
                                          const std::vector<std::vector<CkksCiphertext>>& rotated_x,
                                          uint32_t start,
                                          uint32_t end,
                                          std::vector<CkksCiphertext>& result) const {
    for (uint32_t packed_out_ct_idx = start; packed_out_ct_idx < end; packed_out_ct_idx++) {
        CkksCiphertext accumulator(0);

        for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
            for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
                const uint32_t kernel_idx = ki * kernel_shape_[1] + kj;

                const CkksCiphertext& x_ct = rotated_x[packed_out_ct_idx][kernel_idx];
                CkksCiphertext product;
                if (weight_pt_.empty()) {
                    CkksPlaintextRingt w_pt_rt = generate_weight_pt_for_indices(*ctx, packed_out_ct_idx, kernel_idx);
                    auto w_pt = ctx->ringt_to_mul(w_pt_rt, level_);
                    product = ctx->mult_plain_mul(x_ct, w_pt);
                } else {
                    const CkksPlaintextRingt& w_pt_rt = weight_pt_[packed_out_ct_idx][kernel_idx];
                    auto w_pt = ctx->ringt_to_mul(w_pt_rt, level_);
                    product = ctx->mult_plain_mul(x_ct, w_pt);
                }
                if (ki == 0 && kj == 0) {
                    accumulator = std::move(product);
                } else {
                    accumulator = ctx->add(accumulator, product);
                }
            }
        }

        accumulator = ctx->rescale(accumulator, param_.get_default_scale());
        if (bias_pt_.empty()) {
            CkksPlaintextRingt bias_plaintext = generate_bias_pt_for_index(*ctx, packed_out_ct_idx);
            accumulator = ctx->add_plain_ringt(accumulator, bias_plaintext);
        } else {
            const CkksPlaintextRingt& bias_plaintext = bias_pt_[packed_out_ct_idx];
            accumulator = ctx->add_plain_ringt(accumulator, bias_plaintext);
        }
        result[packed_out_ct_idx] = std::move(accumulator);
    }
}

std::vector<CkksCiphertext> Conv2DPackedDepthwiseLayer::run_core(CkksContext& ctx,
                                                                 const std::vector<CkksCiphertext>& x) const {
    std::vector<CkksCiphertext> input_ciphertexts;
    input_ciphertexts.reserve(x.size());
    for (const auto& ct : x) {
        input_ciphertexts.push_back(ct.copy());
    }

    const int num_inputs = input_ciphertexts.size();
    std::vector<std::vector<CkksCiphertext>> spatially_rotated_x(num_inputs);

    parallel_for(num_inputs, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        std::vector<CkksCiphertext> row_rotations =
            populate_rotations_2_sides(ctx_copy, input_ciphertexts[ct_idx], kernel_shape_[0], input_rotate_units_[0]);

        for (auto& row_ct : row_rotations) {
            auto col_rotations = populate_rotations_2_sides(ctx_copy, row_ct, kernel_shape_[1], input_rotate_units_[1]);
            std::move(col_rotations.begin(), col_rotations.end(), std::back_inserter(spatially_rotated_x[ct_idx]));
        }
    });

    std::vector<CkksCiphertext> result(n_packed_out_ct_);

    parallel_for(n_packed_out_ct_, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        mult_add(&ctx_copy, spatially_rotated_x, ct_idx, ct_idx + 1, result);
    });

    return result;
}

// ============================================================================
// Public Run Methods
// ============================================================================

Feature2DEncrypted Conv2DPackedDepthwiseLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(x.context, x.level);

    result.data = run_core(ctx, x.data);

    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.shape[0] = x.shape[0] / stride_[0];
    result.shape[1] = x.shape[1] / stride_[1];
    result.skip[0] = x.skip[0] * stride_[0];
    result.skip[1] = x.skip[1] * stride_[1];
    result.level = x.level - 1;
    return result;
}

// ============================================================================
// Plaintext Depthwise Convolution (for Testing/Verification)
// ============================================================================

Array<double, 3> Conv2DPackedDepthwiseLayer::run_plaintext(const Array<double, 3>& x, double multiplier) {
    const double weight_scale = 1.0 / multiplier;

    const std::array<uint32_t, 2> padding{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    Array<double, 3> padded_input({n_out_channel_, input_shape_[0] + padding[0] * 2, input_shape_[1] + padding[1] * 2},
                                  0.0);

    for (uint32_t ch = 0; ch < n_out_channel_; ++ch) {
        for (uint32_t i = 0; i < input_shape_[0]; ++i) {
            for (uint32_t j = 0; j < input_shape_[1]; ++j) {
                padded_input.set(ch, i + padding[0], j + padding[1], x.get(ch, i, j));
            }
        }
    }

    const std::array<uint32_t, 2> output_shape{input_shape_[0] / stride_[0], input_shape_[1] / stride_[1]};

    Array<double, 3> result({n_out_channel_, output_shape[0], output_shape[1]});

#ifdef _OPENMP
#    pragma omp parallel for collapse(3) schedule(static)
#endif
    for (uint32_t out_ch = 0; out_ch < n_out_channel_; ++out_ch) {
        for (uint32_t out_i = 0; out_i < output_shape[0]; ++out_i) {
            for (uint32_t out_j = 0; out_j < output_shape[1]; ++out_j) {
                const double value = compute_depthwise_element(out_ch, out_i, out_j, padded_input, weight_scale);
                result.set(out_ch, out_i, out_j, value);
            }
        }
    }

    return result;
}

// ============================================================================
// Helper Function: Compute Depthwise Element
// ============================================================================

double Conv2DPackedDepthwiseLayer::compute_depthwise_element(uint32_t out_ch,
                                                             uint32_t out_i,
                                                             uint32_t out_j,
                                                             const Array<double, 3>& padded_input,
                                                             double weight_scale) const {
    double sum = bias_.get(out_ch);

    // Depthwise operation: convolve with single channel (out_ch == in_ch)
    for (uint32_t ki = 0; ki < kernel_shape_[0]; ++ki) {
        for (uint32_t kj = 0; kj < kernel_shape_[1]; ++kj) {
            const uint32_t input_i = out_i * stride_[0] + ki;
            const uint32_t input_j = out_j * stride_[1] + kj;

            const double input_val = padded_input.get(out_ch, input_i, input_j);
            const double weight_val = weight_.get(out_ch, 0, ki, kj) * weight_scale;

            sum += input_val * weight_val;
        }
    }

    return sum;
}

// ============================================================================
// Lazy Mode Weight Preparation
// ============================================================================

void Conv2DPackedDepthwiseLayer::prepare_weight_lazy() {
    const std::array<uint32_t, 2> padding{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    // Cache values for on-demand generation
    N = param_.get_n();
    cached_input_block_size = input_shape_ct[0] * input_shape_ct[1];

    // Generate kernel masks
    kernel_masks_.clear();
    for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
        for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
            std::vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);

            for (uint32_t i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (uint32_t j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    const bool valid_i =
                        (ki * skip_[0] + i_s >= padding[0]) && (ki * skip_[0] + i_s - padding[0] < input_shape_ct[0]);
                    const bool valid_j =
                        (kj * skip_[1] + j_s >= padding[1]) && (kj * skip_[1] + j_s - padding[1] < input_shape_ct[1]);
                    const bool aligned_stride =
                        (i_s % (skip_[0] * stride_[0]) == 0) && (j_s % (skip_[1] * stride_[1]) == 0);

                    if (valid_i && valid_j && aligned_stride) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
            kernel_masks_.push_back(std::move(mask));
        }
    }

    // Cache rotation units
    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[0] * 1);
}

CkksPlaintextRingt
Conv2DPackedDepthwiseLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j) const {
    const uint32_t packed_out_ct_idx = ct_idx;
    const uint32_t ki = j / kernel_shape_[1];
    const uint32_t kj = j % kernel_shape_[1];

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    const auto& mask = kernel_masks_[j];
    const double encode_pt_scale = modified_scale_;

    std::vector<double> packed_weights;
    packed_weights.reserve(n_channel_per_ct_ * input_shape_ct[0] * input_shape_ct[1]);

    for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
        const uint32_t out_ch_idx = packed_out_ct_idx * n_channel_per_ct_ + pack_idx;
        if (out_ch_idx < n_out_channel_) {
            const double weight_val = weight_.get(out_ch_idx, 0, ki, kj);
            for (uint32_t slot = 0; slot < input_shape_ct[0] * input_shape_ct[1]; slot++) {
                packed_weights.push_back(weight_val * mask[slot]);
            }
        } else {
            packed_weights.insert(packed_weights.end(), input_shape_ct[0] * input_shape_ct[1], 0.0);
        }
    }

    return ctx.encode_ringt(packed_weights, encode_pt_scale);
}

CkksPlaintextRingt Conv2DPackedDepthwiseLayer::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
    const uint32_t packed_out_ct_idx = bpt_idx;

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    const double bias_scale = param_.get_default_scale();

    std::vector<double> packed_bias;
    for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
        const uint32_t out_ch_idx = packed_out_ct_idx * n_channel_per_ct_ + pack_idx;

        for (uint32_t i = 0; i < input_shape_ct[0]; i++) {
            for (uint32_t j = 0; j < input_shape_ct[1]; j++) {
                const bool is_output_position =
                    (i % (skip_[0] * stride_[0]) == 0) && (j % (skip_[1] * stride_[1]) == 0);
                if (is_output_position && out_ch_idx < n_out_channel_) {
                    packed_bias.push_back(bias_.get(out_ch_idx));
                } else {
                    packed_bias.push_back(0.0);
                }
            }
        }
    }

    return ctx.encode_ringt(packed_bias, bias_scale);
}
