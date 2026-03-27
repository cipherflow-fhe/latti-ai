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

#include "conv2d_packed_layer.h"
#include "util.h"

#include <array>
#include <cmath>
#include <deque>
#include <future>
#include <thread>
#include <immintrin.h>

using namespace std;
using namespace cxx_sdk_v2;

// ============================================================================
// Constructor and Destructor
// ============================================================================

Conv2DPackedLayer::Conv2DPackedLayer(const CkksParameter& param,
                                     const Duo& input_shape,
                                     const Array<double, 4>& weight,
                                     const Array<double, 1>& bias,
                                     const Duo& stride,
                                     const Duo& skip,
                                     uint32_t n_channel_per_ct,
                                     uint32_t level,
                                     double residual_scale)
    : Conv2DLayer(param, input_shape, weight, bias, stride, skip), n_channel_per_ct_(n_channel_per_ct),
      n_packed_ct_in_(div_ceil(n_in_channel_, n_channel_per_ct)),
      n_packed_ct_out_(div_ceil(n_out_channel_, n_channel_per_ct)),
      weight_scale_(param_.get_q(level) * residual_scale) {
    level_ = level;
}

// ============================================================================
// Weight Preparation
// ============================================================================

void Conv2DPackedLayer::prepare_weight() {
    const std::array<uint32_t, 2> padding_shape{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    const double encode_pt_scale = weight_scale_;
    const double bias_scale = param_.get_default_scale();

    kernel_masks_.clear();
    for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
        for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
            std::vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);

            for (uint32_t i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (uint32_t j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    const bool valid_i = (ki * skip_[0] + i_s >= padding_shape[0]) &&
                                         (ki * skip_[0] + i_s - padding_shape[0] < input_shape_ct[0]);
                    const bool valid_j = (kj * skip_[1] + j_s >= padding_shape[1]) &&
                                         (kj * skip_[1] + j_s - padding_shape[1] < input_shape_ct[1]);
                    const bool aligned_stride = (i_s % stride_[0] == 0) && (j_s % stride_[1] == 0);
                    const bool aligned_skip_stride =
                        (i_s % (skip_[0] * stride_[0]) == 0) && (j_s % (skip_[1] * stride_[1]) == 0);
                    const bool aligned_skip = (i_s % skip_[0] == 0) && (j_s % skip_[1] == 0);

                    if (valid_i && valid_j && aligned_stride && aligned_skip_stride && aligned_skip) {
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

    weight_pt_.clear();
    bias_pt_.clear();

    weight_pt_.resize(n_packed_ct_out_);
    const uint32_t kernel_size = kernel_shape_[0] * kernel_shape_[1];

    for (uint32_t i = 0; i < n_packed_ct_out_; i++) {
        weight_pt_[i].resize(n_packed_ct_in_ * n_channel_per_ct_);
        for (uint32_t j = 0; j < n_packed_ct_in_ * n_channel_per_ct_; j++) {
            CkksPlaintextRingt dummy(0);
            weight_pt_[i][j].push_back(std::move(dummy));
        }
        CkksPlaintextRingt bias_dummy(0);
        bias_pt_.push_back(std::move(bias_dummy));
    }

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    ctx.resize_copies(n_packed_ct_out_);

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
#endif
    for (int packed_out_ct_idx = 0; packed_out_ct_idx < static_cast<int>(n_packed_ct_out_); packed_out_ct_idx++) {
        CkksContext& ctx_copy = ctx.get_copy(packed_out_ct_idx);

        for (uint32_t packed_in_ct_idx = 0; packed_in_ct_idx < n_packed_ct_in_; packed_in_ct_idx++) {
            for (uint32_t rotate_idx = 0; rotate_idx < n_channel_per_ct_; rotate_idx++) {
                std::vector<CkksPlaintextRingt> encoded_kernels;

                for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
                    for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
                        const uint32_t mask_idx = ki * kernel_shape_[1] + kj;
                        const auto& mask = kernel_masks_[mask_idx];

                        std::vector<double> packed_weights;
                        packed_weights.reserve(param_.get_n() / 2);

                        for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
                            const uint32_t out_ch_idx = packed_out_ct_idx * n_channel_per_ct_ + pack_idx;
                            const uint32_t in_ch_idx = packed_in_ct_idx * n_channel_per_ct_ +
                                                       (rotate_idx + pack_idx + n_channel_per_ct_) % n_channel_per_ct_;

                            // prepare plaintext weight for (out_ch_idx, in_ch_idx)-SISO convolution
                            if (in_ch_idx < n_in_channel_ && out_ch_idx < n_out_channel_) {
                                const double weight_val = weight_.get(out_ch_idx, in_ch_idx, ki, kj);
                                for (uint32_t slot_idx = 0; slot_idx < input_shape_ct[0] * input_shape_ct[1];
                                     slot_idx++) {
                                    packed_weights.push_back(weight_val * mask[slot_idx]);
                                }
                            } else {
                                packed_weights.insert(packed_weights.end(), input_shape_ct[0] * input_shape_ct[1], 0.0);
                            }
                        }

                        auto encoded = ctx_copy.encode_ringt(packed_weights, encode_pt_scale);
                        encoded_kernels.push_back(std::move(encoded));
                    }
                }
                weight_pt_[packed_out_ct_idx][packed_in_ct_idx * n_channel_per_ct_ + rotate_idx] =
                    std::move(encoded_kernels);
            }
        }

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

#ifdef DEBUG_CONV2D_PACKED
    std::printf("Weight preparation complete: weight_pt size = [%zu][%zu][%zu]\n", weight_pt_.size(),
                weight_pt_.empty() ? 0 : weight_pt_[0].size(),
                (weight_pt_.empty() || weight_pt_[0].empty()) ? 0 : weight_pt_[0][0].size());
#endif
}

// ============================================================================
// Multiply-Accumulate Operation
// ============================================================================

void Conv2DPackedLayer::mult_add(CkksContext* ctx,
                                 std::vector<std::vector<CkksCiphertext>>& rotated_x,
                                 uint32_t start,
                                 uint32_t end,
                                 std::vector<CkksCiphertext>& result) {
    for (uint32_t packed_out_ct_idx = start; packed_out_ct_idx < end; packed_out_ct_idx++) {
        CkksCiphertext accumulator(0);

        for (uint32_t in_ch_idx = 0; in_ch_idx < n_packed_ct_in_ * n_channel_per_ct_; in_ch_idx++) {
            for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
                for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
                    const uint32_t kernel_idx = ki * kernel_shape_[1] + kj;

                    const auto& x_ct = rotated_x[in_ch_idx][kernel_idx];
                    CkksCiphertext product;
                    if (weight_pt_.empty()) {
                        CkksPlaintextRingt w_pt_rt =
                            generate_weight_pt_for_indices(*ctx, packed_out_ct_idx, in_ch_idx, kernel_idx);
                        auto w_pt = ctx->ringt_to_mul(w_pt_rt, level_);
                        product = ctx->mult_plain_mul(x_ct, w_pt);
                    } else {
                        const auto& w_pt_rt = weight_pt_[packed_out_ct_idx][in_ch_idx][kernel_idx];
                        auto w_pt = ctx->ringt_to_mul(w_pt_rt, level_);
                        product = ctx->mult_plain_mul(x_ct, w_pt);
                    }

                    if (in_ch_idx == 0 && ki == 0 && kj == 0) {
                        accumulator = std::move(product);
                    } else {
                        accumulator = ctx->add(accumulator, product);
                    }
                }
            }
        }

        accumulator = ctx->rescale(accumulator, param_.get_default_scale());
        if (bias_pt_.empty()) {
            CkksPlaintextRingt bias_plaintext = generate_bias_pt_for_index(*ctx, packed_out_ct_idx);
            accumulator = ctx->add_plain_ringt(accumulator, bias_plaintext);
        } else {
            const auto& bias_plaintext = bias_pt_[packed_out_ct_idx];
            accumulator = ctx->add_plain_ringt(accumulator, bias_plaintext);
        }
        result[packed_out_ct_idx] = std::move(accumulator);
    }
}

// ============================================================================
// Core Convolution Computation
// ============================================================================

std::vector<CkksCiphertext> Conv2DPackedLayer::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    std::vector<CkksCiphertext> channel_rotated_x;
    const uint32_t x_size = x.size();
    std::vector<std::vector<CkksCiphertext>> rotation_batches(x_size);

    const uint32_t rotation_unit = (input_shape_[0] * skip_[0]) * (input_shape_[1] * skip_[1]);

    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotation_batches[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_channel_per_ct_ - 1, rotation_unit);
    });

    for (auto& batch : rotation_batches) {
        std::move(batch.begin(), batch.end(), std::back_inserter(channel_rotated_x));
    }

    const int rotated_size = channel_rotated_x.size();
    std::vector<std::vector<CkksCiphertext>> spatially_rotated_x(rotated_size);

    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        std::vector<CkksCiphertext> row_rotations =
            populate_rotations_2_sides(ctx_copy, channel_rotated_x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);

        for (auto& row_ct : row_rotations) {
            auto col_rotations = populate_rotations_2_sides(ctx_copy, row_ct, kernel_shape_[1], input_rotate_units_[1]);
            std::move(col_rotations.begin(), col_rotations.end(), std::back_inserter(spatially_rotated_x[ct_idx]));
        }
    });

    std::vector<CkksCiphertext> result(n_packed_ct_out_);

    parallel_for(n_packed_ct_out_, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        mult_add(&ctx_copy, spatially_rotated_x, ct_idx, ct_idx + 1, result);
    });

    return result;
}

// ============================================================================
// Public Run Methods
// ============================================================================

Feature2DEncrypted Conv2DPackedLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
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
// Lazy Mode Weight Preparation
// ============================================================================

void Conv2DPackedLayer::prepare_weight_lazy() {
    const std::array<uint32_t, 2> padding_shape{kernel_shape_[0] / 2, kernel_shape_[1] / 2};

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    // Cache values for on-demand generation
    N = param_.get_n();
    cached_n_packed_in_ct = n_packed_ct_in_;
    cached_n_packed_out_ct = n_packed_ct_out_;
    cached_input_block_size = input_shape_ct[0] * input_shape_ct[1];

    // Generate kernel masks
    kernel_masks_.clear();
    for (uint32_t ki = 0; ki < kernel_shape_[0]; ki++) {
        for (uint32_t kj = 0; kj < kernel_shape_[1]; kj++) {
            std::vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);

            for (uint32_t i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (uint32_t j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    const bool valid_i = (ki * skip_[0] + i_s >= padding_shape[0]) &&
                                         (ki * skip_[0] + i_s - padding_shape[0] < input_shape_ct[0]);
                    const bool valid_j = (kj * skip_[1] + j_s >= padding_shape[1]) &&
                                         (kj * skip_[1] + j_s - padding_shape[1] < input_shape_ct[1]);
                    const bool aligned_stride = (i_s % stride_[0] == 0) && (j_s % stride_[1] == 0);
                    const bool aligned_skip_stride =
                        (i_s % (skip_[0] * stride_[0]) == 0) && (j_s % (skip_[1] * stride_[1]) == 0);
                    const bool aligned_skip = (i_s % skip_[0] == 0) && (j_s % skip_[1] == 0);

                    if (valid_i && valid_j && aligned_stride && aligned_skip_stride && aligned_skip) {
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

    // Note: We don't pre-generate weight_pt_ or bias_pt_ in lazy mode
    // They will be generated on-demand via generate_weight_pt_for_indices() and generate_bias_pt_for_index()
}

CkksPlaintextRingt Conv2DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const {
    const uint32_t packed_in_ct_idx = j / n_channel_per_ct_;
    const uint32_t rotate_idx = j % n_channel_per_ct_;
    const uint32_t ki = k / kernel_shape_[1];
    const uint32_t kj = k % kernel_shape_[1];

    const std::array<uint32_t, 2> input_shape_ct{input_shape_[0] * skip_[0], input_shape_[1] * skip_[1]};

    const auto& mask = kernel_masks_[k];
    const double encode_pt_scale = weight_scale_;

    std::vector<double> packed_weights;
    packed_weights.reserve(param_.get_n() / 2);

    for (uint32_t pack_idx = 0; pack_idx < n_channel_per_ct_; pack_idx++) {
        const uint32_t out_ch_idx = ct_idx * n_channel_per_ct_ + pack_idx;
        const uint32_t in_ch_idx =
            packed_in_ct_idx * n_channel_per_ct_ + (rotate_idx + pack_idx + n_channel_per_ct_) % n_channel_per_ct_;

        if (in_ch_idx < n_in_channel_ && out_ch_idx < n_out_channel_) {
            const double weight_val = weight_.get(out_ch_idx, in_ch_idx, ki, kj);
            for (uint32_t slot_idx = 0; slot_idx < input_shape_ct[0] * input_shape_ct[1]; slot_idx++) {
                packed_weights.push_back(weight_val * mask[slot_idx]);
            }
        } else {
            packed_weights.insert(packed_weights.end(), input_shape_ct[0] * input_shape_ct[1], 0.0);
        }
    }

    return ctx.encode_ringt(packed_weights, encode_pt_scale);
}

CkksPlaintextRingt Conv2DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
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
