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

#include "conv1d_packed_layer.h"
#include "conv2d_layer.h"
#include "../common.h"
#include "util.h"

Conv1DPackedLayer::Conv1DPackedLayer(const CkksParameter& param_in,
                                     const uint32_t input_shape_in,
                                     const Array<double, 3>& weight_in,
                                     const Array<double, 1>& bias_in,
                                     const uint32_t stride_in,
                                     const uint32_t skip_in,
                                     uint32_t pack_in,
                                     uint32_t level_in,
                                     double residual_scale)
    : param(param_in.copy()), weight(weight_in.copy()), bias(bias_in.copy()) {
    input_shape = input_shape_in;
    skip = skip_in;

    n_channel_per_ct = pack_in;
    level = level_in;
    stride = stride_in;
    weight_scale = param.get_q(level) * residual_scale;
    n_channel_out = weight.get_shape()[0];
    n_channel_in = weight.get_shape()[1];
    kernel_shape = weight.get_shape()[2];
    n_packed_in_channel = div_ceil(n_channel_in, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_channel_out, n_channel_per_ct);
}

Conv1DPackedLayer::~Conv1DPackedLayer() {}

void Conv1DPackedLayer::prepare_weight() {
    uint32_t shape_ct = input_shape * skip;
    uint32_t half_kernel_shape = kernel_shape / 2;
    uint32_t slot_count = div_ceil(param.get_n(), 2);
    vector<vector<double>> kernel_mask(kernel_shape);

    for (int i = 0; i < kernel_shape; i++) {
        kernel_mask[i].resize(shape_ct, 1);
        if (half_kernel_shape <= i && i < (kernel_shape - half_kernel_shape)) {
            continue;
        } else {
            for (int j = 0; j < shape_ct; j++) {
                if (i < half_kernel_shape && j < (half_kernel_shape - i) * skip) {
                    kernel_mask[i][j] = 0;
                } else if (i >= (kernel_shape - half_kernel_shape) &&
                           j >= (shape_ct - (i - half_kernel_shape) * skip)) {
                    kernel_mask[i][j] = 0;
                } else {
                    if (j % stride * skip == 0) {
                        kernel_mask[i][j] = 1.0;
                    } else {
                        kernel_mask[i][j] = 0;
                    }
                }
            }
        }
    }

    vector<vector<vector<vector<double>>>> kernel_temp;
    vector<vector<double>> bias_tmp;
    uint32_t rot_n_channel_per_ct = (n_channel_in < n_channel_per_ct) ? n_channel_in : n_channel_per_ct;
    kernel_temp.resize(n_packed_out_channel);
    weight_pt.resize(n_packed_out_channel);
    bias_pt.resize(n_packed_out_channel);
    bias_tmp.resize(n_packed_out_channel);
    for (int packed_out_channel_idx = 0; packed_out_channel_idx < n_packed_out_channel; packed_out_channel_idx++) {
        kernel_temp[packed_out_channel_idx].resize(rot_n_channel_per_ct);
        weight_pt[packed_out_channel_idx].resize(rot_n_channel_per_ct);
        for (int packed_in_channel_idx = 0; packed_in_channel_idx < n_packed_in_channel; packed_in_channel_idx++) {
            for (int rot_idx = 0; rot_idx < rot_n_channel_per_ct; rot_idx++) {
                kernel_temp[packed_out_channel_idx][packed_in_channel_idx * rot_n_channel_per_ct + rot_idx].resize(
                    kernel_shape);
                weight_pt[packed_out_channel_idx][packed_in_channel_idx * rot_n_channel_per_ct + rot_idx].resize(
                    kernel_shape);
                for (int pack_idx = 0; pack_idx < n_channel_per_ct; pack_idx++) {
                    int out_channel_idx = packed_out_channel_idx * n_channel_per_ct + pack_idx;
                    int in_channel_idx = packed_in_channel_idx * n_channel_per_ct +
                                         (rot_idx + pack_idx + n_channel_per_ct) % n_channel_in;
                    for (int kernel_idx = 0; kernel_idx < kernel_shape; kernel_idx++) {
                        if (out_channel_idx < n_channel_out && in_channel_idx < n_channel_in) {
                            for (int k = 0; k < shape_ct; k++) {
                                kernel_temp[packed_out_channel_idx][packed_in_channel_idx * n_channel_per_ct + rot_idx]
                                           [kernel_idx]
                                               .push_back(weight.get(out_channel_idx, in_channel_idx, kernel_idx) *
                                                          kernel_mask[kernel_idx][k]);
                            }
                        } else {
                            for (int k = 0; k < shape_ct; k++) {
                                kernel_temp[packed_out_channel_idx][packed_in_channel_idx * n_channel_per_ct + rot_idx]
                                           [kernel_idx]
                                               .push_back(0);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int packed_out_channel_idx = 0; packed_out_channel_idx < n_packed_out_channel; packed_out_channel_idx++) {
        for (int pack_idx = 0; pack_idx < n_channel_per_ct; pack_idx++) {
            int idx = packed_out_channel_idx * n_channel_per_ct + pack_idx;
            for (int k = 0; k < shape_ct; k++) {
                if (idx < n_channel_out && k % stride == 0) {
                    bias_tmp[packed_out_channel_idx].push_back(bias.get(idx));
                } else {
                    bias_tmp[packed_out_channel_idx].push_back(0);
                }
            }
        }
    }

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    for (int i = 0; i < n_packed_out_channel; i++) {
        for (int j = 0; j < rot_n_channel_per_ct; j++) {
            for (int k = 0; k < kernel_shape; k++) {
                weight_pt[i][j][k] = ctx.encode_ringt(kernel_temp[i][j][k], weight_scale);
            }
        }
        bias_pt[i] = ctx.encode(bias_tmp[i], level, ctx.get_parameter().get_default_scale());
    }
}

void Conv1DPackedLayer::prepare_weight_lazy() {
    uint32_t half_kernel_shape = kernel_shape / 2;
    uint32_t shape_ct = input_shape * skip;

    // Generate kernel masks
    kernel_masks_.clear();
    for (int i = 0; i < kernel_shape; i++) {
        std::vector<double> mask;
        mask.reserve(shape_ct);

        if (half_kernel_shape <= i && i < (kernel_shape - half_kernel_shape)) {
            for (int j = 0; j < shape_ct; j++) {
                if (j % stride == 0) {
                    mask.push_back(1.0);
                } else {
                    mask.push_back(0.0);
                }
            }
        } else {
            for (int j = 0; j < shape_ct; j++) {
                if (i < half_kernel_shape && j < (half_kernel_shape - i) * skip) {
                    mask.push_back(0.0);
                } else if (i >= (kernel_shape - half_kernel_shape) &&
                           j >= (shape_ct - (i - half_kernel_shape) * skip)) {
                    mask.push_back(0.0);
                } else {
                    if (j % stride == 0) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
        }

        kernel_masks_.push_back(std::move(mask));
    }
}

CkksPlaintextRingt Conv1DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const {
    uint32_t shape_ct = input_shape * skip;
    uint32_t rot_n_channel_per_ct = (n_channel_in < n_channel_per_ct) ? n_channel_in : n_channel_per_ct;

    const auto& mask = kernel_masks_[k];
    const double encode_pt_scale = weight_scale;

    std::vector<double> packed_weights;
    packed_weights.reserve(param.get_n() / 2);

    for (int pack_idx = 0; pack_idx < n_channel_per_ct; pack_idx++) {
        int out_channel_idx = ct_idx * n_channel_per_ct + pack_idx;
        int in_channel_idx = (j / rot_n_channel_per_ct) * n_channel_per_ct +
                             (j % rot_n_channel_per_ct + pack_idx + n_channel_per_ct) % n_channel_in;

        if (out_channel_idx < n_channel_out && in_channel_idx < n_channel_in) {
            const double weight_val = weight.get(out_channel_idx, in_channel_idx, k);
            for (int slot_idx = 0; slot_idx < shape_ct; slot_idx++) {
                packed_weights.push_back(weight_val * mask[slot_idx]);
            }
        } else {
            packed_weights.insert(packed_weights.end(), shape_ct, 0.0);
        }
    }

    return ctx.encode_ringt(packed_weights, encode_pt_scale);
}

CkksPlaintextRingt Conv1DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
    uint32_t shape_ct = input_shape * skip;
    const double bias_scale = param.get_default_scale();

    std::vector<double> packed_bias;
    for (int pack_idx = 0; pack_idx < n_channel_per_ct; pack_idx++) {
        int idx = bpt_idx * n_channel_per_ct + pack_idx;

        for (int k = 0; k < shape_ct; k++) {
            if (idx < n_channel_out && k % stride == 0) {
                packed_bias.push_back(bias.get(idx));
            } else {
                packed_bias.push_back(0.0);
            }
        }
    }
    return ctx.encode_ringt(packed_bias, bias_scale);
}

Feature1DEncrypted Conv1DPackedLayer::run(CkksContext& ctx, Feature1DEncrypted& x) {
    Feature1DEncrypted result(x.context, x.level);
    std::cout << std::fixed << std::setprecision(10);
    result.data = move(run_core(ctx, x.data));
    result.n_channel = n_channel_out;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.shape = x.shape / stride;
    result.skip = x.skip * stride;
    result.level = x.level - 1;
    return result;
}

vector<CkksCiphertext> Conv1DPackedLayer::run_core(CkksContext& ctx, std::vector<CkksCiphertext>& x) {
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    vector<CkksCiphertext> input_rotated_x;
    uint32_t rot_num = (n_channel_in < n_channel_per_ct) ? n_channel_in : n_channel_per_ct;
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = Conv2DLayer::populate_rotations_1_side(ctx_copy, x[x_id], rot_num - 1, input_shape * skip);
    });

    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    int rotated_size = input_rotated_x.size();
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            Conv2DLayer::populate_rotations_2_sides(ctx_copy, input_rotated_x[ct_idx], kernel_shape, skip);
        move(rotations.begin(), rotations.end(), back_inserter(rotated_x[ct_idx]));
    });

    std::vector<cxx_sdk_v2::CkksCiphertext> result(n_packed_out_channel);
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        mult_add(&ctx_copy, rotated_x, ct_idx, ct_idx + 1, result);
    });

    return result;
}

void Conv1DPackedLayer::mult_add(CkksContext* ctx,
                                 vector<vector<CkksCiphertext>>& rotated_x,
                                 uint32_t start,
                                 uint32_t end,
                                 vector<CkksCiphertext>& result) {
    for (int packed_out_channel_idx = start; packed_out_channel_idx < end; packed_out_channel_idx++) {
        CkksCiphertext accumulator(0);

        for (int in_channel_idx = 0; in_channel_idx < rotated_x.size(); in_channel_idx++) {
            for (int kernel_idx = 0; kernel_idx < kernel_shape; kernel_idx++) {
                const auto& x_ct = rotated_x[in_channel_idx][kernel_idx];

                CkksCiphertext product;
                if (weight_pt.empty()) {
                    // Lazy mode: generate weight on-demand
                    CkksPlaintextRingt w_pt_rt =
                        generate_weight_pt_for_indices(*ctx, packed_out_channel_idx, in_channel_idx, kernel_idx);
                    auto w_pt = ctx->ringt_to_mul(w_pt_rt, level);
                    product = ctx->mult_plain_mul(x_ct, w_pt);
                } else {
                    // Normal mode: use pre-generated weight
                    const auto& w_pt_rt = weight_pt[packed_out_channel_idx][in_channel_idx][kernel_idx];
                    auto w_pt = ctx->ringt_to_mul(w_pt_rt, level);
                    product = ctx->mult_plain_mul(x_ct, w_pt);
                }

                if (in_channel_idx == 0 && kernel_idx == 0) {
                    accumulator = move(product);
                } else {
                    accumulator = ctx->add(accumulator, product);
                }
            }
        }

        // Add bias
        if (bias_pt.empty()) {
            // Lazy mode: generate bias on-demand
            CkksPlaintextRingt bias_plaintext = generate_bias_pt_for_index(*ctx, packed_out_channel_idx);
            accumulator = ctx->add_plain_ringt(accumulator, bias_plaintext);
        } else {
            // Normal mode: use pre-generated bias
            const auto& bias_plaintext = bias_pt[packed_out_channel_idx];
            accumulator = ctx->add_plain(accumulator, bias_plaintext);
        }

        result[packed_out_channel_idx] = move(accumulator);
    }
}

Array<double, 2> Conv1DPackedLayer::plaintext_call(const Array<double, 2>& x) {
    Array<double, 2> output({n_channel_out, input_shape / stride});
    uint32_t padding_shape = kernel_shape / 2;
    Array<double, 2> padding_input({n_channel_in, input_shape + padding_shape * 2});

    for (int i = 0; i < n_channel_in; i++) {
        for (int j = 0; j < input_shape + padding_shape * 2; j++) {
            if (j < padding_shape or (j - padding_shape) >= input_shape) {
                padding_input.set(i, j, 0);
            } else {
                padding_input.set(i, j, x.get(i, j - padding_shape));
            }
        }
    }
    for (int i = 0; i < n_channel_out; i++) {
        for (int j = 0; j < input_shape / stride; j++) {
            double s = bias[i];
            for (int ic = 0; ic < n_channel_in; ic++) {
                for (int k = 0; k < kernel_shape; k++) {
                    s += padding_input.get(ic, j * stride + k) * weight.get(i, ic, k);
                }
            }
            output.set(i, j, s);
        }
    }
    return output;
}
