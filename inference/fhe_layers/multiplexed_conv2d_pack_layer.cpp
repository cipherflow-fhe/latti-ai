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

#include <math.h>
#include "conv2d_layer.h"
#include "../common.h"
#include "multiplexed_conv2d_pack_layer.h"

CkksCiphertext sum_slot(CkksContext& ctx, CkksCiphertext& x, uint32_t m, uint32_t p) {
    CkksCiphertext result = x.copy();
    for (int j = 1; j < std::floor(log2(m)) + 1; j++) {
        auto res = ctx.rotate(result, pow(2, j - 1) * p);
        result = ctx.add(result, res);
    }

    for (int j = 0; j < std::floor(log2(m)) - 1; j++) {
        if (int(std::floor(m / pow(2, j))) % 2 == 1) {
            auto res = ctx.rotate(result, std::floor(m / pow(2, j + 1)) * pow(2, j + 1) * p);
            result = ctx.add(result, res);
        }
    }
    return result;
}

vector<double> ParMultiplexedConv2DPackedLayer::select_tensor(int num) const {
    vector<double> tensor;
    for (int k = 0; k < n_block_per_ct; k++) {
        for (int i = 0; i < input_shape_[0] * skip_[0]; i++) {
            for (int j = 0; j < input_shape_[1] * skip_[1]; j++) {
                if ((i % (skip_[0] * stride_[0])) < zero_inserted_skip[0] &&
                    (j % (skip_[1] * stride_[1])) < zero_inserted_skip[1] &&
                    k * zero_inserted_skip[0] * zero_inserted_skip[1] +
                            zero_inserted_skip[1] * (i % zero_inserted_skip[0]) + (j % (zero_inserted_skip[1])) ==
                        num) {
                    tensor.push_back(1.0);
                } else {
                    tensor.push_back(0.0);
                }
            }
        }
    }

    return tensor;
}

ParMultiplexedConv2DPackedLayer::ParMultiplexedConv2DPackedLayer(const CkksParameter& param_in,
                                                                 const Duo& input_shape_in,
                                                                 const Array<double, 4>& weight_in,
                                                                 const Array<double, 1>& bias_in,
                                                                 const Duo& stride_in,
                                                                 const Duo& skip_in,
                                                                 uint32_t n_channel_per_ct_in,
                                                                 uint32_t level_in,
                                                                 double residual_scale,
                                                                 const Duo& upsample_factor_in)
    : Conv2DLayer(param_in, input_shape_in, weight_in, bias_in, stride_in, skip_in) {
    upsample_factor[0] = upsample_factor_in[0];
    upsample_factor[1] = upsample_factor_in[1];
    zero_inserted_skip[0] = skip_in[0] * stride_in[0] / upsample_factor_in[0];
    zero_inserted_skip[1] = skip_in[1] * stride_in[1] / upsample_factor_in[1];
    n_channel_per_ct = n_channel_per_ct_in;
    n_packed_in_channel = div_ceil(n_in_channel_, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_out_channel_, n_channel_per_ct * stride_in[0] * stride_in[1] /
                                                        (upsample_factor[0] * upsample_factor[1]));
    n_block_per_ct = div_ceil(n_channel_per_ct, (skip_[0] * skip_[1]));
    level = level_in;
    weight_scale = param_.get_q(level) * residual_scale;
    N = param_in.get_n();
}

ParMultiplexedConv2DPackedLayer::~ParMultiplexedConv2DPackedLayer() {}

void ParMultiplexedConv2DPackedLayer::prepare_weight_for_reduct_rot() {
    uint32_t pad0 = std::floor(kernel_shape_[0] / 2);
    uint32_t pad1 = std::floor(kernel_shape_[1] / 2);

    uint32_t padding_shape[] = {pad0, pad1};
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape_[0] * skip_[0];
    input_shape_ct[1] = input_shape_[1] * skip_[1];
    kernel_masks_.clear();
    double scale_new = 0.0;
    double bias_scale = 0.0;

    for (int i = 0; i < kernel_shape_[0]; i++) {
        for (int j = 0; j < kernel_shape_[1]; j++) {
            vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);
            for (int i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (int j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    if (i * skip_[0] + i_s - padding_shape[0] * skip_[0] >= 0 &&
                        i * skip_[0] + i_s - padding_shape[0] * skip_[0] < input_shape_ct[0] &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] >= 0 &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] < input_shape_ct[1]) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
            kernel_masks_.push_back(mask);
        }
    }

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[1] * 1);
    weight_pt.clear();
    bias_pt.clear();

    uint32_t skip_out_0 = skip_[0] * stride_[0] / upsample_factor[0];
    uint32_t skip_out_1 = skip_[1] * stride_[1] / upsample_factor[1];
    uint32_t skip_out_prod = skip_out_0 * skip_out_1;

    // Reduct_rot needs one intermediate ct per (output_ct_group, sub_pos) pair
    uint32_t n_weight_pt = n_packed_out_channel * skip_out_prod;
    weight_pt.resize(n_weight_pt);

    for (int i = 0; i < n_weight_pt; i++) {
        weight_pt[i].resize(n_packed_in_channel * n_block_per_ct);
    }
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    ctx.resize_copies(n_weight_pt);

    int kernel_size = kernel_shape_[0] * kernel_shape_[1];
    int input_block_size = input_shape_ct[0] * input_shape_ct[1];
    parallel_for(n_weight_pt, th_nums, ctx, [&](CkksContext& ctx_copy, int weight_pt_num_idx) {
        // Reduct_rot ordering: channels ordered by final multiplexed position
        // sub_pos = weight_pt_num_idx % skip_out_prod
        // output_ct_group = weight_pt_num_idx / skip_out_prod
        uint32_t sub_pos = weight_pt_num_idx % skip_out_prod;
        uint32_t output_ct_group = weight_pt_num_idx / skip_out_prod;

        for (int packed_in_channel_idx = 0; packed_in_channel_idx < n_packed_in_channel; ++packed_in_channel_idx) {
            int base_channel_in = packed_in_channel_idx * n_channel_per_ct;
            for (int block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
                vector<CkksPlaintextRingt> a1(kernel_size);
                int total_skip = skip_[0] * skip_[1];

                for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
                    auto& mask = kernel_masks_[kernel_idx];
                    vector<double> w(N / 2);
                    for (int linear_idx = 0; linear_idx < n_block_per_ct * input_block_size; ++linear_idx) {
                        int t = linear_idx / input_block_size;
                        int shape_linear = linear_idx % input_block_size;
                        int shape_i = shape_linear / input_shape_ct[1];
                        int shape_j = shape_linear % input_shape_ct[1];
                        int kernel_shape_i = kernel_idx / kernel_shape_[1];
                        int kernel_shape_j = kernel_idx % kernel_shape_[1];

                        uint32_t channel_in =
                            base_channel_in + (block_idx * total_skip + t * total_skip + (shape_j % skip_[1]) +
                                               (shape_i % skip_[0]) * skip_[1]) %
                                                  n_channel_per_ct;
                        // Reduct_rot channel_out: ordered by final multiplexed position
                        // block in output = (t + n_block_per_ct) % n_block_per_ct
                        // channel = output_ct_group * n_channel_per_ct_out + block * skip_out_prod + sub_pos
                        uint32_t channel_out = output_ct_group * n_block_per_ct * skip_out_prod +
                                               ((t + n_block_per_ct) % n_block_per_ct) * skip_out_prod + sub_pos;
                        w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                                            0 :
                                            weight_.get(channel_out, channel_in, kernel_shape_i, kernel_shape_j) *
                                                mask[shape_i * input_shape_ct[1] + shape_j];
                    }
                    a1[kernel_idx] = ctx_copy.encode_ringt(w, weight_scale);
                }
                weight_pt[weight_pt_num_idx][packed_in_channel_idx * n_block_per_ct + block_idx] = move(a1);
            }
        }
    });
    bias_level_down = 2;
    int mask_size = min(n_block_per_ct, n_out_channel_);
    vector<vector<double>> feature_tmp_pack(n_packed_out_channel);
    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
        bias_level_down = 1;
    } else {
        // Per-block mask: for each ct_idx, each block i selects its channel's sub_pos
        mask_pt.resize(n_weight_pt);
        parallel_for(n_weight_pt, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            uint32_t sub_pos_ct = ct_idx % skip_out_prod;
            mask_pt[ct_idx].resize(mask_size);
            for (int i = 0; i < mask_size; i++) {
                // In reduct_rot, block i of ct_idx has local channel index = i * skip_out_prod + sub_pos_ct
                uint32_t channel_local = i * skip_out_prod + sub_pos_ct;
                auto si = select_tensor(channel_local);
                mask_pt[ct_idx][i] = ctx_copy.encode_ringt(si, ctx_copy.get_parameter().get_q(level - 1));
            }
        });
    }

    Duo bias_skip;
    bias_skip[0] = zero_inserted_skip[0];
    bias_skip[1] = zero_inserted_skip[1];
    int skip_prod = bias_skip[0] * bias_skip[1];
    int bias_n_channel_per_ct = n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        const int total_block_size = n_block_per_ct * input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1];
        feature_tmp_pack[n_packed_out_channel_idx].resize(ctx_copy.get_parameter().get_n() / 2);

        for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
            int j = linear_idx / (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
            int residual = linear_idx % (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
            int h = residual / (input_shape_[1] * skip_[1]);
            int k = residual % (input_shape_[1] * skip_[1]);

            int channel = n_packed_out_channel_idx * bias_n_channel_per_ct + j * skip_prod +
                          bias_skip[1] * (h % bias_skip[0]) + k % bias_skip[1];
            if (channel >= n_out_channel_ || (h % (stride_[0] * skip_[0])) >= bias_skip[0] ||
                (k % (stride_[1] * skip_[1])) >= bias_skip[1])
                continue;

            int index =
                j * (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]) + h * input_shape_[1] * skip_[1] + k;
            feature_tmp_pack[n_packed_out_channel_idx][index] = bias_.get(channel);
        }
        bias_pt[n_packed_out_channel_idx] =
            ctx_copy.encode_ringt(feature_tmp_pack[n_packed_out_channel_idx], param_.get_default_scale());
    });
}

void ParMultiplexedConv2DPackedLayer::prepare_weight_for_post_skip_rotation() {
    uint32_t pad0 = std::floor(kernel_shape_[0] / 2);
    uint32_t pad1 = std::floor(kernel_shape_[1] / 2);

    uint32_t padding_shape[] = {pad0, pad1};
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape_[0] * skip_[0];
    input_shape_ct[1] = input_shape_[1] * skip_[1];
    kernel_masks_.clear();
    double scale_new = 0.0;
    double bias_scale = 0.0;

    for (int i = 0; i < kernel_shape_[0]; i++) {
        for (int j = 0; j < kernel_shape_[1]; j++) {
            vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);
            for (int i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (int j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    if (i * skip_[0] + i_s - padding_shape[0] * skip_[0] >= 0 &&
                        i * skip_[0] + i_s - padding_shape[0] * skip_[0] < input_shape_ct[0] &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] >= 0 &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] < input_shape_ct[1]) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
            kernel_masks_.push_back(mask);
        }
    }

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[1] * 1);
    weight_pt.clear();
    bias_pt.clear();

    uint32_t n_weight_pt = div_ceil(n_out_channel_, n_block_per_ct);
    weight_pt.resize(n_weight_pt);

    for (int i = 0; i < n_weight_pt; i++) {
        weight_pt[i].resize(n_packed_in_channel * n_block_per_ct);
    }
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    ctx.resize_copies(n_weight_pt);

    int kernel_size = kernel_shape_[0] * kernel_shape_[1];
    int input_block_size = input_shape_ct[0] * input_shape_ct[1];
    parallel_for(n_weight_pt, th_nums, ctx, [&](CkksContext& ctx_copy, int weight_pt_num_idx) {
        for (int packed_in_channel_idx = 0; packed_in_channel_idx < n_packed_in_channel; ++packed_in_channel_idx) {
            int base_channel_in = packed_in_channel_idx * n_channel_per_ct;
            for (int block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
                vector<CkksPlaintextRingt> a1(kernel_size);
                int total_skip = skip_[0] * skip_[1];

                for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
                    auto& mask = kernel_masks_[kernel_idx];
                    vector<double> w(N / 2);
                    vector<Duo> wp;
                    for (int linear_idx = 0; linear_idx < n_block_per_ct * input_block_size; ++linear_idx) {
                        int t = linear_idx / input_block_size;
                        int shape_linear = linear_idx % input_block_size;
                        int shape_i = shape_linear / input_shape_ct[1];
                        int shape_j = shape_linear % input_shape_ct[1];
                        int kernel_shape_i = kernel_idx / kernel_shape_[1];
                        int kernel_shape_j = kernel_idx % kernel_shape_[1];

                        uint32_t channel_in =
                            base_channel_in + (block_idx * total_skip + t * total_skip + (shape_j % skip_[1]) +
                                               (shape_i % skip_[0]) * skip_[1]) %
                                                  n_channel_per_ct;
                        uint32_t channel_out =
                            weight_pt_num_idx * n_block_per_ct + (t + n_block_per_ct) % n_block_per_ct;
                        Duo sp = {channel_out, channel_in};
                        wp.push_back(sp);
                        w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                                            0 :
                                            weight_.get(channel_out, channel_in, kernel_shape_i, kernel_shape_j) *
                                                mask[shape_i * input_shape_ct[1] + shape_j];
                    }
                    a1[kernel_idx] = ctx_copy.encode_ringt(w, weight_scale);
                }
                weight_pt[weight_pt_num_idx][packed_in_channel_idx * n_block_per_ct + block_idx] = move(a1);
            }
        }
    });
    bias_level_down = 2;
    int mask_size = min(n_block_per_ct, n_out_channel_);
    vector<vector<double>> feature_tmp_pack(n_packed_out_channel);
    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
        bias_level_down = 1;
    } else {
        mask_pt.resize(weight_pt.size());
        parallel_for(weight_pt.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            mask_pt[ct_idx].resize(mask_size);
            for (int i = 0; i < mask_size; i++) {
                auto si = select_tensor((ct_idx * n_block_per_ct + i) % (n_channel_per_ct * stride_[0] * stride_[1] /
                                                                         (upsample_factor[0] * upsample_factor[1])));
                mask_pt[ct_idx][i] = ctx_copy.encode_ringt(si, ctx_copy.get_parameter().get_q(level - 1));
            }
        });
    }

    Duo bias_skip;
    bias_skip[0] = zero_inserted_skip[0];
    bias_skip[1] = zero_inserted_skip[1];
    int skip_prod = bias_skip[0] * bias_skip[1];
    int bias_n_channel_per_ct = n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        const int total_block_size = n_block_per_ct * input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1];
        feature_tmp_pack[n_packed_out_channel_idx].resize(ctx_copy.get_parameter().get_n() / 2);

        for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
            int j = linear_idx / (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
            int residual = linear_idx % (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
            int h = residual / (input_shape_[1] * skip_[1]);
            int k = residual % (input_shape_[1] * skip_[1]);

            int channel = n_packed_out_channel_idx * bias_n_channel_per_ct + j * skip_prod +
                          bias_skip[1] * (h % bias_skip[0]) + k % bias_skip[1];
            if (channel >= n_out_channel_ || (h % (stride_[0] * skip_[0])) >= bias_skip[0] ||
                (k % (stride_[1] * skip_[1])) >= bias_skip[1])
                continue;

            int index =
                j * (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]) + h * input_shape_[1] * skip_[1] + k;
            feature_tmp_pack[n_packed_out_channel_idx][index] = bias_.get(channel);
        }
        bias_pt[n_packed_out_channel_idx] =
            ctx_copy.encode_ringt(feature_tmp_pack[n_packed_out_channel_idx], param_.get_default_scale());
    });
}

void ParMultiplexedConv2DPackedLayer::prepare_weight_for_post_skip_rotation_lazy() {
    uint32_t pad0 = std::floor(kernel_shape_[0] / 2);
    uint32_t pad1 = std::floor(kernel_shape_[1] / 2);

    uint32_t padding_shape[] = {pad0, pad1};
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape_[0] * skip_[0];
    input_shape_ct[1] = input_shape_[1] * skip_[1];

    // Cache commonly used values for on-demand generation
    cached_input_shape_ct[0] = input_shape_ct[0];
    cached_input_shape_ct[1] = input_shape_ct[1];
    cached_input_block_size = input_shape_ct[0] * input_shape_ct[1];
    cached_kernel_size = kernel_shape_[0] * kernel_shape_[1];
    cached_total_skip = skip_[0] * skip_[1];

    // Cache bias-related values
    cached_bias_skip[0] = zero_inserted_skip[0];
    cached_bias_skip[1] = zero_inserted_skip[1];
    cached_skip_prod = cached_bias_skip[0] * cached_bias_skip[1];
    cached_bias_n_channel_per_ct =
        n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    cached_total_block_size = n_block_per_ct * input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1];

    kernel_masks_.clear();

    for (int i = 0; i < kernel_shape_[0]; i++) {
        for (int j = 0; j < kernel_shape_[1]; j++) {
            vector<double> mask;
            mask.reserve(input_shape_ct[0] * input_shape_ct[1]);
            for (int i_s = 0; i_s < input_shape_ct[0]; i_s++) {
                for (int j_s = 0; j_s < input_shape_ct[1]; j_s++) {
                    if (i * skip_[0] + i_s - padding_shape[0] * skip_[0] >= 0 &&
                        i * skip_[0] + i_s - padding_shape[0] * skip_[0] < input_shape_ct[0] &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] >= 0 &&
                        j * skip_[1] + j_s - padding_shape[1] * skip_[1] < input_shape_ct[1]) {
                        mask.push_back(1.0);
                    } else {
                        mask.push_back(0.0);
                    }
                }
            }
            kernel_masks_.push_back(mask);
        }
    }

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[1] * 1);

    // Set bias_level_down based on stride and skip
    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
        bias_level_down = 1;
    } else {
        bias_level_down = 2;
    }
}

CkksPlaintextRingt
ParMultiplexedConv2DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const {
    // Extract indices from j
    int packed_in_channel_idx = j / n_block_per_ct;
    int block_idx = j % n_block_per_ct;
    int kernel_idx = k;

    // Use cached values
    auto& mask = kernel_masks_[kernel_idx];
    vector<double> w(N / 2, 0.0);
    int base_channel_in = packed_in_channel_idx * n_channel_per_ct;

    for (int linear_idx = 0; linear_idx < n_block_per_ct * cached_input_block_size; ++linear_idx) {
        int t = linear_idx / cached_input_block_size;
        int shape_linear = linear_idx % cached_input_block_size;
        int shape_i = shape_linear / cached_input_shape_ct[1];
        int shape_j = shape_linear % cached_input_shape_ct[1];
        int kernel_shape_i = kernel_idx / kernel_shape_[1];
        int kernel_shape_j = kernel_idx % kernel_shape_[1];

        uint32_t channel_in = base_channel_in + (block_idx * cached_total_skip + t * cached_total_skip +
                                                 (shape_j % skip_[1]) + (shape_i % skip_[0]) * skip_[1]) %
                                                    n_channel_per_ct;
        uint32_t channel_out = ct_idx * n_block_per_ct + (t + n_block_per_ct) % n_block_per_ct;

        w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                            0 :
                            weight_.get(channel_out, channel_in, kernel_shape_i, kernel_shape_j) *
                                mask[shape_i * cached_input_shape_ct[1] + shape_j];
    }

    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt ParMultiplexedConv2DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
    // Use cached values
    vector<double> bias_vec(N / 2, 0.0);

    for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        int j = linear_idx / (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
        int residual = linear_idx % (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]);
        int h = residual / (input_shape_[1] * skip_[1]);
        int k = residual % (input_shape_[1] * skip_[1]);

        int channel = bpt_idx * cached_bias_n_channel_per_ct + j * cached_skip_prod +
                      cached_bias_skip[1] * (h % cached_bias_skip[0]) + k % cached_bias_skip[1];
        if (channel >= n_out_channel_ || (h % (stride_[0] * skip_[0])) >= cached_bias_skip[0] ||
            (k % (stride_[1] * skip_[1])) >= cached_bias_skip[1])
            continue;

        int index = j * (input_shape_[0] * skip_[0] * input_shape_[1] * skip_[1]) + h * input_shape_[1] * skip_[1] + k;
        bias_vec[index] = bias_.get(channel);
    }

    return ctx.encode_ringt(bias_vec, ctx.get_parameter().get_default_scale());
}

// Generate mask vector for given indices on-demand
CkksPlaintextRingt
ParMultiplexedConv2DPackedLayer::generate_mask_pt_for_indices(CkksContext& ctx, int ct_idx, int i) const {
    auto si = select_tensor((ct_idx * n_block_per_ct + i) %
                            (n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1])));
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level - 1));
}

vector<CkksCiphertext> ParMultiplexedConv2DPackedLayer::run_core(CkksContext& ctx,
                                                                 const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1,
                                                      (input_shape_[0] * skip_[0]) * (input_shape_[1] * skip_[1]));
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    vector<CkksCiphertext> input_rotated_x_skip;
    uint32_t x_size_skip = input_rotated_x.size();
    vector<vector<CkksCiphertext>> rotated_tmp_skip(x_size_skip);

    parallel_for(x_size_skip, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp_skip[x_id] = populate_rotations_1_side(ctx_copy, input_rotated_x[x_id], skip_[1] - 1, 1);
    });
    for (auto& y : rotated_tmp_skip) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x_skip));
    }

    int rotated_size = input_rotated_x_skip.size();
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations = populate_rotations_2_sides(ctx_copy, input_rotated_x_skip[ct_idx],
                                                                      kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    parallel_for(weight_pt.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s(0);
        for (int j = 0; j < weight_pt[ct_idx].size(); j++) {
            for (int k = 0; k < weight_pt[ct_idx][j].size(); k++) {
                auto& w_pt_rt = weight_pt[ct_idx][j][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                auto res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt);
                if (j == 0 && k == 0) {
                    s = move(res);
                } else {
                    s = ctx_copy.add(s, res);
                }
            }
        }

        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
        s = sum_slot(ctx_copy, s, skip_[0], skip_[1] * input_shape_[1]);
        vector<int32_t> steps;
        for (int i = 0; i < n_block_per_ct; i++) {
            int32_t p = (ct_idx * n_block_per_ct + i) %
                        (n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]));
            int32_t r_num0 = floor(p / (zero_inserted_skip[0] * zero_inserted_skip[1])) * skip_[0] * skip_[1] *
                             input_shape_[0] * input_shape_[1];
            int32_t r_num1 = floor((p % (zero_inserted_skip[0] * zero_inserted_skip[1])) / zero_inserted_skip[1]) *
                             input_shape_[1] * skip_[1];
            int32_t r_num = -r_num0 - r_num1 - p % zero_inserted_skip[1] +
                            i * skip_[0] * skip_[1] * input_shape_[0] * input_shape_[1];
            steps.push_back(r_num);
        }
        auto s_rots = ctx_copy.rotate(s, steps);
        for (int i = 0; i < n_block_per_ct; i++) {
            auto si = select_tensor((ct_idx * n_block_per_ct + i) % (n_channel_per_ct * stride_[0] * stride_[1] /
                                                                     (upsample_factor[0] * upsample_factor[1])));
            auto p_ss = ctx_copy.encode(si, level - 1, ctx_copy.get_parameter().get_q(level - 1));
            auto c_m_s = ctx_copy.mult_plain(s_rots[steps[i]], p_ss);
            if ((ct_idx * n_block_per_ct + i) < n_out_channel_) {
                result_ct[ct_idx * n_block_per_ct + i] =
                    move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
            }
        }
    });
    vector<CkksCiphertext> res;
    CkksCiphertext sp;
    for (int i = 0; i < result_ct.size(); i++) {
        int p = i % (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
        auto c_m_s = result_ct[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
            int bpt_idx = i / (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
            sp = ctx.add_plain_ringt(sp, bias_pt[bpt_idx]);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1])) == 0 ||
            i == result_ct.size() - 1) {
            res.push_back(move(sp));
        }
    }
    return res;
}

vector<CkksCiphertext>
ParMultiplexedConv2DPackedLayer::run_core_for_post_skip_rotation(CkksContext& ctx,
                                                                 const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1,
                                                      (input_shape_[0] * skip_[0]) * (input_shape_[1] * skip_[1]));
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    int rotated_size = input_rotated_x.size();
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, input_rotated_x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    vector<CkksCiphertext> res;
    uint32_t n_weight = weight_pt.empty() ? div_ceil(n_out_channel_, n_block_per_ct) : weight_pt.size();
    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
        res.resize(n_weight);
    }
    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s(0);
        uint32_t n_j = weight_pt.empty() ? n_packed_in_channel * n_block_per_ct : weight_pt[ct_idx].size();
        for (int j = 0; j < n_j; j++) {
            uint32_t n_k = weight_pt.empty() ? cached_kernel_size : weight_pt[ct_idx][j].size();
            for (int k = 0; k < n_k; k++) {
                CkksCiphertext res;
                if (weight_pt.empty()) {
                    auto w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, j, k);
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                    res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt);
                } else {
                    auto w_pt_rt = ctx_copy.ringt_to_mul(weight_pt[ct_idx][j][k], level);
                    res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt_rt);
                }
                if (j == 0 && k == 0) {
                    s = move(res);
                } else {
                    s = ctx_copy.add(s, res);
                }
            }
        }

        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
        if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_for_index(ctx_copy, ct_idx);
                res[ct_idx] = ctx.add_plain_ringt(s, b_pt);
            } else {
                res[ct_idx] = ctx.add_plain_ringt(s, bias_pt[ct_idx]);
            }
        } else {
            s = sum_slot(ctx_copy, s, skip_[0], skip_[1] * input_shape_[1]);
            s = sum_slot(ctx_copy, s, skip_[1], 1);
            vector<int32_t> steps;
            for (int i = 0; i < n_block_per_ct; i++) {
                int32_t n_block = (ct_idx * n_block_per_ct + i) % (n_channel_per_ct * stride_[0] * stride_[1] /
                                                                   (upsample_factor[0] * upsample_factor[1]));
                int32_t n_skip = floor(n_block / (zero_inserted_skip[0] * zero_inserted_skip[1])) * skip_[0] *
                                 skip_[1] * input_shape_[0] * input_shape_[1];
                int32_t n_skip_residue =
                    floor((n_block % (zero_inserted_skip[0] * zero_inserted_skip[1])) / zero_inserted_skip[1]) *
                    input_shape_[1] * skip_[1];
                int32_t rot_step = -n_skip - n_skip_residue - n_block % zero_inserted_skip[1] +
                                   i * skip_[0] * skip_[1] * input_shape_[0] * input_shape_[1];
                steps.push_back(rot_step);
            }
            auto s_rots = ctx_copy.rotate(s, steps);
            for (int i = 0; i < n_block_per_ct; i++) {
                if ((ct_idx * n_block_per_ct + i) < n_out_channel_) {
                    if (mask_pt.empty()) {
                        auto m_pt_rt = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], m_pt);
                        result_ct[ct_idx * n_block_per_ct + i] =
                            move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
                    } else {
                        auto& m_pt_rt = mask_pt[ct_idx][i];
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], m_pt);
                        result_ct[ct_idx * n_block_per_ct + i] =
                            move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
                    }
                }
            }
        }
    });
    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
    } else {
        CkksCiphertext sp;
        for (int i = 0; i < result_ct.size(); i++) {
            int p = i % (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
            auto c_m_s = result_ct[i].copy();
            if (p == 0) {
                sp = move(c_m_s);
                int bpt_idx =
                    i / (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1]));
                if (bias_pt.empty()) {
                    auto b_pt = generate_bias_pt_for_index(ctx, bpt_idx);
                    sp = ctx.add_plain_ringt(sp, b_pt);
                } else {
                    sp = ctx.add_plain_ringt(sp, bias_pt[bpt_idx]);
                }
            } else {
                sp = ctx.add(sp, c_m_s);
            }
            if ((i + 1) % (stride_[0] * stride_[1] * n_channel_per_ct / (upsample_factor[0] * upsample_factor[1])) ==
                    0 ||
                i == result_ct.size() - 1) {
                res.push_back(move(sp));
            }
        }
    }
    return res;
}

Feature2DEncrypted ParMultiplexedConv2DPackedLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0] / stride_[0] * upsample_factor[0];
    result.shape[1] = x.shape[1] / stride_[1] * upsample_factor[1];
    result.skip[0] = x.skip[0] * stride_[0] / upsample_factor[0];
    result.skip[1] = x.skip[1] * stride_[1] / upsample_factor[1];
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    result.level = x.level - 2;
    result.data = run_core(ctx, x.data);
    return result;
}

Feature2DEncrypted ParMultiplexedConv2DPackedLayer::run_for_post_skip_rotation(CkksContext& ctx,
                                                                               const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0] / stride_[0] * upsample_factor[0];
    result.shape[1] = x.shape[1] / stride_[1] * upsample_factor[1];
    result.skip[0] = x.skip[0] * stride_[0] / upsample_factor[0];
    result.skip[1] = x.skip[1] * stride_[1] / upsample_factor[1];
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    result.level = x.level - bias_level_down;
    result.data = run_core_for_post_skip_rotation(ctx, x.data);
    return result;
}

vector<CkksCiphertext> ParMultiplexedConv2DPackedLayer::run_core_for_reduct_rot(CkksContext& ctx,
                                                                                const std::vector<CkksCiphertext>& x) {
    // 1. Block direction rotations (same as post_skip)
    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1,
                                                      (input_shape_[0] * skip_[0]) * (input_shape_[1] * skip_[1]));
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    // 2. Kernel direction rotations (same as post_skip)
    int rotated_size = input_rotated_x.size();
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, input_rotated_x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    // 3. Multiply-accumulate + rescale + sum_slot + mask
    uint32_t skip_out_0 = skip_[0] * stride_[0] / upsample_factor[0];
    uint32_t skip_out_1 = skip_[1] * stride_[1] / upsample_factor[1];
    uint32_t skip_out_prod = skip_out_0 * skip_out_1;

    uint32_t n_weight = weight_pt.size();

    if (stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1) {
        // No mask needed, directly add bias
        vector<CkksCiphertext> res(n_weight);
        parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            CkksCiphertext s(0);
            for (int j = 0; j < weight_pt[ct_idx].size(); j++) {
                for (int k = 0; k < weight_pt[ct_idx][j].size(); k++) {
                    auto& w_pt_rt = weight_pt[ct_idx][j][k];
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                    auto mult_res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt);
                    if (j == 0 && k == 0) {
                        s = move(mult_res);
                    } else {
                        s = ctx_copy.add(s, mult_res);
                    }
                }
            }
            s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
            res[ct_idx] = ctx.add_plain_ringt(s, bias_pt[ct_idx]);
        });
        return res;
    }

    // stride/skip > 1: mult-accumulate + rescale + sum_slot + per-block rotate+mask
    uint32_t n_channel_per_ct_out = n_block_per_ct * skip_out_prod;
    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        uint32_t sub_pos = ct_idx % skip_out_prod;
        uint32_t output_ct_group = ct_idx / skip_out_prod;

        CkksCiphertext s(0);
        for (int j = 0; j < weight_pt[ct_idx].size(); j++) {
            for (int k = 0; k < weight_pt[ct_idx][j].size(); k++) {
                auto& w_pt_rt = weight_pt[ct_idx][j][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                auto mult_res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt);
                if (j == 0 && k == 0) {
                    s = move(mult_res);
                } else {
                    s = ctx_copy.add(s, mult_res);
                }
            }
        }

        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
        s = sum_slot(ctx_copy, s, skip_[0], skip_[1] * input_shape_[1]);
        s = sum_slot(ctx_copy, s, skip_[1], 1);

        // Per-block rotation + mask (same structure as post_skip but with reduct_rot channel ordering)
        vector<int32_t> steps;
        for (int i = 0; i < n_block_per_ct; i++) {
            // channel_local = i * skip_out_prod + sub_pos (reduct_rot ordering within ct)
            uint32_t channel_local = i * skip_out_prod + sub_pos;
            int32_t r_num0 = floor(channel_local / (zero_inserted_skip[0] * zero_inserted_skip[1])) * skip_[0] *
                             skip_[1] * input_shape_[0] * input_shape_[1];
            int32_t r_num1 =
                floor((channel_local % (zero_inserted_skip[0] * zero_inserted_skip[1])) / zero_inserted_skip[1]) *
                input_shape_[1] * skip_[1];
            int32_t rot_step = -r_num0 - r_num1 - channel_local % zero_inserted_skip[1] +
                               i * skip_[0] * skip_[1] * input_shape_[0] * input_shape_[1];
            steps.push_back(rot_step);
        }
        auto s_rots = ctx_copy.rotate(s, steps);
        for (int i = 0; i < n_block_per_ct; i++) {
            uint32_t channel_out = output_ct_group * n_channel_per_ct_out + i * skip_out_prod + sub_pos;
            if (channel_out < n_out_channel_) {
                auto& m_pt_rt = mask_pt[ct_idx][i];
                auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level - 1);
                auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i]], m_pt);
                result_ct[channel_out] = move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
            }
        }
    });

    // 4. Accumulate n_channel_per_ct_out results per output ct, then add bias
    vector<CkksCiphertext> res;
    CkksCiphertext sp;
    for (int i = 0; i < result_ct.size(); i++) {
        int p = i % n_channel_per_ct_out;
        auto c_m_s = result_ct[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
            int bpt_idx = i / n_channel_per_ct_out;
            sp = ctx.add_plain_ringt(sp, bias_pt[bpt_idx]);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % n_channel_per_ct_out == 0 || i == result_ct.size() - 1) {
            res.push_back(move(sp));
        }
    }
    return res;
}

Feature2DEncrypted ParMultiplexedConv2DPackedLayer::run_for_reduct_rot(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0] / stride_[0] * upsample_factor[0];
    result.shape[1] = x.shape[1] / stride_[1] * upsample_factor[1];
    result.skip[0] = x.skip[0] * stride_[0] / upsample_factor[0];
    result.skip[1] = x.skip[1] * stride_[1] / upsample_factor[1];
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * stride_[0] * stride_[1] / (upsample_factor[0] * upsample_factor[1]);
    result.level = x.level - bias_level_down;
    result.data = run_core_for_reduct_rot(ctx, x.data);
    return result;
}
