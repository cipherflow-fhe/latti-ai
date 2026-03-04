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
#include "multiplexed_conv2d_pack_layer_depthwise.h"

ParMultiplexedConv2DPackedLayerDepthwise::ParMultiplexedConv2DPackedLayerDepthwise(const CkksParameter& param_in,
                                                                                   const Duo& input_shape_in,
                                                                                   const Array<double, 4>& weight_in,
                                                                                   const Array<double, 1>& bias_in,
                                                                                   const Duo& stride_in,
                                                                                   const Duo& skip_in,
                                                                                   uint32_t n_channel_per_ct_in,
                                                                                   uint32_t level_in,
                                                                                   double residual_scale)
    : Conv2DLayer(param_in, input_shape_in, weight_in, bias_in, stride_in, skip_in) {
    n_channel_per_ct = n_channel_per_ct_in;
    n_packed_in_channel = div_ceil(n_out_channel_, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_out_channel_, n_channel_per_ct * stride_[0] * stride_[1]);
    n_block_per_ct = std::ceil(n_channel_per_ct / (skip_[0] * skip_[1]));
    level = level_in;
    weight_scale = param_.get_q(level) * residual_scale;
}

ParMultiplexedConv2DPackedLayerDepthwise::~ParMultiplexedConv2DPackedLayerDepthwise() {}

void ParMultiplexedConv2DPackedLayerDepthwise::prepare_weight() {
    uint32_t pad0 = std::floor(kernel_shape_[0] / 2);
    uint32_t pad1 = std::floor(kernel_shape_[1] / 2);

    uint32_t padding_shape[] = {pad0, pad1};
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape_[0] * skip_[0];
    input_shape_ct[1] = input_shape_[1] * skip_[1];
    kernel_masks_.clear();
    double scale_new = 0;
    double bias_scale = 0;

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
    input_rotate_units_.push_back(skip_[0] * 1);
    input_rotate_ranges_.clear();
    input_rotate_ranges_.push_back(padding_shape[1]);
    input_rotate_ranges_.push_back(padding_shape[0]);
    weight_pt.clear();
    bias_pt.clear();

    weight_pt.resize(n_packed_in_channel);
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    ctx.resize_copies(n_packed_in_channel);

    int kernel_size = kernel_shape_[0] * kernel_shape_[1];
    int input_block_size = input_shape_ct[0] * input_shape_ct[1];
    int skip_prod = skip_[0] * skip_[1];
    parallel_for(n_packed_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        for (int packed_in_channel_idx = 0; packed_in_channel_idx < n_packed_in_channel; ++packed_in_channel_idx) {
            int base_channel_in = packed_in_channel_idx * n_channel_per_ct;
            vector<CkksPlaintextRingt> a1(kernel_size);
            for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
                auto& mask = kernel_masks_[kernel_idx];
                vector<double> w(n_block_per_ct * input_block_size);

                for (int linear_idx = 0; linear_idx < n_block_per_ct * input_block_size; ++linear_idx) {
                    int t = linear_idx / input_block_size;
                    int shape_linear = linear_idx % input_block_size;
                    int shapei = shape_linear / input_shape_ct[1];
                    int shapej = shape_linear % input_shape_ct[1];

                    int channel_in = 0;
                    int channel_out = n_packed_out_channel_idx * n_channel_per_ct + t * skip_prod +
                                      (skip_[0] * (shapei % skip_[0]) + shapej % skip_[0]);

                    w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                                        0 :
                                        weight_.get(channel_out, channel_in, kernel_idx / kernel_shape_[1],
                                                    kernel_idx % kernel_shape_[1]) *
                                            mask[shapei * input_shape_ct[1] + shapej];
                }
                a1[kernel_idx] = move(ctx_copy.encode_ringt(w, weight_scale));
            }
            weight_pt[n_packed_out_channel_idx] = move(a1);
        }
    });
    vector<vector<double>> feature_tmp_pack(n_packed_out_channel);

    Duo bias_shape;
    Duo bias_skip;
    bias_shape[0] = input_shape_[0] / stride_[0];
    bias_shape[1] = input_shape_[1] / stride_[1];
    bias_skip[0] = skip_[0] * stride_[0];
    bias_skip[1] = skip_[1] * stride_[1];
    int bis_skip_prod = bias_skip[0] * bias_skip[1];
    int bias_n_channel_per_ct = n_channel_per_ct * stride_[0] * stride_[1];
    int bias_level_down = 2;
    if (stride_[0] == 1) {
        bias_level_down = 1;
    }
    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        const int total_block_size = n_block_per_ct * bias_shape[0] * bias_skip[0] * bias_shape[1] * bias_skip[1];
        feature_tmp_pack[n_packed_out_channel_idx].resize(ctx_copy.get_parameter().get_n() / 2);

        for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
            int j = linear_idx / (bias_shape[0] * bias_skip[0] * bias_shape[1] * bias_skip[1]);
            int residual = linear_idx % (bias_shape[0] * bias_skip[0] * bias_shape[1] * bias_skip[1]);
            int h = residual / (bias_shape[1] * bias_skip[1]);
            int k = residual % (bias_shape[1] * bias_skip[1]);

            int channel = n_packed_out_channel_idx * bias_n_channel_per_ct + j * bis_skip_prod +
                          (bias_skip[0] * (h % bias_skip[0]) + k % bias_skip[0]);
            if (channel >= n_out_channel_)
                continue;

            int index = j * (bias_shape[0] * bias_skip[0] * bias_shape[1] * bias_skip[1]) +
                        (h * bias_shape[0] * bias_skip[0] + k);
            feature_tmp_pack[n_packed_out_channel_idx][index] = bias_.get(channel);
        }
        bias_pt[n_packed_out_channel_idx] =
            ctx_copy.encode_ringt(feature_tmp_pack[n_packed_out_channel_idx], param_.get_default_scale());
    });
    mask_pt.resize(n_out_channel_);
    parallel_for(n_packed_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        if (stride_[0] == 1) {
        } else {
            for (int i = 0; i < n_channel_per_ct; i++) {
                if ((ct_idx * n_channel_per_ct + i) < n_out_channel_) {
                    auto si =
                        select_tensor((ct_idx * n_channel_per_ct + i) % (n_channel_per_ct * stride_[0] * stride_[1]));
                    mask_pt[ct_idx * n_channel_per_ct + i] =
                        ctx_copy.encode_ringt(si, ctx_copy.get_parameter().get_q(level - 1));
                }
            }
        }
    });
}

void ParMultiplexedConv2DPackedLayerDepthwise::prepare_weight_lazy() {
    uint32_t pad0 = std::floor(kernel_shape_[0] / 2);
    uint32_t pad1 = std::floor(kernel_shape_[1] / 2);

    uint32_t padding_shape[] = {pad0, pad1};
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape_[0] * skip_[0];
    input_shape_ct[1] = input_shape_[1] * skip_[1];
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
    input_rotate_units_.push_back(skip_[0] * 1);
    input_rotate_ranges_.clear();
    input_rotate_ranges_.push_back(padding_shape[1]);
    input_rotate_ranges_.push_back(padding_shape[0]);

    int kernel_size = kernel_shape_[0] * kernel_shape_[1];
    int input_block_size = input_shape_ct[0] * input_shape_ct[1];
    int skip_prod = skip_[0] * skip_[1];
    int N = param_.get_n();

    // Cache commonly used values for on-demand generation
    cached_input_shape_ct[0] = input_shape_ct[0];
    cached_input_shape_ct[1] = input_shape_ct[1];
    cached_input_block_size = input_block_size;
    cached_kernel_size = kernel_size;
    cached_skip_prod = skip_prod;

    // Cache bias-related values
    Duo bias_shape;
    Duo bias_skip;
    bias_shape[0] = input_shape_[0] / stride_[0];
    bias_shape[1] = input_shape_[1] / stride_[1];
    bias_skip[0] = skip_[0] * stride_[0];
    bias_skip[1] = skip_[1] * stride_[1];
    cached_bias_skip = bias_skip;
    int bis_skip_prod = bias_skip[0] * bias_skip[1];
    cached_bias_n_channel_per_ct = n_channel_per_ct * stride_[0] * stride_[1];
    cached_total_block_size = n_block_per_ct * bias_shape[0] * bias_skip[0] * bias_shape[1] * bias_skip[1];

    // Note: weight_rearranged, bias_rearranged, and mask_rearranged are no longer generated here.
    // They will be generated on-demand in run_core using helper functions.
}

vector<double> ParMultiplexedConv2DPackedLayerDepthwise::select_tensor(int num) const {
    vector<double> tensor;
    for (int k = 0; k < n_block_per_ct; k++) {
        for (int i = 0; i < input_shape_[0] * skip_[0]; i++) {
            for (int j = 0; j < input_shape_[1] * skip_[1]; j++) {
                if (k * skip_[0] * stride_[0] * skip_[1] * stride_[0] +
                        skip_[0] * stride_[0] * (i % (skip_[0] * stride_[0])) + (j % (stride_[0] * skip_[0])) ==
                    num) {
                    tensor.push_back(1);
                } else {
                    tensor.push_back(0);
                }
            }
        }
    }

    return tensor;
}

CkksPlaintextRingt
ParMultiplexedConv2DPackedLayerDepthwise::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                         int n_packed_out_channel_idx,
                                                                         int kernel_idx) const {
    auto& mask = kernel_masks_[kernel_idx];
    vector<double> w(n_block_per_ct * cached_input_block_size, 0.0);

    for (int linear_idx = 0; linear_idx < n_block_per_ct * cached_input_block_size; ++linear_idx) {
        int t = linear_idx / cached_input_block_size;
        int shape_linear = linear_idx % cached_input_block_size;
        int shapei = shape_linear / cached_input_shape_ct[1];
        int shapej = shape_linear % cached_input_shape_ct[1];

        int channel_in = 0;
        int channel_out = n_packed_out_channel_idx * n_channel_per_ct + t * cached_skip_prod +
                          (skip_[0] * (shapei % skip_[0]) + shapej % skip_[0]);

        w[linear_idx] =
            (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                0 :
                weight_.get(channel_out, channel_in, kernel_idx / kernel_shape_[1], kernel_idx % kernel_shape_[1]) *
                    mask[shapei * cached_input_shape_ct[1] + shapej];
    }
    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt ParMultiplexedConv2DPackedLayerDepthwise::generate_bias_pt_for_index(CkksContext& ctx,
                                                                                        int bpt_idx) const {
    int N = param_.get_n();
    vector<double> bias_vec(N / 2, 0.0);

    // Compute bias_shape locally
    Duo bias_shape;
    bias_shape[0] = input_shape_[0] / stride_[0];
    bias_shape[1] = input_shape_[1] / stride_[1];
    int bis_skip_prod = cached_bias_skip[0] * cached_bias_skip[1];

    for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        int j = linear_idx / (bias_shape[0] * cached_bias_skip[0] * bias_shape[1] * cached_bias_skip[1]);
        int residual = linear_idx % (bias_shape[0] * cached_bias_skip[0] * bias_shape[1] * cached_bias_skip[1]);
        int h = residual / (bias_shape[1] * cached_bias_skip[1]);
        int k = residual % (bias_shape[1] * cached_bias_skip[1]);

        int channel = bpt_idx * cached_bias_n_channel_per_ct + j * bis_skip_prod +
                      (cached_bias_skip[0] * (h % cached_bias_skip[0]) + k % cached_bias_skip[0]);
        if (channel >= n_out_channel_)
            continue;

        int index = j * (bias_shape[0] * cached_bias_skip[0] * bias_shape[1] * cached_bias_skip[1]) +
                    (h * bias_shape[0] * cached_bias_skip[0] + k);
        bias_vec[index] = bias_.get(channel);
    }
    int bias_level_down = 2;
    if (stride_[0] == 1 && stride_[1] == 1) {
        bias_level_down = 1;
    }
    return ctx.encode_ringt(bias_vec, ctx.get_parameter().get_default_scale());
}

CkksPlaintextRingt
ParMultiplexedConv2DPackedLayerDepthwise::generate_mask_pt_for_indices(CkksContext& ctx, int ct_idx, int i) const {
    auto si = select_tensor((ct_idx * n_channel_per_ct + i) % (n_channel_per_ct * stride_[0] * stride_[1]));
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level - 1));
}

vector<CkksCiphertext> ParMultiplexedConv2DPackedLayerDepthwise::run_core(CkksContext& ctx,
                                                                          const std::vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    // 1. rotation of kernel direction
    int rotated_size = x.size();
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    vector<CkksCiphertext> res;
    uint32_t n_weight = weight_pt.empty() ? n_packed_in_channel : weight_pt.size();
    if (stride_[0] == 1) {
        res.resize(n_weight);
    }
    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s(0);
        uint32_t n_k = weight_pt.empty() ? cached_kernel_size : weight_pt[ct_idx].size();
        for (int k = 0; k < n_k; k++) {
            CkksCiphertext r_tmp;
            if (weight_pt.empty()) {
                auto w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, k);
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                r_tmp = ctx_copy.mult_plain_mul(rotated_x[ct_idx][k], w_pt);
            } else {
                auto& w_pt_rt = weight_pt[ct_idx][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                r_tmp = ctx_copy.mult_plain_mul(rotated_x[ct_idx][k], w_pt);
            }
            if (k == 0) {
                s = move(r_tmp);
            } else {
                s = ctx_copy.add(s, r_tmp);
            }
        }
        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
        if (stride_[0] == 1) {
            res[ct_idx] = move(s);
        } else {
            vector<int32_t> steps;
            for (int i = 0; i < n_channel_per_ct; i += skip_[0]) {
                int32_t r_n_block = floor((ct_idx * n_channel_per_ct + i) / int(pow(skip_[0] * stride_[0], 2)));
                int32_t r_n_block_residue = (ct_idx * n_channel_per_ct + i) % int(pow(skip_[0] * stride_[0], 2));
                int32_t r_n_stride_skip = floor(r_n_block_residue / (stride_[0] * skip_[0]));
                int32_t r_n_stride_skip_residue = r_n_block_residue % (stride_[0] * skip_[0]);

                int32_t n_block = floor((ct_idx * n_channel_per_ct + i) / int(pow(skip_[0], 2)));
                int32_t n_block_residue = (ct_idx * n_channel_per_ct + i) % int(pow(skip_[0], 2));
                int32_t n_stride_skip = floor(n_block_residue / skip_[0]);
                int32_t n_stride_skip_residue = n_block_residue % skip_[0];
                int32_t rot_step = (r_n_block - n_block) * int(pow(skip_[0], 2)) * (input_shape_[0] * input_shape_[1]) +
                                   (r_n_stride_skip - n_stride_skip) * (skip_[0] * input_shape_[0]) +
                                   (r_n_stride_skip_residue - n_stride_skip_residue);
                steps.push_back(-rot_step);
            }
            auto s_rots = ctx_copy.rotate(s, steps);
            for (int i = 0; i < n_channel_per_ct; i++) {
                if (mask_pt.empty()) {
                    if ((ct_idx * n_channel_per_ct + i) < n_out_channel_) {
                        auto m_pt_rt = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[int(i / skip_[0])]], m_pt);
                        result_ct[ct_idx * n_channel_per_ct + i] =
                            move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
                    }
                } else {
                    if ((ct_idx * n_channel_per_ct + i) < n_out_channel_) {
                        auto& m_pt_rt = mask_pt[ct_idx * n_channel_per_ct + i];
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[int(i / skip_[0])]], m_pt);
                        result_ct[ct_idx * n_channel_per_ct + i] =
                            move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
                    }
                }
            }
        }
    });
    if (stride_[0] == 1) {
        for (int i = 0; i < res.size(); i++) {
            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_for_index(ctx, i);
                res[i] = ctx.add_plain_ringt(res[i], b_pt);
            } else {
                res[i] = ctx.add_plain_ringt(res[i], bias_pt[i]);
            }
        }
        return res;
    }

    CkksCiphertext sp;
    for (int i = 0; i < result_ct.size(); i++) {
        int p = i % (stride_[0] * stride_[1] * n_channel_per_ct);
        auto c_m_s = move(result_ct[i]);
        if (p == 0) {
            sp = move(c_m_s);
            int bias_idx = i / (stride_[0] * stride_[1] * n_channel_per_ct);
            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_for_index(ctx, bias_idx);
                sp = ctx.add_plain_ringt(sp, b_pt);
            } else {
                sp = ctx.add_plain_ringt(sp, bias_pt[bias_idx]);
            }
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % (stride_[0] * stride_[1] * n_channel_per_ct) == 0 || i == result_ct.size() - 1) {
            res.push_back(move(sp));
        }
    }
    return res;
}

Feature2DEncrypted ParMultiplexedConv2DPackedLayerDepthwise::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    int bias_level_down = 2;
    if (stride_[0] == 1) {
        bias_level_down = 1;
    }
    result.shape[0] = x.shape[0] / stride_[0];
    result.shape[1] = x.shape[1] / stride_[1];
    result.skip[0] = x.skip[0] * stride_[0];
    result.skip[1] = x.skip[1] * stride_[1];
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * stride_[0] * stride_[1];
    result.level = x.level - bias_level_down;
    result.data = run_core(ctx, x.data);
    return result;
}

Array<double, 3> ParMultiplexedConv2DPackedLayerDepthwise::run_plaintext(const Array<double, 3>& x, double multiplier) {
    double value = 1.0 / multiplier;
    uint32_t padding_shape[]{kernel_shape_[0] / 2, kernel_shape_[1] / 2};
    Array<double, 3> padded_input(
        {n_out_channel_, input_shape_[0] + padding_shape[0] * 2, input_shape_[1] + padding_shape[1] * 2}, 0.0);
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel_; in_channel_idx++) {
        for (int i = 0; i < input_shape_[0]; i++) {
            for (int j = 0; j < input_shape_[1]; j++) {
                padded_input.set(in_channel_idx, i + padding_shape[0], j + padding_shape[1],
                                 x.get(in_channel_idx, i, j));
            }
        }
    }

    uint32_t output_shape[]{input_shape_[0] / stride_[0], input_shape_[1] / stride_[1]};
    Array<double, 3> result({n_out_channel_, output_shape[0], output_shape[1]});
    for (int out_channel_idx = 0; out_channel_idx < n_out_channel_; out_channel_idx++) {
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                double r = bias_[out_channel_idx];
                for (int ki = 0; ki < kernel_shape_[0]; ki++) {
                    for (int kj = 0; kj < kernel_shape_[1]; kj++) {
                        r += padded_input.get(out_channel_idx, i * stride_[0] + ki, j * stride_[1] + kj) *
                             (weight_.get(out_channel_idx, 0, ki, kj) * value);
                    }
                }
                result.set(out_channel_idx, i, j, r);
            }
        }
    }
    return result;
}
