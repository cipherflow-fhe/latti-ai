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
#include "layer_util.h"
#include "../util.h"
#include "multiplexed_conv2d_pack_layer.h"
#include "multiplexed_conv2d_pack_layer_depthwise.h"

using namespace std;
using namespace cxx_sdk_v2;

MultiplexedConv2DPackedLayerDepthwise::MultiplexedConv2DPackedLayerDepthwise(const CkksParameter& param_in,
                                                                             const Duo& input_shape_in,
                                                                             const Array<double, 4>& weight_in,
                                                                             const Array<double, 1>& bias_in,
                                                                             const Duo& stride_in,
                                                                             const Duo& skip_in,
                                                                             uint32_t n_channel_per_ct_in,
                                                                             uint32_t level_in,
                                                                             double residual_scale)
    : Conv2DLayer(param_in, input_shape_in, weight_in, bias_in, stride_in, skip_in) {
    const uint32_t output_channels_per_ct = n_channel_per_ct_in * prod(stride_);

    n_channel_per_ct = n_channel_per_ct_in;
    n_packed_in_channel = div_ceil(n_out_channel_, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_out_channel_, output_channels_per_ct);
    n_block_per_ct = div_ceil(n_channel_per_ct, prod(skip_));
    level_ = level_in;
    weight_scale = param_.get_q(level_) * residual_scale;
}

void MultiplexedConv2DPackedLayerDepthwise::prepare_weight() {
    prepare_weight_lazy();

    int kernel_size = cached_kernel_size;
    weight_pt.clear();
    bias_pt.clear();
    weight_pt.resize(n_packed_in_channel);
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);

    parallel_for(n_packed_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        weight_pt[n_packed_out_channel_idx].resize(kernel_size);
        for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
            weight_pt[n_packed_out_channel_idx][kernel_idx] =
                generate_weight_pt_for_indices(ctx_copy, n_packed_out_channel_idx, kernel_idx);
        }
    });

    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        bias_pt[n_packed_out_channel_idx] = generate_bias_pt_for_index(ctx_copy, n_packed_out_channel_idx);
    });

    if (stride_[0] != 1) {
        mask_pt.resize(n_out_channel_);
        parallel_for(n_packed_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            for (int i = 0; i < n_channel_per_ct; i++) {
                if ((ct_idx * n_channel_per_ct + i) < n_out_channel_) {
                    mask_pt[ct_idx * n_channel_per_ct + i] = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
                }
            }
        });
    }
}

void MultiplexedConv2DPackedLayerDepthwise::prepare_weight_lazy() {
    const Duo padding_shape = kernel_shape_ / 2;
    const Duo input_shape_ct = input_shape_ * skip_;
    kernel_masks_.clear();

    for (const Duo& kernel_pos : duo_range(kernel_shape_)) {
        vector<double> mask;
        mask.reserve(prod(input_shape_ct));
        for (const Duo& input_pos : duo_range(input_shape_ct)) {
            const int64_t shifted_i = static_cast<int64_t>(kernel_pos[0] * skip_[0] + input_pos[0]) -
                                      static_cast<int64_t>(padding_shape[0] * skip_[0]);
            const int64_t shifted_j = static_cast<int64_t>(kernel_pos[1] * skip_[1] + input_pos[1]) -
                                      static_cast<int64_t>(padding_shape[1] * skip_[1]);
            if (0 <= shifted_i && shifted_i < input_shape_ct[0] && 0 <= shifted_j && shifted_j < input_shape_ct[1]) {
                mask.push_back(1.0);
            } else {
                mask.push_back(0.0);
            }
        }
        kernel_masks_.push_back(move(mask));
    }

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[0]);

    // Cache commonly used values for on-demand generation
    this->cached_input_shape_ct = input_shape_ct;
    cached_input_block_size = prod(input_shape_ct);
    cached_kernel_size = prod(kernel_shape_);
    cached_skip_prod = prod(skip_);

    // Cache bias-related values
    const Duo bias_shape = input_shape_ / stride_;
    cached_bias_skip = skip_ * stride_;
    cached_bias_n_channel_per_ct = n_channel_per_ct * prod(stride_);
    cached_total_block_size = n_block_per_ct * prod(bias_shape * cached_bias_skip);

    // Note: weight_rearranged, bias_rearranged, and mask_rearranged are no longer generated here.
    // They will be generated on-demand in run_core using helper functions.
}

vector<double> MultiplexedConv2DPackedLayerDepthwise::select_tensor(int num) const {
    const Duo input_shape_ct = input_shape_ * skip_;
    const Duo stride_skip = stride_ * skip_;
    const uint32_t stride_skip_prod = prod(stride_skip);

    vector<double> tensor;
    for (int block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
        for (const Duo& input_pos : duo_range(input_shape_ct)) {
            if (block_idx * stride_skip_prod + stride_skip[0] * (input_pos[0] % stride_skip[0]) +
                    (input_pos[1] % stride_skip[0]) ==
                num) {
                tensor.push_back(1.0);
            } else {
                tensor.push_back(0.0);
            }
        }
    }

    return tensor;
}

CkksPlaintextRingt MultiplexedConv2DPackedLayerDepthwise::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                                         int n_packed_out_channel_idx,
                                                                                         int kernel_idx) const {
    auto& mask = kernel_masks_[kernel_idx];
    vector<double> w(n_block_per_ct * cached_input_block_size, 0.0);
    const Duo kernel_pos = div_mod(static_cast<uint32_t>(kernel_idx), kernel_shape_[1]);

    for (uint32_t linear_idx = 0; linear_idx < n_block_per_ct * cached_input_block_size; ++linear_idx) {
        const Duo block_pos = div_mod(linear_idx, cached_input_block_size);
        const Duo input_pos = div_mod(block_pos[1], cached_input_shape_ct[1]);

        const uint32_t channel_in = 0;
        const uint32_t channel_out = static_cast<uint32_t>(n_packed_out_channel_idx) * n_channel_per_ct +
                                     block_pos[0] * cached_skip_prod +
                                     (skip_[0] * (input_pos[0] % skip_[0]) + input_pos[1] % skip_[0]);

        w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                            0 :
                            weight_.get(channel_out, channel_in, kernel_pos[0], kernel_pos[1]) *
                                mask[input_pos[0] * cached_input_shape_ct[1] + input_pos[1]];
    }
    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt MultiplexedConv2DPackedLayerDepthwise::generate_bias_pt_for_index(CkksContext& ctx,
                                                                                     int bpt_idx) const {
    const int N = param_.get_n();
    const Duo bias_shape = input_shape_ / stride_;
    const Duo bias_block_shape = bias_shape * cached_bias_skip;
    const uint32_t bias_block_size = prod(bias_block_shape);
    const uint32_t bias_skip_prod = prod(cached_bias_skip);
    vector<double> bias_vec(N / 2, 0.0);

    for (uint32_t linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        const Duo block_pos = div_mod(linear_idx, bias_block_size);
        const Duo output_pos = div_mod(block_pos[1], bias_block_shape[1]);
        const Duo channel_offset = output_pos % cached_bias_skip;

        const uint32_t channel = static_cast<uint32_t>(bpt_idx) * cached_bias_n_channel_per_ct +
                                 block_pos[0] * bias_skip_prod + cached_bias_skip[0] * channel_offset[0] +
                                 channel_offset[1];
        if (channel >= n_out_channel_) {
            continue;
        }

        bias_vec[linear_idx] = bias_.get(channel);
    }
    return ctx.encode_ringt(bias_vec, ctx.get_parameter().get_default_scale());
}

CkksPlaintextRingt
MultiplexedConv2DPackedLayerDepthwise::generate_mask_pt_for_indices(CkksContext& ctx, int ct_idx, int i) const {
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_);
    auto si = select_tensor((ct_idx * n_channel_per_ct + i) % output_channels_per_ct);
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level_ - 1));
}

vector<CkksCiphertext> MultiplexedConv2DPackedLayerDepthwise::run_core(CkksContext& ctx,
                                                                       const std::vector<CkksCiphertext>& x) {
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_);
    const uint32_t input_feature_size = prod(input_shape_);

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
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                r_tmp = ctx_copy.mult_plain_mul(rotated_x[ct_idx][k], w_pt);
            } else {
                auto& w_pt_rt = weight_pt[ct_idx][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
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
            const uint32_t stride_skip_prod = prod(skip_ * stride_);
            const uint32_t skip_prod = prod(skip_);
            for (int i = 0; i < n_channel_per_ct; i += skip_[0]) {
                const int32_t channel_idx = ct_idx * n_channel_per_ct + i;
                const int32_t rotated_block = floor(channel_idx / stride_skip_prod);
                const int32_t rotated_residual = channel_idx % stride_skip_prod;
                const int32_t rotated_row = floor(rotated_residual / (stride_[0] * skip_[0]));
                const int32_t rotated_col = rotated_residual % (stride_[0] * skip_[0]);

                const int32_t base_block = floor(channel_idx / skip_prod);
                const int32_t base_residual = channel_idx % skip_prod;
                const int32_t base_row = floor(base_residual / skip_[0]);
                const int32_t base_col = base_residual % skip_[0];
                const int32_t rot_step = (rotated_block - base_block) * skip_prod * input_feature_size +
                                         (rotated_row - base_row) * (skip_[0] * input_shape_[0]) +
                                         (rotated_col - base_col);
                steps.push_back(-rot_step);
            }
            auto s_rots = ctx_copy.rotate(s, steps);
            for (int i = 0; i < n_channel_per_ct; i++) {
                if ((ct_idx * n_channel_per_ct + i) < n_out_channel_) {
                    if (mask_pt.empty()) {
                        auto m_pt_rt = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level_ - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i / skip_[0]]], m_pt);
                        result_ct[ct_idx * n_channel_per_ct + i] =
                            move(ctx_copy.rescale(c_m_s, ctx_copy.get_parameter().get_default_scale()));
                    } else {
                        auto& m_pt_rt = mask_pt[ct_idx * n_channel_per_ct + i];
                        auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level_ - 1);
                        auto c_m_s = ctx_copy.mult_plain_mul(s_rots[steps[i / skip_[0]]], m_pt);
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
        int p = i % output_channels_per_ct;
        auto c_m_s = move(result_ct[i]);
        if (p == 0) {
            sp = move(c_m_s);
            int bias_idx = i / output_channels_per_ct;
            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_for_index(ctx, bias_idx);
                sp = ctx.add_plain_ringt(sp, b_pt);
            } else {
                sp = ctx.add_plain_ringt(sp, bias_pt[bias_idx]);
            }
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % output_channels_per_ct == 0 || i == result_ct.size() - 1) {
            res.push_back(move(sp));
        }
    }
    return res;
}

Feature2DEncrypted MultiplexedConv2DPackedLayerDepthwise::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    int bias_level_down = (stride_[0] == 1) ? 1 : 2;
    result.shape = x.shape / stride_;
    result.skip = x.skip * stride_;
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * prod(stride_);
    result.level = x.level - bias_level_down;
    result.data = run_core(ctx, x.data);
    return result;
}

Array<double, 3> MultiplexedConv2DPackedLayerDepthwise::run_plaintext(const Array<double, 3>& x, double multiplier) {
    const double value = 1.0 / multiplier;
    const Duo padding_shape = kernel_shape_ / 2;
    const Duo padded_shape = input_shape_ + padding_shape * 2;
    const Duo output_shape = input_shape_ / stride_;
    Array<double, 3> padded_input({n_out_channel_, padded_shape[0], padded_shape[1]}, 0.0);
    for (int in_channel_idx = 0; in_channel_idx < n_out_channel_; in_channel_idx++) {
        for (const Duo& input_pos : duo_range(input_shape_)) {
            const Duo padded_pos = input_pos + padding_shape;
            padded_input.set(in_channel_idx, padded_pos[0], padded_pos[1],
                             x.get(in_channel_idx, input_pos[0], input_pos[1]));
        }
    }

    Array<double, 3> result({n_out_channel_, output_shape[0], output_shape[1]});
    for (int out_channel_idx = 0; out_channel_idx < n_out_channel_; out_channel_idx++) {
        for (const Duo& output_pos : duo_range(output_shape)) {
            double sum = bias_[out_channel_idx];
            const Duo input_base = output_pos * stride_;
            for (const Duo& kernel_pos : duo_range(kernel_shape_)) {
                const Duo input_pos = input_base + kernel_pos;
                sum += padded_input.get(out_channel_idx, input_pos[0], input_pos[1]) *
                       (weight_.get(out_channel_idx, 0, kernel_pos[0], kernel_pos[1]) * value);
            }
            result.set(out_channel_idx, output_pos[0], output_pos[1], sum);
        }
    }
    return result;
}
