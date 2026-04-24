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

using namespace std;
using namespace lattisense;

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

vector<double> MultiplexedConv2DPackedLayer::select_tensor(int num) const {
    const Duo input_shape_ct = input_shape_ * skip_;
    const Duo skip_stride = skip_ * stride_;
    const uint32_t zero_inserted_skip_prod = prod(zero_inserted_skip);

    vector<double> tensor;
    for (int block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
        for (const Duo& input_pos : duo_range(input_shape_ct)) {
            if ((input_pos[0] % skip_stride[0]) < zero_inserted_skip[0] &&
                (input_pos[1] % skip_stride[1]) < zero_inserted_skip[1] &&
                block_idx * zero_inserted_skip_prod + zero_inserted_skip[1] * (input_pos[0] % zero_inserted_skip[0]) +
                        (input_pos[1] % zero_inserted_skip[1]) ==
                    num) {
                tensor.push_back(1.0);
            } else {
                tensor.push_back(0.0);
            }
        }
    }

    return tensor;
}

MultiplexedConv2DPackedLayer::MultiplexedConv2DPackedLayer(const CkksParameter& param_in,
                                                           const Duo& input_shape_in,
                                                           Array<double, 4>&& weight_in,
                                                           Array<double, 1>&& bias_in,
                                                           const Duo& stride_in,
                                                           const Duo& skip_in,
                                                           uint32_t n_channel_per_ct_in,
                                                           uint32_t level_in,
                                                           double residual_scale,
                                                           const Duo& external_upsample_factor_in)
    : Conv2DLayer(param_in, input_shape_in, move(weight_in), move(bias_in), stride_in, skip_in),
      external_upsample_factor(external_upsample_factor_in),
      zero_inserted_skip(skip_in * stride_in / external_upsample_factor_in) {
    const uint32_t output_channels_per_ct = n_channel_per_ct_in * prod(stride_in) / prod(external_upsample_factor);

    n_channel_per_ct = n_channel_per_ct_in;
    n_packed_in_channel = div_ceil(n_in_channel_, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_out_channel_, output_channels_per_ct);
    n_block_per_ct = div_ceil(n_channel_per_ct, prod(skip_));
    need_repack_ = !(stride_ == Duo{1, 1} && skip_ == Duo{1, 1});
    level_ = level_in;
    weight_scale = param_.get_q(level_) * residual_scale;
    N = param_in.get_n();
}

void MultiplexedConv2DPackedLayer::prepare_weight_for_post_skip_rotation() {
    prepare_weight_for_post_skip_rotation_lazy();

    uint32_t n_weight_pt = div_ceil(n_out_channel_, n_block_per_ct);
    int kernel_size = prod(kernel_shape_);
    weight_pt.clear();
    bias_pt.clear();
    weight_pt.resize(n_weight_pt);
    for (int i = 0; i < n_weight_pt; i++) {
        weight_pt[i].resize(n_packed_in_channel * n_block_per_ct);
    }
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);

    parallel_for(n_weight_pt, th_nums, ctx, [&](CkksContext& ctx_copy, int weight_pt_num_idx) {
        for (int j = 0; j < n_packed_in_channel * n_block_per_ct; ++j) {
            weight_pt[weight_pt_num_idx][j].resize(kernel_size);
            for (int k = 0; k < kernel_size; ++k) {
                weight_pt[weight_pt_num_idx][j][k] = generate_weight_pt_for_indices(ctx_copy, weight_pt_num_idx, j, k);
            }
        }
    });

    // mask_pt is already populated by prepare_weight_for_post_skip_rotation_lazy().

    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        bias_pt[n_packed_out_channel_idx] = generate_bias_pt_for_index(ctx_copy, n_packed_out_channel_idx);
    });
}

void MultiplexedConv2DPackedLayer::prepare_weight_for_post_skip_rotation_lazy() {
    const Duo padding_shape = kernel_shape_ / 2;
    const Duo input_shape_ct = input_shape_ * skip_;

    // Cache commonly used values for on-demand generation
    cached_input_shape_ct = input_shape_ct;
    cached_input_block_size = prod(input_shape_ct);
    cached_kernel_size = prod(kernel_shape_);
    cached_total_skip = prod(skip_);

    // Cache bias-related values
    cached_bias_skip = zero_inserted_skip;
    cached_skip_prod = prod(cached_bias_skip);
    cached_bias_n_channel_per_ct = n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    cached_total_block_size = n_block_per_ct * prod(input_shape_ct);

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
    input_rotate_units_.push_back(skip_[1]);

    bias_level_down = (stride_ == Duo{1, 1} && skip_ == Duo{1, 1}) ? 1 : 2;

    // mask_pt is small (<= n_block_per_ct entries) and shared across ct_idx,
    // so generate it offline even in lazy mode.
    mask_pt.clear();
    if (!(stride_[0] == 1 && stride_[1] == 1 && skip_[0] == 1 && skip_[1] == 1)) {
        uint32_t n_mask = min(n_block_per_ct, n_out_channel_);
        mask_pt.resize(n_mask);
        CkksContext ctx = CkksContext::create_empty_context(this->param_);
        parallel_for(n_mask, th_nums, ctx,
                     [&](CkksContext& ctx_copy, int i) { mask_pt[i] = generate_mask_pt_for_indices(ctx_copy, i); });
    }
}

CkksPlaintextRingt MultiplexedConv2DPackedLayer::generate_weight_pt_for_indices_reduct_rot(CkksContext& ctx,
                                                                                           int ct_idx,
                                                                                           int j,
                                                                                           int k) const {
    // ct_idx = output_ct_group * skip_out_prod + sub_pos
    uint32_t sub_pos = ct_idx % cached_skip_out_prod;
    uint32_t output_ct_group = ct_idx / cached_skip_out_prod;

    int packed_in_channel_idx = j / n_block_per_ct;
    int block_idx = j % n_block_per_ct;
    int kernel_idx = k;

    auto& mask = kernel_masks_[kernel_idx];
    int base_channel_in = packed_in_channel_idx * n_channel_per_ct;
    int total_skip = skip_[0] * skip_[1];

    vector<double> w(N / 2, 0.0);
    for (int linear_idx = 0; linear_idx < n_block_per_ct * cached_input_block_size; ++linear_idx) {
        int t = linear_idx / cached_input_block_size;
        int shape_linear = linear_idx % cached_input_block_size;
        int shape_i = shape_linear / cached_input_shape_ct[1];
        int shape_j = shape_linear % cached_input_shape_ct[1];
        int kernel_shape_i = kernel_idx / kernel_shape_[1];
        int kernel_shape_j = kernel_idx % kernel_shape_[1];

        uint32_t channel_in = base_channel_in + (block_idx * total_skip + t * total_skip + (shape_j % skip_[1]) +
                                                 (shape_i % skip_[0]) * skip_[1]) %
                                                    n_channel_per_ct;
        uint32_t channel_out = output_ct_group * n_block_per_ct * cached_skip_out_prod +
                               ((t + n_block_per_ct) % n_block_per_ct) * cached_skip_out_prod + sub_pos;

        w[linear_idx] = (channel_in >= n_in_channel_ || channel_out >= n_out_channel_) ?
                            0 :
                            weight_.get(channel_out, channel_in, kernel_shape_i, kernel_shape_j) *
                                mask[shape_i * cached_input_shape_ct[1] + shape_j];
    }
    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt MultiplexedConv2DPackedLayer::generate_bias_pt_for_index_reduct_rot(CkksContext& ctx,
                                                                                       int bpt_idx) const {
    // Same layout as post_skip bias: indexed by n_packed_out_channel
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

CkksPlaintextRingt
MultiplexedConv2DPackedLayer::generate_mask_pt_for_indices_reduct_rot(CkksContext& ctx, int ct_idx, int i) const {
    // In reduct_rot, block i of ct_idx has channel_local = i * skip_out_prod + sub_pos
    uint32_t sub_pos = ct_idx % cached_skip_out_prod;
    uint32_t channel_local = i * cached_skip_out_prod + sub_pos;
    auto si = select_tensor(channel_local);
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level_ - 1));
}

CkksPlaintextRingt
MultiplexedConv2DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int j, int k) const {
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

CkksPlaintextRingt MultiplexedConv2DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
    const Duo input_shape_ct = input_shape_ * skip_;
    const Duo skip_stride = stride_ * skip_;

    vector<double> bias_vec(N / 2, 0.0);

    for (uint32_t linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        const Duo block_pos = div_mod(linear_idx, static_cast<uint32_t>(cached_input_block_size));
        const Duo input_pos = div_mod(block_pos[1], input_shape_ct[1]);

        const uint32_t channel =
            static_cast<uint32_t>(bpt_idx) * cached_bias_n_channel_per_ct + block_pos[0] * cached_skip_prod +
            cached_bias_skip[1] * (input_pos[0] % cached_bias_skip[0]) + input_pos[1] % cached_bias_skip[1];
        if (channel >= n_out_channel_ || (input_pos[0] % skip_stride[0]) >= cached_bias_skip[0] ||
            (input_pos[1] % skip_stride[1]) >= cached_bias_skip[1]) {
            continue;
        }

        bias_vec[linear_idx] = bias_.get(channel);
    }

    return ctx.encode_ringt(bias_vec, ctx.get_parameter().get_default_scale());
}

// Generate mask vector for block i on-demand.
// Used in the mask-then-rotate pipeline: the mask keeps sub_pos 0 of block i
// in `s` (shared across all ct_idx; rotation amount carries the ct_idx-specific
// target offset).
CkksPlaintextRingt MultiplexedConv2DPackedLayer::generate_mask_pt_for_indices(CkksContext& ctx, int i) const {
    const uint32_t zero_inserted_skip_prod = prod(zero_inserted_skip);
    auto si = select_tensor(i * zero_inserted_skip_prod);
    return ctx.encode_ringt(si, ctx.get_parameter().get_q(level_ - 1));
}

vector<CkksCiphertext> MultiplexedConv2DPackedLayer::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    const Duo input_shape_ct = input_shape_ * skip_;
    const uint32_t input_ct_size = prod(input_shape_ct);
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    const uint32_t input_feature_size = prod(input_shape_);

    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1, input_ct_size);
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
    std::vector<std::vector<lattisense::CkksCiphertext>> rotated_x(rotated_size);
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
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
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
            const int32_t channel_in_ct = (ct_idx * n_block_per_ct + i) % output_channels_per_ct;
            const int32_t row_offset =
                floor(channel_in_ct / prod(zero_inserted_skip)) * prod(skip_) * input_feature_size;
            const int32_t col_offset =
                floor((channel_in_ct % prod(zero_inserted_skip)) / zero_inserted_skip[1]) * input_shape_[1] * skip_[1];
            const int32_t rot_step =
                -row_offset - col_offset - channel_in_ct % zero_inserted_skip[1] + i * prod(skip_) * input_feature_size;
            steps.push_back(rot_step);
        }
        auto s_rots = ctx_copy.rotate(s, steps);
        for (int i = 0; i < n_block_per_ct; i++) {
            auto si = select_tensor((ct_idx * n_block_per_ct + i) % output_channels_per_ct);
            auto p_ss = ctx_copy.encode(si, level_ - 1, ctx_copy.get_parameter().get_q(level_ - 1));
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
        int p = i % output_channels_per_ct;
        auto c_m_s = result_ct[i].copy();
        if (p == 0) {
            sp = move(c_m_s);
            int bpt_idx = i / output_channels_per_ct;
            sp = ctx.add_plain_ringt(sp, bias_pt[bpt_idx]);
        } else {
            sp = ctx.add(sp, c_m_s);
        }
        if ((i + 1) % output_channels_per_ct == 0 || i == result_ct.size() - 1) {
            res.push_back(move(sp));
        }
    }
    return res;
}

vector<CkksCiphertext>
MultiplexedConv2DPackedLayer::run_core_for_post_skip_rotation(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    const Duo input_shape_ct = input_shape_ * skip_;
    const uint32_t input_ct_size = prod(input_shape_ct);
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    const uint32_t input_feature_size = prod(input_shape_);
    const bool lazy_encoding = weight_pt.empty();

    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1, input_ct_size);
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    int rotated_size = input_rotated_x.size();
    std::vector<std::vector<lattisense::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, input_rotated_x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    vector<CkksCiphertext> res;
    uint32_t n_weight = lazy_encoding ? div_ceil(n_out_channel_, n_block_per_ct) : weight_pt.size();
    if (!need_repack_) {
        res.resize(n_weight);
    }
    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s(0);
        uint32_t n_j = lazy_encoding ? n_packed_in_channel * n_block_per_ct : weight_pt[ct_idx].size();
        for (int j = 0; j < n_j; j++) {
            uint32_t n_k = lazy_encoding ? cached_kernel_size : weight_pt[ct_idx][j].size();
            for (int k = 0; k < n_k; k++) {
                CkksCiphertext res;
                CkksPlaintextRingt gen_w_pt_rt;
                if (lazy_encoding)
                    gen_w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, j, k);
                const CkksPlaintextRingt& w_pt_rt = lazy_encoding ? gen_w_pt_rt : weight_pt[ct_idx][j][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                res = ctx_copy.mult_plain_mul(rotated_x[j][k], w_pt);
                if (j == 0 && k == 0) {
                    s = move(res);
                } else {
                    s = ctx_copy.add(s, res);
                }
            }
        }

        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
        if (!need_repack_) {
            res[ct_idx] = move(s);
        } else {
            s = sum_slot(ctx_copy, s, skip_[0], skip_[1] * input_shape_[1]);
            s = sum_slot(ctx_copy, s, skip_[1], 1);
            for (int i = 0; i < n_block_per_ct; i++) {
                if ((ct_idx * n_block_per_ct + i) >= n_out_channel_) {
                    continue;
                }
                const int32_t channel_in_ct = (ct_idx * n_block_per_ct + i) % output_channels_per_ct;
                const int32_t row_offset =
                    floor(channel_in_ct / prod(zero_inserted_skip)) * prod(skip_) * input_feature_size;
                const int32_t col_offset = floor((channel_in_ct % prod(zero_inserted_skip)) / zero_inserted_skip[1]) *
                                           input_shape_[1] * skip_[1];
                const int32_t rot_step = -row_offset - col_offset - channel_in_ct % zero_inserted_skip[1] +
                                         i * prod(skip_) * input_feature_size;

                auto m_pt = ctx_copy.ringt_to_mul(mask_pt[i], level_ - 1);
                auto c_m = ctx_copy.mult_plain_mul(s, m_pt);
                c_m = ctx_copy.rescale(c_m, ctx_copy.get_parameter().get_default_scale());
                result_ct[ct_idx * n_block_per_ct + i] = ctx_copy.rotate(c_m, rot_step);
            }
        }
    });
    if (need_repack_) {
        CkksCiphertext sp;
        for (int i = 0; i < result_ct.size(); i++) {
            int p = i % output_channels_per_ct;
            auto c_m_s = result_ct[i].copy();
            if (p == 0) {
                sp = move(c_m_s);
            } else {
                sp = ctx.add(sp, c_m_s);
            }
            if ((i + 1) % output_channels_per_ct == 0 || i == result_ct.size() - 1) {
                res.push_back(move(sp));
            }
        }
    }
    for (int i = 0; i < (int)res.size(); i++) {
        CkksPlaintextRingt gen_b_pt;
        if (lazy_encoding)
            gen_b_pt = generate_bias_pt_for_index(ctx, i);
        const CkksPlaintextRingt& b_pt = lazy_encoding ? gen_b_pt : bias_pt[i];
        res[i] = ctx.add_plain_ringt(res[i], b_pt);
    }
    return res;
}

Feature2DEncrypted MultiplexedConv2DPackedLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape = x.shape / stride_ * external_upsample_factor;
    result.skip = x.skip * stride_ / external_upsample_factor;
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    result.level = x.level - 2;
    result.data = run_core(ctx, x.data);
    return result;
}

Feature2DEncrypted MultiplexedConv2DPackedLayer::run_for_post_skip_rotation(CkksContext& ctx,
                                                                            const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape = x.shape / stride_ * external_upsample_factor;
    result.skip = x.skip * stride_ / external_upsample_factor;
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    result.level = x.level - bias_level_down;
    result.data = run_core_for_post_skip_rotation(ctx, x.data);
    return result;
}

vector<CkksCiphertext> MultiplexedConv2DPackedLayer::run_core_for_reduct_rot(CkksContext& ctx,
                                                                             const std::vector<CkksCiphertext>& x) {
    const Duo input_shape_ct = input_shape_ * skip_;
    const uint32_t input_block_size = prod(input_shape_);
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    const uint32_t skip_out_prod = prod(zero_inserted_skip);

    // 1. Block direction rotations (same as post_skip)
    vector<CkksCiphertext> input_rotated_x;
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = populate_rotations_1_side(ctx_copy, x[x_id], n_block_per_ct - 1, prod(input_shape_ct));
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    // 2. Kernel direction rotations (same as post_skip)
    int rotated_size = input_rotated_x.size();
    std::vector<std::vector<lattisense::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, input_rotated_x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    // 3. Multiply-accumulate + rescale + sum_slot + mask
    uint32_t n_weight = weight_pt.size();

    if (!need_repack_) {
        // No mask needed, directly add bias
        vector<CkksCiphertext> res(n_weight);
        parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            CkksCiphertext s(0);
            for (int j = 0; j < weight_pt[ct_idx].size(); j++) {
                for (int k = 0; k < weight_pt[ct_idx][j].size(); k++) {
                    auto& w_pt_rt = weight_pt[ct_idx][j][k];
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
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
    const uint32_t n_channel_per_ct_out = n_block_per_ct * skip_out_prod;
    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        uint32_t sub_pos = ct_idx % skip_out_prod;
        uint32_t output_ct_group = ct_idx / skip_out_prod;

        CkksCiphertext s(0);
        for (int j = 0; j < weight_pt[ct_idx].size(); j++) {
            for (int k = 0; k < weight_pt[ct_idx][j].size(); k++) {
                auto& w_pt_rt = weight_pt[ct_idx][j][k];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
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
            const uint32_t channel_local = i * skip_out_prod + sub_pos;
            const int32_t row_offset = floor(channel_local / prod(zero_inserted_skip)) * prod(skip_) * input_block_size;
            const int32_t col_offset =
                floor((channel_local % prod(zero_inserted_skip)) / zero_inserted_skip[1]) * input_shape_[1] * skip_[1];
            const int32_t rot_step =
                -row_offset - col_offset - channel_local % zero_inserted_skip[1] + i * prod(skip_) * input_block_size;
            steps.push_back(rot_step);
        }
        auto s_rots = ctx_copy.rotate(s, steps);
        for (int i = 0; i < n_block_per_ct; i++) {
            uint32_t channel_out = output_ct_group * n_channel_per_ct_out + i * skip_out_prod + sub_pos;
            if (channel_out < n_out_channel_) {
                // TODO: reduct_rot path uses a different mask layout; kept compiling
                // against the post_skip-layout mask_pt for now. reduct_rot has no
                // prepare_weight function wired up, so this path is effectively dead.
                auto& m_pt_rt = mask_pt[i];
                auto m_pt = ctx_copy.ringt_to_mul(m_pt_rt, level_ - 1);
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

Feature2DEncrypted MultiplexedConv2DPackedLayer::run_for_reduct_rot(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape = x.shape / stride_ * external_upsample_factor;
    result.skip = x.skip * stride_ / external_upsample_factor;
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * prod(stride_) / prod(external_upsample_factor);
    result.level = x.level - bias_level_down;
    result.data = run_core_for_reduct_rot(ctx, x.data);
    return result;
}
