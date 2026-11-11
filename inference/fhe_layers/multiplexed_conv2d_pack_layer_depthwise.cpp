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
#include "interleaved_seed.h"

using namespace std;
using namespace lattisense;

MultiplexedConv2DPackedLayerDepthwise::MultiplexedConv2DPackedLayerDepthwise(const CkksParameter& param_in,
                                                                             const Duo& input_shape_in,
                                                                             Array<double, 4>&& weight_in,
                                                                             Array<double, 1>&& bias_in,
                                                                             const Duo& stride_in,
                                                                             const Duo& skip_in,
                                                                             uint32_t n_channel_per_ct_in,
                                                                             uint32_t level_in,
                                                                             double residual_scale,
                                                                             const Duo& seed_weight_subgrid_stride_in,
                                                                             bool joint_seed_requests)
    : Conv2DLayer(param_in, input_shape_in, move(weight_in), move(bias_in), stride_in) {
    skip_ = skip_in;
    if ((skip_[0] & (skip_[0] - 1)) != 0 || (skip_[1] & (skip_[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: " + str(skip_));
    }
    n_groups_ = n_out_channel_;
    n_in_channel_ = n_out_channel_;
    const uint32_t output_channels_per_ct = n_channel_per_ct_in * prod(stride_);

    n_channel_per_ct = n_channel_per_ct_in;
    n_packed_in_channel = div_ceil(n_out_channel_, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_out_channel_, output_channels_per_ct);
    n_block_per_ct = div_ceil(n_channel_per_ct, prod(skip_));
    need_repack_ = (stride_ != Duo{1, 1});
    level_ = level_in;
    weight_scale = param_.q()[level_] * residual_scale;
    seed_weight_subgrid_stride_ = seed_weight_subgrid_stride_in;
    use_seed_weight_pack_ = interleaved_seed::valid_stride(seed_weight_subgrid_stride_, input_shape_);
    joint_seed_requests_ = joint_seed_requests;
    if (joint_seed_requests_ && n_packed_in_channel * prod(kernel_shape_) == 1)
        use_seed_weight_pack_ = false;
}

void MultiplexedConv2DPackedLayerDepthwise::prepare_weight() {
    prepare_weight_lazy();

    const int rows = joint_seed_weights() ? 1 : n_packed_in_channel;
    const int kernel_size = joint_seed_weights() ? seed_weight_count() : seed_weight_kernel_count();
    weight_pt.clear();
    bias_pt.clear();
    weight_pt.resize(rows);
    bias_pt.resize(n_packed_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);

    parallel_for(rows, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        weight_pt[n_packed_out_channel_idx].resize(kernel_size);
        for (int kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
            weight_pt[n_packed_out_channel_idx][kernel_idx] =
                use_seed_weight_pack_ ?
                    generate_weight_pt_for_seed_indices(ctx_copy, n_packed_out_channel_idx, kernel_idx) :
                    generate_weight_pt_for_indices(ctx_copy, n_packed_out_channel_idx, kernel_idx);
        }
    });

    parallel_for(n_packed_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int n_packed_out_channel_idx) {
        bias_pt[n_packed_out_channel_idx] = generate_bias_pt_for_index(ctx_copy, n_packed_out_channel_idx);
    });

    if (stride_[0] != 1) {
        mask_pt.resize(n_out_channel_);
        parallel_for(n_packed_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
            for (int i = 0; i < static_cast<int>(n_channel_per_ct); ++i) {
                const int channel_global = ct_idx * static_cast<int>(n_channel_per_ct) + i;
                if (channel_global >= static_cast<int>(n_out_channel_)) {
                    break;
                }
                mask_pt[channel_global] = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
            }
        });
    }
}

void MultiplexedConv2DPackedLayerDepthwise::prepare_weight_lazy() {
    const Duo input_shape_ct = input_shape_ * skip_;
    kernel_masks_.clear();

    input_rotate_units_.clear();
    input_rotate_units_.push_back(skip_[0] * input_shape_ct[1]);
    input_rotate_units_.push_back(skip_[1]);

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

    mask_pt.clear();
}

vector<double> MultiplexedConv2DPackedLayerDepthwise::select_tensor(int num) const {
    return select_tensor_for_config(input_shape_, skip_, stride_, n_channel_per_ct, num);
}

vector<double> MultiplexedConv2DPackedLayerDepthwise::select_tensor_for_config(const Duo& input_shape,
                                                                               const Duo& skip,
                                                                               const Duo& stride,
                                                                               uint32_t channels_per_ct,
                                                                               int num) {
    const Duo input_shape_ct = input_shape * skip;
    const Duo stride_skip = stride * skip;
    const uint32_t stride_skip_prod = prod(stride_skip);

    vector<double> tensor;
    for (int block_idx = 0; block_idx < div_ceil(channels_per_ct, prod(skip)); ++block_idx) {
        for (const Duo& input_pos : duo_range(input_shape_ct)) {
            if (block_idx * stride_skip_prod + stride_skip[1] * (input_pos[0] % stride_skip[0]) +
                    (input_pos[1] % stride_skip[1]) ==
                num) {
                tensor.push_back(1.0);
            } else {
                tensor.push_back(0.0);
            }
        }
    }

    return tensor;
}

vector<double> MultiplexedConv2DPackedLayerDepthwise::generate_weight_values_for_indices(int n_packed_out_channel_idx,
                                                                                         int kernel_idx,
                                                                                         bool boundary) const {
    const Duo padding_shape = kernel_shape_ / 2;
    const Duo kernel_pos = div_mod(static_cast<uint32_t>(kernel_idx), kernel_shape_[1]);

    vector<double> w(param_.max_slots(), 0.0);

    for (uint32_t linear_idx = 0; linear_idx < n_block_per_ct * cached_input_block_size; ++linear_idx) {
        const Duo block_pos = div_mod(linear_idx, cached_input_block_size);
        const Duo input_pos = div_mod(block_pos[1], cached_input_shape_ct[1]);

        // Inline boundary check (replaces pre-computed kernel_masks_)
        int64_t shifted_i = (int64_t)(kernel_pos[0] * skip_[0] + input_pos[0]) - (int64_t)(padding_shape[0] * skip_[0]);
        int64_t shifted_j = (int64_t)(kernel_pos[1] * skip_[1] + input_pos[1]) - (int64_t)(padding_shape[1] * skip_[1]);
        if (boundary && (shifted_i < 0 || shifted_i >= cached_input_shape_ct[0] || shifted_j < 0 ||
                         shifted_j >= cached_input_shape_ct[1])) {
            continue;
        }

        const uint32_t channel_in = 0;
        const uint32_t channel_out = static_cast<uint32_t>(n_packed_out_channel_idx) * n_channel_per_ct +
                                     block_pos[0] * cached_skip_prod +
                                     (skip_[1] * (input_pos[0] % skip_[0]) + input_pos[1] % skip_[1]);

        if (channel_in < n_in_channel_ && channel_out < n_out_channel_) {
            w[linear_idx] = weight_.get(channel_out, channel_in, kernel_pos[0], kernel_pos[1]);
        }
    }
    return w;
}

CkksPlaintext MultiplexedConv2DPackedLayerDepthwise::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                                    int ct_idx,
                                                                                    int kernel_index) const {
    return fhe_api::encode_ringt(ctx, generate_weight_values_for_indices(ct_idx, kernel_index, true), weight_scale);
}

CkksPlaintext MultiplexedConv2DPackedLayerDepthwise::generate_weight_pt_for_seed_indices(CkksContext& ctx,
                                                                                         int ct_idx,
                                                                                         int seed_kernel) const {
    const int requests = joint_seed_weights() ? n_packed_in_channel * cached_kernel_size : cached_kernel_size;
    if (ct_idx < 0 || (joint_seed_weights() ? ct_idx != 0 : ct_idx >= static_cast<int>(n_packed_in_channel)) ||
        seed_kernel < 0 || seed_kernel >= static_cast<int>(div_ceil(requests, prod(seed_weight_subgrid_stride_)))) {
        throw std::out_of_range("depthwise seed request index");
    }
    auto seed = interleaved_seed::pack_kernels(requests, seed_kernel, seed_weight_subgrid_stride_, input_shape_, skip_,
                                               n_block_per_ct, param_.max_slots(), [&](int request) {
                                                   const int group =
                                                       joint_seed_weights() ? request / cached_kernel_size : ct_idx;
                                                   const int kernel = request % cached_kernel_size;
                                                   return generate_weight_values_for_indices(group, kernel, false);
                                               });
    return fhe_api::encode_ringt(ctx, seed, weight_scale);
}

vector<double>
MultiplexedConv2DPackedLayerDepthwise::seed_mask_values(int subgrid, int kernel_index, bool boundary) const {
    const auto step =
        DuoInt{static_cast<int>(kernel_index / kernel_shape_[1]) - static_cast<int>(kernel_shape_[0] / 2),
               static_cast<int>(kernel_index % kernel_shape_[1]) - static_cast<int>(kernel_shape_[1] / 2)};
    return interleaved_seed::mask(input_shape_, skip_, n_block_per_ct, param_.max_slots(), seed_weight_subgrid_stride_,
                                  subgrid, step, boundary);
}

CkksPlaintext MultiplexedConv2DPackedLayerDepthwise::generate_mask_pt_for_seed_indices(CkksContext& ctx,
                                                                                       int subgrid,
                                                                                       int kernel_index,
                                                                                       bool boundary) const {
    const int mask_level = level_ + seed_weight_level_cost() - (boundary ? 1 : 0);
    return fhe_api::encode_ringt(ctx, seed_mask_values(subgrid, kernel_index, boundary), param_.q()[mask_level]);
}

CkksPlaintext MultiplexedConv2DPackedLayerDepthwise::generate_bias_pt_for_index(CkksContext& ctx, int bpt_idx) const {
    const int n_real_num_slot = param_.max_slots();
    const Duo bias_shape = input_shape_ / stride_;
    const Duo bias_block_shape = bias_shape * cached_bias_skip;
    const uint32_t bias_block_size = prod(bias_block_shape);
    const uint32_t bias_skip_prod = prod(cached_bias_skip);
    vector<double> bias_vec(n_real_num_slot, 0.0);

    for (uint32_t linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        const Duo block_pos = div_mod(linear_idx, bias_block_size);
        const Duo output_pos = div_mod(block_pos[1], bias_block_shape[1]);
        const Duo channel_offset = output_pos % cached_bias_skip;

        const uint32_t channel = static_cast<uint32_t>(bpt_idx) * cached_bias_n_channel_per_ct +
                                 block_pos[0] * bias_skip_prod + cached_bias_skip[1] * channel_offset[0] +
                                 channel_offset[1];
        if (channel >= n_out_channel_) {
            continue;
        }

        bias_vec[linear_idx] = bias_.get(channel);
    }
    return fhe_api::encode_ringt(ctx, bias_vec, ctx.parameter().default_scale());
}

CkksPlaintext
MultiplexedConv2DPackedLayerDepthwise::generate_mask_pt_for_indices(CkksContext& ctx, int ct_idx, int i) const {
    return fhe_api::encode_ringt(
        ctx, mask_tensor_for_config(input_shape_, skip_, stride_, n_channel_per_ct, param_.max_slots(), ct_idx, i),
        ctx.parameter().q()[level_ - 1]);
}

vector<double> MultiplexedConv2DPackedLayerDepthwise::mask_tensor_for_config(const Duo& input_shape,
                                                                             const Duo& skip,
                                                                             const Duo& stride,
                                                                             uint32_t channels_per_ct,
                                                                             uint32_t slots_in,
                                                                             int ct_idx,
                                                                             int i) {
    if (slots_in == 0 || prod(skip) == 0 || prod(stride) == 0 || channels_per_ct == 0 || ct_idx < 0 || i < 0 ||
        i >= static_cast<int>(channels_per_ct) ||
        uint64_t(div_ceil(channels_per_ct, prod(skip))) * prod(input_shape * skip) > slots_in) {
        throw std::invalid_argument("invalid depthwise mask geometry or index");
    }
    const int32_t slots = static_cast<int32_t>(slots_in);
    const uint32_t output_channels_per_ct = channels_per_ct * prod(stride);
    const uint32_t stride_skip_prod = prod(skip * stride);
    const uint32_t skip_prod = prod(skip);
    const uint32_t input_feature_size = prod(input_shape);

    // Recompute step_k for the aligned channel index (matches run_core's steps[i/skip[1]]).
    const int32_t i_aligned = (i / static_cast<int>(skip[1])) * static_cast<int>(skip[1]);
    const int32_t channel_idx = ct_idx * static_cast<int>(channels_per_ct) + i_aligned;
    const int32_t rotated_block = channel_idx / static_cast<int32_t>(stride_skip_prod);
    const int32_t rotated_residual = channel_idx % static_cast<int32_t>(stride_skip_prod);
    const int32_t rotated_row = rotated_residual / static_cast<int32_t>(stride[1] * skip[1]);
    const int32_t rotated_col = rotated_residual % static_cast<int32_t>(stride[1] * skip[1]);
    const int32_t base_block = channel_idx / static_cast<int32_t>(skip_prod);
    const int32_t base_residual = channel_idx % static_cast<int32_t>(skip_prod);
    const int32_t base_row = base_residual / static_cast<int32_t>(skip[1]);
    const int32_t base_col = base_residual % static_cast<int32_t>(skip[1]);
    const int32_t rot_step = (rotated_block - base_block) * static_cast<int32_t>(skip_prod * input_feature_size) +
                             (rotated_row - base_row) * static_cast<int32_t>(skip[1] * input_shape[1]) +
                             (rotated_col - base_col);
    const int32_t step_k = -rot_step;

    // Target-position mask (legacy select_tensor layout).
    auto target = select_tensor_for_config(input_shape, skip, stride, channels_per_ct,
                                           (ct_idx * channels_per_ct + i) % output_channels_per_ct);

    // Source-position mask = vec_rotate(target, -step_k) on the full slot vector:
    //   source[(j + step_k) mod slots] = target[j]
    // matches the SDK convention rotate(c, k)[j] = c[(j+k) mod slots], so that
    // rotate(source * s, step_k) reproduces target * rotate(s, step_k).
    const int32_t shift = ((step_k % slots) + slots) % slots;
    vector<double> source(static_cast<size_t>(slots), 0.0);
    for (size_t j = 0; j < target.size(); ++j) {
        if (target[j] != 0.0) {
            source[(static_cast<int32_t>(j) + shift) % slots] = target[j];
        }
    }
    return source;
}

vector<CkksCiphertext> MultiplexedConv2DPackedLayerDepthwise::run_core(CkksContext& ctx,
                                                                       const std::vector<CkksCiphertext>& x) {
    const uint32_t output_channels_per_ct = n_channel_per_ct * prod(stride_);
    const uint32_t input_feature_size = prod(input_shape_);

    vector<CkksCiphertext> result_ct;
    result_ct.resize(n_out_channel_);

    // 1. rotation of kernel direction
    int rotated_size = x.size();
    std::vector<std::vector<lattisense::CkksCiphertext>> rotated_x(rotated_size);
    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        vector<CkksCiphertext> rotations =
            populate_rotations_2_sides(ctx_copy, x[ct_idx], kernel_shape_[0], input_rotate_units_[0]);
        for (auto& r : rotations) {
            auto x = populate_rotations_2_sides(ctx_copy, r, kernel_shape_[1], input_rotate_units_[1]);
            move(x.begin(), x.end(), back_inserter(rotated_x[ct_idx]));
        }
    });

    vector<CkksCiphertext> res;
    uint32_t n_weight = n_packed_in_channel;
    if (!need_repack_) {
        res.resize(n_weight);
    }
    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s;
        uint32_t n_k = cached_kernel_size;
        for (int k = 0; k < n_k; k++) {
            CkksCiphertext r_tmp;
            if (use_seed_weight_pack_) {
                const int logical_kernel = k;
                const int request = joint_seed_weights() ? ct_idx * n_k + k : k;
                const int stored_group = joint_seed_weights() ? 0 : ct_idx;
                const int seed_index = request / prod(seed_weight_subgrid_stride_);
                const int subgrid = request % prod(seed_weight_subgrid_stride_);
                CkksPlaintext generated;
                if (weight_pt.empty())
                    generated = generate_weight_pt_for_seed_indices(ctx_copy, stored_group, seed_index);
                const auto& seed = weight_pt.empty() ? generated : weight_pt[stored_group][seed_index];
                auto restored = interleaved_seed::restore_plaintext(
                    ctx_copy, seed, seed_mask_values(subgrid, logical_kernel, false),
                    seed_mask_values(subgrid, logical_kernel, true), input_shape_, skip_, seed_weight_subgrid_stride_,
                    subgrid, level_, weight_scale);
                const auto& mul_pt = restored;
                r_tmp = ctx_copy.mult(rotated_x[ct_idx][k], mul_pt);
            } else if (weight_pt.empty()) {
                auto w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, k);
                const auto& w_pt = w_pt_rt;
                r_tmp = ctx_copy.mult(rotated_x[ct_idx][k], w_pt);
            } else {
                auto& w_pt_rt = weight_pt[ct_idx][k];
                const auto& w_pt = w_pt_rt;
                r_tmp = ctx_copy.mult(rotated_x[ct_idx][k], w_pt);
            }
            if (k == 0) {
                s = move(r_tmp);
            } else {
                s = ctx_copy.add(s, r_tmp);
            }
        }
        s = ctx_copy.rescale(s);
        if (!need_repack_) {
            res[ct_idx] = move(s);
        } else {
            vector<int32_t> steps;
            const uint32_t stride_skip_prod = prod(skip_ * stride_);
            const uint32_t skip_prod = prod(skip_);
            for (int i = 0; i < n_channel_per_ct; i += skip_[1]) {
                const int32_t channel_idx = ct_idx * n_channel_per_ct + i;
                const int32_t rotated_block = floor(channel_idx / stride_skip_prod);
                const int32_t rotated_residual = channel_idx % stride_skip_prod;
                const int32_t rotated_row = floor(rotated_residual / (stride_[1] * skip_[1]));
                const int32_t rotated_col = rotated_residual % (stride_[1] * skip_[1]);

                const int32_t base_block = floor(channel_idx / skip_prod);
                const int32_t base_residual = channel_idx % skip_prod;
                const int32_t base_row = floor(base_residual / skip_[1]);
                const int32_t base_col = base_residual % skip_[1];
                const int32_t rot_step = (rotated_block - base_block) * skip_prod * input_feature_size +
                                         (rotated_row - base_row) * (skip_[1] * input_shape_[1]) +
                                         (rotated_col - base_col);
                steps.push_back(-rot_step);
            }
            const bool lazy_mask = mask_pt.empty();
            for (int i = 0; i < n_channel_per_ct; i++) {
                if ((ct_idx * n_channel_per_ct + i) >= n_out_channel_) {
                    continue;
                }
                CkksPlaintext gen_m_pt;
                if (lazy_mask) {
                    gen_m_pt = generate_mask_pt_for_indices(ctx_copy, ct_idx, i);
                }
                const CkksPlaintext& m_pt_rt = lazy_mask ? gen_m_pt : mask_pt[ct_idx * n_channel_per_ct + i];
                const auto& m_pt = m_pt_rt;
                auto c_m = ctx_copy.mult(s, m_pt);
                c_m = ctx_copy.rescale(c_m);
                result_ct[ct_idx * n_channel_per_ct + i] = ctx_copy.rotate(c_m, steps[i / skip_[1]]);
            }
        }
    });
    if (need_repack_) {
        CkksCiphertext sp;
        for (int i = 0; i < (int)result_ct.size(); i++) {
            int p = i % output_channels_per_ct;
            auto c_m_s = move(result_ct[i]);
            if (p == 0) {
                sp = move(c_m_s);
            } else {
                sp = ctx.add(sp, c_m_s);
            }
            if ((i + 1) % output_channels_per_ct == 0 || i == (int)result_ct.size() - 1) {
                res.push_back(move(sp));
            }
        }
    }
    for (int i = 0; i < (int)res.size(); i++) {
        if (bias_pt.empty()) {
            auto b_pt = generate_bias_pt_for_index(ctx, i);
            res[i] = ctx.add(res[i], b_pt);
        } else {
            res[i] = ctx.add(res[i], bias_pt[i]);
        }
    }
    return res;
}

Feature2DEncrypted MultiplexedConv2DPackedLayerDepthwise::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    int bias_level_down = need_repack_ ? 2 : 1;
    result.shape = x.shape / stride_;
    result.skip = x.skip * stride_;
    result.n_channel = n_out_channel_;
    result.n_channel_per_ct = x.n_channel_per_ct * prod(stride_);
    result.level = x.level - bias_level_down;
    result.data = run_core(ctx, x.data);
    return result;
}
