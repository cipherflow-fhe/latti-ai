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
#include <set>
#include "../util.h"
#include "inverse_multiplexed_conv2d_layer.h"

using namespace std;
using namespace lattisense;

InverseMultiplexedConv2DLayer::InverseMultiplexedConv2DLayer(const CkksParameter& param_in,
                                                             const Duo& input_shape_in,
                                                             Array<double, 4>&& weight_in,
                                                             Array<double, 1>&& bias_in,
                                                             const Array<int, 1>& padding_in,
                                                             const Duo& stride_in,
                                                             const Duo& block_shape_in,
                                                             uint32_t level_in,
                                                             double residual_scale)
    : Layer(param_in), weight(move(weight_in)), bias(move(bias_in)) {
    block_shape = block_shape_in;
    input_shape = input_shape_in;
    std::array<uint64_t, 4UL> weight_shape = weight.get_shape();
    n_out_channel = weight_shape[0];
    n_in_channel = weight_shape[1];
    kernel_shape = {static_cast<uint32_t>(weight_shape[2]), static_cast<uint32_t>(weight_shape[3])};
    if (padding_in.get(0) < 0 && padding_in.get(1) < 0) {
        pad_ = to_int((kernel_shape - Duo{1, 1}) / 2);
    } else if (padding_in.get(0) >= 0 && padding_in.get(1) >= 0) {
        pad_ = {padding_in.get(0), padding_in.get(1)};
    } else {
        throw std::invalid_argument("Invalid padding inputs in InverseMultiplexedConv2DLayer");
    }
    stride = stride_in;
    orig_stride = stride_in;
    Duo output_shape = input_shape / stride_in;
    need_repack = (output_shape[0] < block_shape[0]) || (output_shape[1] < block_shape[1]);
    if (need_repack) {
        stride = input_shape / block_shape;
    }
    output_step = input_shape / (block_shape * stride);
    input_step = stride * output_step;

    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: [" + std::to_string(input_shape[0]) + ", " +
                                    std::to_string(input_shape[1]) + "]");
    }
    if ((stride[0] & (stride[0] - 1)) != 0 || (stride[1] & (stride[1] - 1)) != 0) {
        throw std::invalid_argument("stride must be powers of 2, got: [" + std::to_string(stride[0]) + ", " +
                                    std::to_string(stride[1]) + "]");
    }
    if ((block_shape[0] & (block_shape[0] - 1)) != 0 || (block_shape[1] & (block_shape[1] - 1)) != 0) {
        throw std::invalid_argument("block_shape must be powers of 2, got: [" + std::to_string(block_shape[0]) + ", " +
                                    std::to_string(block_shape[1]) + "]");
    }

    level_ = level_in;
    weight_scale = param_.get_q(level_) * residual_scale;
    N = param_in.get_n();
}

void InverseMultiplexedConv2DLayer::prepare_weight() {
    prepare_weight_lazy();

    weight_pt.clear();
    bias_pt.clear();
    weight_pt.resize(n_out_channel);
    for (int i = 0; i < (int)weight_pt.size(); i++) {
        weight_pt[i].resize(n_in_channel);
    }
    bias_pt.resize(n_out_channel);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);

    int total_kernel_count = kernel_shape[0] * kernel_shape[1] * output_step[0] * output_step[1];
    parallel_for(n_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int out_channel_idx) {
        for (int in_channel_idx = 0; in_channel_idx < n_in_channel; ++in_channel_idx) {
            vector<CkksPlaintextRingt> a1(total_kernel_count);
            for (int kernel_count = 0; kernel_count < total_kernel_count; ++kernel_count) {
                a1[kernel_count] =
                    generate_weight_pt_for_indices(ctx_copy, out_channel_idx, in_channel_idx, kernel_count);
            }
            weight_pt[out_channel_idx][in_channel_idx] = move(a1);
        }
    });
    parallel_for(n_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int out_channel_idx) {
        bias_pt[out_channel_idx] = generate_bias_pt_for_index(ctx_copy, out_channel_idx);
    });
}

void InverseMultiplexedConv2DLayer::prepare_weight_lazy() {
    input_rotate_units.clear();
    input_rotate_units.push_back(block_shape[1]);
    input_rotate_units.push_back(1);

    // Cache computed values for on-demand generation
    cached_input_block_size = prod(block_shape);
    cached_kernel_total_count = prod(kernel_shape) * prod(output_step);
    cached_total_block_size = prod(block_shape);
    if (need_repack) {
        Duo out_skip = block_shape / (input_shape / orig_stride);
        vector<double> mask_vec(N / 2, 0.0);
        for (uint32_t row = 0; row < block_shape[0]; row += out_skip[0]) {
            for (uint32_t col = 0; col < block_shape[1]; col += out_skip[1]) {
                mask_vec[row * block_shape[1] + col] = 1.0;
            }
        }
        CkksContext ctx_tmp = CkksContext::create_empty_context(this->param_);
        repack_mask_pt = ctx_tmp.encode_ringt(mask_vec, ctx_tmp.get_parameter().get_q(level_ - 1));
    }
}

CkksPlaintextRingt InverseMultiplexedConv2DLayer::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                                 int out_channel_idx,
                                                                                 int in_channel_idx,
                                                                                 int kernel_count) const {
    int current_count = 0;
    Duo saved_seg_pos = {}, saved_uv_pos = {}, saved_r2_pos = {};
    bool found = false;

    for (uint32_t r_i2 = 0; r_i2 < output_step[0] && !found; r_i2++) {
        for (uint32_t r_j2 = 0; r_j2 < output_step[1] && !found; r_j2++) {
            for (uint32_t seg0 = 0; seg0 < stride[0] && !found; seg0++) {
                for (uint32_t seg1 = 0; seg1 < stride[1] && !found; seg1++) {
                    Duo seg_pos = {seg0, seg1};
                    if (seg_pos[0] >= kernel_shape[0] || seg_pos[1] >= kernel_shape[1])
                        continue;
                    Duo split_ks = (kernel_shape + stride - Duo{1, 1} - seg_pos) / stride;
                    for (uint32_t u_s = 0; u_s < split_ks[0] && !found; u_s++) {
                        for (uint32_t v_s = 0; v_s < split_ks[1] && !found; v_s++) {
                            if (current_count == kernel_count) {
                                saved_seg_pos = seg_pos;
                                saved_uv_pos = {u_s, v_s};
                                saved_r2_pos = {r_i2, r_j2};
                                found = true;
                            } else {
                                current_count++;
                            }
                        }
                    }
                }
            }
        }
    }

    Duo kernel_idx = saved_uv_pos * stride + saved_seg_pos;

    // Compute step on-the-fly instead of reading from pre-computed kernel_masks.
    DuoInt val = to_int(saved_seg_pos) - pad_ + to_int(stride) * to_int(saved_uv_pos + saved_r2_pos);
    DuoInt begin_idx = (val % input_step + to_int(input_step)) % input_step;
    DuoInt step = (val - begin_idx) / to_int(input_step);

    int i_start = std::max(0, -step[0]);
    int i_end = std::min((int)block_shape[0], (int)block_shape[0] - step[0]);
    int j_start = std::max(0, -step[1]);
    int j_end = std::min((int)block_shape[1], (int)block_shape[1] - step[1]);

    double w_val = weight.get(out_channel_idx, in_channel_idx, kernel_idx[0], kernel_idx[1]);
    vector<double> w(N / 2, 0.0);
    for (int i_s = i_start; i_s < i_end; ++i_s) {
        for (int j_s = j_start; j_s < j_end; ++j_s) {
            w[i_s * (int)block_shape[1] + j_s] = w_val;
        }
    }
    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt InverseMultiplexedConv2DLayer::generate_bias_pt_for_index(CkksContext& ctx,
                                                                             int out_channel_idx) const {
    vector<double> bias_vec(N / 2, 0.0);
    for (int linear_idx = 0; linear_idx < cached_total_block_size; ++linear_idx) {
        bias_vec[linear_idx] = bias.get(out_channel_idx);
    }
    return ctx.encode_ringt(bias_vec, ctx.get_parameter().get_default_scale());
}

CkksPlaintextRingt InverseMultiplexedConv2DLayer::generate_repack_mask_pt(CkksContext& ctx) const {
    Duo out_skip = block_shape / (input_shape / orig_stride);
    vector<double> mask_vec(N / 2, 0.0);
    for (uint32_t row = 0; row < block_shape[0]; row += out_skip[0]) {
        for (uint32_t col = 0; col < block_shape[1]; col += out_skip[1]) {
            mask_vec[row * block_shape[1] + col] = 1.0;
        }
    }
    return ctx.encode_ringt(mask_vec, ctx.get_parameter().get_q(level_ - 1));
}

std::vector<uint32_t> InverseMultiplexedConv2DLayer::get_used_input_indices() const {
    std::set<uint32_t> used;
    for (uint32_t n_in_ch = 0; n_in_ch < n_in_channel; n_in_ch++) {
        uint32_t base = n_in_ch * prod(stride) * prod(output_step);
        for (const Duo& r2_pos : duo_range(output_step)) {
            for (const Duo& seg_pos : duo_range(stride)) {
                if (seg_pos[0] >= kernel_shape[0] || seg_pos[1] >= kernel_shape[1])
                    continue;
                Duo split_ks = (kernel_shape + stride - Duo{1, 1} - seg_pos) / stride;
                for (const Duo& uv_pos : duo_range(split_ks)) {
                    DuoInt val = to_int(seg_pos) - pad_ + to_int(stride) * to_int(uv_pos + r2_pos);
                    DuoInt begin_idx = (val % input_step + to_int(input_step)) % input_step;
                    used.insert(base + begin_idx[0] * input_step[1] + begin_idx[1]);
                }
            }
        }
    }
    return std::vector<uint32_t>(used.begin(), used.end());
}

vector<CkksCiphertext> InverseMultiplexedConv2DLayer::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    std::vector<std::vector<lattisense::CkksCiphertext>> rotated_x(n_in_channel);

    parallel_for(n_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int in_channel_idx) {
        uint32_t base_in_ct_idx = in_channel_idx * prod(stride) * prod(output_step);
        for (const Duo& r2_pos : duo_range(output_step)) {
            for (const Duo& seg_pos : duo_range(stride)) {
                if (seg_pos[0] >= kernel_shape[0] || seg_pos[1] >= kernel_shape[1])
                    continue;
                Duo split_ks = (kernel_shape + stride - Duo{1, 1} - seg_pos) / stride;
                for (const Duo& uv_pos : duo_range(split_ks)) {
                    DuoInt val = to_int(seg_pos) - pad_ + to_int(stride) * to_int(uv_pos + r2_pos);
                    DuoInt begin_idx = (val % input_step + to_int(input_step)) % input_step;
                    DuoInt step = (val - begin_idx) / to_int(input_step);
                    uint32_t in_ct_idx = base_in_ct_idx + begin_idx[0] * input_step[1] + begin_idx[1];
                    long rot_step = (long)step[0] * block_shape[1] + step[1];
                    rotated_x[in_channel_idx].push_back(ctx_copy.rotate(x[in_ct_idx], rot_step));
                }
            }
        }
    });

    int n_channel_per_ct_out;
    if (2 * prod(input_shape / stride) < N) {
        n_channel_per_ct_out = N / (2 * prod(input_shape / stride));
    } else {
        n_channel_per_ct_out = 1;
    }

    uint32_t n_weight = weight_pt.empty() ? n_out_channel : weight_pt.size();
    vector<CkksCiphertext> temp_res(n_weight * prod(output_step));

    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        for (const Duo& r2_pos : duo_range(output_step)) {
            CkksCiphertext s(0);
            int r2_linear = r2_pos[0] * output_step[1] + r2_pos[1];
            int out_ct_idx = ct_idx * prod(output_step) + r2_linear;
            int base_idx = r2_linear * prod(kernel_shape);
            uint32_t n_j = weight_pt.empty() ? n_in_channel : weight_pt[ct_idx].size();
            for (int j = 0; j < n_j; j++) {
                for (int k = 0; k < kernel_shape[0] * kernel_shape[1]; k++) {
                    if (weight_pt.empty()) {
                        auto w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, j, k + base_idx);
                        auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                        lattisense::CkksCiphertext one_mult_res =
                            ctx_copy.mult_plain_mul(rotated_x[j][k + base_idx], w_pt);
                        if (j == 0 && k == 0) {
                            s = move(one_mult_res);
                        } else {
                            s = ctx_copy.add(s, one_mult_res);
                        }
                    } else {
                        lattisense::CkksPlaintextRingt& w_pt_rt = weight_pt[ct_idx][j][k + base_idx];
                        lattisense::CkksPlaintextMul w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                        lattisense::CkksCiphertext one_mult_res =
                            ctx_copy.mult_plain_mul(rotated_x[j][k + base_idx], w_pt);
                        if (j == 0 && k == 0) {
                            s = move(one_mult_res);
                        } else {
                            s = ctx_copy.add(s, one_mult_res);
                        }
                    }
                }
            }
            s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());
            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_for_index(ctx_copy, ct_idx);
                s = ctx_copy.add_plain_ringt(s, b_pt);
            } else {
                s = ctx_copy.add_plain_ringt(s, bias_pt[ct_idx]);
            }
            temp_res[out_ct_idx] = move(s);
        }
    });

    if (need_repack) {
        Duo out_skip = block_shape / (input_shape / orig_stride);
        uint32_t n_channel_per_block = prod(out_skip);
        uint32_t n_block_per_ct_out = (N / 2) / prod(block_shape);
        uint32_t n_channel_per_ct_repack = n_channel_per_block * n_block_per_ct_out;
        uint32_t n_out_ct = div_ceil(n_out_channel, n_channel_per_ct_repack);

        // Step 1: mask all channels with the shared repack mask
        parallel_for(n_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int c) {
            auto mask_mul = ctx_copy.ringt_to_mul(repack_mask_pt, level_ - 1);
            temp_res[c] = ctx_copy.mult_plain_mul(temp_res[c], mask_mul);
        });

        // Step 2: rotate + accumulate, grouped by output CT
        vector<CkksCiphertext> res(n_out_ct);
        parallel_for(n_out_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int out_ct_idx) {
            CkksCiphertext sum(0);
            bool first = true;
            for (uint32_t ch_in_ct = 0; ch_in_ct < n_channel_per_ct_repack; ch_in_ct++) {
                uint32_t c = out_ct_idx * n_channel_per_ct_repack + ch_in_ct;
                if (c >= n_out_channel)
                    break;

                uint32_t block_idx_val = ch_in_ct / n_channel_per_block;
                uint32_t ch_in_block = ch_in_ct % n_channel_per_block;
                uint32_t cx = ch_in_block / out_skip[1];
                uint32_t cy = ch_in_block % out_skip[1];

                long rot_step = -((long)cx * block_shape[1] + cy + (long)block_idx_val * prod(block_shape));
                CkksCiphertext rotated;
                if (rot_step == 0) {
                    rotated = temp_res[c].copy();
                } else {
                    rotated = ctx_copy.rotate(temp_res[c], rot_step);
                }

                if (first) {
                    sum = move(rotated);
                    first = false;
                } else {
                    sum = ctx_copy.add(sum, rotated);
                }
            }
            res[out_ct_idx] = ctx_copy.rescale(sum, ctx_copy.get_parameter().get_default_scale());
        });
        return res;
    }

    vector<CkksCiphertext> res(div_ceil(n_weight, (uint32_t)n_channel_per_ct_out) * prod(output_step));
    if (n_channel_per_ct_out == 1) {
        res = move(temp_res);
    } else {
        for (int out_ct_idx = 0; out_ct_idx < temp_res.size(); out_ct_idx++) {
            int pack_out_ct_idx = out_ct_idx / n_channel_per_ct_out;
            int channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out;
            if (channel_idx_in_ct == 0) {
                res[pack_out_ct_idx] = move(temp_res[out_ct_idx]);
            } else {
                long step = -1 * channel_idx_in_ct * prod(input_shape / stride);
                auto s_rot = ctx.rotate(temp_res[out_ct_idx], step);
                res[pack_out_ct_idx] = ctx.add(res[pack_out_ct_idx], move(s_rot));
            }
        }
    }
    return res;
}

Feature2DEncrypted InverseMultiplexedConv2DLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape = x.shape / orig_stride;
    result.n_channel = n_out_channel;
    if (need_repack) {
        result.skip = block_shape / result.shape;
        result.n_channel_per_ct = N / (2 * prod(result.shape));
        result.level = x.level - 2;
    } else {
        result.skip = {1, 1};
        result.n_channel_per_ct = 2 * prod(result.shape) < N ? N / (2 * prod(result.shape)) : 1;
        result.level = x.level - 1;
    }
    result.data = run_core(ctx, x.data);
    return result;
}

Array<double, 3> InverseMultiplexedConv2DLayer::run_plaintext(const Array<double, 3>& x, double multiplier) {
    double value = 1.0 / multiplier;

    auto x_shape = x.get_shape();
    input_shape = {static_cast<uint32_t>(x_shape[1]), static_cast<uint32_t>(x_shape[2])};
    vector<vector<vector<double>>> padded_input(
        n_in_channel,
        vector<vector<double>>(input_shape[0] + pad_[0] * 2, vector<double>(input_shape[1] + pad_[1] * 2, 0.0)));
    for (int in_channel_idx = 0; in_channel_idx < n_in_channel; in_channel_idx++) {
        for (const Duo& pos : duo_range(input_shape)) {
            padded_input[in_channel_idx][pos[0] + pad_[0]][pos[1] + pad_[1]] = x.get(in_channel_idx, pos[0], pos[1]);
        }
    }
    Duo output_shape = input_shape / orig_stride;
    Array<double, 3> result({n_out_channel, output_shape[0], output_shape[1]});
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
    for (int out_channel_idx = 0; out_channel_idx < n_out_channel; out_channel_idx++) {
        vector<vector<double>> output_channel(output_shape[0], vector<double>(output_shape[1], bias[out_channel_idx]));
        for (int in_channel_idx = 0; in_channel_idx < n_in_channel; in_channel_idx++) {
            for (const Duo& out_pos : duo_range(output_shape)) {
                for (const Duo& k_pos : duo_range(kernel_shape)) {
                    output_channel[out_pos[0]][out_pos[1]] +=
                        padded_input[in_channel_idx][out_pos[0] * orig_stride[0] + k_pos[0]]
                                    [out_pos[1] * orig_stride[1] + k_pos[1]] *
                        (weight.get(out_channel_idx, in_channel_idx, k_pos[0], k_pos[1]) * value);
                }
            }
        }
        for (const Duo& out_pos : duo_range(output_shape)) {
            result.set(out_channel_idx, out_pos[0], out_pos[1], output_channel[out_pos[0]][out_pos[1]]);
        }
    }
    return result;
}
