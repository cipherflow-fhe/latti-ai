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
#include "../common.h"
#include "inverse_multiplexed_conv2d_layer.h"

InverseMultiplexedConv2DLayer::InverseMultiplexedConv2DLayer(const CkksParameter& param_in,
                                                             const Duo& input_shape_in,
                                                             const Array<double, 4>& weight_in,
                                                             const Array<double, 1>& bias_in,
                                                             const Array<int, 1>& padding_in,
                                                             const Duo& stride_in,
                                                             const Duo& stride_next_in,
                                                             const Duo& skip_in,
                                                             const Duo& block_shape_in,
                                                             uint32_t level_in,
                                                             double residual_scale)
    : param(param_in.copy()) {
    block_shape[0] = block_shape_in[0];
    block_shape[1] = block_shape_in[1];
    input_shape[0] = input_shape_in[0];
    input_shape[1] = input_shape_in[1];
    std::array<uint64_t, 4UL> weight_shape = weight_in.get_shape();
    n_out_channel = weight_shape[0];
    n_in_channel = weight_shape[1];
    kernel_shape[0] = weight_shape[2];
    kernel_shape[1] = weight_shape[3];
    if (padding_in.get(0) < 0 && padding_in.get(1) < 0) {
        padding_shape = {(kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2};
    } else if (padding_in.get(0) >= 0 && padding_in.get(1) >= 0) {
        padding_shape[0] = padding_in.get(0);
        padding_shape[1] = padding_in.get(1);
    } else {
        throw std::invalid_argument("Invalid padding inputs in InverseMultiplexedConv2DLayer");
    }
    stride[0] = stride_in[0];
    stride[1] = stride_in[1];
    stride_next[0] = stride_next_in[0];
    stride_next[1] = stride_next_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];

    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: [" + std::to_string(input_shape[0]) + ", " +
                                    std::to_string(input_shape[1]) + "]");
    }
    if ((stride[0] & (stride[0] - 1)) != 0 || (stride[1] & (stride[1] - 1)) != 0) {
        throw std::invalid_argument("stride must be powers of 2, got: [" + std::to_string(stride[0]) + ", " +
                                    std::to_string(stride[1]) + "]");
    }
    if ((stride_next[0] & (stride_next[0] - 1)) != 0 || (stride_next[1] & (stride_next[1] - 1)) != 0) {
        throw std::invalid_argument("stride_next must be powers of 2, got: [" + std::to_string(stride_next[0]) + ", " +
                                    std::to_string(stride_next[1]) + "]");
    }
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: [" + std::to_string(skip[0]) + ", " +
                                    std::to_string(skip[1]) + "]");
    }
    if ((block_shape[0] & (block_shape[0] - 1)) != 0 || (block_shape[1] & (block_shape[1] - 1)) != 0) {
        throw std::invalid_argument("block_shape must be powers of 2, got: [" + std::to_string(block_shape[0]) + ", " +
                                    std::to_string(block_shape[1]) + "]");
    }

    weight = weight_in.copy();
    bias = bias_in.copy();
    level = level_in;
    weight_scale = param.get_q(level) * residual_scale;
    N = param_in.get_n();
}

InverseMultiplexedConv2DLayer::~InverseMultiplexedConv2DLayer() {}

void InverseMultiplexedConv2DLayer::prepare_weight() {
    int pad0 = static_cast<int>(padding_shape[0]);
    int pad1 = static_cast<int>(padding_shape[1]);
    int stride0 = static_cast<int>(stride[0]);
    int stride1 = static_cast<int>(stride[1]);
    int stride_next0 = static_cast<int>(stride_next[0]);
    int stride_next1 = static_cast<int>(stride_next[1]);
    int kernel_shape0 = static_cast<int>(kernel_shape[0]);
    int kernel_shape1 = static_cast<int>(kernel_shape[1]);

    kernel_masks.clear();
    kernel_masks.resize(kernel_shape[0] * kernel_shape[1] * stride_next[0] * stride_next[1]);
    for (int i = 0; i < kernel_shape[0] * kernel_shape[1] * stride_next[0] * stride_next[1]; i++) {
        kernel_masks[i].resize(N / 2);
    }
    int mask_count = 0;
    for (int r_i2 = 0; r_i2 < stride_next[0]; r_i2++) {
        for (int r_j2 = 0; r_j2 < stride_next[1]; r_j2++) {
            for (int row_seg_idx = 0; row_seg_idx < stride[0]; row_seg_idx++) {
                for (int col_seg_idx = 0; col_seg_idx < stride[1]; col_seg_idx++) {
                    if (row_seg_idx >= kernel_shape[0] || col_seg_idx >= kernel_shape[1]) {
                        continue;
                    }
                    int split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) / stride0 + 1;
                    int split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) / stride1 + 1;
                    for (int u_s = 0; u_s < split_kernel_shape0; u_s++) {
                        for (int v_s = 0; v_s < split_kernel_shape1; v_s++) {
                            int begin_row_idx =
                                (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * stride_next0);
                            begin_row_idx = (begin_row_idx + stride0 * stride_next0) % (stride0 * stride_next0);
                            int begin_col_idx =
                                (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * stride_next1);
                            begin_col_idx = (begin_col_idx + stride1 * stride_next1) % (stride1 * stride_next1);
                            int row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) /
                                           (stride0 * stride_next0);
                            int col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) /
                                           (stride1 * stride_next1);
                            for (int i_s = 0; i_s < block_shape[0]; i_s++) {
                                for (int j_s = 0; j_s < block_shape[1]; j_s++) {
                                    if (i_s + row_step >= 0 && i_s + row_step < block_shape[0] && j_s + col_step >= 0 &&
                                        j_s + col_step < block_shape[1]) {
                                        int linear_idx = i_s * block_shape[1] + j_s;
                                        kernel_masks[mask_count][linear_idx] = 1.0;
                                    }
                                }
                            }
                            mask_count = mask_count + 1;
                        }
                    }
                }
            }
        }
    }
    input_rotate_units.clear();
    input_rotate_units.push_back(block_shape[1]);
    input_rotate_units.push_back(1);
    weight_pt.clear();
    bias_pt.clear();

    weight_pt.resize(n_out_channel);

    for (int i = 0; i < weight_pt.size(); i++) {
        weight_pt[i].resize(n_in_channel);
    }
    for (int i = 0; i < n_out_channel; i++) {
        CkksPlaintextRingt s(0);
        bias_pt.push_back(move(s));
    }

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    ctx.resize_copies(n_out_channel);

    int kernel_size = kernel_shape[0] * kernel_shape[1];
    int input_block_size = block_shape[0] * block_shape[1];
    parallel_for(n_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int out_channel_idx) {
        for (int in_channel_idx = 0; in_channel_idx < n_in_channel; ++in_channel_idx) {
            vector<CkksPlaintextRingt> a1(kernel_size * stride_next[0] * stride_next[1]);
            int kernel_count = 0;
            for (int r_i2 = 0; r_i2 < stride_next[0]; r_i2++) {
                for (int r_j2 = 0; r_j2 < stride_next[1]; r_j2++) {
                    for (int row_seg_idx = 0; row_seg_idx < stride[0]; row_seg_idx++) {
                        for (int col_seg_idx = 0; col_seg_idx < stride[1]; col_seg_idx++) {
                            int split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) / stride0 + 1;
                            int split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) / stride1 + 1;
                            for (int u_s = 0; u_s < split_kernel_shape0; u_s++) {
                                for (int v_s = 0; v_s < split_kernel_shape1; v_s++) {
                                    int kernel_idx_i = u_s * stride[0] + row_seg_idx;
                                    int kernel_idx_j = v_s * stride[1] + col_seg_idx;
                                    auto& mask = kernel_masks[kernel_count];
                                    vector<double> w(N / 2);
                                    for (int linear_idx = 0; linear_idx < input_block_size; ++linear_idx) {
                                        int shape_i = linear_idx / block_shape[1];
                                        int shape_j = linear_idx % block_shape[1];
                                        w[linear_idx] =
                                            weight.get(out_channel_idx, in_channel_idx, kernel_idx_i, kernel_idx_j) *
                                            mask[shape_i * block_shape[1] + shape_j];
                                    }
                                    a1[kernel_count] = ctx_copy.encode_ringt(w, weight_scale);
                                    kernel_count = kernel_count + 1;
                                }
                            }
                        }
                    }
                }
            }
            weight_pt[out_channel_idx][in_channel_idx] = move(a1);
        }
    });
    vector<vector<double>> feature_tmp_pack(n_out_channel);
    parallel_for(n_out_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int out_channel_idx) {
        const int total_block_size = block_shape[0] * block_shape[1];
        feature_tmp_pack[out_channel_idx].resize(N / 2);
        for (int linear_idx = 0; linear_idx < total_block_size; ++linear_idx) {
            feature_tmp_pack[out_channel_idx][linear_idx] = bias.get(out_channel_idx);
        }
        bias_pt[out_channel_idx] =
            ctx_copy.encode_ringt(feature_tmp_pack[out_channel_idx], ctx_copy.get_parameter().get_default_scale());
    });
}

void InverseMultiplexedConv2DLayer::prepare_weight_lazy() {
    int pad0 = static_cast<int>(padding_shape[0]);
    int pad1 = static_cast<int>(padding_shape[1]);
    int stride0 = static_cast<int>(stride[0]);
    int stride1 = static_cast<int>(stride[1]);
    int stride_next0 = static_cast<int>(stride_next[0]);
    int stride_next1 = static_cast<int>(stride_next[1]);
    int kernel_shape0 = static_cast<int>(kernel_shape[0]);
    int kernel_shape1 = static_cast<int>(kernel_shape[1]);

    kernel_masks.clear();
    kernel_masks.resize(kernel_shape[0] * kernel_shape[1] * stride_next[0] * stride_next[1]);
    for (int i = 0; i < kernel_shape[0] * kernel_shape[1] * stride_next[0] * stride_next[1]; i++) {
        kernel_masks[i].resize(N / 2);
    }
    int mask_count = 0;
    for (int r_i2 = 0; r_i2 < stride_next[0]; r_i2++) {
        for (int r_j2 = 0; r_j2 < stride_next[1]; r_j2++) {
            for (int row_seg_idx = 0; row_seg_idx < stride[0]; row_seg_idx++) {
                for (int col_seg_idx = 0; col_seg_idx < stride[1]; col_seg_idx++) {
                    if (row_seg_idx >= kernel_shape[0] || col_seg_idx >= kernel_shape[1]) {
                        continue;
                    }
                    int split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) / stride0 + 1;
                    int split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) / stride1 + 1;
                    for (int u_s = 0; u_s < split_kernel_shape0; u_s++) {
                        for (int v_s = 0; v_s < split_kernel_shape1; v_s++) {
                            int begin_row_idx =
                                (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * stride_next0);
                            begin_row_idx = (begin_row_idx + stride0 * stride_next0) % (stride0 * stride_next0);
                            int begin_col_idx =
                                (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * stride_next1);
                            begin_col_idx = (begin_col_idx + stride1 * stride_next1) % (stride1 * stride_next1);
                            int row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) /
                                           (stride0 * stride_next0);
                            int col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) /
                                           (stride1 * stride_next1);
                            for (int i_s = 0; i_s < block_shape[0]; i_s++) {
                                for (int j_s = 0; j_s < block_shape[1]; j_s++) {
                                    if (i_s + row_step >= 0 && i_s + row_step < block_shape[0] && j_s + col_step >= 0 &&
                                        j_s + col_step < block_shape[1]) {
                                        int linear_idx = i_s * block_shape[1] + j_s;
                                        kernel_masks[mask_count][linear_idx] = 1.0;
                                    }
                                }
                            }
                            mask_count = mask_count + 1;
                        }
                    }
                }
            }
        }
    }
    input_rotate_units.clear();
    input_rotate_units.push_back(block_shape[1]);
    input_rotate_units.push_back(1);

    // Cache computed values for on-demand generation
    cached_input_block_size = block_shape[0] * block_shape[1];
    cached_kernel_total_count = kernel_shape[0] * kernel_shape[1] * stride_next[0] * stride_next[1];
    cached_total_block_size = block_shape[0] * block_shape[1];
}

CkksPlaintextRingt InverseMultiplexedConv2DLayer::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                                 int out_channel_idx,
                                                                                 int in_channel_idx,
                                                                                 int kernel_count) const {
    int pad0 = static_cast<int>(padding_shape[0]);
    int pad1 = static_cast<int>(padding_shape[1]);
    int stride0 = static_cast<int>(stride[0]);
    int stride1 = static_cast<int>(stride[1]);
    int stride_next0 = static_cast<int>(stride_next[0]);
    int stride_next1 = static_cast<int>(stride_next[1]);
    int kernel_shape0 = static_cast<int>(kernel_shape[0]);
    int kernel_shape1 = static_cast<int>(kernel_shape[1]);

    int current_count = 0;
    int saved_r_i2 = 0, saved_r_j2 = 0, saved_row_seg_idx = 0, saved_col_seg_idx = 0, saved_u_s = 0, saved_v_s = 0;
    bool found = false;

    for (int r_i2 = 0; r_i2 < stride_next[0] && !found; r_i2++) {
        for (int r_j2 = 0; r_j2 < stride_next[1] && !found; r_j2++) {
            for (int row_seg_idx = 0; row_seg_idx < stride[0] && !found; row_seg_idx++) {
                for (int col_seg_idx = 0; col_seg_idx < stride[1] && !found; col_seg_idx++) {
                    if (row_seg_idx >= kernel_shape[0] || col_seg_idx >= kernel_shape[1]) {
                        continue;
                    }
                    int split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) / stride0 + 1;
                    int split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) / stride1 + 1;
                    for (int u_s = 0; u_s < split_kernel_shape0 && !found; u_s++) {
                        for (int v_s = 0; v_s < split_kernel_shape1 && !found; v_s++) {
                            if (current_count == kernel_count) {
                                saved_r_i2 = r_i2;
                                saved_r_j2 = r_j2;
                                saved_row_seg_idx = row_seg_idx;
                                saved_col_seg_idx = col_seg_idx;
                                saved_u_s = u_s;
                                saved_v_s = v_s;
                                found = true;
                                break;
                            }
                            current_count++;
                        }
                    }
                }
            }
        }
    }

    int kernel_idx_i = saved_u_s * stride[0] + saved_row_seg_idx;
    int kernel_idx_j = saved_v_s * stride[1] + saved_col_seg_idx;
    auto& mask = kernel_masks[kernel_count];

    vector<double> w(N / 2, 0.0);
    for (int linear_idx = 0; linear_idx < cached_input_block_size; ++linear_idx) {
        int shape_i = linear_idx / block_shape[1];
        int shape_j = linear_idx % block_shape[1];
        w[linear_idx] = weight.get(out_channel_idx, in_channel_idx, kernel_idx_i, kernel_idx_j) *
                        mask[shape_i * block_shape[1] + shape_j];
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

vector<CkksCiphertext> InverseMultiplexedConv2DLayer::run_core(CkksContext& ctx, const std::vector<CkksCiphertext>& x) {
    std::vector<std::vector<cxx_sdk_v2::CkksCiphertext>> rotated_x(n_in_channel);
    int pad0 = static_cast<int>(padding_shape[0]);
    int pad1 = static_cast<int>(padding_shape[1]);
    int stride0 = static_cast<int>(stride[0]);
    int stride1 = static_cast<int>(stride[1]);
    int stride_next0 = static_cast<int>(stride_next[0]);
    int stride_next1 = static_cast<int>(stride_next[1]);
    int kernel_shape0 = static_cast<int>(kernel_shape[0]);
    int kernel_shape1 = static_cast<int>(kernel_shape[1]);
    int block_shape1 = static_cast<int>(block_shape[1]);

    parallel_for(n_in_channel, th_nums, ctx, [&](CkksContext& ctx_copy, int in_channel_idx) {
        int base_in_ct_idx = in_channel_idx * stride[0] * stride[1] * stride_next[0] * stride_next[1];
        for (int r_i2 = 0; r_i2 < stride_next[0]; r_i2++) {
            for (int r_j2 = 0; r_j2 < stride_next[1]; r_j2++) {
                for (int row_seg_idx = 0; row_seg_idx < stride[0]; row_seg_idx++) {
                    for (int col_seg_idx = 0; col_seg_idx < stride[1]; col_seg_idx++) {
                        int split_kernel_shape0 = (kernel_shape0 - 1 - row_seg_idx) / stride0 + 1;
                        int split_kernel_shape1 = (kernel_shape1 - 1 - col_seg_idx) / stride1 + 1;
                        for (int u_s = 0; u_s < split_kernel_shape0; u_s++) {
                            for (int v_s = 0; v_s < split_kernel_shape1; v_s++) {
                                int begin_row_idx =
                                    (row_seg_idx - pad0 + stride0 * (u_s + r_i2)) % (stride0 * stride_next0);
                                begin_row_idx = (begin_row_idx + stride0 * stride_next0) % (stride0 * stride_next0);
                                int begin_col_idx =
                                    (col_seg_idx - pad1 + stride1 * (v_s + r_j2)) % (stride1 * stride_next1);
                                begin_col_idx = (begin_col_idx + stride1 * stride_next1) % (stride1 * stride_next1);
                                int begin_idx = begin_row_idx * stride1 * stride_next1 + begin_col_idx;
                                int in_ct_idx = base_in_ct_idx + begin_idx;
                                int row_step = (row_seg_idx - pad0 + stride0 * (u_s + r_i2) - begin_row_idx) /
                                               (stride0 * stride_next0);
                                int col_step = (col_seg_idx - pad1 + stride1 * (v_s + r_j2) - begin_col_idx) /
                                               (stride1 * stride_next1);

                                long step = row_step * block_shape1 + col_step;
                                CkksCiphertext res_temp = ctx_copy.rotate(x[in_ct_idx], step);
                                rotated_x[in_channel_idx].push_back(move(res_temp));
                            }
                        }
                    }
                }
            }
        }
    });

    int n_channel_per_ct_out;
    if (2 * input_shape[0] / stride[0] * input_shape[1] / stride[1] < N) {
        n_channel_per_ct_out = N / (2 * input_shape[0] / stride[0] * input_shape[1] / stride[1]);
    } else {
        n_channel_per_ct_out = 1;
    }

    uint32_t n_weight = weight_pt.empty() ? n_out_channel : weight_pt.size();
    vector<CkksCiphertext> temp_res(n_weight * stride_next[0] * stride_next[1]);

    parallel_for(n_weight, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        for (int r_i2 = 0; r_i2 < stride_next[0]; r_i2++) {
            for (int r_j2 = 0; r_j2 < stride_next[1]; r_j2++) {
                CkksCiphertext s(0);
                int out_ct_idx = ct_idx * stride_next[0] * stride_next[1] + r_i2 * stride_next[1] + r_j2;
                int base_idx = (r_i2 * stride_next[1] + r_j2) * kernel_shape[0] * kernel_shape[1];
                uint32_t n_j = weight_pt.empty() ? n_in_channel : weight_pt[ct_idx].size();
                for (int j = 0; j < n_j; j++) {
                    for (int k = 0; k < kernel_shape[0] * kernel_shape[1]; k++) {
                        if (weight_pt.empty()) {
                            auto w_pt_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, j, k + base_idx);
                            auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                            cxx_sdk_v2::CkksCiphertext one_mult_res =
                                ctx_copy.mult_plain_mul(rotated_x[j][k + base_idx], w_pt);
                            if (j == 0 && k == 0) {
                                s = move(one_mult_res);
                            } else {
                                s = ctx_copy.add(s, one_mult_res);
                            }
                        } else {
                            cxx_sdk_v2::CkksPlaintextRingt& w_pt_rt = weight_pt[ct_idx][j][k + base_idx];
                            cxx_sdk_v2::CkksPlaintextMul w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                            cxx_sdk_v2::CkksCiphertext one_mult_res =
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
        }
    });

    vector<CkksCiphertext> res(n_weight / n_channel_per_ct_out * stride_next[0] * stride_next[1]);
    if (n_channel_per_ct_out == 1) {
        res = move(temp_res);
    } else {
        for (int out_ct_idx = 0; out_ct_idx < temp_res.size(); out_ct_idx++) {
            int pack_out_ct_idx = out_ct_idx / n_channel_per_ct_out;
            int channel_idx_in_ct = out_ct_idx % n_channel_per_ct_out;
            if (channel_idx_in_ct == 0) {
                res[pack_out_ct_idx] = move(temp_res[out_ct_idx]);
            } else {
                long step = -1 * channel_idx_in_ct * input_shape[0] / stride[0] * input_shape[1] / stride[1];
                auto s_rot = ctx.rotate(temp_res[out_ct_idx], step);
                res[pack_out_ct_idx] = ctx.add(res[pack_out_ct_idx], move(s_rot));
            }
        }
    }
    return res;
}

Feature2DEncrypted InverseMultiplexedConv2DLayer::run(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature2DEncrypted result(&ctx, x.level);
    result.shape[0] = x.shape[0] / stride[0];
    result.shape[1] = x.shape[1] / stride[1];
    result.skip[0] = 1;
    result.skip[1] = 1;
    result.n_channel = n_out_channel;
    if (2 * result.shape[0] * result.shape[1] < N) {
        result.n_channel_per_ct = N / (2 * result.shape[0] * result.shape[1]);
    } else {
        result.n_channel_per_ct = 1;
    }
    result.level = x.level - 1;
    result.data = run_core(ctx, x.data);
    return result;
}

Array<double, 3> InverseMultiplexedConv2DLayer::run_plaintext(const Array<double, 3>& x, double multiplier) {
    double value = 1.0 / multiplier;

    auto x_shape = x.get_shape();
    input_shape[0] = x_shape[1];
    input_shape[1] = x_shape[2];
    vector<vector<vector<double>>> padded_input(
        n_in_channel, vector<vector<double>>(input_shape[0] + padding_shape[0] * 2,
                                             vector<double>(input_shape[1] + padding_shape[1] * 2, 0.0)));
    for (int in_channel_idx = 0; in_channel_idx < n_in_channel; in_channel_idx++) {
        for (int i = 0; i < input_shape[0]; i++) {
            for (int j = 0; j < input_shape[1]; j++) {
                padded_input[in_channel_idx][i + padding_shape[0]][j + padding_shape[1]] = x.get(in_channel_idx, i, j);
            }
        }
    }
    uint32_t output_shape[]{input_shape[0] / stride[0], input_shape[1] / stride[1]};
    Array<double, 3> result({n_out_channel, output_shape[0], output_shape[1]});
    for (int out_channel_idx = 0; out_channel_idx < n_out_channel; out_channel_idx++) {
        vector<vector<double>> output_channel(output_shape[0], vector<double>(output_shape[1], bias[out_channel_idx]));
        for (int in_channel_idx = 0; in_channel_idx < n_in_channel; in_channel_idx++) {
            for (int i = 0; i < output_shape[0]; i++) {
                for (int j = 0; j < output_shape[1]; j++) {
                    for (int ki = 0; ki < kernel_shape[0]; ki++) {
                        for (int kj = 0; kj < kernel_shape[1]; kj++) {
                            output_channel[i][j] +=
                                padded_input[in_channel_idx][i * stride[0] + ki][j * stride[1] + kj] *
                                (weight.get(out_channel_idx, in_channel_idx, ki, kj) * value);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < output_shape[0]; i++) {
            for (int j = 0; j < output_shape[1]; j++) {
                result.set(out_channel_idx, i, j, output_channel[i][j]);
            }
        }
    }
    return result;
}
