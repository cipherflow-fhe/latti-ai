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

#include "multiplexed_conv1d_pack_layer.h"
#include "conv2d_layer.h"
#include "../common.h"
#include "util.h"
#include <cmath>

using namespace std;

ParMultiplexedConv1DPackedLayer::ParMultiplexedConv1DPackedLayer(const CkksParameter& param_in,
                                                                 uint32_t input_shape_in,
                                                                 const Array<double, 3>& weight_in,
                                                                 const Array<double, 1>& bias_in,
                                                                 uint32_t stride_in,
                                                                 uint32_t skip_in,
                                                                 uint32_t n_channel_per_ct_in,
                                                                 uint32_t level_in,
                                                                 double residual_scale)
    : param(param_in.copy()), weight(weight_in.copy()), bias(bias_in.copy()) {
    input_shape = input_shape_in;
    skip = skip_in;
    stride = stride_in;
    n_channel_per_ct = n_channel_per_ct_in;
    level = level_in;

    weight_scale = param.get_q(level) * residual_scale;
    n_channel_out = weight.get_shape()[0];
    n_channel_in = weight.get_shape()[1];
    kernel_shape = weight.get_shape()[2];

    n_mult_pack_per_ct = n_channel_per_ct;
    n_packed_in_channel = div_ceil(n_channel_in, n_channel_per_ct);
    n_packed_out_channel = div_ceil(n_channel_out, n_channel_per_ct);
    cached_input_block_size = input_shape * skip;
}

ParMultiplexedConv1DPackedLayer::~ParMultiplexedConv1DPackedLayer() {}

vector<double> ParMultiplexedConv1DPackedLayer::select_tensor(int num) const {
    uint32_t skip_out = skip * stride;
    uint32_t output_shape = input_shape / stride;
    uint32_t output_shape_with_skip = output_shape * skip_out;
    int target_block = num / skip_out;
    int target_offset = num % skip_out;

    uint32_t n_groups_out = n_channel_per_ct / skip_out;
    if (n_groups_out == 0)
        n_groups_out = 1;
    vector<double> tensor(n_groups_out * output_shape_with_skip, 0.0);
    for (int out_idx = 0; out_idx < (int)output_shape; out_idx++) {
        int slot_idx = target_block * (int)output_shape_with_skip + out_idx * (int)skip_out + target_offset;
        tensor[slot_idx] = 1.0;
    }
    return tensor;
}

CkksPlaintextRingt ParMultiplexedConv1DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx,
                                                                                   int wg,
                                                                                   int w_idx,
                                                                                   int kernel_idx) const {
    int n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    uint32_t input_block_size = cached_input_block_size;

    int packed_in_idx = w_idx / n_block_per_ct;
    int block_idx = w_idx % n_block_per_ct;
    int base_channel_in = packed_in_idx * n_channel_per_ct;

    const auto& mask = kernel_masks_[kernel_idx];
    vector<double> w(param.get_n() / 2, 0.0);

    for (int linear_idx = 0; linear_idx < n_block_per_ct * (int)input_block_size; linear_idx++) {
        int t = linear_idx / (int)input_block_size;
        int shape_linear = linear_idx % (int)input_block_size;
        int channel_index = shape_linear % skip;
        int data_idx = shape_linear / skip;

        uint32_t channel_in = base_channel_in + (block_idx * skip + t * skip + channel_index) % n_channel_per_ct;
        uint32_t channel_out = wg * n_block_per_ct + t;

        if (channel_in < n_channel_in && channel_out < n_channel_out) {
            w[linear_idx] = weight.get(channel_out, channel_in, kernel_idx) * mask[data_idx];
        }
    }

    return ctx.encode_ringt(w, weight_scale);
}

CkksPlaintextRingt ParMultiplexedConv1DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int idx) const {
    bool needs_rearrange = (skip > 1 || stride > 1);
    int n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    uint32_t input_block_size = cached_input_block_size;
    vector<double> bias_data(param.get_n() / 2, 0.0);

    if (!needs_rearrange) {
        // Simple case: idx = wg (weight group index)
        for (int t = 0; t < n_block_per_ct; t++) {
            int out_ch_idx = idx * n_block_per_ct + t;
            if (out_ch_idx < (int)n_channel_out) {
                for (int data_idx = 0; data_idx < (int)input_shape; data_idx++) {
                    bias_data[t * (int)input_block_size + data_idx] = bias.get(out_ch_idx);
                }
            }
        }
        return ctx.encode_ringt(bias_data, ctx.get_parameter().get_default_scale());
    } else {
        // Rearrange case: idx = po (packed output index)
        uint32_t skip_out = skip * stride;
        uint32_t output_shape = input_shape / stride;
        for (int ch_local = 0; ch_local < (int)n_channel_per_ct; ch_local++) {
            int out_ch = idx * n_channel_per_ct + ch_local;
            if (out_ch < (int)n_channel_out) {
                int group = ch_local / (int)skip_out;
                int ch_offset = ch_local % (int)skip_out;
                for (int out_idx = 0; out_idx < (int)output_shape; out_idx++) {
                    int slot_idx = group * (int)(output_shape * skip_out) + out_idx * (int)skip_out + ch_offset;
                    bias_data[slot_idx] = bias.get(out_ch);
                }
            }
        }
        return ctx.encode_ringt(bias_data, ctx.get_parameter().get_q(level - 1));
    }
}

CkksPlaintext ParMultiplexedConv1DPackedLayer::generate_select_tensor_pt_for_index(CkksContext& ctx, int t) const {
    uint32_t input_block_size = cached_input_block_size;
    vector<double> mask(param.get_n() / 2, 0.0);
    for (int out_idx = 0; out_idx < (int)(input_shape / stride); out_idx++) {
        int slot_idx = t * (int)input_block_size + out_idx * (int)stride * (int)skip;
        mask[slot_idx] = 1.0;
    }
    return ctx.encode(mask, level - 1, ctx.get_parameter().get_q(level - 1));
}

void ParMultiplexedConv1DPackedLayer::prepare_weight() {
    uint32_t shape_with_skip = input_shape * skip;
    uint32_t half_kernel_shape = kernel_shape / 2;
    uint32_t n_groups = n_channel_per_ct / skip;

    vector<vector<double>> kernel_mask(kernel_shape);
    for (int i = 0; i < kernel_shape; i++) {
        kernel_mask[i].resize(input_shape, 0.0);
        for (int data_idx = 0; data_idx < input_shape; data_idx++) {
            bool valid_pos = true;
            if (i < half_kernel_shape && data_idx < (half_kernel_shape - i)) {
                valid_pos = false;
            } else if (i >= (kernel_shape - half_kernel_shape) && data_idx >= (input_shape - (i - half_kernel_shape))) {
                valid_pos = false;
            }
            if (valid_pos && data_idx % stride == 0) {
                kernel_mask[i][data_idx] = 1.0;
            }
        }
    }

    CkksContext ctx = CkksContext::create_empty_context(this->param);

    int n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    uint32_t n_weight_pt = div_ceil(n_channel_out, n_block_per_ct);
    uint32_t input_block_size = input_shape * skip;
    weight_pt.clear();
    weight_pt.resize(n_weight_pt);
    for (int i = 0; i < n_weight_pt; i++) {
        weight_pt[i].resize(n_packed_in_channel * n_block_per_ct);
    }

    for (int out_ch = 0; out_ch < (int)n_weight_pt; out_ch++) {
        for (int packed_in_idx = 0; packed_in_idx < (int)n_packed_in_channel; packed_in_idx++) {
            int base_channel_in = packed_in_idx * n_channel_per_ct;
            for (int block_idx = 0; block_idx < n_block_per_ct; ++block_idx) {
                int w_idx = packed_in_idx * n_block_per_ct + block_idx;
                weight_pt[out_ch][w_idx].resize(kernel_shape);
                for (int kernel_idx = 0; kernel_idx < (int)kernel_shape; kernel_idx++) {
                    vector<double> w(param.get_n() / 2, 0.0);

                    for (int linear_idx = 0; linear_idx < n_block_per_ct * input_block_size; linear_idx++) {
                        int t = linear_idx / input_block_size;
                        int shape_linear = linear_idx % input_block_size;
                        int channel_index = shape_linear % skip;
                        int data_idx = shape_linear / skip;

                        uint32_t channel_in =
                            base_channel_in + (block_idx * skip + t * skip + channel_index) % n_channel_per_ct;
                        uint32_t channel_out = out_ch * n_block_per_ct + t;

                        if (channel_in < n_channel_in && channel_out < n_channel_out) {
                            w[linear_idx] =
                                weight.get(channel_out, channel_in, kernel_idx) * kernel_mask[kernel_idx][data_idx];
                        }
                    }
                    weight_pt[out_ch][w_idx][kernel_idx] = ctx.encode_ringt(w, weight_scale);
                }
            }
        }
    }

    bool needs_rearrange = (skip > 1 || stride > 1);

    if (!needs_rearrange) {
        bias_pt.resize(n_weight_pt);
        for (int wg = 0; wg < (int)n_weight_pt; wg++) {
            vector<double> bias_data(param.get_n() / 2, 0.0);
            for (int t = 0; t < n_block_per_ct; t++) {
                int out_ch_idx = wg * n_block_per_ct + t;
                if (out_ch_idx < (int)n_channel_out) {
                    for (int data_idx = 0; data_idx < (int)input_shape; data_idx++) {
                        bias_data[t * (int)input_block_size + data_idx] = bias.get(out_ch_idx);
                    }
                }
            }
            bias_pt[wg] = ctx.encode_ringt(bias_data, ctx.get_parameter().get_default_scale());
        }
    } else {
        uint32_t skip_out = skip * stride;
        uint32_t output_shape = input_shape / stride;
        uint32_t n_packed_out = div_ceil(n_channel_out, n_channel_per_ct);

        bias_pt.resize(n_packed_out);
        for (int po = 0; po < (int)n_packed_out; po++) {
            vector<double> bias_data(param.get_n() / 2, 0.0);
            for (int ch_local = 0; ch_local < (int)n_channel_per_ct; ch_local++) {
                int out_ch = po * n_channel_per_ct + ch_local;
                if (out_ch < (int)n_channel_out) {
                    int group = ch_local / (int)skip_out;
                    int ch_offset = ch_local % (int)skip_out;
                    for (int out_idx = 0; out_idx < (int)output_shape; out_idx++) {
                        int slot_idx = group * (int)(output_shape * skip_out) + out_idx * (int)skip_out + ch_offset;
                        bias_data[slot_idx] = bias.get(out_ch);
                    }
                }
            }
            bias_pt[po] = ctx.encode_ringt(bias_data, ctx.get_parameter().get_q(level - 1));
        }

        block_select_pt.resize(n_block_per_ct);
        for (int t = 0; t < n_block_per_ct; t++) {
            vector<double> mask(param.get_n() / 2, 0.0);
            for (int out_idx = 0; out_idx < (int)(input_shape / stride); out_idx++) {
                int slot_idx = t * (int)input_block_size + out_idx * (int)stride * (int)skip;
                mask[slot_idx] = 1.0;
            }
            block_select_pt[t] = ctx.encode_ringt(mask, ctx.get_parameter().get_q(level - 1));
        }
    }
}

void ParMultiplexedConv1DPackedLayer::prepare_weight_for_lazy() {
    uint32_t half_kernel_shape = kernel_shape / 2;
    // Generate kernel masks
    kernel_masks_.clear();
    kernel_masks_.resize(kernel_shape);
    for (int i = 0; i < (int)kernel_shape; i++) {
        kernel_masks_[i].resize(input_shape, 0.0);
        for (int data_idx = 0; data_idx < (int)input_shape; data_idx++) {
            bool valid_pos = true;
            if (i < (int)half_kernel_shape && data_idx < (int)(half_kernel_shape - i)) {
                valid_pos = false;
            } else if (i >= (int)(kernel_shape - half_kernel_shape) &&
                       data_idx >= (int)(input_shape - (i - half_kernel_shape))) {
                valid_pos = false;
            }
            if (valid_pos && data_idx % stride == 0) {
                kernel_masks_[i][data_idx] = 1.0;
            }
        }
    }

    // Clear pre-computed plaintexts (will be generated on-demand)
    weight_pt.clear();
    bias_pt.clear();
    block_select_pt.clear();
}

Array<double, 2> ParMultiplexedConv1DPackedLayer::plaintext_call(const Array<double, 2>& x) {
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

Feature1DEncrypted ParMultiplexedConv1DPackedLayer::run(CkksContext& ctx, Feature1DEncrypted& x) {
    Feature1DEncrypted result(x.context, x.level);
    result.data = move(run_core(ctx, x.data));
    result.n_channel = n_channel_out;
    result.shape = x.shape / stride;
    result.skip = x.skip * stride;

    bool needs_rearrange = (skip > 1 || stride > 1);

    if (!needs_rearrange) {
        result.n_channel_per_ct = n_channel_per_ct;
        result.level = x.level - 1;
    } else {
        result.n_channel_per_ct = n_channel_per_ct;
        result.level = x.level - 2;
    }
    return result;
}

vector<CkksCiphertext> ParMultiplexedConv1DPackedLayer::run_core(CkksContext& ctx, std::vector<CkksCiphertext>& x) {
    uint32_t x_size = x.size();
    int n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    uint32_t input_block_size = input_shape * skip;
    uint32_t n_out_groups = div_ceil(n_channel_out, n_block_per_ct);

    // ======== 1: kernel ========
    int rotated_size = x.size();
    std::vector<std::vector<CkksCiphertext>> rotated_x(rotated_size);

    parallel_for(rotated_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        rotated_x[ct_idx] = Conv2DLayer::populate_rotations_2_sides(ctx_copy, x[ct_idx], kernel_shape, skip);
    });

    // ======== 2: mult + add ========
    std::vector<CkksCiphertext> result(n_out_groups);

    parallel_for(n_out_groups, th_nums, ctx, [&](CkksContext& ctx_copy, int wg) {
        CkksCiphertext s(0);
        bool first = true;

        for (int in_ct = 0; in_ct < (int)n_packed_in_channel; in_ct++) {
            for (int b = 0; b < n_block_per_ct; b++) {
                for (int k = 0; k < (int)kernel_shape; k++) {
                    CkksCiphertext to_mult;
                    if (b == 0) {
                        to_mult = rotated_x[in_ct][k].copy();
                    } else {
                        to_mult = ctx_copy.rotate(rotated_x[in_ct][k], b * (int)input_block_size);
                    }

                    int w_idx = in_ct * n_block_per_ct + b;
                    CkksCiphertext product;
                    if (weight_pt.empty()) {
                        auto w_rt = generate_weight_pt_for_indices(ctx_copy, wg, w_idx, k);
                        auto w = ctx_copy.ringt_to_mul(w_rt, level);
                        product = ctx_copy.mult_plain_mul(to_mult, w);
                    } else {
                        const auto& w_rt = weight_pt[wg][w_idx][k];
                        auto w = ctx_copy.ringt_to_mul(w_rt, level);
                        product = ctx_copy.mult_plain_mul(to_mult, w);
                    }

                    if (first) {
                        s = move(product);
                        first = false;
                    } else {
                        s = ctx_copy.add(s, product);
                    }
                }
            }
        }

        // ======== 3: skip ========
        for (int r = 1; r < (int)skip; r *= 2) {
            auto rot = ctx_copy.rotate(s, r);
            s = ctx_copy.add(s, rot);
        }

        // ======== 4: rescale ========
        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());

        result[wg] = move(s);
    });

    // ======== 5: add bias ========
    bool needs_rearrange = (skip > 1 || stride > 1);

    if (!needs_rearrange) {
        // Reason: skip==1 && stride== 1 add bias
        for (int wg = 0; wg < (int)n_out_groups; wg++) {
            if (bias_pt.empty()) {
                auto b_rt = generate_bias_pt_for_index(ctx, wg);
                result[wg] = ctx.add_plain_ringt(result[wg], b_rt);
            } else {
                result[wg] = ctx.add_plain_ringt(result[wg], bias_pt[wg]);
            }
        }
        return result;
    }

    // ======== skip>1 or stride>1: select + rotate ========
    uint32_t skip_out = skip * stride;
    uint32_t output_shape = input_shape / stride;
    uint32_t n_packed_out = div_ceil(n_channel_out, n_channel_per_ct);

    std::vector<CkksCiphertext> merged_result(n_packed_out);

    parallel_for(n_packed_out, th_nums, ctx, [&](CkksContext& ctx_copy, int po) {
        CkksCiphertext combined(0);
        bool first = true;

        for (int ch_local = 0; ch_local < (int)n_channel_per_ct; ch_local++) {
            int out_ch = po * (int)n_channel_per_ct + ch_local;
            if (out_ch >= (int)n_channel_out)
                break;

            int wg = out_ch / n_block_per_ct;
            int t = out_ch % n_block_per_ct;
            if (wg >= (int)n_out_groups)
                break;

            CkksCiphertext masked;
            if (block_select_pt.empty()) {
                auto bs_pt = generate_select_tensor_pt_for_index(ctx_copy, t);
                masked = ctx_copy.mult_plain(result[wg], bs_pt);
            } else {
                auto bs_pt = ctx_copy.ringt_to_mul(block_select_pt[t], level - 1);
                masked = ctx_copy.mult_plain_mul(result[wg], bs_pt);
            }
            masked = ctx_copy.rescale(masked, ctx_copy.get_parameter().get_default_scale());

            int group = ch_local / (int)skip_out;
            int ch_offset = ch_local % (int)skip_out;
            int source_base = t * (int)input_block_size;
            int target_base = group * (int)(output_shape * skip_out) + ch_offset;
            int rotation = target_base - source_base;

            if (rotation != 0) {
                masked = ctx_copy.rotate(masked, -rotation);
            }

            if (first) {
                combined = move(masked);
                first = false;
            } else {
                combined = ctx_copy.add(combined, masked);
            }
        }

        if (bias_pt.empty()) {
            auto b_rt = generate_bias_pt_for_index(ctx_copy, po);
            combined = ctx_copy.add_plain_ringt(combined, b_rt);
        } else {
            combined = ctx_copy.add_plain_ringt(combined, bias_pt[po]);
        }
        merged_result[po] = move(combined);
    });

    return merged_result;
}
