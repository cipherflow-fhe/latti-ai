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

#include "multiplexed_conv1d_depthwise_pack_layer.h"
#include "conv2d_layer.h"
#include "layer_util.h"
#include "util.h"
#include <cmath>

using namespace std;
using namespace cxx_sdk_v2;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MultiplexedDWConv1DPackedLayer::MultiplexedDWConv1DPackedLayer(const CkksParameter& param_in,
                                                               uint32_t input_shape_in,
                                                               Array<double, 3>&& weight_in,
                                                               Array<double, 1>&& bias_in,
                                                               uint32_t stride_in,
                                                               uint32_t skip_in,
                                                               uint32_t n_channel_per_ct_in,
                                                               uint32_t level_in,
                                                               double residual_scale)
    : Layer(param_in), weight(move(weight_in)), bias(move(bias_in)) {
    input_shape = input_shape_in;
    skip = skip_in;
    stride = stride_in;
    n_channel_per_ct = n_channel_per_ct_in;
    level_ = level_in;

    weight_scale = param_.get_q(level_) * residual_scale;

    // weight shape: [n_channel, 1, kernel_shape]
    n_channel = weight.get_shape()[0];
    kernel_shape = weight.get_shape()[2];

    n_block_per_ct = div_ceil(n_channel_per_ct, skip);
    n_packed_ct = div_ceil(n_channel, n_channel_per_ct);
    cached_input_block_size = input_shape * skip;
}

// ---------------------------------------------------------------------------
// select_tensor  (same semantics as the normal Conv1D version)
// ---------------------------------------------------------------------------

vector<double> MultiplexedDWConv1DPackedLayer::select_tensor(int num) const {
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

// ---------------------------------------------------------------------------
// generate_weight_pt_for_indices  (lazy / on-demand)
//
// DW difference vs. normal Conv1D:
//   - ct_idx selects which packed group of channels we are handling.
//   - channel_in == channel_out for every slot (depthwise).
//   - No inner loop over packed_in_idx; each ct only uses its own weights.
// ---------------------------------------------------------------------------

CkksPlaintextRingt
MultiplexedDWConv1DPackedLayer::generate_weight_pt_for_indices(CkksContext& ctx, int ct_idx, int kernel_idx) const {
    uint32_t input_block_size = cached_input_block_size;
    const auto& mask = kernel_masks_[kernel_idx];
    vector<double> w(param_.get_n() / 2, 0.0);

    // ct_idx corresponds to block 0 of this packed group; we iterate over all
    // n_block_per_ct blocks within the ciphertext.
    for (int linear_idx = 0; linear_idx < n_block_per_ct * (int)input_block_size; linear_idx++) {
        int t = linear_idx / (int)input_block_size;  // block index within ct
        int shape_linear = linear_idx % (int)input_block_size;
        int channel_index = shape_linear % skip;  // sub-position within skip group
        int data_idx = shape_linear / skip;

        // In DW conv channel_in == channel_out for every slot.
        uint32_t channel = ct_idx * n_channel_per_ct + (t * skip + channel_index) % n_channel_per_ct;

        if (channel < n_channel) {
            // weight[channel, 0, kernel_idx]
            w[linear_idx] = weight.get(channel, 0, kernel_idx) * mask[data_idx];
        }
    }

    return ctx.encode_ringt(w, weight_scale);
}

// ---------------------------------------------------------------------------
// generate_bias_pt_for_index  (lazy / on-demand)
//
// Mirrors the two-path logic from the normal Conv1D layer.
// ---------------------------------------------------------------------------

CkksPlaintextRingt MultiplexedDWConv1DPackedLayer::generate_bias_pt_for_index(CkksContext& ctx, int idx) const {
    bool needs_rearrange = (skip > 1 || stride > 1);
    uint32_t input_block_size = cached_input_block_size;
    vector<double> bias_data(param_.get_n() / 2, 0.0);

    if (!needs_rearrange) {
        // idx == ct_idx (== packed output group index)
        for (int t = 0; t < n_block_per_ct; t++) {
            int ch = idx * n_block_per_ct + t;
            if (ch < (int)n_channel) {
                for (int data_idx = 0; data_idx < (int)input_shape; data_idx++) {
                    bias_data[t * (int)input_block_size + data_idx] = bias.get(ch);
                }
            }
        }
        return ctx.encode_ringt(bias_data, ctx.get_parameter().get_default_scale());
    } else {
        // idx == packed output ct index after rearrange
        uint32_t skip_out = skip * stride;
        uint32_t output_shape = input_shape / stride;
        for (int ch_local = 0; ch_local < (int)n_channel_per_ct; ch_local++) {
            int ch = idx * n_channel_per_ct + ch_local;
            if (ch < (int)n_channel) {
                int group = ch_local / (int)skip_out;
                int ch_offset = ch_local % (int)skip_out;
                for (int out_idx = 0; out_idx < (int)output_shape; out_idx++) {
                    int slot_idx = group * (int)(output_shape * skip_out) + out_idx * (int)skip_out + ch_offset;
                    bias_data[slot_idx] = bias.get(ch);
                }
            }
        }
        return ctx.encode_ringt(bias_data, ctx.get_parameter().get_q(level_ - 1));
    }
}

// ---------------------------------------------------------------------------
// generate_select_tensor_pt_for_index  (lazy / on-demand)
// ---------------------------------------------------------------------------

CkksPlaintext MultiplexedDWConv1DPackedLayer::generate_select_tensor_pt_for_index(CkksContext& ctx,
                                                                                  int local_ch) const {
    int t = local_ch / (int)skip;
    int j = local_ch % (int)skip;
    uint32_t input_block_size = cached_input_block_size;
    vector<double> mask(param_.get_n() / 2, 0.0);
    for (int out_idx = 0; out_idx < (int)(input_shape / stride); out_idx++) {
        int slot_idx = t * (int)input_block_size + out_idx * (int)stride * (int)skip + j;
        mask[slot_idx] = 1.0;
    }
    return ctx.encode(mask, level_ - 1, ctx.get_parameter().get_q(level_ - 1));
}

// ---------------------------------------------------------------------------
// prepare_weight
// ---------------------------------------------------------------------------

void MultiplexedDWConv1DPackedLayer::prepare_weight() {
    prepare_weight_for_lazy();

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    uint32_t input_block_size = input_shape * skip;

    weight_pt.resize(n_packed_ct);
    for (int ct_idx = 0; ct_idx < (int)n_packed_ct; ct_idx++) {
        weight_pt[ct_idx].resize(kernel_shape);
        for (int kernel_idx = 0; kernel_idx < (int)kernel_shape; kernel_idx++) {
            weight_pt[ct_idx][kernel_idx] = generate_weight_pt_for_indices(ctx, ct_idx, kernel_idx);
        }
    }

    bool needs_rearrange = (skip > 1 || stride > 1);

    if (!needs_rearrange) {
        bias_pt.resize(n_packed_ct);
        for (int ct_idx = 0; ct_idx < (int)n_packed_ct; ct_idx++) {
            bias_pt[ct_idx] = generate_bias_pt_for_index(ctx, ct_idx);
        }
    } else {
        uint32_t n_packed_out = div_ceil(n_channel, n_channel_per_ct);
        bias_pt.resize(n_packed_out);
        for (int po = 0; po < (int)n_packed_out; po++) {
            bias_pt[po] = generate_bias_pt_for_index(ctx, po);
        }

        block_select_pt.resize(min(n_channel_per_ct, n_channel));
        for (int local_ch = 0; local_ch < (int)min(n_channel_per_ct, n_channel); local_ch++) {
            int t = local_ch / (int)skip;
            int j = local_ch % (int)skip;
            vector<double> mask(param_.get_n() / 2, 0.0);
            for (int out_idx = 0; out_idx < (int)(input_shape / stride); out_idx++) {
                int slot_idx = t * (int)input_block_size + out_idx * (int)stride * (int)skip + j;
                mask[slot_idx] = 1.0;
            }
            block_select_pt[local_ch] = ctx.encode_ringt(mask, ctx.get_parameter().get_q(level_ - 1));
        }
    }
}

// ---------------------------------------------------------------------------
// prepare_weight_for_lazy
// ---------------------------------------------------------------------------

void MultiplexedDWConv1DPackedLayer::prepare_weight_for_lazy() {
    uint32_t half_kernel_shape = kernel_shape / 2;

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

    weight_pt.clear();
    bias_pt.clear();
    block_select_pt.clear();
}

// ---------------------------------------------------------------------------
// run_plaintext  (reference implementation for correctness verification)
// ---------------------------------------------------------------------------

Array<double, 2> MultiplexedDWConv1DPackedLayer::run_plaintext(const Array<double, 2>& x) {
    uint32_t output_shape = input_shape / stride;
    Array<double, 2> output({n_channel, output_shape});
    uint32_t padding = kernel_shape / 2;

    for (int ch = 0; ch < (int)n_channel; ch++) {
        for (int j = 0; j < (int)output_shape; j++) {
            double s = bias.get(ch);
            for (int k = 0; k < (int)kernel_shape; k++) {
                int src = j * stride + k - padding;
                if (src >= 0 && src < (int)input_shape) {
                    s += x.get(ch, src) * weight.get(ch, 0, k);
                }
            }
            output.set(ch, j, s);
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

Feature1DEncrypted MultiplexedDWConv1DPackedLayer::run(CkksContext& ctx, Feature1DEncrypted& x) {
    Feature1DEncrypted result(x.context, x.level);
    result.data = move(run_core(ctx, x.data));
    result.n_channel = n_channel;
    result.shape = x.shape / stride;
    result.skip = x.skip * stride;
    result.n_channel_per_ct = n_channel_per_ct;

    bool needs_rearrange = (skip > 1 || stride > 1);
    result.level = x.level - (needs_rearrange ? 2 : 1);

    return result;
}

// ---------------------------------------------------------------------------
// run_core
//
// DW simplification vs. normal Conv1D run_core:
//   - No outer loop over n_packed_in_channel; ct_idx == input ct == output ct.
//   - No cross-ct weight rotation; each ct is self-contained.
//   - The skip-reduction and rearrange stages are identical to the normal layer.
// ---------------------------------------------------------------------------

vector<CkksCiphertext> MultiplexedDWConv1DPackedLayer::run_core(CkksContext& ctx, vector<CkksCiphertext>& x) {
    uint32_t input_block_size = cached_input_block_size;

    // ======== 1: kernel rotations ========
    int n_ct = x.size();
    vector<vector<CkksCiphertext>> rotated_x(n_ct);

    parallel_for(n_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        rotated_x[ct_idx] = populate_rotations_2_sides(ctx_copy, x[ct_idx], kernel_shape, skip);
    });

    // ======== 2: mult + add over kernel positions (no cross-ct sum) ========
    vector<CkksCiphertext> result(n_packed_ct);

    parallel_for(n_packed_ct, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_idx) {
        CkksCiphertext s(0);
        bool first = true;

        for (int k = 0; k < (int)kernel_shape; k++) {
            CkksCiphertext product;
            if (weight_pt.empty()) {
                auto w_rt = generate_weight_pt_for_indices(ctx_copy, ct_idx, k);
                auto w = ctx_copy.ringt_to_mul(w_rt, level_);
                product = ctx_copy.mult_plain_mul(rotated_x[ct_idx][k], w);
            } else {
                const auto& w_rt = weight_pt[ct_idx][k];
                auto w = ctx_copy.ringt_to_mul(w_rt, level_);
                product = ctx_copy.mult_plain_mul(rotated_x[ct_idx][k], w);
            }

            if (first) {
                s = move(product);
                first = false;
            } else {
                s = ctx_copy.add(s, product);
            }
        }

        // ======== 3: skip reduction (NOT done for DW conv — each slot already holds
        //            the result for its own channel; summing would mix channels) ========

        // ======== 4: rescale ========
        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());

        result[ct_idx] = move(s);
    });

    // ======== 5: add bias ========
    bool needs_rearrange = (skip > 1 || stride > 1);

    if (!needs_rearrange) {
        for (int ct_idx = 0; ct_idx < (int)n_packed_ct; ct_idx++) {
            if (bias_pt.empty()) {
                auto b_rt = generate_bias_pt_for_index(ctx, ct_idx);
                result[ct_idx] = ctx.add_plain_ringt(result[ct_idx], b_rt);
            } else {
                result[ct_idx] = ctx.add_plain_ringt(result[ct_idx], bias_pt[ct_idx]);
            }
        }
        return result;
    }

    // ======== skip>1 or stride>1: select + rotate ========
    uint32_t skip_out = skip * stride;
    uint32_t output_shape = input_shape / stride;
    uint32_t n_packed_out = div_ceil(n_channel, n_channel_per_ct);

    vector<CkksCiphertext> merged_result(n_packed_out);

    parallel_for(n_packed_out, th_nums, ctx, [&](CkksContext& ctx_copy, int po) {
        CkksCiphertext combined(0);
        bool first = true;

        for (int ch_local = 0; ch_local < (int)n_channel_per_ct; ch_local++) {
            int ch = po * (int)n_channel_per_ct + ch_local;
            if (ch >= (int)n_channel)
                break;

            // DW: result CT index == po (= ch / n_channel_per_ct)
            int ct_idx = po;
            int local_ch = ch_local;       // ch % n_channel_per_ct
            int t = local_ch / (int)skip;  // block within CT
            int j = local_ch % (int)skip;  // channel_index within skip group

            CkksCiphertext masked;
            if (block_select_pt.empty()) {
                auto bs_pt = generate_select_tensor_pt_for_index(ctx_copy, local_ch);
                masked = ctx_copy.mult_plain(result[ct_idx], bs_pt);
            } else {
                auto bs_pt = ctx_copy.ringt_to_mul(block_select_pt[local_ch], level_ - 1);
                masked = ctx_copy.mult_plain_mul(result[ct_idx], bs_pt);
            }
            masked = ctx_copy.rescale(masked, ctx_copy.get_parameter().get_default_scale());

            int group = ch_local / (int)skip_out;
            int ch_offset = ch_local % (int)skip_out;
            int source_base = t * (int)input_block_size + j;  // slot of (out_idx=0, channel j)
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
