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

#include "dense_packed_layer.h"
#include "conv2d_layer.h"
#include "layer_util.h"
#include "util.h"
#include <chrono>
#include <numeric>
#include <vector>

using namespace std;
using namespace lattisense;

DensePackedLayer::DensePackedLayer(const CkksParameter& param_in,
                                   const Array<double, 2>& weight_in,
                                   const Array<double, 1>& bias_in,
                                   uint32_t pack_in,
                                   uint32_t level_in,
                                   int mark_in,
                                   double residual_scale)
    : Layer(param_in) {
    auto weight_shape = weight_in.get_shape();
    n_out_feature = weight_shape[0];
    n_in_feature = weight_shape[1];
    weight = weight_in.copy();
    bias = bias_in.copy();
    n_channel_per_ct = pack_in;
    n_packed_in_feature = div_ceil(n_in_feature, n_channel_per_ct);
    n_packed_out_feature = div_ceil(n_out_feature, n_channel_per_ct);
    level_ = level_in;
    mark = mark_in;
    modified_scale = param_.get_q(level_) * residual_scale;
}

void DensePackedLayer::prepare_weight_0d_skip(uint32_t skip_0d) {
    prepare_weight_0d_skip_lazy(skip_0d);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    weight_pt.clear();
    bias_pt.clear();

    weight_pt.resize(n_packed_out_feature);
    bias_pt.resize(n_packed_out_feature);
    for (uint32_t i = 0; i < n_packed_out_feature; i++) {
        weight_pt[i].resize(n_packed_in_feature * n_channel_per_ct);
    }

    uint32_t total_pairs = n_packed_out_feature * n_packed_in_feature;
    parallel_for(total_pairs, th_nums, ctx, [&](CkksContext& ctx_copy, int flat_idx) {
        uint32_t packed_out_idx = flat_idx / n_packed_in_feature;
        uint32_t packed_in_idx = flat_idx % n_packed_in_feature;
        uint32_t base = packed_in_idx * n_channel_per_ct;
        for (uint32_t d = 0; d < n_channel_per_ct; d++) {
            weight_pt[packed_out_idx][base + d] = generate_weight_0d_pt_for_indices(ctx_copy, packed_out_idx, base + d);
        }
    });

    parallel_for(n_packed_out_feature, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_idx) {
        bias_pt[packed_out_idx] = generate_bias_0d_pt_for_index(ctx_copy, packed_out_idx);
    });
}

void DensePackedLayer::prepare_weight_0d_skip_lazy(uint32_t skip_0d) {
    skip = skip_0d;
    bsgs_bs = (uint32_t)ceil(sqrt((double)n_channel_per_ct));
    bsgs_gs = div_ceil(n_channel_per_ct, bsgs_bs);

    if (!normal_dense) {
        modified_scale = modified_scale * ENC_TO_SHARE_SCALE / param_.get_default_scale();
    }
}

CkksPlaintextRingt DensePackedLayer::generate_weight_0d_pt_for_indices(CkksContext& ctx,
                                                                       uint32_t packed_out_idx,
                                                                       uint32_t weight_idx) const {
    uint32_t packed_in_idx = weight_idx / n_channel_per_ct;
    uint32_t d = weight_idx % n_channel_per_ct;
    uint32_t g = d / bsgs_bs;
    uint32_t b = d % bsgs_bs;

    vector<double> w;
    for (uint32_t j = 0; j < n_channel_per_ct; j++) {
        uint32_t out_local = (j - g * bsgs_bs + n_channel_per_ct) % n_channel_per_ct;
        uint32_t in_local = (j + b) % n_channel_per_ct;
        uint32_t out_ch = packed_out_idx * n_channel_per_ct + out_local;
        uint32_t in_ch = packed_in_idx * n_channel_per_ct + in_local;
        if (in_ch < n_in_feature && out_ch < n_out_feature) {
            w.push_back(weight.get(out_ch, in_ch));
        } else {
            w.push_back(0.0);
        }
        w.insert(w.end(), skip - 1, 0.0);
    }
    return ctx.encode_ringt(w, modified_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_bias_0d_pt_for_index(CkksContext& ctx, uint32_t packed_out_idx) const {
    double bias_scale = normal_dense ? param_.get_default_scale() : ENC_TO_SHARE_SCALE;

    vector<double> bv;
    for (uint32_t j = 0; j < n_channel_per_ct; j++) {
        uint32_t out_ch = packed_out_idx * n_channel_per_ct + j;
        if (out_ch < n_out_feature) {
            bv.push_back(bias[out_ch]);
        } else {
            bv.push_back(0.0);
        }
        bv.insert(bv.end(), skip - 1, 0.0);
    }
    return ctx.encode_ringt(bv, bias_scale);
}

void DensePackedLayer::prepare_weight_for_2d_multiplexed_lazy(const Duo& input_shape_in,
                                                              const Duo& skip_in,
                                                              const Duo& invalid_fill_in) {
    special_input_shape[0] = input_shape_in[0];
    special_input_shape[1] = input_shape_in[1];
    special_skip[0] = skip_in[0];
    special_skip[1] = skip_in[1];
    special_invalid_fill[0] = invalid_fill_in[0];
    special_invalid_fill[1] = invalid_fill_in[1];
    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    input_shape_ct_mult[0] = special_input_shape[0] * special_skip[0];
    input_shape_ct_mult[1] = special_input_shape[1] * special_skip[1];
    N_half = ctx.get_parameter().get_n() / 2;
    n_block_per_ct = div_ceil(N_half, input_shape_ct_mult[0] * input_shape_ct_mult[1]);
    int valid_skip_0 = special_skip[0] / invalid_fill_in[0];
    int valid_skip_1 = special_skip[1] / invalid_fill_in[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel = n_in_feature / (special_input_shape[0] * special_input_shape[1]);
    n_block_input = div_ceil(n_channel, n_channel_per_block);
}

CkksPlaintextRingt DensePackedLayer::generate_weight_pt_mult_pack_for_indices(CkksContext& ctx,
                                                                              int packed_out_feature_idx,
                                                                              int n_block_input_idx) const {
    int valid_skip_0 = special_skip[0] / special_invalid_fill[0];
    int valid_skip_1 = special_skip[1] / special_invalid_fill[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel_per_block_col = valid_skip_1;
    int spatial_size = special_input_shape[0] * special_input_shape[1];

    vector<double> w(N_half, 0);
    for (int i = 0; i < N_half; i++) {
        int block_i = packed_out_feature_idx * n_block_per_ct + i / (input_shape_ct_mult[0] * input_shape_ct_mult[1]);
        int shape_linear = i % (input_shape_ct_mult[0] * input_shape_ct_mult[1]);
        int shape_i = shape_linear / input_shape_ct_mult[1];
        int shape_j = shape_linear % input_shape_ct_mult[1];
        int cx = shape_i % special_skip[0];
        int cy = shape_j % special_skip[1];
        int x = shape_i / special_skip[0];
        int y = shape_j / special_skip[1];
        if (cx < valid_skip_0 && cy < valid_skip_1 && x < (int)special_input_shape[0] &&
            y < (int)special_input_shape[1] && block_i < n_out_feature) {
            int rotated_block = ((n_block_input_idx + i / (input_shape_ct_mult[0] * input_shape_ct_mult[1])) %
                                     std::min(n_block_input, n_block_per_ct) +
                                 int(n_block_input_idx / std::min(n_block_input, n_block_per_ct)) *
                                     std::min(n_block_input, n_block_per_ct));
            int in_ch = rotated_block * n_channel_per_block + cx * n_channel_per_block_col + cy;
            int line_i = in_ch * spatial_size + x * special_input_shape[1] + y;
            if (line_i >= n_in_feature || block_i > n_out_feature) {
                w[i] = 0;
            } else {
                w[i] = weight.get(block_i, line_i);
            }
        }
    }
    return ctx.encode_ringt(w, modified_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_bias_pt_mult_pack_for_index(CkksContext& ctx,
                                                                          int packed_out_feature_idx) const {
    vector<double> b(N_half, 0);
    for (int i = 0; i < N_half; i++) {
        int block_i = packed_out_feature_idx * n_block_per_ct + i / (input_shape_ct_mult[0] * input_shape_ct_mult[1]);
        int shape_linear = i % (input_shape_ct_mult[0] * input_shape_ct_mult[1]);
        int shape_i = shape_linear / input_shape_ct_mult[1];
        int shape_j = shape_linear % input_shape_ct_mult[1];
        if (shape_i == 0 && shape_j == 0 && block_i < n_out_feature) {
            b[i] = bias.get(block_i);
        }
    }
    return ctx.encode_ringt(b, ctx.get_parameter().get_default_scale());
}

void DensePackedLayer::prepare_weight_for_2d_multiplexed(const Duo& input_shape_in,
                                                         const Duo& skip_in,
                                                         const Duo& invalid_fill_in) {
    prepare_weight_for_2d_multiplexed_lazy(input_shape_in, skip_in, invalid_fill_in);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    int n_packed_out_feature_for_mult_pack = div_ceil(n_out_feature, n_block_per_ct);
    weight_pt.resize(n_packed_out_feature_for_mult_pack);
    bias_pt.resize(n_packed_out_feature_for_mult_pack);

    parallel_for(
        n_packed_out_feature_for_mult_pack, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
            weight_pt[packed_out_feature_idx].resize(n_block_input);
            for (int n_block_input_idx = 0; n_block_input_idx < n_block_input; n_block_input_idx++) {
                weight_pt[packed_out_feature_idx][n_block_input_idx] =
                    generate_weight_pt_mult_pack_for_indices(ctx_copy, packed_out_feature_idx, n_block_input_idx);
            }
            bias_pt[packed_out_feature_idx] = generate_bias_pt_mult_pack_for_index(ctx_copy, packed_out_feature_idx);
        });
}

vector<CkksCiphertext> DensePackedLayer::run_core_mult_pack(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    uint32_t x_size = x.size();
    int n_packed_out_feature_for_mult_pack =
        weight_pt.empty() ? div_ceil(n_out_feature, n_block_per_ct) : (int)weight_pt.size();

    int block_size = input_shape_ct_mult[0] * input_shape_ct_mult[1];

    // Replicate input data across CT blocks so dense only needs n_block_input iterations
    int n_rot_factor = (n_block_input > 0 && n_block_input < n_block_per_ct) ? n_block_per_ct / n_block_input : 1;
    int n_rep_iters = (n_rot_factor > 1) ? static_cast<int>(std::floor(std::log2(n_rot_factor))) : 0;

    vector<CkksCiphertext> x_rep(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        x_rep[x_id] = x[x_id].copy();
        for (int r = 0; r < n_rep_iters; r++) {
            x_rep[x_id] =
                ctx_copy.add(x_rep[x_id], ctx_copy.rotate(x_rep[x_id], -(int)pow(2, r) * n_block_input * block_size));
        }
    });

    // rotated_cts[x_id][rot] = x_rep[x_id] rotated by rot * block_size slots.
    vector<vector<CkksCiphertext>> rotated_cts(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        int n_rotations = std::min(n_block_input, n_block_per_ct);
        rotated_cts[x_id] = populate_rotations_1_side(ctx_copy, x_rep[x_id], n_rotations - 1, block_size);
    });

    vector<CkksCiphertext> result;
    result.resize(n_packed_out_feature_for_mult_pack);

    parallel_for(
        n_packed_out_feature_for_mult_pack, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
            CkksCiphertext s(0);
            int num_inputs = weight_pt.empty() ? n_block_input : weight_pt[packed_out_feature_idx].size();
            int modulus = std::min(n_block_input, n_block_per_ct);
            for (int in_feature_idx = 0; in_feature_idx < num_inputs; in_feature_idx++) {
                // in_feature_idx encodes (group, offset): group = which input ct, offset = rotation within ct.
                int group = in_feature_idx / modulus;
                int offset = in_feature_idx % modulus;
                auto& x_ct = rotated_cts[group][offset];

                CkksPlaintextRingt w_pt_rt_owned;
                const CkksPlaintextRingt* w_ptr;
                if (weight_pt.empty()) {
                    w_pt_rt_owned =
                        generate_weight_pt_mult_pack_for_indices(ctx_copy, packed_out_feature_idx, in_feature_idx);
                    w_ptr = &w_pt_rt_owned;
                } else {
                    w_ptr = &weight_pt[packed_out_feature_idx][in_feature_idx];
                }
                auto w_pt = ctx_copy.ringt_to_mul(*w_ptr, level_);
                auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
                if (in_feature_idx == 0) {
                    s = move(p);
                } else {
                    s = ctx_copy.add(s, p);
                }
            }
            s = move(ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale()));

            CkksPlaintextRingt b_pt_owned;
            const CkksPlaintextRingt* b_ptr;
            if (bias_pt.empty()) {
                b_pt_owned = generate_bias_pt_mult_pack_for_index(ctx_copy, packed_out_feature_idx);
                b_ptr = &b_pt_owned;
            } else {
                b_ptr = &bias_pt[packed_out_feature_idx];
            }
            s = ctx_copy.add_plain_ringt(s, *b_ptr);

            // Fold across all block_size = shape*skip positions (spatial + channel sub-positions).
            int n_fold = input_shape_ct_mult[0] * input_shape_ct_mult[1];
            while (n_fold > 1) {
                CkksCiphertext rotated = ctx_copy.rotate(s, n_fold / 2);
                s = ctx_copy.add(s, rotated);
                n_fold /= 2;
            }
            result[packed_out_feature_idx] = move(s);
        });
    return result;
}

Feature0DEncrypted DensePackedLayer::run_2d_multiplexed(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(run_core_mult_pack(ctx, x.data));
    result.skip = x.skip;
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = ctx.get_parameter().get_n() / 2 / result.skip;
    result.level = x.level - 1;
    return result;
}

vector<CkksCiphertext> DensePackedLayer::run_core_0d(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    uint32_t x_size = x.size();

    // Step 1: Baby-step rotations (bs-1 rotations per input CT)
    vector<vector<CkksCiphertext>> baby_rots(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int ct_id) {
        baby_rots[ct_id] = populate_rotations_1_side(ctx_copy, x[ct_id], bsgs_bs - 1, skip);
    });

    vector<CkksCiphertext> result;
    result.resize(n_packed_out_feature);

    // Step 2: For each output group, accumulate with BSGS
    parallel_for(n_packed_out_feature, th_nums, ctx, [&](CkksContext& ctx_copy, int out_idx) {
        CkksCiphertext total(0);
        bool total_init = false;

        for (uint32_t ct_in = 0; ct_in < x_size; ct_in++) {
            for (uint32_t g = 0; g < bsgs_gs; g++) {
                // Inner sum over baby-steps
                CkksCiphertext inner(0);
                bool inner_init = false;
                uint32_t b_end = std::min(bsgs_bs, n_channel_per_ct - g * bsgs_bs);

                for (uint32_t b = 0; b < b_end; b++) {
                    uint32_t d = g * bsgs_bs + b;
                    uint32_t weight_idx = ct_in * n_channel_per_ct + d;

                    CkksPlaintextRingt w_pt_rt_owned;
                    const CkksPlaintextRingt* w_ptr;
                    if (weight_pt.empty()) {
                        w_pt_rt_owned = generate_weight_0d_pt_for_indices(ctx_copy, out_idx, weight_idx);
                        w_ptr = &w_pt_rt_owned;
                    } else {
                        w_ptr = &weight_pt[out_idx][weight_idx];
                    }
                    auto w_pt = ctx_copy.ringt_to_mul(*w_ptr, level_);
                    CkksCiphertext p = ctx_copy.mult_plain_mul(baby_rots[ct_in][b], w_pt);

                    if (!inner_init) {
                        inner = move(p);
                        inner_init = true;
                    } else {
                        inner = ctx_copy.add(inner, p);
                    }
                }

                // Giant-step rotation (g=0 不需要旋转)
                if (g > 0) {
                    inner = ctx_copy.rotate(inner, g * bsgs_bs * skip);
                }

                if (!total_init) {
                    total = move(inner);
                    total_init = true;
                } else {
                    total = ctx_copy.add(total, inner);
                }
            }
        }

        total = move(ctx_copy.rescale(total, ctx_copy.get_parameter().get_default_scale()));

        CkksPlaintextRingt b_pt_owned;
        const CkksPlaintextRingt* b_ptr;
        if (bias_pt.empty()) {
            b_pt_owned = generate_bias_0d_pt_for_index(ctx_copy, out_idx);
            b_ptr = &b_pt_owned;
        } else {
            b_ptr = &bias_pt[out_idx];
        }
        total = ctx_copy.add_plain_ringt(total, *b_ptr);

        result[out_idx] = move(total);
    });
    return result;
}

Feature0DEncrypted DensePackedLayer::run_0d_skip(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(run_core_0d(ctx, x.data));
    result.skip = skip;
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = n_channel_per_ct;
    result.level = x.level - 1;
    return result;
}

Array<double, 1> DensePackedLayer::plaintext_call(const Array<double, 1>& x, double multiplier) {
    Array<double, 1> result({n_out_feature});
    double value = 1.0 / multiplier;

    for (int out_feature_idx = 0; out_feature_idx < n_out_feature; out_feature_idx++) {
        double s = bias[out_feature_idx];
        for (int in_feature_idx = 0; in_feature_idx < n_in_feature; in_feature_idx++) {
            s += weight.get(out_feature_idx, in_feature_idx) * x[in_feature_idx];
        }
        result[out_feature_idx] = s * value;
    }
    return result;
}

void DensePackedLayer::prepare_weight_for_1d_multiplexed(uint32_t input_shape_in,
                                                         uint32_t skip_in,
                                                         uint32_t invalid_fill_in) {
    prepare_weight_for_1d_multiplexed_lazy(input_shape_in, skip_in, invalid_fill_in);

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    int n_packed_out = div_ceil(n_out_feature, n_block_per_ct_1d);
    weight_pt.resize(n_packed_out);
    bias_pt.resize(n_packed_out);

    parallel_for(n_packed_out, th_nums, ctx, [&](CkksContext& ctx_copy, int out_group) {
        weight_pt[out_group].resize(n_block_input_1d);
        for (int rot_idx = 0; rot_idx < n_block_input_1d; rot_idx++) {
            weight_pt[out_group][rot_idx] = generate_weight_pt_1d_mult_for_indices(ctx_copy, out_group, rot_idx);
        }
        bias_pt[out_group] = generate_bias_pt_1d_mult_for_index(ctx_copy, out_group);
    });
}

CkksPlaintextRingt
DensePackedLayer::generate_weight_pt_1d_mult_for_indices(CkksContext& ctx, int out_group, int rot_idx) const {
    uint32_t block_stride = skip_1d;
    uint32_t block_size = input_shape_1d * block_stride;
    int valid_sub = (int)skip_1d / (int)invalid_fill_1d;

    int group = rot_idx / n_block_per_ct_1d;
    int offset = rot_idx % n_block_per_ct_1d;

    vector<double> w(N_half, 0.0);
    for (int i = 0; i < N_half; i++) {
        int local_block = i / (int)block_size;
        int pos_in_block = i % (int)block_size;
        int data_idx = pos_in_block / (int)block_stride;
        int sub = pos_in_block % (int)block_stride;

        if (sub >= valid_sub)
            continue;

        int out_ch = out_group * n_block_per_ct_1d + local_block;
        if (out_ch >= n_out_feature)
            continue;

        int rotated_block = (offset + local_block) % n_block_per_ct_1d + group * n_block_per_ct_1d;
        int in_ch = rotated_block * valid_sub + sub;
        int in_flat = in_ch * (int)input_shape_1d + data_idx;

        if (in_flat < n_in_feature) {
            w[i] = weight.get(out_ch, in_flat);
        }
    }
    return ctx.encode_ringt(w, modified_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_bias_pt_1d_mult_for_index(CkksContext& ctx, int out_group) const {
    uint32_t block_stride = skip_1d;
    uint32_t block_size = input_shape_1d * block_stride;

    vector<double> b(N_half, 0.0);
    for (int local_block = 0; local_block < n_block_per_ct_1d; local_block++) {
        int out_ch = out_group * n_block_per_ct_1d + local_block;
        if (out_ch < n_out_feature) {
            int slot = local_block * (int)block_size;
            b[slot] = bias.get(out_ch);
        }
    }
    return ctx.encode_ringt(b, param_.get_default_scale());
}

void DensePackedLayer::prepare_weight_for_1d_multiplexed_lazy(uint32_t input_shape_in,
                                                              uint32_t skip_in,
                                                              uint32_t invalid_fill_in) {
    input_shape_1d = input_shape_in;
    skip_1d = skip_in;
    invalid_fill_1d = invalid_fill_in;

    CkksContext ctx = CkksContext::create_empty_context(this->param_);
    N_half = ctx.get_parameter().get_n() / 2;

    uint32_t block_stride = skip_1d;
    uint32_t block_size = input_shape_1d * block_stride;

    int n_blocks = N_half / (int)block_size;
    int valid_sub = (int)skip_1d / (int)invalid_fill_1d;
    int n_valid_per_ct_1d = n_blocks * valid_sub;
    n_block_per_ct_1d = n_blocks;

    int n_actual_channels = n_in_feature / (int)input_shape_1d;
    n_block_input_1d = div_ceil(n_actual_channels, n_valid_per_ct_1d) * n_block_per_ct_1d;

    normal_dense = false;
    is_1d_multiplexed = true;
}

Feature0DEncrypted DensePackedLayer::run_1d_multiplexed(CkksContext& ctx, const Feature0DEncrypted& x) {
    uint32_t block_stride = skip_1d;  // skip already contains invalid_fill
    uint32_t block_size = input_shape_1d * block_stride;

    int n_packed_out = weight_pt.empty() ? div_ceil(n_out_feature, n_block_per_ct_1d) : (int)weight_pt.size();

    // Pre-compute rotations of each input ciphertext (rotate by multiples of block_size)
    int x_size = (int)x.data.size();
    vector<vector<CkksCiphertext>> rotated_cts(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_cts[x_id] = populate_rotations_1_side(ctx_copy, x.data[x_id], n_block_per_ct_1d - 1, (int)block_size);
    });

    vector<CkksCiphertext> result(n_packed_out);
    parallel_for(n_packed_out, th_nums, ctx, [&](CkksContext& ctx_copy, int out_group) {
        CkksCiphertext s(0);
        int num_rot = weight_pt.empty() ? n_block_input_1d : (int)weight_pt[out_group].size();
        for (int rot_idx = 0; rot_idx < num_rot; rot_idx++) {
            int group = rot_idx / n_block_per_ct_1d;
            int offset = rot_idx % n_block_per_ct_1d;
            auto& x_ct = rotated_cts[group][offset];

            CkksPlaintextRingt w_pt_rt_owned;
            const CkksPlaintextRingt* w_ptr;
            if (weight_pt.empty()) {
                w_pt_rt_owned = generate_weight_pt_1d_mult_for_indices(ctx_copy, out_group, rot_idx);
                w_ptr = &w_pt_rt_owned;
            } else {
                w_ptr = &weight_pt[out_group][rot_idx];
            }
            auto w_pt = ctx_copy.ringt_to_mul(*w_ptr, level_);
            auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
            if (rot_idx == 0) {
                s = move(p);
            } else {
                s = ctx_copy.add(s, p);
            }
        }
        s = ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale());

        CkksPlaintextRingt b_pt_owned;
        const CkksPlaintextRingt* b_ptr;
        if (bias_pt.empty()) {
            b_pt_owned = generate_bias_pt_1d_mult_for_index(ctx_copy, out_group);
            b_ptr = &b_pt_owned;
        } else {
            b_ptr = &bias_pt[out_group];
        }
        s = ctx_copy.add_plain_ringt(s, *b_ptr);

        // Fold: sum over the block_size slots within each output block position.
        // Each output channel's value is spread over block_size slots; fold to slot 0.
        int n_fold = (int)block_size;
        while (n_fold > 1) {
            CkksCiphertext rotated = ctx_copy.rotate(s, n_fold / 2);
            s = ctx_copy.add(s, rotated);
            n_fold /= 2;
        }
        result[out_group] = move(s);
    });

    Feature0DEncrypted out(x.context, x.level);
    out.data = move(result);
    out.skip = (int)block_size;
    out.n_channel = n_out_feature;
    out.dim = 0;
    out.n_channel_per_ct = n_block_per_ct_1d;
    out.level = x.level - 1;
    return out;
}
