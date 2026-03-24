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
#include "util.h"
#include <chrono>
#include <numeric>
#include <vector>

using namespace std;

DensePackedLayer::DensePackedLayer(const CkksParameter& param_in,
                                   const Array<double, 2>& weight_in,
                                   const Array<double, 1>& bias_in,
                                   uint32_t pack_in,
                                   uint32_t level_in,
                                   int mark_in,
                                   double residual_scale)
    : param(param_in.copy()) {
    auto weight_shape = weight_in.get_shape();
    n_out_feature = weight_shape[0];
    n_in_feature = weight_shape[1];
    weight = weight_in.copy();
    bias = bias_in.copy();
    pack = pack_in;
    n_packed_in_feature = div_ceil(n_in_feature, pack);
    n_packed_out_feature = div_ceil(n_out_feature, pack);
    level_ = level_in;
    mark = mark_in;
    modified_scale = param.get_q(level_) * residual_scale;
}

void DensePackedLayer::prepare_weight_skip_0d(uint32_t skip_0d) {
    skip_0d_val = skip_0d;

    // BSGS decomposition: pack = bs * gs, bs ≈ √pack
    bsgs_bs = (uint32_t)ceil(sqrt((double)pack));
    bsgs_gs = div_ceil(pack, bsgs_bs);

    CkksContext ctx = CkksContext::create_empty_context(this->param);
    weight_pt.clear();
    bias_pt.clear();

    double bias_scale = 0;
    if (!normal_dense) {
        modified_scale = modified_scale * ENC_TO_SHARE_SCALE / param.get_default_scale();
        bias_scale = ENC_TO_SHARE_SCALE;
    } else {
        bias_scale = param.get_default_scale();
    }

    for (uint32_t packed_out_idx = 0; packed_out_idx < n_packed_out_feature; packed_out_idx++) {
        vector<CkksPlaintextRingt> a1;
        for (uint32_t packed_in_idx = 0; packed_in_idx < n_packed_in_feature; packed_in_idx++) {
            // weight plaintext 按 diagonal d = g*bs + b 顺序排列
            for (uint32_t d = 0; d < pack; d++) {
                uint32_t g = d / bsgs_bs;
                uint32_t b = d % bsgs_bs;

                vector<double> w;
                for (uint32_t j = 0; j < pack; j++) {
                    // BSGS: giant-step rotation 后 slot i 拿到 slot (i+g*bs)%pack
                    // 因此 weight 在 slot j 需要对应 output channel (j - g*bs + pack) % pack
                    uint32_t out_local = (j - g * bsgs_bs + pack) % pack;
                    uint32_t in_local = (j + b) % pack;
                    uint32_t out_ch = packed_out_idx * pack + out_local;
                    uint32_t in_ch = packed_in_idx * pack + in_local;
                    if (in_ch < n_in_feature && out_ch < n_out_feature) {
                        w.push_back(weight.get(out_ch, in_ch));
                    } else {
                        w.push_back(0.0);
                    }
                    w.insert(w.end(), skip_0d - 1, 0.0);
                }
                auto w_pt = ctx.encode_ringt(w, modified_scale);
                a1.push_back(move(w_pt));
            }
        }
        weight_pt.push_back(move(a1));

        vector<double> bv;
        for (uint32_t j = 0; j < pack; j++) {
            uint32_t out_ch = packed_out_idx * pack + j;
            if (out_ch < n_out_feature) {
                bv.push_back(bias[out_ch]);
            } else {
                bv.push_back(0.0);
            }
            bv.insert(bv.end(), skip_0d - 1, 0.0);
        }
        auto b_pt = ctx.encode_ringt(bv, bias_scale);
        bias_pt.push_back(move(b_pt));
    }
}

void DensePackedLayer::prepare_weight_skip_0d_lazy(uint32_t skip_0d) {
    skip_0d_val = skip_0d;
    bsgs_bs = (uint32_t)ceil(sqrt((double)pack));
    bsgs_gs = div_ceil(pack, bsgs_bs);

    if (!normal_dense) {
        modified_scale = modified_scale * ENC_TO_SHARE_SCALE / param.get_default_scale();
    }
}

CkksPlaintextRingt DensePackedLayer::generate_weight_0d_pt_for_indices(CkksContext& ctx,
                                                                       uint32_t packed_out_idx,
                                                                       uint32_t weight_idx) const {
    uint32_t packed_in_idx = weight_idx / pack;
    uint32_t d = weight_idx % pack;
    uint32_t g = d / bsgs_bs;
    uint32_t b = d % bsgs_bs;

    vector<double> w;
    for (uint32_t j = 0; j < pack; j++) {
        uint32_t out_local = (j - g * bsgs_bs + pack) % pack;
        uint32_t in_local = (j + b) % pack;
        uint32_t out_ch = packed_out_idx * pack + out_local;
        uint32_t in_ch = packed_in_idx * pack + in_local;
        if (in_ch < n_in_feature && out_ch < n_out_feature) {
            w.push_back(weight.get(out_ch, in_ch));
        } else {
            w.push_back(0.0);
        }
        w.insert(w.end(), skip_0d_val - 1, 0.0);
    }
    return ctx.encode_ringt(w, modified_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_bias_0d_pt_for_index(CkksContext& ctx, uint32_t packed_out_idx) const {
    double bias_scale = normal_dense ? param.get_default_scale() : ENC_TO_SHARE_SCALE;

    vector<double> bv;
    for (uint32_t j = 0; j < pack; j++) {
        uint32_t out_ch = packed_out_idx * pack + j;
        if (out_ch < n_out_feature) {
            bv.push_back(bias[out_ch]);
        } else {
            bv.push_back(0.0);
        }
        bv.insert(bv.end(), skip_0d_val - 1, 0.0);
    }
    return ctx.encode_ringt(bv, bias_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_weight1_pt_for_indices(CkksContext& ctx,
                                                                     int packed_out_feature_idx,
                                                                     int in_feature_idx) const {
    int total_per_packed_in = pack;
    if (total_per_packed_in == 0) {
        throw std::runtime_error("pack is 0 in generate_weight1_pt_for_indices!");
    }
    int packed_in_feature_idx = in_feature_idx / total_per_packed_in;
    int rotate_idx = in_feature_idx % total_per_packed_in;

    vector<double> w;
    for (int pack_idx = 0; pack_idx < pack; pack_idx++) {
        int out_feature_idx = packed_out_feature_idx * pack + pack_idx;
        int in_feat_idx = packed_in_feature_idx * pack + (rotate_idx + pack_idx + pack) % pack;
        if (in_feat_idx < n_in_feature && out_feature_idx < n_out_feature) {
            int start = in_feat_idx * cached_per_channel_num;
            int end = (in_feat_idx + 1) * cached_per_channel_num;
            int T = 0;
            for (int k = 0; k < cached_input_shape_ct_1[0]; k++) {
                for (int m = 0; m < cached_input_shape_ct_1[1]; m++) {
                    if (k % skip[0] == 0 && m % skip[1] == 0 && start + T < n_in_feature) {
                        int out = start + T;
                        w.push_back(weight.get(out_feature_idx, out));
                        T += 1;
                    } else {
                        w.push_back(0);
                    }
                }
            }
        } else {
            w.insert(w.end(), cached_input_shape_ct_1[0] * cached_input_shape_ct_1[1], 0);
        }
    }
    return ctx.encode_ringt(w, modified_scale);
}

CkksPlaintextRingt DensePackedLayer::generate_bias1_pt_for_index(CkksContext& ctx, int packed_out_feature_idx) const {
    vector<double> b;
    for (int pack_idx = 0; pack_idx < pack; pack_idx++) {
        int out_feature_idx = packed_out_feature_idx * pack + pack_idx;
        if (out_feature_idx >= n_out_feature) {
            break;
        }
        for (int k = 0; k < cached_input_shape_ct_1[0] * cached_input_shape_ct_1[1]; k++) {
            if (k == 0) {
                b.push_back(bias[out_feature_idx] * 1);
            } else {
                b.push_back(0);
            }
        }
    }
    return ctx.encode_ringt(b, ctx.get_parameter().get_default_scale());
}

void DensePackedLayer::prepare_weight_for_multiplexed_lazy(const Duo& input_shape_in,
                                                           const Duo& skip_in,
                                                           const Duo& invalid_fill_in) {
    input_shape[0] = input_shape_in[0];
    input_shape[1] = input_shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];
    cached_invalid_fill[0] = invalid_fill_in[0];
    cached_invalid_fill[1] = invalid_fill_in[1];
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    cached_input_shape_ct_mult[0] = input_shape[0] * skip[0];
    cached_input_shape_ct_mult[1] = input_shape[1] * skip[1];
    cached_N_half = ctx.get_parameter().get_n() / 2;
    cached_n_num_pre_ct = div_ceil(cached_N_half, cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
    int valid_skip_0 = skip[0] / invalid_fill_in[0];
    int valid_skip_1 = skip[1] / invalid_fill_in[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel = n_in_feature / (input_shape[0] * input_shape[1]);
    cached_n_block_input = div_ceil(n_channel, cached_n_num_pre_ct * n_channel_per_block) * cached_n_num_pre_ct;
}

CkksPlaintextRingt DensePackedLayer::generate_weight_pt_mult_pack_for_indices(CkksContext& ctx,
                                                                              int packed_out_feature_idx,
                                                                              int n_block_input_idx) const {
    int valid_skip_0 = skip[0] / cached_invalid_fill[0];
    int valid_skip_1 = skip[1] / cached_invalid_fill[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel_per_block_col = valid_skip_1;
    int spatial_size = input_shape[0] * input_shape[1];

    vector<double> w(cached_N_half, 0);
    for (int i = 0; i < cached_N_half; i++) {
        int block_i = packed_out_feature_idx * cached_n_num_pre_ct +
                      i / (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_linear = i % (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_i = shape_linear / cached_input_shape_ct_mult[1];
        int shape_j = shape_linear % cached_input_shape_ct_mult[1];
        int cx = shape_i % skip[0];
        int cy = shape_j % skip[1];
        int x = shape_i / skip[0];
        int y = shape_j / skip[1];
        if (cx < valid_skip_0 && cy < valid_skip_1 && x < (int)input_shape[0] && y < (int)input_shape[1] &&
            block_i < n_out_feature) {
            int rotated_block =
                ((n_block_input_idx + i / (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]) +
                  cached_n_num_pre_ct) %
                     cached_n_num_pre_ct +
                 int(n_block_input_idx / cached_n_num_pre_ct) * cached_n_num_pre_ct);
            int in_ch = rotated_block * n_channel_per_block + cx * n_channel_per_block_col + cy;
            int line_i = in_ch * spatial_size + x * input_shape[1] + y;
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
    vector<double> b(cached_N_half, 0);
    for (int i = 0; i < cached_N_half; i++) {
        int block_i = packed_out_feature_idx * cached_n_num_pre_ct +
                      i / (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_linear = i % (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_i = shape_linear / cached_input_shape_ct_mult[1];
        int shape_j = shape_linear % cached_input_shape_ct_mult[1];
        if (shape_i == 0 && shape_j == 0 && block_i < n_out_feature) {
            b[i] = bias.get(block_i);
        }
    }
    return ctx.encode_ringt(b, ctx.get_parameter().get_default_scale());
}

void DensePackedLayer::prepare_weight_for_multiplexed(const Duo& input_shape_in,
                                                      const Duo& skip_in,
                                                      const Duo& invalid_fill_in) {
    input_shape[0] = input_shape_in[0];
    input_shape[1] = input_shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];
    cached_invalid_fill[0] = invalid_fill_in[0];
    cached_invalid_fill[1] = invalid_fill_in[1];
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    int N_half = ctx.get_parameter().get_n() / 2;
    int n_num_pre_ct = div_ceil(N_half, input_shape_ct[0] * input_shape_ct[1]);

    // ParMultiplexedPack: valid channels per mini-block
    int valid_skip_0 = skip[0] / invalid_fill_in[0];
    int valid_skip_1 = skip[1] / invalid_fill_in[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel_per_block_col = valid_skip_1;
    int n_channel = n_in_feature / (input_shape[0] * input_shape[1]);
    int spatial_size = input_shape[0] * input_shape[1];

    int n_packed_out_feature_for_mult_apck = div_ceil(n_out_feature, n_num_pre_ct);
    weight_pt.resize(n_packed_out_feature_for_mult_apck);
    bias_pt.resize(n_packed_out_feature_for_mult_apck);
    int n_block_input = div_ceil(n_channel, n_num_pre_ct * n_channel_per_block) * n_num_pre_ct;

    parallel_for(
        n_packed_out_feature_for_mult_apck, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
            weight_pt[packed_out_feature_idx].resize(n_block_input);

            // Encode bias once (independent of n_block_input_idx)
            vector<double> b(N_half, 0);
            for (int i = 0; i < N_half; i++) {
                int block_i = packed_out_feature_idx * n_num_pre_ct + i / (input_shape_ct[0] * input_shape_ct[1]);
                int shape_linear = i % (input_shape_ct[0] * input_shape_ct[1]);
                int shape_i = shape_linear / input_shape_ct[1];
                int shape_j = shape_linear % input_shape_ct[1];
                if (shape_i == 0 && shape_j == 0 && block_i < n_out_feature) {
                    b[i] = bias.get(block_i);
                }
            }
            bias_pt[packed_out_feature_idx] = ctx_copy.encode_ringt(b, param.get_default_scale());

            for (int n_block_input_idx = 0; n_block_input_idx < n_block_input; n_block_input_idx++) {
                vector<double> w(N_half, 0);
                for (int i = 0; i < N_half; i++) {
                    int block_i = packed_out_feature_idx * n_num_pre_ct + i / (input_shape_ct[0] * input_shape_ct[1]);
                    int shape_linear = i % (input_shape_ct[0] * input_shape_ct[1]);
                    int shape_i = shape_linear / input_shape_ct[1];
                    int shape_j = shape_linear % input_shape_ct[1];
                    int cx = shape_i % skip[0];
                    int cy = shape_j % skip[1];
                    int x = shape_i / skip[0];
                    int y = shape_j / skip[1];
                    if (cx < valid_skip_0 && cy < valid_skip_1 && x < (int)input_shape[0] && y < (int)input_shape[1] &&
                        block_i < n_out_feature) {
                        int local_block = i / (input_shape_ct[0] * input_shape_ct[1]);
                        int group = n_block_input_idx / n_num_pre_ct;
                        int offset = n_block_input_idx % n_num_pre_ct;
                        int rotated_block = (offset + local_block) % n_num_pre_ct + group * n_num_pre_ct;
                        int in_ch = rotated_block * n_channel_per_block + cx * n_channel_per_block_col + cy;
                        int line_i = in_ch * spatial_size + x * input_shape[1] + y;
                        if (line_i >= n_in_feature || block_i > n_out_feature) {
                            w[i] = 0;
                        } else {
                            w[i] = weight.get(block_i, line_i);
                        }
                    }
                }
                weight_pt[packed_out_feature_idx][n_block_input_idx] =
                    ctx_copy.encode_ringt(w, param.get_default_scale());
            }
        });
}

vector<CkksCiphertext> DensePackedLayer::run_core_mult_pack(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> input_rotated_x;
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    uint32_t x_size = x.size();
    int N_half = ctx.get_parameter().get_n() / 2;
    int n_num_pre_ct = div_ceil(N_half, input_shape_ct[0] * input_shape_ct[1]);
    int valid_skip_0 = skip[0] / cached_invalid_fill[0];
    int valid_skip_1 = skip[1] / cached_invalid_fill[1];
    int n_channel_per_block = valid_skip_0 * valid_skip_1;
    int n_channel = n_in_feature / (input_shape[0] * input_shape[1]);
    int n_block_input = div_ceil(n_channel, n_num_pre_ct * n_channel_per_block) * n_num_pre_ct;
    int n_packed_out_feature_for_mult_pack = div_ceil(n_out_feature, n_num_pre_ct);
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] = Conv2DLayer::populate_rotations_1_side(ctx_copy, x[x_id], n_block_input - 1,
                                                                   input_shape_ct[0] * input_shape_ct[1]);
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    vector<CkksCiphertext> result;
    result.resize(n_packed_out_feature_for_mult_pack);

    parallel_for(
        n_packed_out_feature_for_mult_pack, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
            CkksCiphertext s(0);
            int num_inputs = weight_pt.empty() ? cached_n_block_input : weight_pt[packed_out_feature_idx].size();
            for (int in_feature_idx = 0; in_feature_idx < num_inputs; in_feature_idx++) {
                auto& x_ct = input_rotated_x[in_feature_idx];

                if (weight_pt.empty()) {
                    auto w_pt_rt =
                        generate_weight_pt_mult_pack_for_indices(ctx_copy, packed_out_feature_idx, in_feature_idx);
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                    auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
                    if (in_feature_idx == 0) {
                        s = move(p);
                    } else {
                        s = ctx_copy.add(s, p);
                    }
                } else {
                    auto& w_pt_rt = weight_pt[packed_out_feature_idx][in_feature_idx];
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                    auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
                    if (in_feature_idx == 0) {
                        s = move(p);
                    } else {
                        s = ctx_copy.add(s, p);
                    }
                }
            }
            s = move(ctx_copy.rescale(s, ctx_copy.get_parameter().get_default_scale()));

            if (bias_pt.empty()) {
                auto b_pt = generate_bias_pt_mult_pack_for_index(ctx_copy, packed_out_feature_idx);
                s = ctx_copy.add_plain_ringt(s, b_pt);
            } else {
                auto& b_pt = bias_pt[packed_out_feature_idx];
                s = ctx_copy.add_plain_ringt(s, b_pt);
            }

            uint32_t n_term = input_shape_ct[0] * input_shape_ct[1];
            while (n_term > 1) {
                CkksCiphertext rotated = ctx_copy.rotate(s, n_term / 2);
                s = ctx_copy.add(s, rotated);
                n_term /= 2;
            }
            result[packed_out_feature_idx] = move(s);
        });
    return result;
}

Feature0DEncrypted DensePackedLayer::run_multiplexed(CkksContext& ctx, const Feature0DEncrypted& x) {
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
        baby_rots[ct_id] = Conv2DLayer::populate_rotations_1_side(ctx_copy, x[ct_id], bsgs_bs - 1, skip_0d_val);
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
                uint32_t b_end = std::min(bsgs_bs, pack - g * bsgs_bs);

                for (uint32_t b = 0; b < b_end; b++) {
                    uint32_t d = g * bsgs_bs + b;
                    uint32_t weight_idx = ct_in * pack + d;

                    CkksCiphertext p(0);
                    if (weight_pt.empty()) {
                        auto w_pt_rt = generate_weight_0d_pt_for_indices(ctx_copy, out_idx, weight_idx);
                        auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                        p = ctx_copy.mult_plain_mul(baby_rots[ct_in][b], w_pt);
                    } else {
                        auto& w_pt_rt = weight_pt[out_idx][weight_idx];
                        auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level_);
                        p = ctx_copy.mult_plain_mul(baby_rots[ct_in][b], w_pt);
                    }

                    if (!inner_init) {
                        inner = move(p);
                        inner_init = true;
                    } else {
                        inner = ctx_copy.add(inner, p);
                    }
                }

                // Giant-step rotation (g=0 不需要旋转)
                if (g > 0) {
                    inner = ctx_copy.rotate(inner, g * bsgs_bs * skip_0d_val);
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

        if (bias_pt.empty()) {
            auto b_pt = generate_bias_0d_pt_for_index(ctx_copy, out_idx);
            total = ctx_copy.add_plain_ringt(total, b_pt);
        } else {
            auto& b_pt = bias_pt[out_idx];
            total = ctx_copy.add_plain_ringt(total, b_pt);
        }

        result[out_idx] = move(total);
    });
    return result;
}

Feature0DEncrypted DensePackedLayer::run_skip_0d(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(run_core_0d(ctx, x.data));
    result.skip = x.skip;
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = x.n_channel_per_ct;
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
