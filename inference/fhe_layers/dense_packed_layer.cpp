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
                                   const Duo& input_shape_in,
                                   const Duo& skip_in,
                                   const Array<double, 2>& weight_in,
                                   const Array<double, 1>& bias_in,
                                   uint32_t pack_in,
                                   uint32_t level_in,
                                   int mark_in,
                                   double residual_scale)
    : param(param_in.copy()) {
    input_shape[0] = input_shape_in[0];
    input_shape[1] = input_shape_in[1];
    skip[0] = skip_in[0];
    skip[1] = skip_in[1];

    if ((input_shape[0] & (input_shape[0] - 1)) != 0 || (input_shape[1] & (input_shape[1] - 1)) != 0) {
        throw std::invalid_argument("input_shape must be powers of 2, got: [" + std::to_string(input_shape[0]) + ", " +
                                    std::to_string(input_shape[1]) + "]");
    }
    if ((skip[0] & (skip[0] - 1)) != 0 || (skip[1] & (skip[1] - 1)) != 0) {
        throw std::invalid_argument("skip must be powers of 2, got: [" + std::to_string(skip[0]) + ", " +
                                    std::to_string(skip[1]) + "]");
    }

    auto weight_shape = weight_in.get_shape();
    n_out_feature = weight_shape[0];
    n_in_feature = weight_shape[1];
    weight = weight_in.copy();
    bias = bias_in.copy();
    pack = pack_in;
    n_packed_in_feature = div_ceil(n_in_feature, pack);
    n_packed_out_feature = div_ceil(n_out_feature, pack);
    level = level_in;
    mark = mark_in;
    modified_scale = param.get_q(level) * residual_scale;
}

DensePackedLayer::~DensePackedLayer() {}

void DensePackedLayer::prepare_weight1() {
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    weight_pt.clear();
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    int per_channel_num = (input_shape_ct[0] / skip[0]) * (input_shape_ct[1] / skip[1]);
    double encode_pt_scale = modified_scale;
    double bias_scale = param.get_default_scale();

    for (int packed_out_feature_idx = 0; packed_out_feature_idx < n_packed_out_feature; packed_out_feature_idx++) {
        vector<CkksPlaintextRingt> a1;
        for (int packed_in_feature_idx = 0; packed_in_feature_idx < div_ceil(n_packed_in_feature, per_channel_num);
             packed_in_feature_idx++) {
            for (int rotate_idx = 0; rotate_idx < pack; rotate_idx++) {
                vector<double> w;
                for (int pack_idx = 0; pack_idx < pack; pack_idx++) {
                    int out_feature_idx = packed_out_feature_idx * pack + pack_idx;
                    int in_feature_idx = packed_in_feature_idx * pack + (rotate_idx + pack_idx + pack) % pack;
                    if (in_feature_idx < n_in_feature && out_feature_idx < n_out_feature) {
                        int start = in_feature_idx * per_channel_num;
                        int end = (in_feature_idx + 1) * per_channel_num;
                        int T = 0;
                        for (int k = 0; k < input_shape_ct[0]; k++) {
                            for (int m = 0; m < input_shape_ct[1]; m++) {
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
                        w.insert(w.end(), input_shape_ct[0] * input_shape_ct[1], 0);
                    }
                }
                auto w_pt = ctx.encode_ringt(w, encode_pt_scale);
                a1.push_back(move(w_pt));
            }
        }
        weight_pt.push_back(move(a1));

        vector<double> b;
        for (int pack_idx = 0; pack_idx < pack; pack_idx++) {
            int out_feature_idx = packed_out_feature_idx * pack + pack_idx;
            if (out_feature_idx >= n_out_feature) {
                break;
            }
            for (int k = 0; k < input_shape_ct[0] * input_shape_ct[1]; k++) {
                if (k == 0) {
                    b.push_back(bias[out_feature_idx] * 1);
                } else {
                    b.push_back(0);
                }
            }
        }
        auto b_pt = ctx.encode_ringt(b, bias_scale);
        bias_pt.push_back(move(b_pt));
    }
}

void DensePackedLayer::prepare_weight1_lazy() {
    cached_input_shape_ct_1[0] = input_shape[0] * skip[0];
    cached_input_shape_ct_1[1] = input_shape[1] * skip[1];
    cached_per_channel_num = (cached_input_shape_ct_1[0] / skip[0]) * (cached_input_shape_ct_1[1] / skip[1]);
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

void DensePackedLayer::prepare_weight_for_mult_pack_lazy() {
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    cached_input_shape_ct_mult[0] = input_shape[0] * skip[0];
    cached_input_shape_ct_mult[1] = input_shape[1] * skip[1];
    cached_N_half = ctx.get_parameter().get_n() / 2;
    cached_n_num_pre_ct = div_ceil(cached_N_half, cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
    cached_n_block_input =
        div_ceil(n_in_feature * input_shape[0] * input_shape[1], cached_N_half) * cached_n_num_pre_ct;
}

CkksPlaintextRingt DensePackedLayer::generate_weight_pt_mult_pack_for_indices(CkksContext& ctx,
                                                                              int packed_out_feature_idx,
                                                                              int n_block_input_idx) const {
    vector<double> w(cached_N_half, 0);
    for (int i = 0; i < cached_N_half; i++) {
        int block_i = packed_out_feature_idx * cached_n_num_pre_ct +
                      i / (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_linear = i % (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]);
        int shape_i = shape_linear / cached_input_shape_ct_mult[1];
        int shape_j = shape_linear % cached_input_shape_ct_mult[1];
        if (shape_i < skip[0] && shape_j < skip[1] && block_i < n_out_feature) {
            int line_i = ((n_block_input_idx + i / (cached_input_shape_ct_mult[0] * cached_input_shape_ct_mult[1]) +
                           cached_n_num_pre_ct) %
                              cached_n_num_pre_ct +
                          int(n_block_input_idx / cached_n_num_pre_ct) * cached_n_num_pre_ct) *
                             (skip[0] * skip[1]) +
                         shape_i * skip[1] + shape_j;
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
        if (shape_i < skip[0] && shape_j < skip[1] && block_i < n_out_feature) {
            if (shape_i == 0 && shape_j == 0) {
                b[i] = bias.get(block_i);
            }
        }
    }
    return ctx.encode_ringt(b, ctx.get_parameter().get_default_scale());
}

void DensePackedLayer::prepare_weight_for_mult_pack() {
    CkksContext ctx = CkksContext::create_empty_context(this->param);
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    int n_num_pre_ct = div_ceil(ctx.get_parameter().get_n() / 2, input_shape_ct[0] * input_shape_ct[1]);
    int n_packed_out_feature_for_mult_apck = div_ceil(n_out_feature, n_num_pre_ct);
    weight_pt.resize(n_packed_out_feature_for_mult_apck);
    bias_pt.resize(n_packed_out_feature_for_mult_apck);
    int n_block_input =
        div_ceil(n_in_feature * input_shape[0] * input_shape[1], ctx.get_parameter().get_n() / 2) * n_num_pre_ct;

    parallel_for(
        n_packed_out_feature_for_mult_apck, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
            weight_pt[packed_out_feature_idx].resize(n_block_input);
            for (int n_block_input_idx = 0; n_block_input_idx < n_block_input; n_block_input_idx++) {
                vector<double> w(ctx_copy.get_parameter().get_n() / 2, 0);
                vector<double> b(ctx_copy.get_parameter().get_n() / 2, 0);
                for (int i = 0; i < ctx_copy.get_parameter().get_n() / 2; i++) {
                    int block_i = packed_out_feature_idx * n_num_pre_ct + i / (input_shape_ct[0] * input_shape_ct[1]);
                    int shape_linear = i % (input_shape_ct[0] * input_shape_ct[1]);
                    int shape_i = shape_linear / input_shape_ct[1];
                    int shape_j = shape_linear % input_shape_ct[1];
                    if (shape_i < skip[0] && shape_j < skip[1] && block_i < n_out_feature) {
                        if (shape_i == 0 && shape_j == 0) {
                            b[i] = bias.get(block_i);
                        }
                        int line_i = ((n_block_input_idx + i / (input_shape_ct[0] * input_shape_ct[1]) + n_num_pre_ct) %
                                          n_num_pre_ct +
                                      int(n_block_input_idx / n_num_pre_ct) * n_num_pre_ct) *
                                         (skip[0] * skip[1]) +
                                     shape_i * skip[1] + shape_j;
                        if (line_i >= n_in_feature or block_i > n_out_feature) {
                            w[i] = 0;
                        } else {
                            w[i] = weight.get(block_i, line_i);
                        }
                    }
                }
                weight_pt[packed_out_feature_idx][n_block_input_idx] =
                    ctx_copy.encode_ringt(w, param.get_default_scale());
                bias_pt[packed_out_feature_idx] = ctx_copy.encode_ringt(b, param.get_default_scale());
            }
        });
}

vector<CkksCiphertext> DensePackedLayer::call(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_diff;
    time_start = chrono::high_resolution_clock::now();
    vector<CkksCiphertext> input_rotated_x;
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    uint32_t x_size = x.size();
    vector<vector<CkksCiphertext>> rotated_tmp(x_size);
    parallel_for(x_size, th_nums, ctx, [&](CkksContext& ctx_copy, int x_id) {
        rotated_tmp[x_id] =
            Conv2DLayer::populate_rotations_1_side(ctx_copy, x[x_id], pack - 1, input_shape_ct[0] * input_shape_ct[1]);
    });
    for (auto& y : rotated_tmp) {
        move(y.begin(), y.end(), back_inserter(input_rotated_x));
    }

    vector<CkksCiphertext> result;
    result.resize(n_packed_out_feature);

    parallel_for(n_packed_out_feature, th_nums, ctx, [&](CkksContext& ctx_copy, int packed_out_feature_idx) {
        CkksCiphertext s(0);
        for (int in_feature_idx = 0; in_feature_idx < input_rotated_x.size(); in_feature_idx++) {
            auto& x_ct = input_rotated_x[in_feature_idx];

            if (weight_pt.empty()) {
                auto w_pt_rt = generate_weight1_pt_for_indices(ctx_copy, packed_out_feature_idx, in_feature_idx);
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
                if (in_feature_idx == 0) {
                    s = move(p);
                } else {
                    s = ctx_copy.add(s, p);
                }
            } else {
                auto& w_pt_rt = weight_pt[packed_out_feature_idx][in_feature_idx];
                auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
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
            auto b_pt = generate_bias1_pt_for_index(ctx_copy, packed_out_feature_idx);
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
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    return result;
}

Feature0DEncrypted DensePackedLayer::call(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(call(ctx, x.data));
    result.skip = (x.shape[0] * x.skip[0]) * (x.shape[1] * x.skip[1]);
    result.pack_type = 0;
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = x.level - 1;

    return result;
}

vector<CkksCiphertext> DensePackedLayer::run_core_mult_pack(CkksContext& ctx, const vector<CkksCiphertext>& x) {
    vector<CkksCiphertext> input_rotated_x;
    uint32_t input_shape_ct[2];
    input_shape_ct[0] = input_shape[0] * skip[0];
    input_shape_ct[1] = input_shape[1] * skip[1];
    uint32_t x_size = x.size();
    int n_pack = div_ceil(ctx.get_parameter().get_n() / 2, input_shape[0] * input_shape[1]);
    int n_block_input = div_ceil(n_pack, skip[0] * skip[1]);
    int n_num_pre_ct = div_ceil(ctx.get_parameter().get_n() / 2, input_shape_ct[0] * input_shape_ct[1]);
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
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
                    auto p = ctx_copy.mult_plain_mul(x_ct, w_pt);
                    if (in_feature_idx == 0) {
                        s = move(p);
                    } else {
                        s = ctx_copy.add(s, p);
                    }
                } else {
                    auto& w_pt_rt = weight_pt[packed_out_feature_idx][in_feature_idx];
                    auto w_pt = ctx_copy.ringt_to_mul(w_pt_rt, level);
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

Feature0DEncrypted DensePackedLayer::run_mult_park(CkksContext& ctx, const Feature2DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(run_core_mult_pack(ctx, x.data));
    result.skip = (x.shape[0] * x.skip[0]) * (x.shape[1] * x.skip[1]);
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = x.level - 1;
    return result;
}

Feature0DEncrypted DensePackedLayer::run_mult_park(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(run_core_mult_pack(ctx, x.data));
    result.skip = x.skip;
    result.n_channel = n_out_feature;
    result.dim = x.dim;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = x.level - 1;
    return result;
}

Feature0DEncrypted DensePackedLayer::call(CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(x.context, x.level);
    result.data = move(call(ctx, x.data));
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
