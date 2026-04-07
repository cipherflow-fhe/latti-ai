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

#include "poly_relu1d.h"
#include <cmath>

using namespace std;
using namespace lattisense;

// ======================== PolyRelu1D ========================

PolyRelu1D::PolyRelu1D(const CkksParameter& param_in,
                       const Array<double, 2>& weight_in,
                       uint32_t level_in,
                       int order_in,
                       int skip_in,
                       int shape_in)
    : PolyReluBase(param_in,
                   weight_in,
                   param_in.get_n() / 2 / (shape_in * skip_in),  // n_channel_per_ct for mode 1
                   level_in,
                   order_in),
      skip(skip_in), shape(shape_in), shape_with_skip(shape_in * skip_in),
      n_channel_per_ct_mux(param_in.get_n() / 2 / (shape_in * skip_in) * skip_in) {}

// ---- Mode 1: skip pack ----
//   channel ch (CT-local), position i → slot = ch * shape_with_skip + i * skip
//   Weight value is broadcast to all `shape` positions of channel ch.

CkksPlaintextRingt PolyRelu1D::generate_weight_pt_skip1d(CkksContext& ctx, int idx, int ct_idx) const {
    vector<double> buf(N / 2, 0.0);
    for (int ch = 0; ch < (int)n_channel_per_ct; ch++) {
        int channel_idx = ct_idx * n_channel_per_ct + ch;
        if (channel_idx >= cached_channel)
            continue;
        double w = weight.get(idx, channel_idx);
        int block_start = ch * shape_with_skip;
        for (int i = 0; i < shape; i++) {
            buf[block_start + i * skip] = w;
        }
    }
    double pack_scale = cached_bsgs_coeff_scale.at(idx);
    return ctx.encode_ringt(buf, pack_scale);
}

// ---- Mode 2: multiplexed/interleaved pack ----
//   n_channel_per_ct_mux = (N/2 / shape_with_skip) * skip
//   channel j (CT-local), position i → slot = (j/skip)*shape_with_skip + i*skip + (j%skip)
//   Weight value is broadcast to all `shape` positions of channel j.

CkksPlaintextRingt PolyRelu1D::generate_weight_pt_mux1d(CkksContext& ctx, int idx, int ct_idx) const {
    vector<double> buf(N / 2, 0.0);
    for (int j = 0; j < n_channel_per_ct_mux; j++) {
        int channel_idx = ct_idx * n_channel_per_ct_mux + j;
        if (channel_idx >= cached_channel)
            continue;
        double w = weight.get(idx, channel_idx);
        int group = j / skip;
        int sub_pos = j % skip;
        for (int i = 0; i < shape; i++) {
            buf[group * shape_with_skip + i * skip + sub_pos] = w;
        }
    }
    double pack_scale = cached_bsgs_coeff_scale.at(idx);
    return ctx.encode_ringt(buf, pack_scale);
}

CkksPlaintextRingt PolyRelu1D::generate_weight_pt_for_bsgs(CkksContext& ctx, int idx, int ct_idx) const {
    if (is_multiplexed)
        return generate_weight_pt_mux1d(ctx, idx, ct_idx);
    return generate_weight_pt_skip1d(ctx, idx, ct_idx);
}

// ---- prepare (eager) ----

void PolyRelu1D::prepare_weight_bsgs() {
    init_bsgs();
    is_multiplexed = false;

    int n_ct = div_ceil(cached_channel, (int)n_channel_per_ct);
    weight_pt.resize(order + 1);

    CkksContext ctx = CkksContext::create_empty_context(param_);
    ctx.resize_copies(order + 1);
    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        for (int ct_idx = 0; ct_idx < n_ct; ct_idx++) {
            weight_pt[idx].push_back(generate_weight_pt_skip1d(ctx_copy, idx, ct_idx));
        }
    });
}

void PolyRelu1D::prepare_weight_bsgs_mux() {
    init_bsgs();
    is_multiplexed = true;

    int n_ct = div_ceil(cached_channel, n_channel_per_ct_mux);
    weight_pt.resize(order + 1);

    CkksContext ctx = CkksContext::create_empty_context(param_);
    ctx.resize_copies(order + 1);
    parallel_for(order + 1, th_nums, ctx, [&](CkksContext& ctx_copy, int idx) {
        for (int ct_idx = 0; ct_idx < n_ct; ct_idx++) {
            weight_pt[idx].push_back(generate_weight_pt_mux1d(ctx_copy, idx, ct_idx));
        }
    });
}

// ---- prepare (lazy) ----

void PolyRelu1D::prepare_weight_bsgs_lazy() {
    init_bsgs();
    is_multiplexed = false;
    weight_pt.clear();
}

void PolyRelu1D::prepare_weight_bsgs_mux_lazy() {
    init_bsgs();
    is_multiplexed = true;
    weight_pt.clear();
}

// ---- run ----

Feature1DEncrypted PolyRelu1D::run(CkksContext& ctx, const Feature1DEncrypted& x) {
    Feature1DEncrypted result(&ctx, x.level);
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.shape = x.shape;
    result.skip = x.skip;
    result.data = run_core_bsgs(ctx, x.data);
    result.level = result.data[0].get_level();
    return result;
}

// Plaintext reference: P(x) = sum_k weight[k][ch] * x[ch][i]^k
Array<double, 2> PolyRelu1D::run_plaintext(const Array<double, 2>& x) {
    int n_ch = x.get_shape()[0];
    int n_pos = x.get_shape()[1];
    Array<double, 2> result({(uint64_t)n_ch, (uint64_t)n_pos});
    for (int ch = 0; ch < n_ch; ch++) {
        for (int i = 0; i < n_pos; i++) {
            double val = x.get(ch, i);
            double p = weight.get(0, ch);
            for (int k = 1; k <= order; k++) {
                p += weight.get(k, ch) * pow(val, k);
            }
            result.set(ch, i, p);
        }
    }
    return result;
}
