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

#include "softmax_layer.h"
#include <cmath>
#include <numeric>

using namespace std;
using namespace cxx_sdk_v2;

SoftmaxLayer::SoftmaxLayer(const std::array<double, 2>& exp_range, int exp_degree,
                           const std::array<double, 2>& inv_range, int inv_degree,
                           int n_channel, int n_channel_per_ct, double temperature)
    : exp_range_(exp_range), exp_degree_(exp_degree),
      inv_range_(inv_range), inv_degree_(inv_degree),
      n_channel_(n_channel), n_channel_per_ct_(n_channel_per_ct),
      temperature_(temperature) {
    n_slots_ = std::min(n_channel, n_channel_per_ct);
}

SoftmaxLayer::~SoftmaxLayer() {}

ls::CkksCiphertext SoftmaxLayer::sum_slots(ls::CkksContext& ctx, const ls::CkksCiphertext& x) {
    int total_slots = ctx.get_parameter().get_n() / 2;

    vector<double> mask(total_slots, 0.0);
    for (int i = 0; i < n_slots_; ++i) {
        mask[i] = 1.0;
    }
    ls::CkksPlaintext pt_mask = ctx.encode(mask, x.get_level(), x.get_scale());
    ls::CkksCiphertext masked_x = ctx.mult_plain(x, pt_mask);
    masked_x = ctx.rescale(masked_x, x.get_scale());

    ls::CkksCiphertext res = masked_x.copy();
    for (int i = 1; i < total_slots; i <<= 1) {
        res = ctx.add(res, ctx.rotate(res, i));
    }
    return res;
}

static double exp_func(double v) { return std::exp(v); }
static double inv_func(double v) { return 1.0 / v; }

static int compute_poly_depth(int degree) {
    if (degree <= 1) return 0;
    return (int)std::ceil(std::log2(degree));
}

Feature0DEncrypted SoftmaxLayer::run(ls::CkksContext& ctx, const Feature0DEncrypted& x) {
    Feature0DEncrypted result(&ctx, x.level);
    result.data.resize(x.data.size());
    
    int exp_depth = compute_poly_depth(exp_degree_);
    int inv_depth = compute_poly_depth(inv_degree_);
    int total_depth = exp_depth + inv_depth + 2;
    if (temperature_ != 1.0) total_depth += 1;

    if (x.level < total_depth) {
        throw std::runtime_error("SoftmaxLayer: input level too low. Need at least " + to_string(total_depth));
    }

    // 1. Temperature scaling: x' = x / T
    parallel_for(x.data.size(), th_nums, ctx, [&](ls::CkksContext& ctx_copy, int i) {
        ls::CkksCiphertext input = x.data[i].copy();

        if (temperature_ != 1.0) {
            int total_slots = ctx_copy.get_parameter().get_n() / 2;
            vector<double> temp_values(total_slots, 1.0 / temperature_);
            ls::CkksPlaintext pt_temp = ctx_copy.encode(temp_values, input.get_level(), input.get_scale());
            input = ctx_copy.mult_plain(input, pt_temp);
            input = ctx_copy.rescale(input, input.get_scale());
        }

        // 2. Approximate exp(x)
        ls::CkksCiphertext exp_x = ctx_copy.poly_eval_function(
            exp_func,
            input, exp_range_[0], exp_range_[1], exp_degree_);
        
        // 2. Sum exp(x_j)
        ls::CkksCiphertext sum_exp = sum_slots(ctx_copy, exp_x);
        
        // 3. Approximate 1/sum
        ls::CkksCiphertext inv_sum = ctx_copy.poly_eval_function(
            inv_func,
            sum_exp, inv_range_[0], inv_range_[1], inv_degree_);
        
        // 4. Align levels before multiplication
        ls::CkksCiphertext exp_x_aligned = exp_x.copy();
        while (exp_x_aligned.get_level() > inv_sum.get_level()) {
            exp_x_aligned = ctx_copy.drop_level(exp_x_aligned, 1);
        }

        // 5. Final multiplication: exp(x_i) * (1/sum)
        ls::CkksCiphertext3 out3 = ctx_copy.mult(exp_x_aligned, inv_sum);
        ls::CkksCiphertext out = ctx_copy.relinearize(out3);
        if (out.get_level() > 0) {
            out = ctx_copy.rescale(out, ctx_copy.get_parameter().get_default_scale());
        }

        result.data[i] = move(out);
    });

    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.skip = x.skip;
    result.level = result.data.empty() ? 0 : result.data[0].get_level();
    return result;
}

Array<double, 1> SoftmaxLayer::run_plaintext(const Array<double, 1>& x) {
    Array<double, 1> res(x.get_shape());
    int size = x.get_size();
    
    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += std::exp(x[i] / temperature_);
    }

    for (int i = 0; i < size; ++i) {
        res.set(i, std::exp(x[i] / temperature_) / sum_exp);
    }

    return res;
}
