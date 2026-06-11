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

#include "compute_distance_layer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

using namespace std;
using namespace lattisense;

namespace {

double dot_product(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("dot_product inputs must have the same size");
    }
    double ret = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

double l2_norm_squared(const vector<double>& x) {
    return dot_product(x, x);
}

vector<double> single_slot(double value) {
    return {value};
}

vector<double> scale_vector(const vector<double>& x, double multiplier) {
    vector<double> ret(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ret[i] = multiplier * x[i];
    }
    return ret;
}

CkksCiphertext sum_slots(CkksContext& ctx, CkksCiphertext ct, int dim) {
    if (dim <= 0) {
        throw invalid_argument("dim must be positive");
    }
    for (int step = 1; step < dim; step <<= 1) {
        auto rotated = ctx.rotate(ct, step);
        ct = ctx.add(ct, rotated);
    }
    return ct;
}

CkksCiphertext multiply_and_rescale(CkksContext& ctx, const CkksCiphertext& lhs, const CkksCiphertext& rhs, double scale) {
    auto product = ctx.mult(lhs, rhs);
    auto relined = ctx.relinearize(product);
    return ctx.rescale(relined, scale);
}

CkksCiphertext multiply_plain_and_rescale(CkksContext& ctx,
                                          const CkksCiphertext& ct,
                                          const CkksPlaintext& pt,
                                          double scale) {
    auto product = ctx.mult_plain(ct, pt);
    return ctx.rescale(product, scale);
}

CkksCiphertext multiply_plain_and_rescale(CkksContext& ctx,
                                          const CkksCiphertext& ct,
                                          const vector<double>& values,
                                          double scale) {
    auto pt = ctx.encode(values, ct.get_level(), scale);
    return multiply_plain_and_rescale(ctx, ct, pt, scale);
}

CkksCiphertext add_plain_scalar(CkksContext& ctx, const CkksCiphertext& ct, double value, double scale) {
    auto pt = ctx.encode(single_slot(value), ct.get_level(), scale);
    return ctx.add_plain(ct, pt);
}

CkksCiphertext drop_to_level(CkksContext& ctx, const CkksCiphertext& ct, int target_level) {
    if (ct.get_level() < target_level) {
        throw invalid_argument("cannot drop ciphertext to a higher level");
    }
    const int levels = ct.get_level() - target_level;
    if (levels == 0) {
        return ct.copy();
    }
    return ctx.drop_level(ct, levels);
}

CkksCiphertext bootstrap_ciphertext(CkksBtpContext& ctx, const CkksCiphertext& ct, double scale) {
    auto input_scale = ct.get_scale();
    auto boot_ct = ct.copy();
    boot_ct.set_scale(scale);
    boot_ct = ctx.bootstrap(boot_ct);
    boot_ct.set_scale(input_scale);
    return boot_ct;
}

CkksCiphertext bootstrap_if_needed(CkksBtpContext& ctx, const CkksCiphertext& ct, double scale) {
    if (ct.get_level() - 3 > 0) {
        return ct.copy();
    }
    return bootstrap_ciphertext(ctx, ct, scale);
}

CkksCiphertext inverse_sqrt_one_iteration(CkksContext& ctx,
                                          const CkksCiphertext& a_ct,
                                          double a_min,
                                          double a_max,
                                          double scale) {
    const double x0 = 0.5 * (1.0 / sqrt(a_min) + 1.0 / sqrt(a_max));
    auto term = multiply_plain_and_rescale(ctx, a_ct, single_slot(-0.5 * x0 * x0 * x0), scale);
    return add_plain_scalar(ctx, term, 1.5 * x0, scale);
}

CkksCiphertext inverse_sqrt_next_iteration(CkksBtpContext& ctx,
                                           const CkksCiphertext& a_ct,
                                           const CkksCiphertext& x_ct,
                                           double scale) {
    auto x_work_ct = bootstrap_if_needed(ctx, x_ct, scale);
    if (x_work_ct.get_level() > a_ct.get_level()) {
        x_work_ct = drop_to_level(ctx, x_work_ct, a_ct.get_level());
    }
    auto x2_ct = multiply_and_rescale(ctx, x_work_ct, x_work_ct, scale);
    auto a_aligned_ct = drop_to_level(ctx, a_ct, x2_ct.get_level());
    auto ax2_ct = multiply_and_rescale(ctx, a_aligned_ct, x2_ct, scale);
    auto factor_ct = multiply_plain_and_rescale(ctx, ax2_ct, single_slot(-0.5), scale);
    factor_ct = add_plain_scalar(ctx, factor_ct, 1.5, scale);
    auto x_aligned_ct = drop_to_level(ctx, x_work_ct, factor_ct.get_level());
    return multiply_and_rescale(ctx, x_aligned_ct, factor_ct, scale);
}

CkksCiphertext inverse_sqrt_iterations(CkksBtpContext& ctx,
                                       const CkksCiphertext& a_ct,
                                       double a_min,
                                       double a_max,
                                       int iterations,
                                       double scale) {
    if (iterations < 1) {
        throw invalid_argument("nr_iterations must be at least 1");
    }
    auto x_ct = inverse_sqrt_one_iteration(ctx, a_ct, a_min, a_max, scale);
    for (int i = 1; i < iterations; ++i) {
        x_ct = inverse_sqrt_next_iteration(ctx, a_ct, x_ct, scale);
    }
    return x_ct;
}

}  // namespace

ComputeDistanceLayer::ComputeDistanceLayer(const CkksParameter& param_in,
                                           uint32_t dim,
                                           double norm2_min,
                                           double norm2_max,
                                           int nr_iterations)
    : Layer(param_in), dim_(dim), norm2_min_(norm2_min), norm2_max_(norm2_max), nr_iterations_(nr_iterations) {
    if (dim_ == 0) {
        throw invalid_argument("dim must be positive");
    }
    if (norm2_min_ <= 0.0) {
        throw invalid_argument("norm2_min must be positive");
    }
    if (norm2_max_ < norm2_min_) {
        throw invalid_argument("norm2_max must be greater than or equal to norm2_min");
    }
    if (nr_iterations_ < 1) {
        throw invalid_argument("nr_iterations must be at least 1");
    }
}

void ComputeDistanceLayer::prepare_weight(const vector<double>& gallery, uint32_t level) {
    if (gallery.size() != dim_) {
        throw invalid_argument("gallery size must match dim");
    }

    CkksContext ctx = CkksContext::create_empty_context(param_);
    const double scale = param_.get_default_scale();
    const auto scaled_gallery = scale_vector(gallery, -2.0);
    gallery_pt_ = ctx.encode(scaled_gallery, level, scale);
    level_ = level;
    has_gallery_pt_ = true;
}

Feature0DEncrypted ComputeDistanceLayer::run(CkksBtpContext& ctx, const Feature0DEncrypted& query) const {
    if (!has_gallery_pt_) {
        throw invalid_argument("gallery plaintext must be prepared before run");
    }
    if (query.data.empty()) {
        throw invalid_argument("query feature must contain at least one ciphertext");
    }
    if (query.n_channel < dim_) {
        throw invalid_argument("query feature channel count must be at least dim");
    }

    const double scale = param_.get_default_scale();
    const auto& query_ct = query.data[0];
    if (gallery_pt_.get_level() != query_ct.get_level()) {
        throw invalid_argument("prepared gallery level must match query ciphertext level");
    }

    auto dot2_ct = multiply_plain_and_rescale(ctx, query_ct, gallery_pt_, scale);
    dot2_ct = sum_slots(ctx, move(dot2_ct), dim_);

    auto norm2_ct = multiply_and_rescale(ctx, query_ct, query_ct, scale);
    norm2_ct = sum_slots(ctx, move(norm2_ct), dim_);

    auto rsqrt_ct = inverse_sqrt_iterations(ctx, norm2_ct, norm2_min_, norm2_max_, nr_iterations_, scale);
    auto dot2_aligned_ct = drop_to_level(ctx, dot2_ct, rsqrt_ct.get_level());
    auto cos2_ct = multiply_and_rescale(ctx, dot2_aligned_ct, rsqrt_ct, scale);

    auto dist2_ct = add_plain_scalar(ctx, cos2_ct, 2.0, scale);

    Feature0DEncrypted result(&ctx, dist2_ct.get_level());
    result.data.push_back(move(dist2_ct));
    result.dim = 0;
    result.n_channel = 1;
    result.n_channel_per_ct = 1;
    result.skip = 1;
    result.level = result.data[0].get_level();
    return result;
}

double ComputeDistanceLayer::run_plaintext(const vector<double>& query, const vector<double>& gallery) const {
    if (query.size() != dim_) {
        throw invalid_argument("query size must match dim");
    }
    if (gallery.size() != dim_) {
        throw invalid_argument("gallery size must match dim");
    }
    const double query_norm = sqrt(l2_norm_squared(query));
    if (query_norm <= 0.0) {
        throw invalid_argument("query norm must be positive");
    }
    return 2.0 - 2.0 * dot_product(query, gallery) / query_norm;
}
