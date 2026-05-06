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
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using namespace cxx_sdk_v2;

namespace {
constexpr int kMinSoftmaxInputLevel = 13;
constexpr double kRecipPolyAnchorClasses = 8.0;
constexpr std::array<double, 6> kExpCoeffs = {
    1.0031377334916605,
    1.0026864218461349,
    0.4860498435309526,
    0.1624711376226941,
    0.05072464694309538,
    0.010053701974162384,
};
constexpr std::array<double, 4> kRecipCoeffs = {
    0.24885999074111392,
    -0.021622407621476325,
    0.0007824595670968044,
    -0.000010035965681343485,
};

const CkksPlaintextRingt& require_one(const vector<CkksPlaintextRingt>& values, const char* name) {
    if (values.size() != 1) {
        throw runtime_error(string("SoftmaxLayer requires exactly one offline arg for ") + name);
    }
    return values[0];
}

const CkksPlaintextMul& require_one(const vector<CkksPlaintextMul>& values, const char* name) {
    if (values.size() != 1) {
        throw runtime_error(string("SoftmaxLayer requires exactly one offline arg for ") + name);
    }
    return values[0];
}

const CkksPlaintext& require_one(const vector<CkksPlaintext>& values, const char* name) {
    if (values.size() != 1) {
        throw runtime_error(string("SoftmaxLayer requires exactly one offline arg for ") + name);
    }
    return values[0];
}

CkksCiphertext rescale_default(CkksContext& ctx, const CkksCiphertext& x) {
    return ctx.rescale(x, ctx.get_parameter().get_default_scale());
}

CkksCiphertext mult_ringt_rescale(CkksContext& ctx, const CkksCiphertext& x, const CkksPlaintextRingt& pt) {
    auto pt_mul = ctx.ringt_to_mul(pt, x.get_level());
    return rescale_default(ctx, ctx.mult_plain_mul(x, pt_mul));
}

CkksCiphertext mult_relin_rescale(CkksContext& ctx, const CkksCiphertext& x, const CkksCiphertext& y) {
    return rescale_default(ctx, ctx.relinearize(ctx.mult(x, y)));
}

CkksCiphertext drop_levels(CkksContext& ctx, const CkksCiphertext& x, int levels) {
    CkksCiphertext result = x.copy();
    for (int i = 0; i < levels; ++i) {
        result = ctx.drop_level(result);
    }
    return result;
}

CkksCiphertext repeated_block_sum(CkksContext& ctx, CkksCiphertext x, uint32_t n_classes) {
    CkksCiphertext total = move(x);
    for (uint32_t step = 1; step < n_classes; step <<= 1U) {
        auto rotated = ctx.advanced_rotate(total, static_cast<int32_t>(step));
        total = ctx.add(total, rotated);
    }
    return total;
}

CkksCiphertext eval_exp_poly_v1(CkksContext& ctx,
                                const CkksCiphertext& x,
                                const CkksPlaintextMul& c5,
                                const CkksPlaintext& c4,
                                const CkksPlaintext& c3,
                                const CkksPlaintext& c2,
                                const CkksPlaintext& c1,
                                const CkksPlaintext& c0) {
    auto x_lm1 = drop_levels(ctx, x, 1);
    auto x_lm2 = drop_levels(ctx, x, 2);
    auto x_lm3 = drop_levels(ctx, x, 3);
    auto x_lm4 = drop_levels(ctx, x, 4);

    auto acc = rescale_default(ctx, ctx.mult_plain_mul(x, c5));
    acc = ctx.add_plain(acc, c4);
    acc = mult_relin_rescale(ctx, acc, x_lm1);
    acc = ctx.add_plain(acc, c3);
    acc = mult_relin_rescale(ctx, acc, x_lm2);
    acc = ctx.add_plain(acc, c2);
    acc = mult_relin_rescale(ctx, acc, x_lm3);
    acc = ctx.add_plain(acc, c1);
    acc = mult_relin_rescale(ctx, acc, x_lm4);
    acc = ctx.add_plain(acc, c0);

    auto exp_half = mult_relin_rescale(ctx, acc, acc);
    return mult_relin_rescale(ctx, exp_half, exp_half);
}

CkksCiphertext eval_recip_poly_v1(CkksContext& ctx,
                                  const CkksCiphertext& x,
                                  const CkksPlaintextMul& c3,
                                  const CkksPlaintext& c2,
                                  const CkksPlaintext& c1,
                                  const CkksPlaintext& c0) {
    auto x_lm1 = drop_levels(ctx, x, 1);
    auto x_lm2 = drop_levels(ctx, x, 2);

    auto acc = rescale_default(ctx, ctx.mult_plain_mul(x, c3));
    acc = ctx.add_plain(acc, c2);
    acc = mult_relin_rescale(ctx, acc, x_lm1);
    acc = ctx.add_plain(acc, c1);
    acc = mult_relin_rescale(ctx, acc, x_lm2);
    return ctx.add_plain(acc, c0);
}
}  // namespace

SoftmaxLayer::SoftmaxLayer(const CkksParameter& param_in, uint32_t n_classes, uint32_t input_level) : Layer(param_in) {
    if (n_classes == 0 && input_level == 0) {
        return;
    }
    prepare_offline_args(n_classes, input_level);
}

void SoftmaxLayer::prepare_offline_args(uint32_t n_classes, uint32_t input_level) {
    if (n_classes == 0) {
        throw runtime_error("SoftmaxLayer::prepare_offline_args requires n_classes > 0");
    }
    if ((n_classes & (n_classes - 1)) != 0) {
        throw runtime_error("SoftmaxLayer currently requires power-of-two class count for rotation-sum.");
    }
    if (input_level < kMinSoftmaxInputLevel) {
        throw runtime_error("SoftmaxLayer requires input level >= 13 for current CKKS approximation.");
    }
    if (input_level > param_.get_max_level()) {
        throw runtime_error("SoftmaxLayer input level exceeds CKKS parameter max level.");
    }

    const int exp_c5_level = static_cast<int>(input_level) - 2;
    const int exp_c4_level = exp_c5_level - 1;
    const int exp_c3_level = exp_c5_level - 2;
    const int exp_c2_level = exp_c5_level - 3;
    const int exp_c1_level = exp_c5_level - 4;
    const int exp_c0_level = exp_c5_level - 5;

    const int recip_c3_level = static_cast<int>(input_level) - 9;
    const int recip_c2_level = recip_c3_level - 1;
    const int recip_c1_level = recip_c3_level - 2;
    const int recip_c0_level = recip_c3_level - 3;

    if (exp_c0_level < 0 || recip_c0_level < 0) {
        throw runtime_error("SoftmaxLayer level underflow when preparing offline args.");
    }

    CkksContext ctx = CkksContext::create_empty_context(param_);
    const int slot_count = static_cast<int>(param_.get_n() / 2);
    auto tile_scalar = [slot_count](double value) { return vector<double>(slot_count, value); };
    auto q = [&](int level) -> double { return static_cast<double>(param_.get_q(level)); };

    const double default_scale = param_.get_default_scale();

    const double scale_exp_1 = default_scale;
    const double scale_exp_2 = scale_exp_1 * default_scale / q(exp_c4_level);
    const double scale_exp_3 = scale_exp_2 * default_scale / q(exp_c3_level);
    const double scale_exp_4 = scale_exp_3 * default_scale / q(exp_c2_level);
    const double scale_exp_5 = scale_exp_4 * default_scale / q(exp_c1_level);
    const double scale_exp_half = scale_exp_5 * scale_exp_5 / q(exp_c0_level);
    const double scale_exp = scale_exp_half * scale_exp_half / q(exp_c0_level - 1);

    const double scale_recip_1 = default_scale;
    const double scale_recip_2 = scale_recip_1 * scale_exp / q(recip_c2_level);
    const double scale_recip_3 = scale_recip_2 * scale_exp / q(recip_c1_level);
    const double scale_recip_c3_mul = q(recip_c3_level) * scale_recip_1 / scale_exp;

    // The reciprocal polynomial is fitted around denominator magnitudes
    // observed for 8-way softmax. Keep higher/lower class counts in-range by
    // evaluating alpha * P(alpha * x), where alpha = 8 / n_classes.
    const double recip_domain_scale = kRecipPolyAnchorClasses / static_cast<double>(n_classes);
    const double recip_scale_pow2 = recip_domain_scale * recip_domain_scale;
    const double recip_scale_pow3 = recip_scale_pow2 * recip_domain_scale;
    const double recip_scale_pow4 = recip_scale_pow3 * recip_domain_scale;
    const std::array<double, 4> recip_coeffs_scaled = {
        kRecipCoeffs[0] * recip_domain_scale,
        kRecipCoeffs[1] * recip_scale_pow2,
        kRecipCoeffs[2] * recip_scale_pow3,
        kRecipCoeffs[3] * recip_scale_pow4,
    };

    pt_quarter.clear();
    pt_inv_classes.clear();
    exp_c5.clear();
    exp_c4.clear();
    exp_c3.clear();
    exp_c2.clear();
    exp_c1.clear();
    exp_c0.clear();
    recip_c3.clear();
    recip_c2.clear();
    recip_c1.clear();
    recip_c0.clear();

    pt_quarter.emplace_back(
        ctx.encode_ringt(tile_scalar(0.25), param_.get_q(static_cast<int>(input_level))));
    pt_inv_classes.emplace_back(
        ctx.encode_ringt(tile_scalar(1.0 / static_cast<double>(n_classes)),
                         param_.get_q(static_cast<int>(input_level) - 1)));

    exp_c5.emplace_back(ctx.encode_mul(tile_scalar(kExpCoeffs[5]), exp_c5_level, q(exp_c5_level)));
    exp_c4.emplace_back(ctx.encode(tile_scalar(kExpCoeffs[4]), exp_c4_level, scale_exp_1));
    exp_c3.emplace_back(ctx.encode(tile_scalar(kExpCoeffs[3]), exp_c3_level, scale_exp_2));
    exp_c2.emplace_back(ctx.encode(tile_scalar(kExpCoeffs[2]), exp_c2_level, scale_exp_3));
    exp_c1.emplace_back(ctx.encode(tile_scalar(kExpCoeffs[1]), exp_c1_level, scale_exp_4));
    exp_c0.emplace_back(ctx.encode(tile_scalar(kExpCoeffs[0]), exp_c0_level, scale_exp_5));

    recip_c3.emplace_back(ctx.encode_mul(tile_scalar(recip_coeffs_scaled[3]), recip_c3_level, scale_recip_c3_mul));
    recip_c2.emplace_back(ctx.encode(tile_scalar(recip_coeffs_scaled[2]), recip_c2_level, scale_recip_1));
    recip_c1.emplace_back(ctx.encode(tile_scalar(recip_coeffs_scaled[1]), recip_c1_level, scale_recip_2));
    recip_c0.emplace_back(ctx.encode(tile_scalar(recip_coeffs_scaled[0]), recip_c0_level, scale_recip_3));

    n_classes_ = n_classes;
    input_level_ = input_level;
}

vector<CkksCiphertext> SoftmaxLayer::run_core(CkksContext& ctx, const vector<CkksCiphertext>& x) const {
    if (x.size() != 1) {
        throw runtime_error("SoftmaxLayer::run_core currently supports exactly one input ciphertext.");
    }
    if (n_classes_ == 0 || input_level_ == 0) {
        throw runtime_error("SoftmaxLayer::run_core requires prepared offline args.");
    }
    if (static_cast<uint32_t>(x[0].get_level()) != input_level_) {
        throw runtime_error("SoftmaxLayer::run_core input level does not match prepared offline args.");
    }

    const auto& pt_quarter_arg = require_one(pt_quarter, "pt_quarter");
    const auto& pt_inv_classes_arg = require_one(pt_inv_classes, "pt_inv_classes");
    const auto& exp_c5_arg = require_one(exp_c5, "exp_c5");
    const auto& exp_c4_arg = require_one(exp_c4, "exp_c4");
    const auto& exp_c3_arg = require_one(exp_c3, "exp_c3");
    const auto& exp_c2_arg = require_one(exp_c2, "exp_c2");
    const auto& exp_c1_arg = require_one(exp_c1, "exp_c1");
    const auto& exp_c0_arg = require_one(exp_c0, "exp_c0");
    const auto& recip_c3_arg = require_one(recip_c3, "recip_c3");
    const auto& recip_c2_arg = require_one(recip_c2, "recip_c2");
    const auto& recip_c1_arg = require_one(recip_c1, "recip_c1");
    const auto& recip_c0_arg = require_one(recip_c0, "recip_c0");

    auto logits_quarter = mult_ringt_rescale(ctx, x[0], pt_quarter_arg);
    auto quarter_sum = repeated_block_sum(ctx, logits_quarter.copy(), n_classes_);
    auto mean_quarter = mult_ringt_rescale(ctx, quarter_sum, pt_inv_classes_arg);
    auto logits_quarter_lm1 = drop_levels(ctx, logits_quarter, 1);
    auto centered_quarter = ctx.sub(logits_quarter_lm1, mean_quarter);

    auto exp_logits = eval_exp_poly_v1(
        ctx, centered_quarter, exp_c5_arg, exp_c4_arg, exp_c3_arg, exp_c2_arg, exp_c1_arg, exp_c0_arg);
    auto denom = repeated_block_sum(ctx, exp_logits.copy(), n_classes_);
    auto inv_denom = eval_recip_poly_v1(ctx, denom, recip_c3_arg, recip_c2_arg, recip_c1_arg, recip_c0_arg);
    auto exp_logits_lm3 = drop_levels(ctx, exp_logits, 3);
    auto softmax = mult_relin_rescale(ctx, exp_logits_lm3, inv_denom);

    vector<CkksCiphertext> result;
    result.push_back(move(softmax));
    return result;
}

Feature0DEncrypted SoftmaxLayer::run(CkksContext& ctx, const Feature0DEncrypted& x) const {
    Feature0DEncrypted result(x.context, x.level);
    result.data = run_core(ctx, x.data);
    result.dim = x.dim;
    result.skip = x.skip;
    result.n_channel = x.n_channel;
    result.n_channel_per_ct = x.n_channel_per_ct;
    result.level = result.data.empty() ? x.level : result.data[0].get_level();
    result.ckks_scale = x.ckks_scale;
    result.multiplier = x.multiplier;
    return result;
}

Array<double, 1> SoftmaxLayer::run_plaintext(const Array<double, 1>& x) const {
    auto shape = x.get_shape();
    Array<double, 1> result(shape);
    if (x.get_size() == 0) {
        return result;
    }

    double max_value = x.get(0);
    for (int i = 1; i < x.get_size(); ++i) {
        max_value = max(max_value, x.get(i));
    }

    vector<double> exp_values(x.get_size(), 0.0);
    double exp_sum = 0.0;
    for (int i = 0; i < x.get_size(); ++i) {
        exp_values[i] = exp(x.get(i) - max_value);
        exp_sum += exp_values[i];
    }

    for (int i = 0; i < x.get_size(); ++i) {
        result.set(i, exp_values[i] / exp_sum);
    }
    return result;
}
