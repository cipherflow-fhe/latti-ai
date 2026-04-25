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
#include <utility>

using namespace std;
using namespace cxx_sdk_v2;

#if defined(LATTI_AI_USE_LATTISENSE_CKKS_SOFTMAX_CPU_KERNEL)
namespace cxx_sdk_v2 {
std::vector<CkksCiphertext> ckks_softmax_cpu(CkksContext& ctx, const std::vector<CkksCiphertext>& x);
}
#endif

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
}  // namespace

SoftmaxLayer::SoftmaxLayer(const CkksParameter& param_in, uint32_t n_classes, uint32_t input_level, KernelFn kernel)
    : Layer(param_in), kernel_(move(kernel)) {
    if (n_classes == 0 && input_level == 0) {
        return;
    }
    prepare_offline_args(n_classes, input_level);
}

void SoftmaxLayer::set_kernel(KernelFn kernel) {
    kernel_ = move(kernel);
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
        ctx.encode_ringt(tile_scalar(1.0 / static_cast<double>(n_classes)), param_.get_q(static_cast<int>(input_level) - 1)));

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
#if defined(LATTI_AI_USE_LATTISENSE_CKKS_SOFTMAX_CPU_KERNEL)
    return ckks_softmax_cpu(ctx, x);
#else
    if (!kernel_) {
        throw runtime_error(
            "SoftmaxLayer kernel is not set. Please bind the lattisense softmax kernel before calling run().");
    }
    return kernel_(ctx, x);
#endif
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
