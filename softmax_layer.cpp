#include "softmax_layer.h"
#include "../util.h"
#include <cmath>
#include <utility>

using namespace std;

SoftmaxLayer::SoftmaxLayer(const ls::CkksParameter& param_in, uint32_t num_channels,
                           int level_in, double scale_in, uint32_t skip)
    : Layer(param_in), num_channels_(num_channels), skip_(skip),
      scale_(scale_in), level_(level_in) {
    n_slots_ = param_in.get_n() / 2;
    n_channel_per_ct_ = n_slots_ / skip_;

    exp_coeffs_ = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0};
    inv_coeffs_ = {9.999, -9.999, 9.999, -9.999};
}

ls::CkksCiphertext SoftmaxLayer::poly_exp(ls::CkksContext& ctx, const ls::CkksCiphertext& x) {
    auto x2_3 = ctx.mult(x, x);
    auto x2 = ctx.relinearize(x2_3);
    x2 = ctx.rescale(x2, 1e-20);

    auto x3_3 = ctx.mult(x2, x);
    auto x3 = ctx.relinearize(x3_3);
    x3 = ctx.rescale(x3, 1e-20);

    auto x4_3 = ctx.mult(x3, x);
    auto x4 = ctx.relinearize(x4_3);
    x4 = ctx.rescale(x4, 1e-20);

    auto x5_3 = ctx.mult(x4, x);
    auto x5 = ctx.relinearize(x5_3);
    x5 = ctx.rescale(x5, 1e-20);

    auto one_pt = ctx.encode({1.0}, level_, scale_);
    auto result = ctx.add_plain(x, one_pt);

    auto half_pt = ctx.encode({0.5}, level_, scale_);
    result = ctx.add(result, ctx.mult_plain(x2, half_pt));

    auto sixth_pt = ctx.encode({1.0/6.0}, level_, scale_);
    result = ctx.add(result, ctx.mult_plain(x3, sixth_pt));

    auto twenty4_pt = ctx.encode({1.0/24.0}, level_, scale_);
    result = ctx.add(result, ctx.mult_plain(x4, twenty4_pt));

    auto one20_pt = ctx.encode({1.0/120.0}, level_, scale_);
    result = ctx.add(result, ctx.mult_plain(x5, one20_pt));
    return result;
}

ls::CkksCiphertext SoftmaxLayer::poly_inv(ls::CkksContext& ctx, const ls::CkksCiphertext& x) {
    auto x2_3 = ctx.mult(x, x);
    auto x2 = ctx.relinearize(x2_3);
    x2 = ctx.rescale(x2, 1e-20);

    auto x3_3 = ctx.mult(x2, x);
    auto x3 = ctx.relinearize(x3_3);
    x3 = ctx.rescale(x3, 1e-20);

    auto c0_pt = ctx.encode({inv_coeffs_[0]}, level_, scale_);
    auto result = ctx.encrypt_asymmetric(c0_pt);

    auto c1_pt = ctx.encode({inv_coeffs_[1]}, level_, scale_);
    auto term1 = ctx.mult_plain(x, c1_pt);
    result = ctx.add(result, term1);

    auto c2_pt = ctx.encode({inv_coeffs_[2]}, level_, scale_);
    auto term2 = ctx.mult_plain(x2, c2_pt);
    result = ctx.add(result, term2);

    auto c3_pt = ctx.encode({inv_coeffs_[3]}, level_, scale_);
    auto term3 = ctx.mult_plain(x3, c3_pt);
    result = ctx.add(result, term3);
    return result;
}

ls::CkksCiphertext SoftmaxLayer::rotate_sum(ls::CkksContext& ctx, const ls::CkksCiphertext& ct,
                                             uint32_t step, uint32_t n_terms) {
    auto one_pt = ctx.encode({1.0}, level_, scale_);
    auto sum = ctx.mult_plain(ct, one_pt);

    for (uint32_t i = 1; i < n_terms; ++i) {
        auto rot = ctx.rotate(ct, i * step);
        sum = ctx.add(std::move(sum), rot);
    }
    return sum;
}

ls::CkksCiphertext SoftmaxLayer::broadcast(ls::CkksContext& ctx, const ls::CkksCiphertext& ct, uint32_t n_slots) {
    auto one_pt = ctx.encode({1.0}, level_, scale_);
    auto result = ctx.mult_plain(ct, one_pt);

    for (uint32_t i = 1; i < n_slots; ++i) {
        auto rot = ctx.rotate(ct, i);
        result = ctx.add(std::move(result), rot);
    }
    return result;
}

Feature0DEncrypted SoftmaxLayer::run(ls::CkksContext& ctx, const Feature0DEncrypted& x) {
    vector<ls::CkksCiphertext> exp_cts;
    exp_cts.reserve(x.data.size());
    for (const auto& ct : x.data) {
        exp_cts.push_back(poly_exp(ctx, ct));
    }

    auto total = std::move(exp_cts[0]);
    for (size_t i = 1; i < exp_cts.size(); ++i) {
        total = ctx.add(std::move(total), exp_cts[i]);
    }

    uint32_t n_effective = (num_channels_ + skip_ - 1) / skip_;
    auto sum = rotate_sum(ctx, total, skip_, n_effective);

    auto inv = poly_inv(ctx, sum);
    auto inv_bcast = broadcast(ctx, inv, n_slots_);

    vector<ls::CkksCiphertext> result_cts;
    result_cts.reserve(exp_cts.size());
    for (auto& exp_ct : exp_cts) {
        auto mul3 = ctx.mult(exp_ct, inv_bcast);
        auto mul = ctx.relinearize(mul3);
        mul = ctx.rescale(mul, 1e-20);
        result_cts.push_back(std::move(mul));
    }

    Feature0DEncrypted result(&ctx, level_);
    result.data = std::move(result_cts);
    result.n_channel = num_channels_;
    result.skip = skip_;
    result.n_channel_per_ct = n_channel_per_ct_;
    return result;
}
