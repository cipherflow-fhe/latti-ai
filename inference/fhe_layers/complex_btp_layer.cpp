/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "complex_btp_layer.h"
#include "complex_pack_layer.h"
#include "complex_unpack_layer.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace lattisense;

namespace {

constexpr double kHalf = 0.5;

template <typename Feature> void copy_common_feature_metadata(const Feature& input, Feature& output, int level) {
    output.dim = input.dim;
    output.n_channel = input.n_channel;
    output.n_channel_per_ct = input.n_channel_per_ct;
    output.level = level;
}

}  // namespace

ComplexBtpLayer::ComplexBtpLayer(const CkksParameter& param_in, bool use_complex_btp)
    : Layer(param_in), use_complex_btp_(use_complex_btp) {}

CkksPlaintextRingt ComplexBtpLayer::generate_half_pt(CkksContext& ctx, int input_level) const {
    if (input_level <= 0 || input_level > ctx.get_parameter().get_max_level()) {
        throw invalid_argument("ComplexBtpLayer half plaintext requested at an invalid input level");
    }
    const int n_slots = ctx.get_parameter().get_n() / 2;
    return ctx.encode_ringt(vector<double>(n_slots, kHalf), ctx.get_parameter().get_q(input_level));
}

CkksCiphertext ComplexBtpLayer::scale_by_half(CkksBtpContext& ctx, const CkksCiphertext& input) const {
    const int input_level = input.get_level();
    if (input_level <= 0) {
        throw invalid_argument("ComplexBtpLayer needs one level before bootstrapping to multiply by 1/2");
    }

    const double input_scale = input.get_scale();
    const int n_slots = ctx.get_parameter().get_n() / 2;
    vector<double> half(n_slots, kHalf);
    auto half_pt = ctx.encode_mul(half, input_level, input_scale);
    auto scaled = ctx.rescale(ctx.mult_plain_mul(input, half_pt), input_scale);

    if (scaled.get_level() != input_level - 1) {
        throw runtime_error("ComplexBtpLayer failed to consume one level for the 1/2 input scaling");
    }
    return scaled;
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>> ComplexBtpLayer::run(CkksBtpContext& ctx,
                                                                          const vector<CkksCiphertext>& packed) const {
    if (packed.empty()) {
        return {};
    }

    vector<CkksCiphertext> refreshed(packed.size());
    vector<CkksCiphertext> refreshed_conjugate(packed.size());
    parallel_for(packed.size(), th_nums, ctx, [&](CkksBtpContext& ctx_copy, int index) {
        // z_half = (a + i*b) / 2.  This is deliberately before BTP.
        auto z_half = scale_by_half(ctx_copy, packed[index]);
        refreshed[index] = ctx_copy.bootstrap(z_half);
        if (use_complex_btp_) {
            // One BTP refreshes the conjugate lane algebraically.
            refreshed_conjugate[index] = ctx_copy.conjugate(refreshed[index]);
        } else {
            // Baseline for benchmarking: refresh the conjugate lane with a
            // second BTP.  This has the same mathematical output but twice
            // the expensive bootstrap count.
            auto z_half_conjugate = ctx_copy.conjugate(z_half);
            refreshed_conjugate[index] = ctx_copy.bootstrap(z_half_conjugate);
        }
    });

    ComplexUnpackLayer unpack_layer(param_);
    return unpack_layer.run(ctx, refreshed, refreshed_conjugate);
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>>
ComplexBtpLayer::run(CkksBtpContext& ctx, const vector<CkksCiphertext>& a, const vector<CkksCiphertext>& b) const {
    ComplexPackLayer pack_layer(param_);
    return run(ctx, pack_layer.run(ctx, a, b));
}

pair<Feature0DEncrypted, Feature0DEncrypted> ComplexBtpLayer::run(CkksBtpContext& ctx,
                                                                  const Feature0DEncrypted& packed) const {
    auto [a, b] = run(ctx, packed.data);
    Feature0DEncrypted a_feature(&ctx, a.empty() ? 0 : a[0].get_level());
    Feature0DEncrypted b_feature(&ctx, b.empty() ? 0 : b[0].get_level());
    copy_common_feature_metadata(packed, a_feature, a.empty() ? 0 : a[0].get_level());
    copy_common_feature_metadata(packed, b_feature, b.empty() ? 0 : b[0].get_level());
    a_feature.skip = packed.skip;
    b_feature.skip = packed.skip;
    a_feature.pack_type = packed.pack_type;
    b_feature.pack_type = packed.pack_type;
    a_feature.data = move(a);
    b_feature.data = move(b);
    return {move(a_feature), move(b_feature)};
}

pair<Feature1DEncrypted, Feature1DEncrypted> ComplexBtpLayer::run(CkksBtpContext& ctx,
                                                                  const Feature1DEncrypted& packed) const {
    auto [a, b] = run(ctx, packed.data);
    const int output_level = a.empty() ? 0 : a[0].get_level();
    Feature1DEncrypted a_feature(&ctx, output_level, packed.skip, packed.invalid_fill);
    Feature1DEncrypted b_feature(&ctx, output_level, packed.skip, packed.invalid_fill);
    copy_common_feature_metadata(packed, a_feature, output_level);
    copy_common_feature_metadata(packed, b_feature, output_level);
    a_feature.shape = packed.shape;
    b_feature.shape = packed.shape;
    a_feature.data = move(a);
    b_feature.data = move(b);
    return {move(a_feature), move(b_feature)};
}

pair<Feature2DEncrypted, Feature2DEncrypted> ComplexBtpLayer::run(CkksBtpContext& ctx,
                                                                  const Feature2DEncrypted& packed) const {
    auto [a, b] = run(ctx, packed.data);
    const int output_level = a.empty() ? 0 : a[0].get_level();
    Feature2DEncrypted a_feature(&ctx, output_level, packed.skip, packed.invalid_fill, packed.packing_type);
    Feature2DEncrypted b_feature(&ctx, output_level, packed.skip, packed.invalid_fill, packed.packing_type);
    copy_common_feature_metadata(packed, a_feature, output_level);
    copy_common_feature_metadata(packed, b_feature, output_level);
    a_feature.shape = packed.shape;
    b_feature.shape = packed.shape;
    a_feature.data = move(a);
    b_feature.data = move(b);
    return {move(a_feature), move(b_feature)};
}

pair<FeatureMatEncrypted, FeatureMatEncrypted> ComplexBtpLayer::run(CkksBtpContext& ctx,
                                                                    const FeatureMatEncrypted& packed) const {
    auto [a, b] = run(ctx, packed.data);
    const int output_level = a.empty() ? 0 : a[0].get_level();
    FeatureMatEncrypted a_feature(&ctx, output_level);
    FeatureMatEncrypted b_feature(&ctx, output_level);
    copy_common_feature_metadata(packed, a_feature, output_level);
    copy_common_feature_metadata(packed, b_feature, output_level);
    a_feature.shape = packed.shape;
    a_feature.head_shape = packed.head_shape;
    a_feature.matmul_block_size = packed.matmul_block_size;
    b_feature.shape = packed.shape;
    b_feature.head_shape = packed.head_shape;
    b_feature.matmul_block_size = packed.matmul_block_size;
    a_feature.data = move(a);
    b_feature.data = move(b);
    return {move(a_feature), move(b_feature)};
}
