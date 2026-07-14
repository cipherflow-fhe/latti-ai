/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "complex_unpack_layer.h"

#include <stdexcept>

using namespace std;
using namespace lattisense;

namespace {

template <typename Feature> void copy_common_metadata(const Feature& input, Feature& output, int level) {
    output.dim = input.dim;
    output.n_channel = input.n_channel;
    output.n_channel_per_ct = input.n_channel_per_ct;
    output.level = level;
}

}  // namespace

ComplexUnpackLayer::ComplexUnpackLayer(const CkksParameter& param_in) : Layer(param_in) {}

pair<CkksCiphertext, CkksCiphertext> ComplexUnpackLayer::run(CkksContext& ctx, const CkksCiphertext& packed) const {
    return run(ctx, packed, ctx.conjugate(packed));
}

pair<CkksCiphertext, CkksCiphertext>
ComplexUnpackLayer::run(CkksContext& ctx, const CkksCiphertext& packed, const CkksCiphertext& conjugate) const {
    auto real = ctx.add(packed, conjugate);
    auto imag = ctx.div_by_i(ctx.sub(packed, conjugate));
    return {move(real), move(imag)};
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>>
ComplexUnpackLayer::run(CkksContext& ctx, const vector<CkksCiphertext>& packed) const {
    vector<CkksCiphertext> real(packed.size());
    vector<CkksCiphertext> imag(packed.size());
    parallel_for(packed.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int index) {
        auto conjugate = ctx_copy.conjugate(packed[index]);
        auto result = run(ctx_copy, packed[index], conjugate);
        real[index] = move(result.first);
        imag[index] = move(result.second);
    });
    return {move(real), move(imag)};
}

pair<vector<CkksCiphertext>, vector<CkksCiphertext>>
ComplexUnpackLayer::run(CkksContext& ctx,
                        const vector<CkksCiphertext>& packed,
                        const vector<CkksCiphertext>& conjugate) const {
    if (packed.size() != conjugate.size()) {
        throw invalid_argument("ComplexUnpackLayer requires equally sized ciphertext vectors");
    }
    vector<CkksCiphertext> real(packed.size());
    vector<CkksCiphertext> imag(packed.size());
    parallel_for(packed.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int index) {
        auto result = run(ctx_copy, packed[index], conjugate[index]);
        real[index] = move(result.first);
        imag[index] = move(result.second);
    });
    return {move(real), move(imag)};
}

pair<Feature0DEncrypted, Feature0DEncrypted> ComplexUnpackLayer::run(CkksContext& ctx,
                                                                     const Feature0DEncrypted& packed) const {
    auto [real, imag] = run(ctx, packed.data);
    int level = real.empty() ? 0 : real[0].get_level();
    Feature0DEncrypted real_feature(&ctx, level);
    Feature0DEncrypted imag_feature(&ctx, level);
    copy_common_metadata(packed, real_feature, level);
    copy_common_metadata(packed, imag_feature, level);
    real_feature.skip = imag_feature.skip = packed.skip;
    real_feature.pack_type = imag_feature.pack_type = packed.pack_type;
    real_feature.data = move(real);
    imag_feature.data = move(imag);
    return {move(real_feature), move(imag_feature)};
}

pair<Feature1DEncrypted, Feature1DEncrypted> ComplexUnpackLayer::run(CkksContext& ctx,
                                                                     const Feature1DEncrypted& packed) const {
    auto [real, imag] = run(ctx, packed.data);
    int level = real.empty() ? 0 : real[0].get_level();
    Feature1DEncrypted real_feature(&ctx, level, packed.skip, packed.invalid_fill);
    Feature1DEncrypted imag_feature(&ctx, level, packed.skip, packed.invalid_fill);
    copy_common_metadata(packed, real_feature, level);
    copy_common_metadata(packed, imag_feature, level);
    real_feature.shape = imag_feature.shape = packed.shape;
    real_feature.data = move(real);
    imag_feature.data = move(imag);
    return {move(real_feature), move(imag_feature)};
}

pair<Feature2DEncrypted, Feature2DEncrypted> ComplexUnpackLayer::run(CkksContext& ctx,
                                                                     const Feature2DEncrypted& packed) const {
    auto [real, imag] = run(ctx, packed.data);
    int level = real.empty() ? 0 : real[0].get_level();
    Feature2DEncrypted real_feature(&ctx, level, packed.skip, packed.invalid_fill, packed.packing_type);
    Feature2DEncrypted imag_feature(&ctx, level, packed.skip, packed.invalid_fill, packed.packing_type);
    copy_common_metadata(packed, real_feature, level);
    copy_common_metadata(packed, imag_feature, level);
    real_feature.shape = imag_feature.shape = packed.shape;
    real_feature.data = move(real);
    imag_feature.data = move(imag);
    return {move(real_feature), move(imag_feature)};
}

pair<FeatureMatEncrypted, FeatureMatEncrypted> ComplexUnpackLayer::run(CkksContext& ctx,
                                                                       const FeatureMatEncrypted& packed) const {
    auto [real, imag] = run(ctx, packed.data);
    int level = real.empty() ? 0 : real[0].get_level();
    FeatureMatEncrypted real_feature(&ctx, level);
    FeatureMatEncrypted imag_feature(&ctx, level);
    copy_common_metadata(packed, real_feature, level);
    copy_common_metadata(packed, imag_feature, level);
    real_feature.shape = imag_feature.shape = packed.shape;
    real_feature.head_shape = imag_feature.head_shape = packed.head_shape;
    real_feature.matmul_block_size = imag_feature.matmul_block_size = packed.matmul_block_size;
    real_feature.data = move(real);
    imag_feature.data = move(imag);
    return {move(real_feature), move(imag_feature)};
}
