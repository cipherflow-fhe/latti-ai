/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "complex_pack_layer.h"

#include <cmath>
#include <stdexcept>

using namespace std;
using namespace lattisense;

namespace {

void check_aligned(const vector<CkksCiphertext>& a, const vector<CkksCiphertext>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("ComplexPackLayer requires equally sized ciphertext vectors");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].get_level() != b[i].get_level()) {
            throw invalid_argument("ComplexPackLayer requires equal ciphertext levels");
        }
        if (!isfinite(a[i].get_scale()) || !isfinite(b[i].get_scale()) ||
            abs(a[i].get_scale() - b[i].get_scale()) > 1e-6 * max(a[i].get_scale(), b[i].get_scale())) {
            throw invalid_argument("ComplexPackLayer requires equal ciphertext scales");
        }
    }
}

template <typename Feature> void copy_common_metadata(const Feature& input, Feature& output, int level) {
    output.dim = input.dim;
    output.n_channel = input.n_channel;
    output.n_channel_per_ct = input.n_channel_per_ct;
    output.level = level;
}

}  // namespace

ComplexPackLayer::ComplexPackLayer(const CkksParameter& param_in) : Layer(param_in) {}

CkksCiphertext ComplexPackLayer::run(CkksContext& ctx, const CkksCiphertext& a, const CkksCiphertext& b) const {
    if (a.get_level() != b.get_level()) {
        throw invalid_argument("ComplexPackLayer requires equal ciphertext levels");
    }
    if (!isfinite(a.get_scale()) || !isfinite(b.get_scale()) ||
        abs(a.get_scale() - b.get_scale()) > 1e-6 * max(a.get_scale(), b.get_scale())) {
        throw invalid_argument("ComplexPackLayer requires equal ciphertext scales");
    }
    return ctx.add(a, ctx.mult_by_i(b));
}

vector<CkksCiphertext>
ComplexPackLayer::run(CkksContext& ctx, const vector<CkksCiphertext>& a, const vector<CkksCiphertext>& b) const {
    check_aligned(a, b);
    vector<CkksCiphertext> result(a.size());
    parallel_for(a.size(), th_nums, ctx, [&](CkksContext& ctx_copy, int index) {
        result[index] = ctx_copy.add(a[index], ctx_copy.mult_by_i(b[index]));
    });
    return result;
}

Feature0DEncrypted
ComplexPackLayer::run(CkksContext& ctx, const Feature0DEncrypted& a, const Feature0DEncrypted& b) const {
    auto data = run(ctx, a.data, b.data);
    Feature0DEncrypted result(&ctx, data.empty() ? 0 : data[0].get_level());
    copy_common_metadata(a, result, data.empty() ? 0 : data[0].get_level());
    result.skip = a.skip;
    result.pack_type = a.pack_type;
    result.data = move(data);
    return result;
}

Feature1DEncrypted
ComplexPackLayer::run(CkksContext& ctx, const Feature1DEncrypted& a, const Feature1DEncrypted& b) const {
    auto data = run(ctx, a.data, b.data);
    int level = data.empty() ? 0 : data[0].get_level();
    Feature1DEncrypted result(&ctx, level, a.skip, a.invalid_fill);
    copy_common_metadata(a, result, level);
    result.shape = a.shape;
    result.data = move(data);
    return result;
}

Feature2DEncrypted
ComplexPackLayer::run(CkksContext& ctx, const Feature2DEncrypted& a, const Feature2DEncrypted& b) const {
    auto data = run(ctx, a.data, b.data);
    int level = data.empty() ? 0 : data[0].get_level();
    Feature2DEncrypted result(&ctx, level, a.skip, a.invalid_fill, a.packing_type);
    copy_common_metadata(a, result, level);
    result.shape = a.shape;
    result.data = move(data);
    return result;
}

FeatureMatEncrypted
ComplexPackLayer::run(CkksContext& ctx, const FeatureMatEncrypted& a, const FeatureMatEncrypted& b) const {
    auto data = run(ctx, a.data, b.data);
    int level = data.empty() ? 0 : data[0].get_level();
    FeatureMatEncrypted result(&ctx, level);
    copy_common_metadata(a, result, level);
    result.shape = a.shape;
    result.head_shape = a.head_shape;
    result.matmul_block_size = a.matmul_block_size;
    result.data = move(data);
    return result;
}
