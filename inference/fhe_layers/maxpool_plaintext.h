// Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../fhe_api.h"
#include "layer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Public polynomial coefficients and layout masks, shared by both deployments.
inline ls::CkksPlaintext encode_maxpool_plaintext(ls::CkksContext& ctx, const json& desc) {
    const auto& param = ctx.parameter();
    const int slots = param.max_slots();
    const int level = desc.at("level").get<int>();
    const double value = desc.at("value").get<double>();
    if (level < 0 || level >= static_cast<int>(param.q().size()) || !std::isfinite(value))
        throw std::runtime_error("Invalid MaxPool plaintext descriptor");
    const double scale = desc.value("scale", double(param.q()[level]));
    if (!std::isfinite(scale) || scale <= 0)
        throw std::runtime_error("Invalid MaxPool encoding scale");
    const auto kind = desc.at("kind").get<std::string>();
    std::vector<double> values(slots, kind == "constant" ? value : 0.0);
    if (kind == "mask") {
        for (int index : desc.at("indices").get<std::vector<int>>()) {
            if (index < 0 || index >= slots) throw std::runtime_error("Invalid MaxPool mask index");
            values[index] = value;
        }
    } else if (kind == "prefix") {
        const int length = desc.at("length").get<int>();
        if (length < 0 || length > slots) throw std::runtime_error("Invalid MaxPool mask length");
        std::fill(values.begin(), values.begin() + length, value);
    } else if (kind != "constant") {
        throw std::runtime_error("Unknown MaxPool plaintext kind: " + kind);
    }
    return desc.value("encoding", std::string("ringt")) == "full_rns"
        ? fhe_api::encode(ctx, values, level, scale)
        : fhe_api::encode_ringt(ctx, values, scale);
}
