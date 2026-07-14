/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <utility>
#include <vector>

#include "layer.h"
#include "../data_structs/feature0d.h"
#include "../data_structs/feature1d.h"
#include "../data_structs/feature2d.h"
#include "../data_structs/feature_mat.h"

/**
 * Split z = a + i*b when the input uses the half-scale convention
 * z = (a + i*b)/2.  The outputs are recovered without a multiplication by 2:
 *
 *   a = z + conj(z)
 *   b = (z - conj(z))/i
 */
class ComplexUnpackLayer : public Layer {
public:
    explicit ComplexUnpackLayer(const ls::CkksParameter& param_in);

    std::pair<std::vector<ls::CkksCiphertext>, std::vector<ls::CkksCiphertext>>
    run(ls::CkksContext& ctx, const std::vector<ls::CkksCiphertext>& packed) const;

    // Variant used by the two-BTP baseline: the conjugate lane has already
    // been refreshed independently.
    std::pair<std::vector<ls::CkksCiphertext>, std::vector<ls::CkksCiphertext>>
    run(ls::CkksContext& ctx,
        const std::vector<ls::CkksCiphertext>& packed,
        const std::vector<ls::CkksCiphertext>& conjugate) const;

    std::pair<ls::CkksCiphertext, ls::CkksCiphertext> run(ls::CkksContext& ctx, const ls::CkksCiphertext& packed) const;

    std::pair<ls::CkksCiphertext, ls::CkksCiphertext>
    run(ls::CkksContext& ctx, const ls::CkksCiphertext& packed, const ls::CkksCiphertext& conjugate) const;

    std::pair<Feature0DEncrypted, Feature0DEncrypted> run(ls::CkksContext& ctx, const Feature0DEncrypted& packed) const;
    std::pair<Feature1DEncrypted, Feature1DEncrypted> run(ls::CkksContext& ctx, const Feature1DEncrypted& packed) const;
    std::pair<Feature2DEncrypted, Feature2DEncrypted> run(ls::CkksContext& ctx, const Feature2DEncrypted& packed) const;
    std::pair<FeatureMatEncrypted, FeatureMatEncrypted> run(ls::CkksContext& ctx,
                                                            const FeatureMatEncrypted& packed) const;
};
